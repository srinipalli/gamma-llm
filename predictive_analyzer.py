import os
import datetime
import time
import random 
from dotenv import load_dotenv

from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# --- 0. Configuration and Environment Setup ---
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not MONGO_URI:
    raise ValueError("MONGO_URI not found in .env file. Please set it.")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file. Please set it.")

# --- MongoDB Connection ---
try:
    mongo_client = MongoClient(MONGO_URI)
    mongo_client.admin.command('ping')
    print("MongoDB connection successful!")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    exit(1)

# Define database and collection names
DB_NAME = "logs" 
LOGS_COLLECTION = "server" # Collection containing your raw server logs/metrics (THIS IS READ-ONLY)
RAG_KB_COLLECTION = "rag_knowledge_base" # Your new collection for RAG data
PREDICTIVE_FLAGS_COLLECTION = "predictive_maintenance_flags" # Collection for current flags

db = mongo_client[DB_NAME]
logs_collection = db[LOGS_COLLECTION]
rag_kb_collection = db[RAG_KB_COLLECTION]
predictive_flags_collection = db[PREDICTIVE_FLAGS_COLLECTION]

# --- Initialize Embedding Model ---
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Embedding model 'all-MiniLM-L6-v2' loaded successfully.")
except Exception as e:
    print(f"Error loading embedding model: {e}")
    print("Please ensure you have an internet connection for the first run to download the model.")
    exit(1)

# --- Initialize LLM (Gemini) ---
try:
    genai.configure(api_key=GEMINI_API_KEY)
    llm_model = genai.GenerativeModel('gemini-1.5-flash') 
    print("Gemini LLM model 'gemini-1.5-flash' configured successfully.")
except Exception as e:
    print(f"Error configuring Gemini LLM: {e}")
    exit(1)

# --- Define Target Servers and Time Window Constants ---
TARGET_SERVERS = {
    "Dev-server1": {"server": "server1", "environment": "Dev"},
    "Prod-server1": {"server": "server1", "environment": "Prod"},
    "Prod-server2": {"server": "server2", "environment": "Prod"},
    "Prod-server3": {"server": "server3", "environment": "Prod"},
    "Prod-server4": {"server": "server4", "environment": "Prod"},
    "QA-server1": {"server": "server1", "environment": "QA"},
    "QA-server2": {"server": "server2", "environment": "QA"},
    "Stage-server1": {"server": "server1", "environment": "Stage"},
    "Stage-server2": {"server": "server2", "environment": "Stage"},
}

TIME_WINDOW_HOURS = 6 # This is the target window for *recent* logs, if available
# For static/old logs, we will fallback to a fixed number of latest logs.
LATEST_LOGS_FALLBACK_COUNT = 5 # If no logs in TIME_WINDOW_HOURS, get the latest N logs

ANALYSIS_INTERVAL_SECONDS = 3600 # 1 hour for actual deployment, 60 for quick testing


# --- Helper Functions ---

def generate_embedding(text: str) -> list[float]:
    """Generates a vector embedding for the given text."""
    return embedding_model.encode(text).tolist()

def get_current_utc_time() -> datetime.datetime:
    """Returns the current UTC datetime."""
    return datetime.datetime.now(datetime.timezone.utc)

def parse_llm_output(llm_text: str) -> dict:
    """
    Parses the LLM's natural language output into a structured dictionary.
    This is a basic parser; for more robust parsing, consider regex or
    asking the LLM to output JSON directly.
    """
    predicted_issue = "Unknown issue"
    preventive_actions = ["No specific actions suggested."]
    confidence_level = "Low" # Default to Low

    lines = llm_text.split('\n')
    for i, line in enumerate(lines):
        if "**Predicted Issue:**" in line:
            predicted_issue = line.replace("**Predicted Issue:**", "").strip()
        elif "**Preventive Actions:**" in line:
            actions_start_index = i + 1
            current_actions = []
            while actions_start_index < len(lines) and lines[actions_start_index].strip().startswith(("-", "*", "1.", "2.")):
                action = lines[actions_start_index].strip().lstrip("-*123456789. ").strip()
                if action: # Only add non-empty actions
                    current_actions.append(action)
                actions_start_index += 1
            if current_actions:
                preventive_actions = current_actions
        elif "**Confidence Level:**" in line:
            # Extract and normalize confidence level
            conf_str = line.replace("**Confidence Level:**", "").strip().lower()
            if "high" in conf_str:
                confidence_level = "High"
            elif "medium" in conf_str:
                confidence_level = "Medium"
            elif "low" in conf_str: # Explicitly check for low
                confidence_level = "Low"
            else: # Fallback for unexpected text
                confidence_level = "Unknown"

    return {
        "predicted_issue": predicted_issue,
        "preventive_actions": preventive_actions,
        "confidence_level": confidence_level
    }

# --- Data Fetching and Summarization from EXISTING logs ---
def get_recent_server_data(server_logical_id: str, time_window_hours: int, fallback_count: int) -> dict:
    """
    Fetches and summarizes recent server data from the 'logs -> server' collection
    for the specified server and time window.
    If no logs found in the window, it falls back to the latest 'fallback_count' logs.
    """
    current_time = get_current_utc_time()
    start_time_window = current_time - datetime.timedelta(hours=time_window_hours)

    server_info = TARGET_SERVERS[server_logical_id]

    # 1. Try to get logs within the defined time window (e.g., last 6 hours)
    recent_logs = list(logs_collection.find({
        "server": server_info["server"],
        "environment": server_info["environment"],
        "createdAt": {"$gte": start_time_window, "$lt": current_time}
    }).sort("createdAt", 1))

    analysis_scope_description = f"last {time_window_hours} hours."

    # 2. If no recent logs, fall back to the latest 'fallback_count' logs available (any age)
    if not recent_logs:
        print(f"No logs found for {server_logical_id} in the {analysis_scope_description}. Falling back to latest {fallback_count} logs.")
        # Retrieve the N latest logs regardless of timestamp
        recent_logs = list(logs_collection.find({
            "server": server_info["server"],
            "environment": server_info["environment"],
        }).sort("createdAt", -1).limit(fallback_count)) # Sort descending to get latest N

        if not recent_logs:
            print(f"No logs whatsoever found for {server_logical_id}.")
            return None # No data at all for this server

        analysis_scope_description = f"latest {fallback_count} available logs (any age, oldest: {recent_logs[-1]['createdAt'].isoformat()[:10]})"
        
    total_logs = len(recent_logs)
    if total_logs == 0: 
        return None

    # Calculate averages from fetched logs
    # Ensure values are converted to float safely, default to 0 if missing/invalid
    avg_cpu = sum(float(log.get('cpu_usage', 0)) for log in recent_logs) / total_logs
    avg_mem = sum(float(log.get('memory_usage', 0)) for log in recent_logs) / total_logs
    avg_disk = sum(float(log.get('disk_utilization', 0)) for log in recent_logs) / total_logs
    avg_temp = sum(float(log.get('cpu_temp', 0)) for log in recent_logs) / total_logs
    avg_cache_miss = sum(float(log.get('cache_miss_rate', 0)) for log in recent_logs) / total_logs
    
    health_statuses = list(set(log.get('server_health', 'Unknown') for log in recent_logs))
    # Collect specific error/critical messages
    important_log_messages = [log.get('log_message', '') for log in recent_logs if log.get('log_type') == 'error_log' or 'Critical' in log.get('server_health', '') or 'Warning' in log.get('server_health', '')]

    summary_parts = []
    
    summary_parts.append(f"CPU usage average: {avg_cpu:.1f}%")
    summary_parts.append(f"Memory usage average: {avg_mem:.1f}%")
    summary_parts.append(f"Disk utilization average: {avg_disk:.1f}%")
    summary_parts.append(f"CPU temperature average: {avg_temp:.1f}C")
    summary_parts.append(f"Cache miss rate average: {avg_cache_miss:.2f}")

    if len(health_statuses) > 1:
        summary_parts.append(f"Varying server health statuses: {', '.join(health_statuses)}")
    else:
        summary_parts.append(f"Server health consistently: '{health_statuses[0]}'")

    if important_log_messages:
        unique_log_messages = list(set(important_log_messages))[:3] # Max 3 unique messages
        summary_parts.append(f"Recent critical/warning log messages: {'; '.join(unique_log_messages)}")
    else:
        summary_parts.append("No critical or warning log messages found.")

    summary_of_recent_state = (f"Current state for {server_logical_id} ({server_info['environment']} environment) "
                               f"analyzing {analysis_scope_description}: " + ". ".join(summary_parts) + ".")
    
    # Provide a few raw recent health/log message snippets for the LLM
    # Use the 'fallback_count' as the number of snippets to provide for brevity/focus
    sample_logs_for_prompt = [f"{log['createdAt'].isoformat()} [{log.get('server_health', 'N/A')}] {log.get('log_message', 'No message')}" for log in recent_logs[-fallback_count:]]
    
    recent_metrics_snapshot = {
        "cpu_usage_avg": avg_cpu,
        "cpu_temp_avg": avg_temp,
        "memory_usage_avg": avg_mem,
        "disk_utilization_avg": avg_disk,
        "cache_miss_rate_avg": avg_cache_miss,
        "server_health_summary": ", ".join(health_statuses)
    }

    return {
        "server_id": server_logical_id,
        "environment": server_info["environment"],
        "server_name": server_info["server"],
        "analysis_timestamp": current_time,
        "summary": summary_of_recent_state,
        "detailed_recent_logs": sample_logs_for_prompt,
        "recent_metrics_snapshot": recent_metrics_snapshot
    }

# --- Main Predictive Maintenance Pipeline Logic ---
def run_predictive_maintenance_pipeline():
    print("\n--- Starting Predictive Maintenance Pipeline ---")
    
    # 1. Check if RAG KB is populated, otherwise prompt to run populate_rag_kb_unique.py
    if rag_kb_collection.count_documents({}) == 0:
        print("\n--- WARNING: RAG Knowledge Base is empty! ---")
        print("Please run 'populate_rag_kb_unique.py' script first to populate the knowledge base.")
        print("Skipping predictive analysis until KB is populated.")
        return # Exit if KB is not populated

    # Main loop for continuous operation
    while True:
        print(f"\n--- Running Analysis Cycle @ {get_current_utc_time().isoformat()} ---")
        
        # 1. Clear previous predictive flags (as per requirement: "only the new one exists")
        print(f"Clearing previous predictive flags from '{PREDICTIVE_FLAGS_COLLECTION}' collection...")
        try:
            predictive_flags_collection.delete_many({})
            print("Previous flags cleared.")
        except Exception as e:
            print(f"Error clearing flags: {e}. Proceeding anyway.")

        print(f"\n--- Analyzing server data for all target servers ---")
        
        flagged_servers_count = 0

        for server_logical_id, server_details in TARGET_SERVERS.items():
          
            current_server_state = get_recent_server_data(server_logical_id, TIME_WINDOW_HOURS, LATEST_LOGS_FALLBACK_COUNT)
            # Handle case where get_recent_server_data returns None (no logs found even with fallback)
            if not current_server_state or (not current_server_state.get("detailed_recent_logs") and not current_server_state.get("recent_metrics_snapshot")):
                 # print(f"Skipping analysis for {server_logical_id} due to insufficient recent data.") # Uncomment for debugging all skips
                 continue # Skip this server if no meaningful data

            # print(f"Current state summary for {server_logical_id}: {current_server_state['summary']}") # Optional: uncomment for verbose debugging of summary

            # 2b. Generate Query Embedding for current state
            query_embedding = generate_embedding(current_server_state["summary"])

            # 2c. Retrieve Similar Historical Patterns using Atlas Vector Search
            try:
                # IMPORTANT: We add a 'server_id' filter to the vector search to only retrieve relevant history for THIS server.
                atlas_search_results = rag_kb_collection.aggregate([
                    {
                        '$vectorSearch': {
                            'queryVector': query_embedding,
                            'path': 'embedding',
                            'numCandidates': 50,  # Number of documents to scan for similarity
                            'limit': 3,           # Number of top similar results to return for LLM context
                            'index': 'vector_index_on_embedding',
                            'filter': { "server_id": server_logical_id } # Filter to only search for this specific server's history
                        }
                    },
                    { # Project only relevant fields to save prompt token space
                        '$project': {
                            'summary_text': 1,
                            'potential_issue_description': 1,
                            'preventive_insight': 1,
                            'score': { '$meta': 'vectorSearchScore' } # Get the similarity score
                        }
                    }
                ])
                retrieved_historical_context = list(atlas_search_results)
                # print(f"Retrieved {len(retrieved_historical_context)} specific historical patterns for {server_logical_id}.") # Uncomment for debugging
            except Exception as e:
                print(f"Error during Atlas Vector Search for {server_logical_id}: {e}")
                retrieved_historical_context = [] 

            # 2d. Construct LLM Prompt
            prompt_sections = [
                "You are an expert infrastructure reliability engineer. Your task is to analyze server health, "
                "identify potential subtle issues, and suggest predictive maintenance actions based on current observations and historical context. "
                "Even if no explicit errors are present, look for patterns and trends that could indicate future problems. "
                "Focus on anticipating issues based on the provided data.",
                f"\n--- Current Server State for Server ID: {current_server_state['server_id']}",
                f"Environment: {current_server_state['environment']}",
                f"Analysis Timestamp: {current_server_state['analysis_timestamp'].isoformat()}",
                f"\nSummary of Recent Observations:\n\"{current_server_state['summary']}\"",
            ]
            
            if current_server_state['detailed_recent_logs']:
                prompt_sections.append("\nSelected Recent Log Snippets (showing health status, messages, metrics):\n" + "\n".join([f"- {log}" for log in current_server_state['detailed_recent_logs']]))
            
            prompt_sections.append("\nRecent Metrics Snapshot (Average/Key Values):\n" + "\n. ".join([f"- {k}: {v:.2f}" if isinstance(v, float) else f"- {k}: {v}" for k, v in current_server_state['recent_metrics_snapshot'].items()]))

            if retrieved_historical_context:
                prompt_sections.append("\n--- Relevant Historical Context (Similar Past Patterns for this Server):")
                for i, item in enumerate(retrieved_historical_context):
                    prompt_sections.append(f"{i+1}. Historical Pattern (Similarity Score: {item.get('score', 'N/A'):.2f}):")
                    prompt_sections.append(f"   - Pattern Description: {item['summary_text']}")
                    prompt_sections.append(f"   - Historically Led To: {item.get('potential_issue_description', 'Not provided')}")
                    prompt_sections.append(f"   - Common Preventive Action: {item.get('preventive_insight', 'Not provided')}")
            else:
                prompt_sections.append("\n--- No highly similar historical patterns found for this server in the RAG knowledge base. ---")
                prompt_sections.append("The LLM will rely solely on general knowledge and the current state to identify potential issues.")


            prompt_sections.append("\n---\nBased on the above information:")
            prompt_sections.append("1.  **Predicted Issue:** What specific future problem is '{server_id}' likely trending towards? Mention specific patterns or precursors from the current state that indicate this. Be concise.".format(**current_server_state))
            prompt_sections.append("2.  **Preventive Actions:** What 2-3 specific, actionable preventive maintenance steps should be taken *now* to mitigate or avoid this predicted issue? List them clearly, e.g., '- Check X', '- Optimize Y'.")
            prompt_sections.append("3.  **Confidence Level:** Rate your confidence in this prediction (High/Medium/Low).")
            # New instruction: If no significant risk is identified, explicitly state it.
            prompt_sections.append("If, based on the data and context, the server appears healthy and not at risk (e.g., matching a 'CLEARLY HEALTHY' pattern), please state 'Server is currently operating within optimal parameters with no identified risk.' and set Confidence Level to 'Low'.")

            llm_prompt = "\n".join(prompt_sections)
            # Uncomment to debug the prompt sent to LLM
            # print(f"\n--- Generated LLM Prompt for {server_logical_id} (for debugging) ---")
            # print(llm_prompt)
            # print("-------------------------------------------")

            # 2e. LLM Inference
            # print(f"Calling Gemini LLM for analysis for {server_logical_id}...") # Optional: uncomment for verbose debugging
            try:
                time.sleep(0.5) # Short delay to avoid hitting API rate limits if running many servers quickly
                response = llm_model.generate_content(llm_prompt)
                llm_analysis_text = response.text
                
                # 2f. Parse LLM Output and Store Flag - ONLY FOR AT-RISK SERVERS
                parsed_analysis = parse_llm_output(llm_analysis_text)
                
                # Decision to FLAG the server based on stricter criteria for "major risk":
                problem_keywords = ["exhaustion", "critical", "failure", "unresponsive", "instability", 
                                    "overload", "leak", "bottleneck", "degradation", "risk", "issue", 
                                    "imminent", "potential problem", "unstable", "concern", "high cpu", 
                                    "high memory", "disk full", "overheating"]
                
                is_problem_predicted = any(keyword in parsed_analysis["predicted_issue"].lower() for keyword in problem_keywords)
                is_explicitly_healthy_message = "optimal parameters with no identified risk" in parsed_analysis["predicted_issue"].lower()

                # Flag only if confidence is High and a problem is predicted AND not explicitly healthy
                if parsed_analysis["confidence_level"] == "High" and \
                   is_problem_predicted and \
                   not is_explicitly_healthy_message:
                    
                    print(f"\n--- Predictive Analysis Result for {server_logical_id} (FLAGGED) ---")
                    print(llm_analysis_text)
                    print("--------------------------------------------------")

                    flag_doc = {
                        "server_id": current_server_state["server_id"],
                        "environment": current_server_state["environment"],
                        "server_name": current_server_state["server_name"],
                        "prediction_timestamp": get_current_utc_time(),
                        "predicted_issue": parsed_analysis["predicted_issue"],
                        "preventive_actions": parsed_analysis["preventive_actions"],
                        "confidence": parsed_analysis["confidence_level"],
                        "raw_llm_output": llm_analysis_text,
                        "current_state_summary": current_server_state["summary"]
                    }
                    predictive_flags_collection.insert_one(flag_doc)
                    print(f"Predictive flag stored for {server_logical_id} with confidence: {parsed_analysis['confidence_level']}.")
                    flagged_servers_count += 1
                else:
                    # Explicitly print for non-flagged servers based on LLM output
                    # Differentiate between no risk identified and medium/low confidence issues
                    if is_explicitly_healthy_message:
                        print(f"Server {server_logical_id}: Operating optimally with no identified risk (LLM explicitly stated).")
                    elif parsed_analysis["confidence_level"] in ["Medium", "Low"] and is_problem_predicted:
                        print(f"Server {server_logical_id}: Potential risk identified with {parsed_analysis['confidence_level']} confidence. Predicted: {parsed_analysis['predicted_issue']}. Not flagged as major risk.")
                    else:
                        print(f"Server {server_logical_id}: Not flagged. {parsed_analysis['predicted_issue'] if parsed_analysis['predicted_issue'] else 'No significant risk identified.'}")
                    pass 

            except Exception as e:
                print(f"Error during LLM inference or analysis for {server_logical_id}: {e}")
                print(f"Skipping predictive flag for {server_logical_id} due to LLM error.")

        print(f"\n--- Analysis Cycle Finished. {flagged_servers_count} servers currently flagged as at-risk. ---")
        print(f"--- Next analysis in {ANALYSIS_INTERVAL_SECONDS} seconds ---")
        time.sleep(ANALYSIS_INTERVAL_SECONDS) # Wait for the next cycle


if __name__ == "__main__":
    run_predictive_maintenance_pipeline()