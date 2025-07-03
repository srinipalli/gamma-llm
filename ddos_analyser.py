import os
import datetime
import time
from dotenv import load_dotenv

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
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
LOGS_COLLECTION_NAME = "network" # Collection containing your raw network logs
PREDICTIONS_COLLECTION_NAME = "attack_predictions" # Collection for DDoS attack predictions/flags

db = mongo_client[DB_NAME]
logs_collection = db[LOGS_COLLECTION_NAME]
predictions_collection = db[PREDICTIONS_COLLECTION_NAME]

# --- Initialize LLM (Gemini) ---
def setup_gemini_model():
    """Configures the Gemini API client and returns an available GenerativeModel, prioritizing gemini-1.5-flash."""
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        print("Gemini API configured successfully.")
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")
        return None
    
    preferred_model_name = "gemini-1.5" 
    
    try:
        print(f"Attempting to use preferred Gemini model: {preferred_model_name}")
        model = genai.GenerativeModel(model_name=preferred_model_name)
        # Quick check if the model is usable for content generation
        # This will raise an error if the model isn't found or doesn't support generateContent
        list(model.generate_content("test", stream=True)) 
        print(f"Successfully loaded Gemini model: {preferred_model_name}")
        return model
    except Exception as e:
        print(f"Failed to load preferred model '{preferred_model_name}': {e}")
        print("Falling back to listing available models...")
        
        # Fallback: Dynamically find any available model
        model_name = None
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    if 'gemini-1.5-flash' in m.name: 
                        model_name = m.name
                        break
                    elif 'gemini-pro' in m.name and not model_name:
                        model_name = m.name
            
            if model_name:
                print(f"Using fallback Gemini model: {model_name}")
                return genai.GenerativeModel(model_name=model_name)
            else:
                print("No suitable Gemini model found that supports 'generateContent'.")
                print("Please check your API key, region, and Google Cloud project settings.")
                print("You might need to enable billing or specific APIs.")
                return None
        except Exception as e:
            print(f"Error listing models or creating model during fallback: {e}")
            return None

llm_model = setup_gemini_model()
if not llm_model:
    print("Exiting script due to LLM setup failure.")
    exit(1)


# --- Define Target Servers (Adapt this to your actual server/environment combinations) ---
# This dictionary helps map logical server IDs to their actual 'server' and 'environment' fields in MongoDB.
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

LATEST_LOGS_COUNT = 5 # Number of latest logs to fetch for analysis
ANALYSIS_INTERVAL_SECONDS = 3600 # 1 hour for actual deployment, 60 for quick testing

# --- Helper Functions ---

def get_current_utc_time() -> datetime.datetime:
    """Returns the current UTC datetime."""
    return datetime.datetime.now(datetime.timezone.utc)

def parse_llm_output_ddos(llm_text: str) -> dict:
    """
    Parses the LLM's natural language output for DDoS prediction into a structured dictionary.
    """
    prediction_summary = "No DDoS activity detected."
    confidence_level = "Low" 

    lines = llm_text.split('\n')
    for line in lines:
        if line.lower().startswith("no ddos activity detected:"):
            prediction_summary = line.strip()
            confidence_level = "Low" # If LLM explicitly says "No DDoS", confidence is low for an attack
            break # No need to parse further for other classifications
        elif line.lower().startswith("potential ddos activity detected:"):
            prediction_summary = line.strip()
            confidence_level = "Medium"
            break
        elif line.lower().startswith("high probability of ddos attack:"):
            prediction_summary = line.strip()
            confidence_level = "High"
            break

    return {
        "ddos_prediction": prediction_summary,
        "confidence": confidence_level
    }

# --- Data Fetching and Analysis ---
def get_latest_network_logs_for_server(server_logical_id: str, limit: int) -> list:
    """
    Retrieves the last 'limit' network log entries for a specific server from the 'network' collection.
    """
    server_info = TARGET_SERVERS.get(server_logical_id)
    if not server_info:
        print(f"Error: Server logical ID '{server_logical_id}' not found in TARGET_SERVERS configuration.")
        return []

    try:
        print(f"Fetching last {limit} network logs for server: {server_info['server']} in environment: {server_info['environment']}...")
        logs = logs_collection.find({
            "server": server_info["server"],
            "environment": server_info["environment"]
        }).sort("createdAt", -1).limit(limit) # Sort descending to get latest
        
        log_list = []
        for log in logs:
            if '_id' in log:
                log['_id'] = str(log['_id']) # Convert ObjectId to string
            log_list.append(log)
        
        if not log_list:
            print(f"No network logs found for {server_logical_id}.")
        return log_list
    except OperationFailure as e:
        print(f"Error fetching logs for {server_logical_id}: {e}")
        return []


def analyze_logs_with_gemini_ddos(model: genai.GenerativeModel, logs: list) -> dict:
    """
    Sends network logs to the Gemini model for DDoS attack analysis and returns the parsed prediction.
    """
    if not model or not logs:
        return {"ddos_prediction": "Analysis skipped due to missing model or logs.", "confidence": "Low"}

    # Format the logs into a readable string for the LLM, focusing on relevant network fields
    log_details = "\n".join([
        f"- Timestamp: {log.get('createdAt')}, Source IP: {log.get('source_ip', 'N/A')}, "
        f"Destination IP: {log.get('destination_ip', 'N/A')}, Bytes Sent: {log.get('bytes_sent', 'N/A')}, "
        f"Bytes Received: {log.get('bytes_received', 'N/A')}, Latency (ms): {log.get('latency_ms', 'N/A')}, "
        f"Throughput (mbps): {log.get('throughput_mbps', 'N/A')}, Status: {log.get('status', 'N/A')}, "
        f"Protocol: {log.get('protocol', 'N/A')}" 
        for log in logs
    ])

    prompt = f"""
    You are an expert cybersecurity analyst. Your task is to analyze the following recent network log entries for a server and determine if there are patterns indicating a Distributed Denial-of-Service (DDoS) attack.

    Pay close attention to these indicators:
    1.  **Traffic Volume Spikes:** A sudden, massive increase in 'bytes_sent' or 'bytes_received'.
    2.  **Source IP Anomalies:** An unusually high number of requests from a single 'source_ip' or a small, suspicious range of IPs, especially if they are new or unusual.
    3.  **Latency Increase:** Significant increase in 'latency_ms', indicating network congestion or server overload.
    4.  **Connection Status:** A high rate of failed connection 'status' codes (e.g., 4xx, 5xx errors for HTTP, or connection timeouts).
    5.  **Throughput Spikes:** Abnormally high 'throughput_mbps' that is not consistent with normal operations.
    6.  **Protocol Anomalies:** Unusual or excessive use of certain protocols.
    7. **Rate Limits/Denials**: Look for explicit denial messages or indicators of rate limiting being hit.

    Based on your analysis, provide a concise, one-sentence prediction. Your response must start with one of these exact classifications, followed by a brief explanation:
    - "No DDoS activity detected: [explanation]"
    - "Potential DDoS activity detected: [explanation]"
    - "High probability of DDoS attack: [explanation]"

    --- Network Logs for Analysis ---
    {log_details}
    """

    try:
        print("Sending network logs to Gemini for DDoS analysis...")
        response = llm_model.generate_content(prompt)
        llm_text = response.text.strip()
        print(f"Gemini analysis complete. Raw LLM response: {llm_text}")
        return parse_llm_output_ddos(llm_text)
    except Exception as e:
        print(f"An error occurred during Gemini API call for DDoS analysis: {e}")
        return {"ddos_prediction": f"Error during analysis: {e}", "confidence": "Low"}


def store_ddos_prediction_flag(server_info: dict, prediction_data: dict, analyzed_log_ids: list):
    """
    Stores the DDoS prediction result as a flag in the 'attack_predictions' collection.
    Only stores if a potential or high probability of DDoS is detected.
    """
    if prediction_data["confidence"] in ["High", "Medium"]:
        print(f"Storing DDoS prediction flag for {server_info['logical_id']}...")
        flag_doc = {
            "server_id": server_info["logical_id"],
            "server_name": server_info["server"],
            "environment": server_info["environment"],
            "prediction_timestamp": get_current_utc_time(),
            "ddos_prediction": prediction_data["ddos_prediction"],
            "confidence": prediction_data["confidence"],
            "analyzed_log_ids": analyzed_log_ids,
        }
        try:
            predictions_collection.insert_one(flag_doc)
            print(f"DDoS flag stored successfully for {server_info['logical_id']} with confidence: {prediction_data['confidence']}.")
        except OperationFailure as e:
            print(f"Error storing DDoS prediction flag for {server_info['logical_id']}: {e}")
    else:
        print(f"No significant DDoS risk detected for {server_info['logical_id']} (Confidence: {prediction_data['confidence']}). Not storing a flag.")


# --- Main DDoS Prediction Pipeline Logic ---
def run_ddos_prediction_pipeline():
    print("\n--- Starting DDoS Prediction Pipeline ---")
    
    # Main loop for continuous operation
    while True:
        print(f"\n--- Running DDoS Analysis Cycle @ {get_current_utc_time().isoformat()} ---")
        
        # 1. Clear previous DDoS flags (to ensure only current flags exist)
        print(f"Clearing previous DDoS prediction flags from '{PREDICTIONS_COLLECTION_NAME}' collection...")
        try:
            predictions_collection.delete_many({})
            print("Previous DDoS flags cleared.")
        except Exception as e:
            print(f"Error clearing DDoS flags: {e}. Proceeding anyway.")

        print(f"\n--- Analyzing network data for all target servers ---")
        
        flagged_servers_count = 0

        for logical_id, server_details in TARGET_SERVERS.items():
            print(f"\n--- Processing Server: {logical_id} ---")
            
            recent_logs = get_latest_network_logs_for_server(logical_id, LATEST_LOGS_COUNT)

            if not recent_logs:
                print(f"Skipping DDoS analysis for {logical_id} due to no recent network logs.")
                continue

            prediction_data = analyze_logs_with_gemini_ddos(llm_model, recent_logs)

            # Extract IDs of analyzed logs for traceability
            analyzed_log_ids = [log['_id'] for log in recent_logs if '_id' in log]

            server_info_for_flag = {
                "logical_id": logical_id,
                "server": server_details["server"],
                "environment": server_details["environment"]
            }
            store_ddos_prediction_flag(server_info_for_flag, prediction_data, analyzed_log_ids)

            if prediction_data["confidence"] in ["High", "Medium"]:
                flagged_servers_count += 1
            
            time.sleep(0.5) # Short delay to avoid hitting API rate limits

        print(f"\n--- DDoS Analysis Cycle Finished. {flagged_servers_count} servers currently flagged for potential/high DDoS risk. ---")
        print(f"--- Next analysis in {ANALYSIS_INTERVAL_SECONDS} seconds ---")
        time.sleep(ANALYSIS_INTERVAL_SECONDS) # Wait for the next cycle


if __name__ == "__main__":
    run_ddos_prediction_pipeline()