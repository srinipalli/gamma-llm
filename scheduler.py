import pymongo
from datetime import datetime, timedelta
from bson.objectid import ObjectId
import json
import schedule
import time
from dateutil import parser
from api import get_gemini_analysis
import logging
import re

# MongoDB connection settings
MONGO_URI = "mongodb+srv://mvishaalgokul8:IMTXb7QXknOIgFaw@infrahealth.vdxwhfq.mongodb.net/"
DB_NAME = "logs"
COLLECTION_NAME = "app"

# Output file paths
RAW_LOGS_FILE = "last_5_min_logs.json"
GEMINI_OUTPUT_FILE = "gemini_analysis.json"

logging.basicConfig(level=logging.INFO)

def extract_json_from_gemini_response(response: str) -> str:
    """
    Extract JSON content from Gemini response that may be wrapped in markdown code blocks
    """
    # Remove Markdown code block if present
    match = re.search(r'``````', response, re.DOTALL)
    if match:
        json_content = match.group(1).strip()
        return json_content
    
    # Fallback: Try to find the first JSON object in the string
    match = re.search(r'(\{.*\})', response, re.DOTALL)
    if match:
        return match.group(1)
    
    return response.strip()  # Return as is if nothing matches

def make_json_serializable(obj):
    """
    Recursively convert datetime and ObjectId fields to serializable formats.
    """
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat() + 'Z'
    elif isinstance(obj, ObjectId):
        return str(obj)
    else:
        return obj

def fetch_and_analyze_logs():
    try:
        # Connect to MongoDB
        client = pymongo.MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]

        iso_time = "2025-06-10T18:05:00Z"  # or any ISO string you want
        now = parser.isoparse(iso_time)
        five_min_ago = now - timedelta(minutes=5)
        start_oid = ObjectId.from_datetime(five_min_ago)
        end_oid = ObjectId.from_datetime(now)

        now_iso = now.isoformat(timespec='seconds') + "Z"
        five_min_ago_iso = five_min_ago.isoformat(timespec='seconds') + "Z"

        logging.info(f"Querying logs from {five_min_ago_iso} to {now_iso}")

        # Query logs
        logs_cursor = collection.find({
            "_id": {
                "$gte": start_oid,
                "$lte": end_oid
            },
        })

        logs = list(logs_cursor)
        # Convert all logs to JSON-serializable format
        logs = [make_json_serializable(log) for log in logs]

        # Save raw logs to file
        with open(RAW_LOGS_FILE, "w") as f:
            json.dump(logs, f, indent=2)

        logging.info(f"Extracted {len(logs)} logs to {RAW_LOGS_FILE}")

        all_analyses = []
        
        # Analyze each log
        for idx, log in enumerate(logs):
            # Build message from all key-value pairs except _id
            message_parts = []
            for key, value in log.items():
                if key != "_id":
                    message_parts.append(f"{key}: {value}")
            message = "\n".join(message_parts)

            if not message.strip():
                continue  # Skip empty

            prompt = f"""
You are an expert server Infrastructure health monitor. Given the following app log data, analyze and respond with only a JSON object having these keys:
- issue (short description of the issue)
- impact (potential impact of the issue)
- resolution (steps to resolve)
- commands (list of shell commands or config changes to fix it)

IMPORTANT: Respond only with a valid JSON object. Do not include markdown code blocks, backticks, or any other text.

Log data:
{json.dumps(log, indent=2)}
"""

            logging.info(f"Sending log #{idx+1} to Gemini...")
            gemini_response = get_gemini_analysis(prompt)
            logging.info(f"Raw Gemini response: {gemini_response!r}")

            # Extract JSON from markdown if present
            cleaned_response = extract_json_from_gemini_response(gemini_response)
            logging.info(f"Cleaned response: {cleaned_response!r}")

            try:
                parsed = json.loads(cleaned_response)
                # Ensure new format keys exist
                parsed["original_log_id"] = log["_id"]
                parsed["original_log"] = {k: v for k, v in log.items() if k != "_id"}
                all_analyses.append(parsed)
                logging.info(f"Successfully parsed JSON for log #{idx+1}")
            except json.JSONDecodeError as e:
                logging.warning(f"Invalid JSON from Gemini for log #{idx+1}: {e}")
                logging.warning(f"Cleaned response was: {cleaned_response}")
                all_analyses.append({
                    "original_log_id": log["_id"],
                    "error": "Gemini did not return valid JSON",
                    "raw_response": gemini_response,
                    "cleaned_response": cleaned_response,
                    "json_error": str(e)
                })

        # Save Gemini output
        with open(GEMINI_OUTPUT_FILE, "w") as f:
            json.dump(all_analyses, f, indent=2)

        logging.info(f"Saved Gemini analysis for {len(all_analyses)} logs to {GEMINI_OUTPUT_FILE}")

        # Save Gemini output to a different MongoDB database
        try:
            analysis_client = pymongo.MongoClient(MONGO_URI)
            analysis_db = analysis_client["llm_response"]
            analysis_collection = analysis_db["LogAnalysis"]
            if all_analyses:
                analysis_collection.insert_many(all_analyses)
                logging.info(f"Inserted {len(all_analyses)} analyses into llm_response.LogAnalysis")
        except Exception as e:
            logging.error(f"Error saving Gemini analyses to llm_response DB: {e}")
    except Exception as e:
        logging.error(f"Error during log fetch/analysis: {e}")

# Run once immediately
fetch_and_analyze_logs()

# Schedule every 5 minutes
schedule.every(5).minutes.do(fetch_and_analyze_logs)

logging.info("Scheduler started. Running every 5 minutes...")
try:
    while True:
        schedule.run_pending()
        time.sleep(1)
except KeyboardInterrupt:
    logging.info("Scheduler stopped.")
