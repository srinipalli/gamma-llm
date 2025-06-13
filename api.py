import google.generativeai as genai
import sys

# Credentials and model setup
GEMINI_API_KEY = "AIzaSyBBh6qma7uR8pJdBOEGHOu1HOTEsyb0Xks"

if not GEMINI_API_KEY:
    print("Error: Gemini API Key not provided or set for analyze_server_health.py.")
    sys.exit(1)

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')

def get_gemini_analysis(full_prompt_string: str) -> str:
    if not gemini_model:
        return "Gemini model was not initialized. Please check available models."
    if not full_prompt_string.strip():
        return "No prompt content available for analysis from Gemini."
    print("Sending prompt to Gemini for analysis...")
    try:
        response = gemini_model.generate_content(full_prompt_string)
        if hasattr(response, 'text'):
            return response.text
        else:
            return "Gemini did not return a text response. It might be due to content policy violations or an empty response."
    except Exception as e:
        return f"An error occurred during the Gemini API call: {e}"

def analyze_logs_with_gemini(logs: list[str]) -> str:
    """
    Accepts a list of logs or strings, builds prompt, calls Gemini, returns analysis.
    """
    # Build a single string with instructions + logs joined
    prompt_instructions = (
        "Respond ONLY with a JSON object with keys: 'summary', 'remediation', 'fix'. "
        "Do NOT include any extra text.\n\n"
    )
    combined_logs = "\n\n".join(logs)
    full_prompt = prompt_instructions + combined_logs

    # Call Gemini with the full prompt
    return get_gemini_analysis(full_prompt)
