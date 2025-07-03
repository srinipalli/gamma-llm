LLM
1.	api.py – has the llm api function
2.	scheduler.py – runs every 5 mins, gets last five mins log from db and send the error logs one by one to llm for analysis and saves the response back in db
3.	predictive_analyzer.py – runs every 6 hours, Gets the last 6 hours logs and the context from knowledge base in db and sends it to llm for getting failiure risk servers
4.	ddos_analyser.py – runs every 6 hours, Gets the last 5 mins network logs of each server and sends it to llm for processing and flagging potential ddos attacks
