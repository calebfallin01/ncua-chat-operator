from ncua_chatbot import NCUAChatbot
import asyncio
import json

async def test():
    chatbot = NCUAChatbot()
    cu_info = {"cu_name": "PenFed", "cu_number": "227"}
    financial_data = chatbot.query_financial_data("227")
    
    if financial_data:
        print("Found financial data in", len(financial_data.get("results", {})), "tables")
        
        if "fs220a_2024_12" in financial_data.get("results", {}):
            print("fs220a_2024_12 table exists")
            record = financial_data["results"]["fs220a_2024_12"][0]
            if "acct_661a" in record:
                print(f"Net income (acct_661a) = {record['acct_661a']}")
            else:
                print("acct_661a field not found in fs220a_2024_12")
                print("Available fields:", list(record.keys())[:10])
        else:
            print("fs220a_2024_12 table not found")
            print("Available tables:", list(financial_data.get("results", {}).keys()))
    
    # Test the interpretation
    answer = chatbot.interpret_results('What is PenFed\'s net income?', cu_info, financial_data)
    print("\nQUESTION: What is PenFed's net income?")
    print("ANSWER:", answer)

# Run the test
asyncio.run(test())
