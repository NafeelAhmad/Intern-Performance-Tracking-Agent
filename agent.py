import os
import gspread
import requests
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

# Load environment variables
load_dotenv(override=True)

INTERN_TASKS_SPREADSHEET_ID = os.getenv("INTERN_TASKS_SPREADSHEET_ID", "").strip("'\" ")
HR_PERFORMANCE_SPREADSHEET_ID = os.getenv("HR_PERFORMANCE_SPREADSHEET_ID", "").strip("'\" ")
GOOGLE_CHAT_WEBHOOK_URL = os.getenv("GOOGLE_CHAT_WEBHOOK_URL", "").strip("'\" ")

# Ensure Google API key is set properly in environment
api_key = (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip("'\" \t\r\n")
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key


print("--- ENV CHECK ---")
if api_key:
    print(f"✅ Found API Key starting with: '{api_key[:8]}...' (Length: {len(api_key.strip())})")
else:
    print("❌ GOOGLE_API_KEY is NOT loaded. Check your .env file!")

print(f"Intern Tasks Sheet ID: {os.getenv('INTERN_TASKS_SPREADSHEET_ID')}")
print(f"HR Sheet ID: {os.getenv('HR_PERFORMANCE_SPREADSHEET_ID')}")
# Authenticate Google Sheets
gc = None
if os.path.exists("service_account.json"):
    gc = gspread.service_account(filename="service_account.json")

# --- TOOLS DEFINITION ---

@tool
def fetch_intern_tasks(week_number: int) -> str:
    """Reads the intern daily task sheet for a specific week number.
    Returns raw task logs submitted by interns for that week.
    """
    if not gc:
        return "Error: service_account.json file missing in root directory."
    try:
        sh = gc.open_by_key(INTERN_TASKS_SPREADSHEET_ID)
        worksheet = sh.worksheet(f"Week {week_number}")
        records = worksheet.get_all_records()
        return str(records)
    except Exception as e:
        return f"Error fetching tasks for Week {week_number}: {str(e)}"

@tool
def update_hr_performance_sheet(intern_name: str, week_number: int, rating: int, feedback: str) -> str:
    """Updates the HR Performance Sheet with score (1-10) and feedback for a week."""
    if not gc:
        return "Error: service_account.json file missing in root directory."
    try:
        sh = gc.open_by_key(HR_PERFORMANCE_SPREADSHEET_ID)
        worksheet = sh.worksheet("Performance Summary")
        worksheet.append_row([intern_name, f"Week {week_number}", rating, feedback])
        return f"Successfully updated performance sheet for {intern_name}."
    except Exception as e:
        return f"Error updating HR sheet: {str(e)}"

@tool
def send_google_chat_notification(message: str) -> str:
    """Sends a message to HR via Google Chat webhook when evaluation is complete."""
    if not GOOGLE_CHAT_WEBHOOK_URL:
        return "Webhook URL missing in .env file, skipping notification."
    
    payload = {"text": message}
    response = requests.post(GOOGLE_CHAT_WEBHOOK_URL, json=payload)
    if response.status_code == 200:
        return "Notification sent to HR successfully."
    return f"Failed to send notification: {response.text}"

# --- REASONING LOOP ---

SYSTEM_PROMPT = """You are an AI Engineering Manager Assistant. Your job is to evaluate intern weekly tasks, output a performance rating (1-10), update HR's sheet, and ping HR on Google Chat.

Evaluation Rubric (1 - 10):
- 9 to 10: Outstanding work. Completed core tasks, tackled stretch goals, showed high initiative.
- 7 to 8: Good/Solid work. Met core expectations on time with clear descriptions.
- 5 to 6: Average / Incomplete. Tasks delayed, low detail in daily logs, or minimal output.
- 1 to 4: Below expectations. Significant lack of activity or unaddressed blockers.

Workflow Strategy:
1. Call `fetch_intern_tasks` for the requested target week.
2. Evaluate tasks for EACH intern found in the records based on the rubric.
3. Call `update_hr_performance_sheet` for EACH intern with their rating and concise 2-sentence feedback.
4. Call `send_google_chat_notification` summarizing that performance evaluation for Week {week_number} is completed.
"""

def run_evaluation(week_number: int):
    tools = [fetch_intern_tasks, update_hr_performance_sheet, send_google_chat_notification]
    tools_by_name = {t.name: t for t in tools}
    
    # Initialize LLM and bind available tools
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        transport="rest",
        temperature=0
    )
    llm_with_tools = llm.bind_tools(tools)
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT.format(week_number=week_number)),
        HumanMessage(content=f"Evaluate all intern daily task entries for Week {week_number}, write ratings/feedback to HR sheet, and notify HR.")
    ]

    print(f"🚀 AI Agent started evaluating Week {week_number}...")

    # Agent ReAct Execution Loop
    while True:
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # Check if LLM requested tool calls
        if not response.tool_calls:
            print(f"\n✅ Agent Execution Completed: {response.content}")
            break

        # Execute requested tools dynamically
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f"🔧 Tool Executing: {tool_name}({tool_args})")
            
            tool_func = tools_by_name[tool_name]
            tool_output = tool_func.invoke(tool_args)
            
            # Feed tool output back as proper ToolMessage
            messages.append(
                ToolMessage(
                    content=str(tool_output),
                    tool_call_id=tool_call["id"]
                )
            )

if __name__ == "__main__":
    run_evaluation(week_number=1)