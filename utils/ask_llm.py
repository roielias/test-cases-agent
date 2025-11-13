import requests
from google.adk.agents import Agent
 
from dotenv import load_dotenv
import os

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
openai_url = os.getenv("OPENAI_URL")
 
# פונקציה ששולחת את השאלה ל-GPT ומחזירה תשובה
def ask_gpt(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {openai_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4o",  # או כל מודל אחר
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = requests.post(openai_url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"Error communicating with GPT: {e}"
    except KeyError:
        return f"Unexpected response format: {response.text}"
 
# יצירת הסוכן
my_agent = Agent(
    name="external_gpt_agent",
    description="Agent that queries GPT via API URL.",
    instruction="Answer user questions by sending them to GPT via API."
)
 
# הפונקציה שמטפלת בבקשות מהסוכן
def agent_handler(user_input: str) -> str:
    return ask_gpt(user_input)
 
# דוגמה לשימוש
if __name__ == "__main__":
    user_question = "מי נשיא ישראל הראשון?"
    answer = agent_handler(user_question)
    print(answer)
 