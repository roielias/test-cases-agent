import os
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
import faiss
import numpy as np
import redis
from pathlib import Path
from dotenv import load_dotenv
from google.adk.agents import Agent
from utils.ask_llm import ask_gpt
import redis



REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
FAISS_DIM = 1536
r = redis.from_url(REDIS_URL)
try:
    response = r.ping()
    print("Redis !", response)  
except redis.ConnectionError:
    print("Error: Could not connect to Redis.")
# ------------------------------
# Load scripts folder
# ------------------------------
def load_scripts(path="scripts"):
    scripts_path = Path(path)
    if not scripts_path.exists():
        raise FileNotFoundError("Scripts folder not found.")

    scripts = {}
    for f in scripts_path.glob("*.json"):
        with open(f, "r", encoding="utf-8") as file:
            scripts[f.stem] = json.load(file)

    return scripts

SCRIPTS = load_scripts()


# ------------------------------
# Embeddings (דרך GPT)
# ------------------------------

from utils.ask_llm import ask_gpt   # נשתמש ב־ask_gpt גם ל־embeddings


def compute_embedding(texts):
    """ משתמש ב־GPT דרך ask_gpt כדי לייצר embedding """

    emb_list = []

    for t in texts:
        prompt = f"Embed the following text into a numeric vector:\n\n{t}"

      
        result = ask_gpt(prompt)

     
        try:
            vector = json.loads(result)  
            emb_list.append(vector)

        except:
            # fallback וקטור אפס
            emb_list.append([0.0] * FAISS_DIM)

    return emb_list


# ------------------------------
# FAISS manager
# ------------------------------
class FaissManager:
    def __init__(self, dim=FAISS_DIM):
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []
        self.names = []

    def add_texts(self, names, texts):
        vectors = np.array(compute_embedding(texts), dtype="float32")
        self.index.add(vectors)
        self.texts.extend(texts)
        self.names.extend(names)

    def search(self, query, top_k=3):
        qv = np.array(compute_embedding([query]), dtype="float32")
        D, I = self.index.search(qv, top_k)
        results = []
        for idx in I[0]:
            if idx < len(self.texts):
                results.append({"name": self.names[idx], "text": self.texts[idx]})
        return results


# ------------------------------
# Initialize FAISS
# ------------------------------
faiss_mgr = FaissManager()
script_names = list(SCRIPTS.keys())
script_texts = [json.dumps(s, ensure_ascii=False) for s in SCRIPTS.values()]
faiss_mgr.add_texts(script_names, script_texts)


# ------------------------------
# Build prompt
# ------------------------------
def build_prompt(session_id, question, relevant):
    context = get_session_context(session_id)
    history = "\n".join([f"User: {q}\nAgent: {a}" for q, a in context])
    docs = "\n".join([item["text"] for item in relevant])

    return f"{history}\n\nRelevant scripts:\n{docs}\n\nUser question:\n{question}"


# ------------------------------
# Redis context
# ------------------------------
def get_session_context(sid):
    data = r.get(f"session:{sid}")
    return pickle.loads(data) if data else []


def save_session_context(sid, context):
    r.set(f"session:{sid}", pickle.dumps(context))


# ------------------------------
# ADK Agent
# ------------------------------
my_agent = Agent(
    name="rag_agent",
    description="RAG agent with FAISS and Redis",
    instruction="Answer with RAG context."
)


# ------------------------------
# Agent handler
# ------------------------------
# ------------------------------
# Agent handler (Updated with RAG debug)
# ------------------------------
def agent_handler(session_id: str, user_question: str) -> str:

    print("\n==============================")
    print(" שאלה חדשה:", user_question)
    print("==============================")

    # --- שלב 1: חיפוש FAISS ---
    relevant = faiss_mgr.search(user_question)

    if relevant:
        print(" FAISS HIT — נמצאו מסמכים רלוונטיים:")
        for i, item in enumerate(relevant):
            print(f"  [{i+1}] מקור: {item['name']}")
    else:
        print(" FAISS MISS — אין תוצאות FAISS")

    # --- שלב 2: שליפת קונטקסט מרדיס ---
    context = get_session_context(session_id)

    if context:
        print("📦 Redis HIT — יש קונטקסט מהיסטוריה:")
        print("תוכן:", context[-1])
        history_text = "\n".join([f"User: {q}\nAgent: {a}" for q, a in context])
    else:
        print(" Redis MISS — אין היסטוריה לשיחה")
        history_text = ""

    # --- בניית פרומפט חזק כדי שהמודל ישתמש בקונטקסט ---
    docs_text = "\n".join([item["text"] for item in relevant])

    prompt_text = f"""
אתה עוזר חכם שמקבל גם קונטקסט מהיסטוריית Redis וגם מסמכים מ-FAISS.
עליך להשתמש בקונטקסט אם הוא קשור לשאלה.

היסטוריית שיחה:
{history_text}

מסמכים רלוונטיים מ-FAISS:
{docs_text}

שאלה:
{user_question}

ענה בצורה ישירה תוך שימוש בחומר לעיל.
"""

    print("\ פרומפט שנשלח ל-LLM:")
    print(prompt_text)

    # --- שלב 3: קריאה ל־GPT ---
    answer = ask_gpt(prompt_text)

    # --- שלב 4: שמירת ההיסטוריה ברדיס ---
    context.append((user_question, answer))
    save_session_context(session_id, context)

    print("\ תשובת הסוכן:", answer)
    print("==============================\n")

    return answer



# ------------------------------
# Example run
# ------------------------------
if __name__ == "__main__":
    print(agent_handler("111", "מי נשיא ארה״ב?"))
    print(agent_handler("111", "ומי נשיא ישראל?"))
    print(agent_handler("221", "שלום מי אתה?"))
    print(agent_handler("221", "מה אמרתי לפני השאלה הזו?"))

