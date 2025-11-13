from typing import List, Dict, Any
from pydantic import BaseModel
import json
import time

# --- ADK IMPORTS ---
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from google.adk.models import Gemini

# ================= Data Models =================

class Criterion(BaseModel):
    id: str
    description: str
    weight: float

class Rubric(BaseModel):
    rubric_id: str
    criteria: List[Criterion]
    score_scale: Dict[str, float]

class ContextDoc(BaseModel):
    id: str
    source: str
    text: str

class EvalInput(BaseModel):
    session_id: str
    test_case_id: str
    test_case_text: str
    context_docs: List[ContextDoc]
    rubric: Rubric   # כאן נכנס כל רובריק דינמי
    request_id: str = None

# ================= Helper Functions =================

def normalize_raw_score(raw: float, rubric_scale: Dict[str, float]) -> float:
    min_s = rubric_scale.get("min", 0)
    max_s = rubric_scale.get("max", 5)
    if raw < min_s or raw > max_s:
        if 0 <= raw <= 100:
            return min_s + (raw / 100.0) * (max_s - min_s)
        return max(min(raw, max_s), min_s)
    return raw

def compute_final_score(scores: Dict[str, float], rubric: Rubric) -> Dict[str, Any]:
    total = 0.0
    max_possible = 0.0
    for c in rubric.criteria:
        w = c.weight
        s = scores.get(c.id, rubric.score_scale.get("min", 0))
        total += s * w
        max_possible += rubric.score_scale.get("max", 5) * w
    normalized = total / max_possible if max_possible > 0 else 0.0
    return {"final_score": total, "normalized_final": normalized}

def build_prompt(inp: EvalInput) -> str:
    rubric_json = inp.rubric.model_dump_json()  # Pydantic V2
    ctx = [{"id": d.id, "source": d.source, "text": d.text[:1200]}
           for d in inp.context_docs[:3]]
    prompt = (
        "You are an automated test evaluator. Given the Rubric, Context docs and Test Case, "
        "return a JSON object with per-criterion scores, comments and evidence.\n\n"
        f"Rubric: {rubric_json}\n\nContext (top-3): {json.dumps(ctx)}\n\nTest case: {inp.test_case_text}\n\n"
        "Return only JSON with keys: scores (list of {criterion_id, score, comment, evidence}), recommendations.\n"
        f"Score scale: min {inp.rubric.score_scale.get('min',0)} max {inp.rubric.score_scale.get('max',5)}."
    )
    return prompt

def parse_llm_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        first = text.find('{')
        last = text.rfind('}')
        if first != -1 and last != -1 and last > first:
            try:
                return json.loads(text[first:last+1])
            except Exception:
                pass
    return None

# ================= ADK Tool Function Definition =================
def evaluate_test_case_func(payload: Dict[str, Any], model_client: Gemini = None) -> Dict[str, Any]:
    start = time.time()
    try:
        inp = EvalInput.parse_obj(payload)
    except Exception as e:
        return {"status": "error", "error": "validation_failed", "details": str(e)}

    prompt = build_prompt(inp)

    if model_client is not None:
        llm_resp = model_client.chat(prompt=prompt, max_tokens=800)
        content = (
            llm_resp.get("content")
            or llm_resp.get("message")
            or llm_resp.get("choices", [{}])[0].get("message", {}).get("content", "")
        )
    else:
        import requests
        try:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                json={"model": "gpt-4o",
                      "messages": [{"role": "user", "content": prompt}]},
                headers={"Authorization": "Bearer <YOUR_API_KEY>"},
                timeout=20
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return {"status": "error", "error": "llm_call_failed", "details": str(e)}

    parsed = parse_llm_json(content)
    if parsed is None:
        return {"status": "error", "error": "llm_parse_failed", "raw_content": content}

    scores_map = {}
    for item in parsed.get("scores", []):
        cid = item.get("criterion_id")
        raw = item.get("score", 0)
        scores_map[cid] = normalize_raw_score(raw, inp.rubric.score_scale)

    final = compute_final_score(scores_map, inp.rubric)
    duration_ms = int((time.time() - start) * 1000)

    return {
        "status": "ok",
        "session_id": inp.session_id,
        "test_case_id": inp.test_case_id,
        "scores": parsed.get("scores", []),
        "final_score": final["final_score"],
        "normalized_final": final["normalized_final"],
        "recommendations": parsed.get("recommendations", []),
        "duration_ms": duration_ms
    }

# ================= ADK Tool Instance =================
evaluate_tool_instance: FunctionTool = FunctionTool(func=evaluate_test_case_func)

# ================= Agent wrapper =================
class EvalAgent(LlmAgent):
    name: str = "evaluator"
    tools: List[FunctionTool] = [evaluate_tool_instance]

# ================= Main =================
if __name__ == "__main__":
    from pprint import pprint

    print("--- Running Test Case Evaluation Tool Directly ---")

    # דוגמה לדינמיקה: אפשר לשים פה כל רובריק
    with open("rubric.json", "r", encoding="utf-8") as f:
        rubric_obj = Rubric.parse_obj(json.load(f))

    dummy_context = [
        {"id": "doc1", "source": "manual",
         "text": "This is a sample context document explaining how to sum numbers."},
        {"id": "doc2", "source": "guide",
         "text": "Additional information about test case evaluation and best practices."}
    ]

    dummy_input = EvalInput(
        session_id="sess123",
        test_case_id="tc001",
        test_case_text="Write a function that sums all numbers in a list.",
        context_docs=[ContextDoc.parse_obj(d) for d in dummy_context],
        rubric=rubric_obj,  # כאן נכנס כל רובריק שתטעין
        request_id="req001"
    )

    try:
        result = evaluate_test_case_func(dummy_input.model_dump(), model_client=None)
        pprint(result)
    except Exception as e:
        print(f"An error occurred during direct function call: {e}")
