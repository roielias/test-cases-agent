import os
import json
import faiss
import numpy as np
import requests
import argparse
from typing import List, Dict, Any, Callable, Optional
from dotenv import load_dotenv

load_dotenv()

# ---------- CONFIG ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_EMBEDDING_URL = os.getenv(
    "OPENAI_EMBEDDING_URL", "https://api.openai.com/v1/embeddings")
FAISS_DIR = os.getenv("FAISS_PATH", "./faiss_store")
INDEX_FILE = os.path.join(FAISS_DIR, "index.faiss")
METADATA_FILE = os.path.join(FAISS_DIR, "metadatas.json")
DOCS_FILE = os.path.join(FAISS_DIR, "docs.json")
# ----------------------------

# ---------- Helper: OpenAI embeddings (HTTP) ----------


def openai_get_embeddings(texts: List[str], model: str = OPENAI_EMBEDDING_MODEL) -> List[List[float]]:
    """Calls OpenAI embeddings endpoint per-text (simple, robust)."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set in env.")
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}",
               "Content-Type": "application/json"}
    out = []
    for text in texts:
        payload = {"model": model, "input": text}
        resp = requests.post(OPENAI_EMBEDDING_URL,
                             json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        j = resp.json()
        emb = j["data"][0]["embedding"]
        out.append(emb)
    return out

# ---------- Placeholder: ADK/Gemini embedding function ----------


def adk_embedding_fn_placeholder(texts: List[str]) -> List[List[float]]:
    """
    Replace this function with a real ADK/Gemini call.
    Example:
        def adk_embedding_fn(texts):
            return [adk_client.embed(text=t)["embedding"] for t in texts]
    """
    raise NotImplementedError(
        "Replace adk_embedding_fn_placeholder with your ADK/Gemini embedding function.")

# ---------- FaissIndexManager ----------


class FaissIndexManager:
    def __init__(self, embedding_fn: Optional[Callable[[List[str]], List[List[float]]]] = None):
        self.embedding_fn = embedding_fn or openai_get_embeddings
        os.makedirs(FAISS_DIR, exist_ok=True)
        self.index = None  # faiss index
        self.metadatas: List[Dict[str, Any]] = []
        self.docs: List[str] = []
        self.dim: Optional[int] = None
        # try to load existing on init
        self.load()

    def _create_index(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.dim = dim

    def save(self):
        if self.index is None:
            raise RuntimeError("No index to save.")
        faiss.write_index(self.index, INDEX_FILE)
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(self.metadatas, f, ensure_ascii=False, indent=2)
        with open(DOCS_FILE, "w", encoding="utf-8") as f:
            json.dump(self.docs, f, ensure_ascii=False, indent=2)

    def load(self) -> bool:
        if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE) and os.path.exists(DOCS_FILE):
            self.index = faiss.read_index(INDEX_FILE)
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                self.metadatas = json.load(f)
            with open(DOCS_FILE, "r", encoding="utf-8") as f:
                self.docs = json.load(f)
            self.dim = self.index.d
            return True
        return False

    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None, batch_size: int = 8):
        if metadatas is None:
            metadatas = [{} for _ in texts]
        assert len(texts) == len(metadatas)
        # compute embeddings in batches
        vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            emb = self.embedding_fn(batch)
            vectors.extend(emb)
        vectors = np.array(vectors).astype("float32")
        if self.index is None:
            self._create_index(vectors.shape[1])
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"Dimension mismatch: index dim {self.dim} vs vector dim {vectors.shape[1]}")
        self.index.add(vectors)
        self.metadatas.extend(metadatas)
        self.docs.extend(texts)
        self.save()

    def search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        if self.index is None:
            raise RuntimeError("Index is empty. Add texts first.")
        q_emb = np.array(self.embedding_fn([query])).astype("float32")
        D, I = self.index.search(q_emb, k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.metadatas):
                continue
            results.append({
                "index": int(idx),
                "score": float(dist),
                "metadata": self.metadatas[idx],
                "text": self.docs[idx]
            })
        return results

    def info(self) -> Dict[str, Any]:
        return {
            "index_exists": self.index is not None,
            "dim": self.dim,
            "n_items": len(self.docs),
            "faiss_path": FAISS_DIR
        }

# ---------- CLI ----------


def main():
    parser = argparse.ArgumentParser(prog="faiss_index_manager")
    sub = parser.add_subparsers(dest="cmd")

    # init (clear/create)
    p_init = sub.add_parser(
        "init", help="Create fresh index (will overwrite existing).")
    p_init.add_argument("--dim", type=int, default=None,
                        help="Dimension (optional, auto-detected if adding).")

    # add
    p_add = sub.add_parser("add", help="Add texts to index.")
    p_add.add_argument("--texts", nargs="+",
                       help="Texts to add (space separated).")
    p_add.add_argument("--meta", nargs="*",
                       help="JSON metadata per text (as strings).")

    # add-from-file
    p_file = sub.add_parser(
        "add-file", help="Add texts from a JSON lines or JSON array file.")
    p_file.add_argument(
        "file", help="path to file (json array of strings or JSONL with {'text':..., 'meta':...}).")

    # search
    p_search = sub.add_parser("search", help="Search query.")
    p_search.add_argument("query", help="Query text.")
    p_search.add_argument("--k", type=int, default=4, help="Top-k results.")

    # info
    p_info = sub.add_parser("info", help="Show index info.")

    # choose embedding backend
    parser.add_argument("--use-adk", action="store_true",
                        help="Use ADK embedding stub (you must implement adk_embedding_fn).")

    args = parser.parse_args()

    # pick embedding function
    embedding_fn = None
    if args.use_adk:
        # replace with real ADK function in your environment
        embedding_fn = adk_embedding_fn_placeholder
    else:
        embedding_fn = openai_get_embeddings

    mgr = FaissIndexManager(embedding_fn=embedding_fn)

    if args.cmd == "init":
        # overwrite existing: remove files
        if os.path.exists(INDEX_FILE):
            os.remove(INDEX_FILE)
        if os.path.exists(METADATA_FILE):
            os.remove(METADATA_FILE)
        if os.path.exists(DOCS_FILE):
            os.remove(DOCS_FILE)
        mgr = FaissIndexManager(embedding_fn=embedding_fn)
        print("Initialized fresh store at:", FAISS_DIR)
        return

    if args.cmd == "add":
        if not args.texts:
            print("No texts provided. Use --texts 'one' 'two' ...")
            return
        metas = []
        if args.meta:
            # try parse each meta as json
            for m in args.meta:
                try:
                    metas.append(json.loads(m))
                except Exception:
                    metas.append({"raw_meta": m})
        else:
            metas = [None] * len(args.texts)
        mgr.add_texts(args.texts, metadatas=metas)
        print(f"Added {len(args.texts)} texts. Total items: {len(mgr.docs)}")
        return

    if args.cmd == "add-file":
        path = args.file
        if not os.path.exists(path):
            print("File not found:", path)
            return
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read().strip()
            try:
                data = json.loads(txt)
                # JSON array of strings or objects
                if isinstance(data, list):
                    texts, metas = [], []
                    for it in data:
                        if isinstance(it, str):
                            texts.append(it)
                            metas.append({})
                        elif isinstance(it, dict) and "text" in it:
                            texts.append(it["text"])
                            metas.append(it.get("meta", {}))
                    mgr.add_texts(texts, metadatas=metas)
                    print(f"Added {len(texts)} texts from JSON array.")
                    return
            except Exception:
                # fallback to jsonl
                f.seek(0)
                texts, metas = [], []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        texts.append(obj.get("text", ""))
                        metas.append(obj.get("meta", {}))
                    except Exception:
                        texts.append(line)
                        metas.append({})
                if texts:
                    mgr.add_texts(texts, metadatas=metas)
                    print(f"Added {len(texts)} texts from file.")
                    return
        print("Couldn't parse file format. Provide JSON array or JSONL.")
        return

    if args.cmd == "search":
        res = mgr.search(args.query, k=args.k)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return

    if args.cmd == "info":
        print(json.dumps(mgr.info(), indent=2, ensure_ascii=False))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
