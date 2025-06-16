# db.py
import os
import faiss
import numpy as np
from openai import AsyncOpenAI
from motor.motor_asyncio import AsyncIOMotorClient
from typing import List, Optional
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# ----------------------------
# 1) MongoDB Setup
# ----------------------------
MONGO_URI = os.getenv("MONGO_URI")
mongo_client = AsyncIOMotorClient(
    MONGO_URI,
    tls=True,
    tlsAllowInvalidCertificates=True
)
mongo_db = mongo_client["robot_db"]
logs_collection = mongo_db["logs"]
users_collection = mongo_db["users"]

# ----------------------------
# 2) FAISS Setup
# ----------------------------
FAISS_INDEX_PATH = "faiss_index.bin"
_index = None           # Our in-memory FAISS index
_embedded_docs = []     # Parallel list of text strings for each vector in _index

# ----------------------------
# 3) OpenAI Embedding
# ----------------------------
key = os.getenv("OPENAI_API_KEY")

async def get_embedding(text: str) -> List[float]:
    """
    Uses OpenAI to get the embedding for a given text.
    """
    client = AsyncOpenAI(api_key=key)
    response = await client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding  # list of floats

# ----------------------------
# 4) Helper to Add Embedding to FAISS
# ----------------------------
async def add_embedding_to_faiss(raw_text: str, embedding: List[float]):
    """
    Adds an *already-computed* embedding to the in-memory FAISS index,
    along with the corresponding text in _embedded_docs.
    """
    global _index, _embedded_docs

    emb_np = np.array([embedding], dtype="float32")  # shape (1, embed_dim)
    if _index is None:
        # Create a new index if not present
        embed_dim = emb_np.shape[1]
        _index = faiss.IndexFlatL2(embed_dim)
        print("[INFO] Created a new in-memory FAISS index.")

    # Add the vector
    _index.add(emb_np)

    # Keep the parallel text in memory
    _embedded_docs.append(raw_text)

    # Optionally persist the index to disk
    await save_faiss_index()

# ----------------------------
# 5) FAISS Init / Save / Retrieve
# ----------------------------
async def init_faiss_index():
    """
    On startup, rebuild the FAISS index from all Mongo logs that have embeddings,
    using the new schema (timestamp & message top-level, everything else under metadata).
    """
    global _index, _embedded_docs

    # 1) Fetch existing logs with embeddings
    logs_with_embeddings = await logs_collection.find(
        {"embedding": {"$exists": True}}
    ).to_list(length=None)

    if not logs_with_embeddings:
        print("[INFO] No existing logs with embeddings in Mongo. Skipping FAISS index build.")
        return

    print(f"[INFO] Found {len(logs_with_embeddings)} logs with embeddings. Rebuilding FAISS index...")

    embeddings = []
    texts = []
    for log_data in logs_with_embeddings:
        # raw embedding vector
        emb = log_data["embedding"]
        embeddings.append(emb)

        # new: pull robotId, module from metadata
        md       = log_data.get("metadata", {})
        robot_id = md.get("robotId", "")
        module   = md.get("module", "")
        message  = log_data.get("message", "")

        # reconstruct the same search text
        raw_text = f"{robot_id} | {module} | {message}"
        texts.append(raw_text)

    # Convert to NumPy array
    embeddings_np = np.array(embeddings, dtype="float32")
    embed_dim     = embeddings_np.shape[1]

    # Build a fresh index
    _index = faiss.IndexFlatL2(embed_dim)
    _index.add(embeddings_np)

    # Store the parallel texts
    _embedded_docs = texts

    # Persist to disk
    faiss.write_index(_index, FAISS_INDEX_PATH)
    print("[INFO] FAISS index built and saved to disk.")


async def save_faiss_index():
    """
    Writes the FAISS index to disk so we can reuse it across server restarts.
    """
    global _index
    if _index is not None:
        faiss.write_index(_index, FAISS_INDEX_PATH)
        # You might also want to store the _embedded_docs somewhere persistent
        print("[INFO] FAISS index saved to disk.")

async def retrieve_similar_docs(query: str, top_k: int = 2) -> List[str]:
    """
    Embeds the user query, searches FAISS for top_k matches, returns the doc texts.
    """
    global _index, _embedded_docs

    if _index is None:
        print("[WARN] No FAISS index in memory. Returning empty list.")
        return []

    query_emb = await get_embedding(query)
    query_vec = np.array([query_emb], dtype="float32")
    distances, indices = _index.search(query_vec, top_k)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if 0 <= idx < len(_embedded_docs):
            results.append(_embedded_docs[idx])

    return results

# ----------------------------
# 6) MongoDB Logs
# ----------------------------
async def store_log_entry(log_data: dict):
    """
    Inserts a log, writes embedding, schedules extraction in BG.
    """
    try:
        # --- core fields unchanged ---
        timestamp = log_data["timestamp"]
        message   = log_data["message"]
        md        = log_data.get("metadata", {})
        robot_id  = md.get("robotId", "")
        module    = md.get("module", "")

        raw_text  = f"{robot_id} | {module} | {message}"
        emb       = await get_embedding(raw_text)

        doc = {
            "timestamp": timestamp,
            "message":   message,
            "metadata":  md,
            "raw_text": raw_text,
            "embedding": emb
        }
        result = await logs_collection.insert_one(doc)
        await add_embedding_to_faiss(raw_text, emb)

        return str(result.inserted_id)

    except Exception as e:
        print(f"[ERROR] Could not store log: {e}")
        return None

