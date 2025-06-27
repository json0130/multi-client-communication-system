# GPT-powered topic extractor + word-cloud from topic counts
import json
import base64
import os
from ast import literal_eval        # fallback parser
from io import BytesIO
from typing import List, Dict
import re

from wordcloud import WordCloud


# ── helpers ─────────────────────────────────────────────────────────────
def _parse_topics(raw: str) -> List[Dict[str, int]]:
    """
    Accepts anything ChatGPT might return:
      • plain JSON
      • JSON inside ``` … ``` or ```json … ```
      • simple list ["Cars", "Food"]
    and converts it to [{"topic": str, "count": int}, …] with count ≥ 1.
    """
    raw = raw.strip()

    # 1) remove code-block fences  ``` … ```   or   ```json … ```
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)

    # 2) try strict JSON first
    try:
        data = json.loads(raw)
    except Exception:
        # 3) fallback to Python-literal list
        try:
            data = literal_eval(raw)
        except Exception:
            data = []

    # 4) normalise to list[dict]
    topics: List[Dict[str, int]] = []
    if isinstance(data, list):
        # a) already dicts with counts
        if all(isinstance(x, dict) and "topic" in x for x in data):
            for i, item in enumerate(data, 1):
                cnt = int(item.get("count", len(data) - i + 1) or 1)
                topics.append({"topic": item["topic"], "count": max(cnt, 1)})
        # b) plain list of strings
        elif all(isinstance(x, str) for x in data):
            step = len(data) or 1
            for i, t in enumerate(data):
                topics.append({"topic": t, "count": step - i})
    return topics


def _build_wordcloud(topics, *, save_to="wordcloud_latest.png") -> str:
    """
    Build a word-cloud PNG.
      • If `save_to` is not None, writes the PNG there (overwrites each call).
      • Always returns the Base-64 string for API use.
    """
    freqs = {d["topic"]: int(d["count"]) for d in topics if int(d["count"]) > 0}
    if not freqs:
        return ""

    wc = WordCloud(width=800, height=800, background_color="white") \
            .generate_from_frequencies(freqs)

    # PIL image (no matplotlib needed)
    img = wc.to_image()

    # 1)  Save to file
    if save_to:
        # ensure parent folders exist
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        img.save(save_to, format="PNG")
        print(f"🖼  Word-cloud saved to {os.path.abspath(save_to)}")

    # 2)  Return base-64
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── coroutine ───────────────────────────────────────────────────────────

async def generate_topics(
    logs_collection,
    gpt_client,
    *,
    min_logs: int = 10,
    max_chars: int = 8000,
) -> Dict[str, object]:
    """
    Returns {"topics": [ {topic, count}, … ], "wordcloud": <b64-png>}
    """

    # 1) ensure we have enough logs
    total = await logs_collection.count_documents({})
    if total < min_logs:
        raise ValueError(f"Need at least {min_logs} logs (found {total}).")

    # 2) fetch newest→oldest messages (truncate for token cost)
    records = (
        await logs_collection.find({}, {"message": 1, "_id": 0})
        .sort("_id", -1)
        .to_list(length=None)
    )
    convo = "\n".join(r["message"] for r in records)
    prompt_convo = convo[:max_chars]

    # 3) build prompt: ask for JSON list with counts, sorted
    prompt = (
        "From the user conversation below, identify the key recurring topics "
        "or interests.  Estimate how often each topic appears (relative count) "
        "and return ONLY a JSON array sorted by descending count, exactly in "
        'this format:\n[\n  {"topic": "Cars", "count": 38},\n  {"topic": "Travel", "count": 27}\n]\n\n'
        f"User conversation:\n{prompt_convo}"
    )

    if not gpt_client.is_available():
        raise RuntimeError("OpenAI client not initialised")

    response = gpt_client.client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        timeout=10,
    )
    raw = response.choices[0].message.content
    #print(f"GPT response: {raw}")  # debug log
    topics = _parse_topics(raw)
    #print(f"Parsed topics: {topics}")  # debug log

    # 4) word-cloud built from topic counts
    wordcloud_b64 = _build_wordcloud(topics)

    return {"topics": topics, "wordcloud": wordcloud_b64}
