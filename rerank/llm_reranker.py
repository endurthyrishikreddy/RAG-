from openai import OpenAI
from google import genai
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
client = OpenAI()
gemini = genai.Client()

def llm_rerank(query, documents):
    """
    documents = list of dicts:
    [{"id": "...", "text": "...", "metadata": {...}}, ...]
    """
    prompt = f"""
You are a ranking model. Rank the following documents by relevance to the query.
Return a JSON list of objects: id and score (0–1).

Query: {query}

Documents:
{documents}
"""
    response = gemini.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        temperature=0
    )

    import json
    try:
        ranking = json.loads(response.choices[0].message.content)
    except Exception:
        print("LLM Formatting Error: Using fallback")
        return documents
    
    # Sort original documents by LLM score
    doc_scores = {item["id"]: item["score"] for item in ranking}
    reranked = sorted(documents, key=lambda d: doc_scores.get(d["id"], 0), reverse=True)

    return reranked
