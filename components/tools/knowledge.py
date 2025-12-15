"""
The knowledge tools library.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from difflib import SequenceMatcher
from pathlib import Path
import json

from components.common import function_tool
from components.utils import ToolRegistry
from typing import List, Dict, Any

def _vector_rank(
    query: str,
    faqs: List[Dict[str, Any]],
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Rank FAQ entries by similarity to a query using TF‑IDF cosine similarity plus a fuzzy title boost.

    The question text is weighted more heavily than the answer to reduce noise from long answers.
    A bigram TF‑IDF model scores matches on the combined (question, answer) text, then a
    SequenceMatcher ratio on the question/title is blended in to break ties and favor close
    question wording.
    
    :param query: The user’s search text.
    :type query: str
    :param faqs: List of FAQ dictionaries containing "question"/"title" and "answer".
    :type faqs: List[Dict[str, Any]]
    :param top_k: Number of top matches to return.
    :type top_k: int
    :return: The top_k FAQ dicts ordered by combined similarity.
    :rtype: List[Dict[str, Any]]
    """
    if not query:
        return faqs[:top_k]
    
    corpus: List[str] = []
    valid_indices: List[int] = []
    questions: List[str] = []
    for idx, faq in enumerate(faqs):
        question = str(faq.get("question") or faq.get("title") or "")
        answer = str(faq.get("answer") or "")
        text = (" ".join([question, question, answer])).strip()
        if text:
            valid_indices.append(idx)
            corpus.append(text)
            questions.append(question)

    if not corpus:
        return faqs[:top_k]
    
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    try:
        tfidf = vectorizer.fit_transform([query] + corpus)
    except ValueError:
        return faqs[:top_k]
    
    query_vec = tfidf[0:1]
    faq_vecs = tfidf[1:]
    cosine_similarities = linear_kernel(query_vec, faq_vecs).flatten()

    def _fuzzy_score(q: str, cand: str) -> float:
        return SequenceMatcher(None, q.lower(), cand.lower()).ratio()

    combined_scores = []
    for sim, idx, qtext in zip(cosine_similarities, valid_indices, questions):
        fuzzy = _fuzzy_score(query, qtext)
        combined = (0.65 * sim) + (0.35 * fuzzy)
        combined_scores.append((combined, idx))

    ranked = sorted(
        combined_scores,
        key=lambda item: item[0],
        reverse=True
    )
    top_indices = [idx for _, idx in ranked[:top_k]]
    return [faqs[i] for i in top_indices]

def _answer_from_file(query: str, path: str, top_k: int=3) -> str:
    data = Path(path).read_text(encoding="utf-8")
    ranked = _vector_rank(query.strip(), json.loads(data), top_k)
    if not ranked:
        return "Wala po akong sagot diyan."
    if not query.strip():
        return "Pakilinaw po ng tanong para mahanap ko ang sagot."
    return str(ranked[0].get("answer") or "Wala po akong sagot diyan.")

@function_tool
def faq_tool(query: str) -> str:
    return _answer_from_file(query, "data/faqs.json", 1)

@function_tool
def mechanic_tool(query: str) -> str:
    return _answer_from_file(query, "data/mechanic_knowledge_base.json", 1)

ToolRegistry.register_tool(
    "knowledge.faq_tool",
    faq_tool,
    category="knowledge",
    description="Answers MechaniGo FAQs using TFIDF."
)

ToolRegistry.register_tool(
    "knowledge.mechanic_tool",
    mechanic_tool,
    category="knowledge",
    description="Answers automotive related inquiries."
)