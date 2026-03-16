from __future__ import annotations

from functools import lru_cache


def get_user_collection_name(user_id: int) -> str:
    return f"user_{user_id}_kb"


@lru_cache(maxsize=30)
def _build_rag_service(top_k: int, collection_name: str):
    # Lazy imports keep backend startup resilient if LLM deps are optional.
    from llm.get_res import rag_service
    from llm.llm import read_llm

    read_llm()
    return rag_service({"top_k": top_k}, collection_name=collection_name)


def get_chat_answer(question: str, top_k: int, user_id: int) -> str:
    service = _build_rag_service(top_k, get_user_collection_name(user_id))
    return service.get_response(question)
