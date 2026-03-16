from anyio import to_thread
from fastapi import APIRouter, Header, HTTPException, status

from backend.app.core.database import AsyncSessionLocal
from backend.app.crued.chat import get_chat_answer
from backend.app.crued.user import verify_token
from backend.app.schemas.chat import ChatMessage, ChatRequest, ChatResponse
from backend.app.utils.user import parse_bearer_token


router = APIRouter()


@router.post("/completion", response_model=ChatResponse)
async def chat_completion(
    payload: ChatRequest,
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> ChatResponse:
    token = parse_bearer_token(authorization)
    async with AsyncSessionLocal() as db:
        user_id = await verify_token(token, db)

    try:
        reply = await to_thread.run_sync(get_chat_answer, payload.question, payload.top_k, user_id)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LLM generation failed: {exc}",
        ) from exc

    history = payload.history + [
        ChatMessage(role="user", content=payload.question),
        ChatMessage(role="assistant", content=reply),
    ]
    return ChatResponse(answer=reply, history=history)
