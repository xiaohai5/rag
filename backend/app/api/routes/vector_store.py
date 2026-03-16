from functools import lru_cache

from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.database import get_db
from backend.app.crued.user import verify_token
from backend.app.crued.vector_store import create_document_record
from backend.app.schemas.vector_store import DocumentItem, UploadResponse
from backend.app.utils.user import parse_bearer_token
from llm.knowledge_base import KnowledgeBaseServce
from llm.llm import read_llm
from llm.load import load_file_to_document


router = APIRouter()


def _collection_name_for_user(user_id: int) -> str:
    return f"user_{user_id}_kb"


@lru_cache(maxsize=256)
def _get_kb_service(collection_name: str) -> KnowledgeBaseServce:
    # Lazy init avoids blocking backend startup/login path.
    read_llm()
    return KnowledgeBaseServce(collection_name=collection_name)


class _MemoryUploadFile:
    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    authorization: str | None = Header(default=None, alias="Authorization"),
    db: AsyncSession = Depends(get_db),
) -> UploadResponse:
    token = parse_bearer_token(authorization)
    user_id = await verify_token(token, db)

    file_bytes = await file.read()
    try:
        docs = load_file_to_document(_MemoryUploadFile(file.filename, file_bytes))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Document parse failed: {exc}",
        ) from exc

    if not docs:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document parse failed: empty document",
        )

    # Keep filename in metadata so list endpoint can reflect true vector-store state.
    for doc in docs:
        doc.metadata = dict(doc.metadata or {})
        doc.metadata["filename"] = file.filename
        doc.metadata["user_id"] = user_id

    try:
        kb_message = _get_kb_service(_collection_name_for_user(user_id)).upload_by_str(docs)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vector store ingest failed: {exc}",
        ) from exc

    await create_document_record(user_id=user_id, filename=file.filename, db=db)

    return UploadResponse(
        filename=file.filename,
        message=kb_message,
    )


@router.get("/documents", response_model=list[DocumentItem])
async def list_documents(
    authorization: str | None = Header(default=None, alias="Authorization"),
    db: AsyncSession = Depends(get_db),
) -> list[DocumentItem]:
    token = parse_bearer_token(authorization)
    user_id = await verify_token(token, db)
    filenames = _get_kb_service(_collection_name_for_user(user_id)).list_uploaded_filenames()
    return [
        DocumentItem(
            id=index + 1,
            filename=name,
            status="recorded",
        )
        for index, name in enumerate(filenames)
    ]
