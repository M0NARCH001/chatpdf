import os
import shutil
import logging
from datetime import datetime
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Header
from sqlmodel import Session
from pydantic import BaseModel
from typing import List
import uuid

logger = logging.getLogger(__name__)

MAX_FILE_SIZE_MB = 15
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

from app.core.ingestion import load_and_split_documents
from app.core.vectorstore import add_documents, list_collections, delete_collection
from app.core.chain import get_rag_chain
from app.core.memory import reset_memory
from app.evaluation.ragas_eval import run_evaluation, load_eval_history
from app.db.database import get_session
from app.core.session_manager import create_session, get_session_by_id, update_display_name, login_or_register
from app.core.group_manager import (
    create_group, join_group, leave_group, delete_group,
    get_my_groups, get_group_details, regenerate_join_code,
    upload_to_group, delete_group_document, query_group,
    get_group_chat_history
)
from app.models.group import GroupMember, Group
from app.models.session import AnonymousUser

router = APIRouter()

# Directory for temporarily storing uploaded files
UPLOAD_DIR = "./temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class ChatRequest(BaseModel):
    question: str
    session_id: str = "default_session"
    collection_id: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]


class SessionStartRequest(BaseModel):
    display_name: str
    password: str


class SessionRenameRequest(BaseModel):
    display_name: str


class GroupCreateRequest(BaseModel):
    name: str

class GroupJoinRequest(BaseModel):
    join_code: str

class GroupChatRequest(BaseModel):
    question: str
    session_id: str = None


@router.post("/session/start")
async def start_session(request: SessionStartRequest, db: Session = Depends(get_session)):
    """Login or register with display_name + password."""
    if len(request.display_name.strip()) < 2:
        raise HTTPException(status_code=422, detail="Display name must be at least 2 characters.")
    if len(request.password) < 8:
        raise HTTPException(status_code=422, detail="Password must be at least 8 characters.")
    try:
        user, error = login_or_register(request.display_name.strip(), request.password, db)
        if error:
            # 429 for the rate-limit lockout, 401 for a wrong password
            status = 429 if error.startswith("Too many") else 401
            raise HTTPException(status_code=status, detail=error)
        return {"session_id": user.session_id, "display_name": user.display_name}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/session/me")
async def get_current_session(x_session_id: str = Header(...), db: Session = Depends(get_session)):
    """Retrieve the current session by ID."""
    user = get_session_by_id(x_session_id, db)
    if not user:
        raise HTTPException(status_code=404, detail="Session expired, please enter your name again")
    return {"session_id": user.session_id, "display_name": user.display_name}

@router.put("/session/rename")
async def rename_session(request: SessionRenameRequest, x_session_id: str = Header(...), db: Session = Depends(get_session)):
    """Update the display name for the current session."""
    user = update_display_name(x_session_id, request.display_name, db)
    if not user:
        raise HTTPException(status_code=404, detail="Session expired, please enter your name again")
    return {"session_id": user.session_id, "display_name": user.display_name}


# --- GROUP ENDPOINTS ---

def require_session(x_session_id: str = Header(...), db: Session = Depends(get_session)):
    user = get_session_by_id(x_session_id, db)
    if not user:
        raise HTTPException(status_code=404, detail="Session expired")
    return user

@router.post("/groups")
async def create_group_endpoint(request: GroupCreateRequest, user=Depends(require_session), db: Session = Depends(get_session)):
    group = create_group(request.name, user.session_id, user.display_name, db)
    return {"group_id": group.id, "join_code": group.join_code, "name": group.name}

@router.get("/groups/mine")
async def get_my_groups_endpoint(user=Depends(require_session), db: Session = Depends(get_session)):
    return get_my_groups(user.session_id, db)

@router.get("/groups/{group_id}")
async def get_group_details_endpoint(group_id: str, user=Depends(require_session), db: Session = Depends(get_session)):
    return get_group_details(group_id, user.session_id, db)

@router.delete("/groups/{group_id}")
async def delete_group_endpoint_auth(group_id: str, user=Depends(require_session), db: Session = Depends(get_session)):
    delete_group(group_id, user.session_id, db)
    return {"message": "Group deleted."}

@router.post("/groups/{group_id}/regen-code")
async def regen_code_endpoint(group_id: str, user=Depends(require_session), db: Session = Depends(get_session)):
    new_code = regenerate_join_code(group_id, user.session_id, db)
    return {"new_join_code": new_code}

@router.post("/groups/join")
async def join_group_endpoint(request: GroupJoinRequest, user=Depends(require_session), db: Session = Depends(get_session)):
    group = join_group(request.join_code, user.session_id, user.display_name, db)
    return {"group_id": group.id, "name": group.name}

@router.delete("/groups/{group_id}/leave")
async def leave_group_endpoint(group_id: str, user=Depends(require_session), db: Session = Depends(get_session)):
    success = leave_group(group_id, user.session_id, db)
    if not success:
        raise HTTPException(status_code=400, detail="Could not leave group.")
    return {"message": "Left group."}

@router.post("/groups/{group_id}/upload")
async def upload_group_doc(group_id: str, files: List[UploadFile] = File(...), user=Depends(require_session), db: Session = Depends(get_session)):
    allowed_extensions = {".pdf", ".txt", ".docx", ".doc"}
    file_paths = []
    for file in files:
        original_name = os.path.basename(file.filename or "")
        ext = os.path.splitext(original_name)[1].lower()
        if ext not in allowed_extensions:
            raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'.")
        safe_filename = f"{uuid.uuid4().hex}_{original_name}"
        file_path = os.path.join(UPLOAD_DIR, safe_filename)
        written = 0
        with open(file_path, "wb") as buffer:
            while chunk := await file.read(1024 * 64):
                written += len(chunk)
                if written > MAX_FILE_SIZE_BYTES:
                    buffer.close()
                    os.remove(file_path)
                    raise HTTPException(status_code=413, detail=f"File '{original_name}' exceeds {MAX_FILE_SIZE_MB} MB limit.")
                buffer.write(chunk)
        file_paths.append((file_path, original_name))

    paths_only = [p for p, _ in file_paths]
    try:
        res = upload_to_group(group_id, user.session_id, user.display_name, paths_only, db)
        return res
    finally:
        for path, _ in file_paths:
            if os.path.exists(path):
                os.remove(path)

@router.delete("/groups/{group_id}/docs/{doc_id}")
async def delete_group_doc_endpoint(group_id: str, doc_id: str, user=Depends(require_session), db: Session = Depends(get_session)):
    success = delete_group_document(doc_id, user.session_id, db)
    if not success:
        raise HTTPException(status_code=404, detail="Doc not found or deletion failed.")
    return {"message": "Document deleted."}

@router.post("/groups/{group_id}/chat")
async def group_chat_endpoint(group_id: str, request: GroupChatRequest, user=Depends(require_session), db: Session = Depends(get_session)):
    res = query_group(group_id, user.session_id, request.question, db)
    return res

@router.get("/groups/{group_id}/chat/history")
async def get_group_chat_history_endpoint(group_id: str, user=Depends(require_session), db: Session = Depends(get_session)):
    history = get_group_chat_history(group_id, user.session_id, db)
    return {"history": history}

@router.get("/groups/invite/{join_code}")
async def get_invite_info(join_code: str, db: Session = Depends(get_session)):
    from sqlmodel import select
    from app.models.group import Group, GroupMember, GroupDocument
    stmt = select(Group).where(Group.join_code == join_code.upper(), Group.is_active == True)
    group = db.exec(stmt).first()
    if not group:
        raise HTTPException(status_code=404, detail="Group not found or inactive")
        
    m_count = len(db.exec(select(GroupMember).where(GroupMember.group_id == group.id)).all())
    d_count = len(db.exec(select(GroupDocument).where(GroupDocument.group_id == group.id)).all())
    return {
        "group_name": group.name,
        "member_count": m_count,
        "doc_count": d_count
    }


# --- END GROUP ENDPOINTS ---

@router.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Accept file(s), run ingestion pipeline, return collection ID."""
    allowed_extensions = {".pdf", ".txt", ".docx", ".doc"}

    file_paths = []

    for file in files:
        # Validate extension using only the original filename (ignore content-type, easy to spoof)
        original_name = os.path.basename(file.filename or "")
        ext = os.path.splitext(original_name)[1].lower()
        if ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{ext}'. Allowed: PDF, TXT, DOCX.",
            )

        # UUID-prefix prevents path traversal and concurrent-upload collisions
        safe_filename = f"{uuid.uuid4().hex}_{original_name}"
        file_path = os.path.join(UPLOAD_DIR, safe_filename)

        try:
            written = 0
            with open(file_path, "wb") as buffer:
                while chunk := await file.read(1024 * 64):  # 64 KB chunks
                    written += len(chunk)
                    if written > MAX_FILE_SIZE_BYTES:
                        buffer.close()
                        os.remove(file_path)
                        raise HTTPException(
                            status_code=413,
                            detail=f"File '{original_name}' exceeds {MAX_FILE_SIZE_MB} MB limit.",
                        )
                    buffer.write(chunk)
            file_paths.append((file_path, original_name))
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Failed to save upload '%s': %s", original_name, e)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save file {original_name}: {e}",
            )

    paths_only = [p for p, _ in file_paths]
    try:
        chunks = load_and_split_documents(paths_only)
        # Restore original filenames in metadata so sources display correctly
        name_map = {p: n for p, n in file_paths}
        for chunk in chunks:
            src = chunk.metadata.get("source", "")
            chunk.metadata["source_file"] = name_map.get(src, os.path.basename(src))

        collection_id = f"collection_{uuid.uuid4().hex[:8]}"
        added = add_documents(collection_id, chunks)

        return {
            "message": "Documents processed successfully",
            "collection_id": collection_id,
            "chunks_processed": len(chunks),
            "new_documents_added": added,
        }

    except Exception as e:
        logger.error("Error processing documents: %s", e)
        raise HTTPException(
            status_code=500, detail=f"Error processing documents: {e}"
        )
    finally:
        for path, _ in file_paths:
            if os.path.exists(path):
                os.remove(path)


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Accept question, construct RAG chain, return answer and sources."""
    try:
        chain = get_rag_chain(request.collection_id, request.session_id)
        response = chain.invoke({"question": request.question})

        answer = response.get("answer", "")
        source_docs = response.get("source_documents", [])

        sources = []
        for doc in source_docs:
            sources.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
            })

        return ChatResponse(answer=answer, sources=sources)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating response: {e}"
        )


@router.get("/collections")
async def get_collections():
    """List all document collections."""
    try:
        collections = list_collections()
        return {"collections": collections}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error listing collections: {e}"
        )


@router.delete("/collections/{collection_id}")
async def delete_collection_endpoint(collection_id: str):
    """Delete a collection."""
    try:
        delete_collection(collection_id)
        return {"message": f"Collection {collection_id} deleted."}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error deleting collection: {e}"
        )


@router.post("/memory/reset/{session_id}")
async def clear_memory(session_id: str):
    """Clear memory for a session."""
    reset_memory(session_id)
    return {"message": f"Memory cleared for session {session_id}"}


@router.post("/evaluate/{collection_id}")
async def evaluate_collection(collection_id: str):
    """Trigger RAGAS evaluation on a collection."""
    try:
        scores = await run_evaluation(collection_id)
        return {"message": "Evaluation completed", "scores": scores}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error running evaluation: {e}"
        )


@router.get("/evaluate/history")
async def get_evaluation_history():
    """Retrieve evaluation history log."""
    try:
        history = load_eval_history()
        return {"history": history}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting history: {e}"
        )
