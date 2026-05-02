import secrets
import logging
from typing import List, Dict, Any
from sqlmodel import Session, select
from fastapi import HTTPException
import os
import json

logger = logging.getLogger(__name__)

from app.models.group import Group, GroupMember, GroupDocument
from app.models.chat_log import ChatLog
from app.core.vectorstore import get_vectorstore, delete_collection, add_documents
from app.core.ingestion import load_and_split_documents
from app.core.chain import get_rag_chain

def create_group(name: str, session_id: str, display_name: str, db: Session) -> Group:
    new_group = Group(
        name=name,
        creator_session_id=session_id,
        creator_display_name=display_name
    )
    db.add(new_group)
    db.commit()
    db.refresh(new_group)
    
    member = GroupMember(
        group_id=new_group.id,
        session_id=session_id,
        display_name=display_name,
        role="creator"
    )
    db.add(member)
    db.commit()
    return new_group

def join_group(join_code: str, session_id: str, display_name: str, db: Session) -> Group:
    statement = select(Group).where(Group.join_code == join_code.upper(), Group.is_active == True)
    group = db.exec(statement).first()
    if not group:
        raise HTTPException(status_code=404, detail="Group not found or inactive.")
    
    member_stmt = select(GroupMember).where(GroupMember.group_id == group.id, GroupMember.session_id == session_id)
    if db.exec(member_stmt).first():
        raise HTTPException(status_code=400, detail="Already a member of this group.")
        
    member = GroupMember(
        group_id=group.id,
        session_id=session_id,
        display_name=display_name,
        role="member"
    )
    db.add(member)
    db.commit()
    db.refresh(group)
    return group

def leave_group(group_id: str, session_id: str, db: Session) -> bool:
    stmt = select(GroupMember).where(GroupMember.group_id == group_id, GroupMember.session_id == session_id)
    member = db.exec(stmt).first()
    if not member:
        return False
        
    group = db.get(Group, group_id)
    role = member.role
    db.delete(member)
    
    doc_stmt = select(GroupDocument).where(GroupDocument.group_id == group_id, GroupDocument.session_id == session_id)
    docs = db.exec(doc_stmt).all()
    
    col_name = f"group_{group_id}"
    vs = get_vectorstore(col_name)
    try:
        vs.delete(where={"session_id": session_id})
    except Exception:
        pass
        
    for d in docs:
        db.delete(d)
    db.commit()
    
    remaining_members = db.exec(
        select(GroupMember).where(GroupMember.group_id == group_id).order_by(GroupMember.joined_at)
    ).all()
    
    if len(remaining_members) == 0:
        if group:
            group.is_active = False
            db.add(group)
        delete_collection(col_name)
    elif role == "creator":
        oldest_member = remaining_members[0]
        oldest_member.role = "creator"
        if group:
            group.creator_session_id = oldest_member.session_id
            group.creator_display_name = oldest_member.display_name
            db.add(group)
            db.add(oldest_member)
            
    db.commit()
    return True

def delete_group(group_id: str, session_id: str, db: Session) -> bool:
    group = db.get(Group, group_id)
    if not group or group.creator_session_id != session_id:
        raise HTTPException(status_code=403, detail="Only creator can delete this group.")
        
    group.is_active = False
    db.add(group)
    
    docs = db.exec(select(GroupDocument).where(GroupDocument.group_id == group_id)).all()
    for d in docs:
        db.delete(d)
    db.commit()
    
    delete_collection(f"group_{group_id}")
    return True

def get_my_groups(session_id: str, db: Session) -> List[Dict[str, Any]]:
    stmt = select(GroupMember, Group).join(Group).where(GroupMember.session_id == session_id, Group.is_active == True)
    results = db.exec(stmt).all()
    
    summary = []
    for member, group in results:
        m_count = len(db.exec(select(GroupMember).where(GroupMember.group_id == group.id)).all())
        d_count = len(db.exec(select(GroupDocument).where(GroupDocument.group_id == group.id)).all())
        summary.append({
            "group_id": group.id,
            "name": group.name,
            "join_code": group.join_code,
            "member_count": m_count,
            "doc_count": d_count,
            "your_role": member.role
        })
    return summary

def get_group_details(group_id: str, session_id: str, db: Session) -> Dict[str, Any]:
    stmt = select(GroupMember).where(GroupMember.group_id == group_id, GroupMember.session_id == session_id)
    member = db.exec(stmt).first()
    if not member:
        raise HTTPException(status_code=403, detail="Not a member of this group.")
        
    group = db.get(Group, group_id)
    if not group or not group.is_active:
        raise HTTPException(status_code=404, detail="Group not found.")
        
    all_members = db.exec(select(GroupMember).where(GroupMember.group_id == group_id)).all()
    all_docs = db.exec(select(GroupDocument).where(GroupDocument.group_id == group_id)).all()
    
    m_list = [{
        "session_id": m.session_id,
        "display_name": m.display_name,
        "role": m.role,
        "docs_contributed": len([d for d in all_docs if d.session_id == m.session_id]),
        "joined_at": m.joined_at
    } for m in all_members]
        
    d_list = [{
        "id": d.id,
        "filename": d.filename,
        "uploader_name": d.display_name,
        "uploader_session_id": d.session_id,
        "chunk_count": d.chunk_count,
        "file_size_kb": d.file_size_kb,
        "uploaded_at": d.uploaded_at
    } for d in all_docs]
        
    return {
        "group_id": group.id,
        "name": group.name,
        "join_code": group.join_code,
        "creator": group.creator_display_name,
        "members": m_list,
        "documents": d_list
    }

def regenerate_join_code(group_id: str, session_id: str, db: Session) -> str:
    group = db.get(Group, group_id)
    if not group or group.creator_session_id != session_id:
        raise HTTPException(status_code=403, detail="Only creator can regenerate code.")
        
    new_code = secrets.token_hex(3).upper()
    group.join_code = new_code
    db.add(group)
    db.commit()
    return new_code

def upload_to_group(group_id: str, session_id: str, display_name: str, file_paths: List[str], db: Session) -> Dict[str, Any]:
    # Verify membership
    stmt = select(GroupMember).where(GroupMember.group_id == group_id, GroupMember.session_id == session_id)
    if not db.exec(stmt).first():
        raise HTTPException(status_code=403, detail="Not a member of this group.")

    # Ingest docs
    chunks = load_and_split_documents(file_paths)
    
    # Inject metadata
    for i, chunk in enumerate(chunks):
        source_path = chunk.metadata.get("source", "")
        filename = os.path.basename(source_path)
        chunk.metadata.update({
            "session_id": session_id,
            "display_name": display_name,
            "group_id": group_id,
            "filename": filename,
            "chunk_index": i
        })
        
    col_name = f"group_{group_id}"
    add_documents(col_name, chunks)
    
    docs_added = []
    for path in file_paths:
        filename = os.path.basename(path)
        file_size_kb = os.path.getsize(path) / 1024.0
        file_chunks = [c for c in chunks if c.metadata.get("filename") == filename]
        
        doc = GroupDocument(
            group_id=group_id,
            session_id=session_id,
            display_name=display_name,
            filename=filename,
            chunk_count=len(file_chunks),
            file_size_kb=file_size_kb,
        )
        db.add(doc)
        docs_added.append(doc)
        
    db.commit()
    return {"message": "Files uploaded successfully", "docs": [d.filename for d in docs_added]}

def delete_group_document(doc_id: str, session_id: str, db: Session) -> bool:
    doc = db.get(GroupDocument, doc_id)
    if not doc:
        return False
        
    group = db.get(Group, doc.group_id)
    if doc.session_id != session_id and group.creator_session_id != session_id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this document.")
        
    col_name = f"group_{doc.group_id}"
    vs = get_vectorstore(col_name)
    try:
        # Note: ChromaDB's local delete with multiple where clauses can be finicky.
        vs.delete(where={"$and": [{"session_id": doc.session_id}, {"filename": doc.filename}]})
    except Exception as e:
        logger.warning("Chroma delete warning for doc '%s': %s", doc.filename, e)

    db.delete(doc)
    db.commit()
    return True

def query_group(group_id: str, session_id: str, question: str, db: Session) -> Dict[str, Any]:
    stmt = select(GroupMember).where(GroupMember.group_id == group_id, GroupMember.session_id == session_id)
    member = db.exec(stmt).first()
    if not member:
        raise HTTPException(status_code=403, detail="Not a member of this group.")

    # Check if group has any documents uploaded
    doc_count = len(db.exec(select(GroupDocument).where(GroupDocument.group_id == group_id)).all())

    if doc_count == 0:
        # No documents yet — respond conversationally without RAG
        answer = (
            f"👋 Hi! I'm the group assistant. This group doesn't have any documents yet. "
            f"Upload some PDFs in the **Documents** tab and I'll be able to answer questions about them! "
            f"In the meantime, feel free to ask me anything general."
        )
        # Save to ChatLog
        try:
            log = ChatLog(
                session_id=session_id,
                display_name=member.display_name,
                group_id=group_id,
                question=question,
                answer=answer,
                sources_used="[]"
            )
            db.add(log)
            db.commit()
        except Exception as e:
            logger.warning("Failed to save ChatLog: %s", e)
        return {"answer": answer, "sources": []}

    # Has documents — use RAG chain
    try:
        chain = get_rag_chain(f"group_{group_id}", f"{session_id}_{group_id}")
        response = chain.invoke({"question": question})
    except Exception as e:
        answer = f"Sorry, I encountered an error processing your question: {str(e)}"
        try:
            log = ChatLog(
                session_id=session_id, display_name=member.display_name,
                group_id=group_id, question=question, answer=answer, sources_used="[]"
            )
            db.add(log)
            db.commit()
        except:
            pass
        return {"answer": answer, "sources": []}

    answer = response.get("answer", "")
    source_docs = response.get("source_documents", [])

    sources = []
    sources_for_db = []
    for doc in source_docs:
        meta = doc.metadata
        filename = meta.get("filename", "Unknown")
        sources.append({
            "filename": filename,
            "page_number": meta.get("page", 0),
            "uploaded_by": meta.get("display_name", "Unknown"),
            "chunk_text": doc.page_content[:150] + "..."
        })
        if filename not in sources_for_db:
            sources_for_db.append(filename)

    # Save to ChatLog DB
    try:
        log = ChatLog(
            session_id=session_id,
            display_name=member.display_name,
            group_id=group_id,
            question=question,
            answer=answer,
            sources_used=json.dumps(sources_for_db)
        )
        db.add(log)
        db.commit()
    except Exception as e:
        logger.warning("Failed to save ChatLog: %s", e)

    return {"answer": answer, "sources": sources}

def get_group_chat_history(group_id: str, session_id: str, db: Session) -> List[Dict]:
    stmt = select(GroupMember).where(GroupMember.group_id == group_id, GroupMember.session_id == session_id)
    if not db.exec(stmt).first():
        raise HTTPException(status_code=403, detail="Not a member of this group.")
        
    # Get all chats by anyone in this group, ordered by time
    chat_stmt = select(ChatLog).where(ChatLog.group_id == group_id).order_by(ChatLog.timestamp)
    logs = db.exec(chat_stmt).all()
    
    formatted_logs = []
    for log in logs:
        # Prepend the author's name to their question to show who asked in the shared UI
        display_q = f"**[{log.display_name}]** {log.question}"
        formatted_logs.append([display_q, log.answer])
        
    return formatted_logs
