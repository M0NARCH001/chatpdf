"""
DocChat AI — Gradio frontend.

Communicates with the FastAPI backend via httpx.
Each browser tab gets its own UUID-based personal session.
"""
import gradio as gr
import httpx
import os
import uuid
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from frontend.components.sidebar import render_sidebar
from frontend.components.chat import render_chat
from frontend.components.source_viewer import render_source_viewer, format_sources
from frontend.components.group_panel import (
    api_call, login_user_simple,
    refresh_group_list, create_group_action, join_group_action,
    load_group_details, upload_group_docs, group_chat_logic
)

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000/api")

# ---------------------------------------------------------------------------
# Personal workspace API helpers
# ---------------------------------------------------------------------------

def process_documents(files):
    if not files:
        return "No files selected.", gr.update()
    try:
        with httpx.Client(timeout=120.0) as client:
            upload_files = [
                ("files", (os.path.basename(f.name), open(f.name, "rb"), "application/octet-stream"))
                for f in files
            ]
            response = client.post(f"{API_BASE_URL}/upload", files=upload_files)
            if response.status_code == 200:
                data = response.json()
                msg = (
                    f"✅ **Upload complete** — "
                    f"{data['chunks_processed']} chunks indexed  \n"
                    f"Collection: `{data['collection_id']}`"
                )
                collections = get_collections()
                return msg, gr.update(choices=collections, value=data["collection_id"])
            else:
                detail = response.json().get("detail", response.text)
                return f"❌ **Upload failed:** {detail}", gr.update()
    except Exception as e:
        return f"❌ **Connection error:** {e}", gr.update()


def get_collections():
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{API_BASE_URL}/collections")
            if response.status_code == 200:
                return response.json().get("collections", [])
    except Exception:
        pass
    return []


def refresh_collections():
    return gr.update(choices=get_collections())


def clear_session_memory(session_id):
    try:
        with httpx.Client(timeout=10.0) as client:
            client.post(f"{API_BASE_URL}/memory/reset/{session_id}")
    except Exception:
        pass
    return [], "*Chat cleared. Ask a new question.*"


def run_evaluation(collection_id):
    if not collection_id:
        return {"error": "Select a collection first."}
    try:
        with httpx.Client(timeout=300.0) as client:
            res = client.post(f"{API_BASE_URL}/evaluate/{collection_id}")
            if res.status_code == 200:
                return res.json().get("scores", {})
            return {"error": res.json().get("detail", res.text)}
    except Exception as e:
        return {"error": str(e)}


def chat_logic(message, history, collection_id, personal_session_id):
    if not message.strip():
        return "", history, "*Ask a question above.*"
    if not collection_id:
        history = history + [[message, "⚠️ Please upload a document and select a collection first."]]
        return "", history, "*No collection selected.*"
    try:
        with httpx.Client(timeout=120.0) as client:
            payload = {
                "question": message,
                "session_id": personal_session_id,
                "collection_id": collection_id,
            }
            res = client.post(f"{API_BASE_URL}/chat", json=payload)
            if res.status_code == 200:
                data = res.json()
                history = history + [[message, data.get("answer", "")]]
                raw_sources = [
                    {
                        "page": s["metadata"].get("page_number", "—"),
                        "file": s["metadata"].get("source_file", "Unknown"),
                        "content": s["content"],
                    }
                    for s in data.get("sources", [])
                ]
                return "", history, format_sources(raw_sources)
            else:
                detail = res.json().get("detail", res.text)
                history = history + [[message, f"❌ Error: {detail}"]]
                return "", history, "*No sources — an error occurred.*"
    except Exception as e:
        history = history + [[message, f"❌ Connection error: {e}"]]
        return "", history, "*No sources — connection failed.*"


# ---------------------------------------------------------------------------
# Custom CSS — light touch; the Gradio Soft theme does the heavy lifting.
# No web fonts (faster load, no CSP issues on HF); just header + cards + spacing.
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
/* Header */
#docchat-header {
    padding: 1.4rem 1.5rem 1.1rem;
    border-bottom: 1px solid var(--border-color-primary);
    margin-bottom: .5rem;
}
#docchat-header h1 { font-size: 1.8rem; margin: 0 0 .2rem; }
#docchat-header p  { font-size: .9rem; opacity: .7; margin: 0; }

/* Cards — sidebar and login share a simple bordered box */
.sidebar-panel, .login-card {
    border: 1px solid var(--border-color-primary);
    border-radius: 10px;
    padding: 1.1rem;
}
.login-card { max-width: 440px; margin: 2rem auto; }

/* Source citation passages */
.source-panel blockquote {
    border-left: 3px solid var(--border-color-accent, #888);
    margin: .4rem 0 0 .3rem;
    padding: .3rem .7rem;
    opacity: .85;
    font-size: .88rem;
}
"""

# ---------------------------------------------------------------------------
# Build the Gradio UI
# ---------------------------------------------------------------------------

GRADIO_MAJOR_VERSION = int(gr.__version__.split(".", 1)[0])
BLOCKS_KWARGS = {"title": "DocChat AI"}
LAUNCH_KWARGS = {}
if GRADIO_MAJOR_VERSION >= 6:
    LAUNCH_KWARGS.update({"theme": gr.themes.Soft(), "css": CUSTOM_CSS})
else:
    BLOCKS_KWARGS.update({"theme": gr.themes.Soft(), "css": CUSTOM_CSS})

with gr.Blocks(**BLOCKS_KWARGS) as demo:

    # Per-browser session ID (generated once on page load, not shared)
    personal_session_id = gr.State(lambda: f"ps_{uuid.uuid4().hex[:12]}")

    # ── Masthead ──────────────────────────────────────────────────────────
    gr.HTML("""
    <div id="docchat-header">
      <h1>📚 DocChat AI</h1>
      <p>Upload your documents and ask questions — get answers with the exact source passages.</p>
    </div>
    """)

    gr.Markdown(
        "**Two ways to use this:**  &nbsp; "
        "🧑 **Personal Workspace** — chat privately with your own files.  &nbsp;•&nbsp; "
        "👥 **Group Collaboration** — share files and chat together with a study group."
    )

    with gr.Tabs():

        # ── Personal Workspace ────────────────────────────────────────────
        with gr.Tab("Personal Workspace"):
            with gr.Row(equal_height=False):

                # Sidebar
                with gr.Column(scale=1, min_width=240, elem_classes="sidebar-panel"):
                    gr.Markdown("### Step 1 — Upload your files")
                    file_input = gr.File(
                        label="Drop PDF, TXT, or DOCX files here (max 15 MB each)",
                        file_count="multiple",
                        file_types=[".pdf", ".txt", ".docx"],
                    )
                    process_btn = gr.Button("Upload & Process", variant="primary")
                    upload_status = gr.Markdown("", elem_classes="upload-status")

                    gr.Markdown("---")
                    gr.Markdown("### Step 2 — Choose what to chat with")
                    collection_dropdown = gr.Dropdown(
                        label="Document set",
                        info="Each upload becomes a set. Pick the one your questions should search.",
                        choices=[], interactive=True,
                    )
                    refresh_btn = gr.Button("Refresh list", size="sm", variant="secondary")

                    gr.Markdown("---")
                    gr.Markdown(
                        "### Answer quality _(optional)_\n"
                        "Runs an automatic accuracy check on the selected set — "
                        "scores how well answers stick to your documents (higher = better)."
                    )
                    eval_btn = gr.Button("Check answer quality", variant="secondary")
                    eval_status = gr.JSON(label="Scores (0–1, higher is better)")

                # Chat + sources
                with gr.Column(scale=3):
                    gr.Markdown("### Step 3 — Ask questions about your files")
                    chatbot = gr.Chatbot(
                        label="",
                        height=480,
                        show_label=False,
                        type="tuples",  # ponytail: UI uses [msg, reply] pairs; migrate to type="messages" when moving to gradio 6
                    )
                    with gr.Row():
                        msg_input = gr.Textbox(
                            show_label=False,
                            placeholder="Ask a question about your documents…",
                            scale=8,
                            container=False,
                            lines=1,
                        )
                        submit_btn = gr.Button("Send", variant="primary", scale=1)
                    clear_btn = gr.Button("Clear conversation", size="sm", variant="secondary")

                    with gr.Accordion("📎 Where did this answer come from?", open=False, elem_classes="source-panel"):
                        source_display = gr.Markdown(
                            value="*Ask a question — the exact passages used to answer will appear here.*"
                        )

        # ── Group Collaboration ───────────────────────────────────────────
        with gr.Tab("Group Collaboration"):
            session_id       = gr.State("")
            display_name_state = gr.State("")

            # Login screen
            with gr.Column(visible=True, elem_classes="login-card") as login_col:
                gr.Markdown("## 👥 Study Groups")
                gr.Markdown(
                    "A shared space where everyone uploads documents and chats with them **together**.\n\n"
                    "Enter a name and password to start:\n"
                    "- **New name** → creates your account\n"
                    "- **Existing name** → logs you back into your groups"
                )
                name_input = gr.Textbox(
                    label="Your Name",
                    info="How other members see you.",
                    placeholder="e.g. Alice", max_lines=1,
                )
                pass_input = gr.Textbox(
                    label="Password (min 8 chars)",
                    placeholder="Your password",
                    type="password",
                    max_lines=1,
                )
                login_btn = gr.Button("Enter Workspace", variant="primary")
                login_err = gr.Markdown()

            # Workspace (post-login)
            with gr.Column(visible=False) as workspace_col:
                workspace_banner = gr.Markdown()

                with gr.Row():
                    # Groups sidebar
                    with gr.Column(scale=1, min_width=280, elem_classes="sidebar-panel"):
                        gr.Markdown("### Your Groups")
                        my_groups_drop = gr.Dropdown(
                            label="", choices=[], interactive=True, show_label=False
                        )
                        refresh_mine_btn = gr.Button("Refresh", size="sm", variant="secondary")

                        gr.Markdown("---")
                        gr.Markdown("#### Start a new group")
                        create_input = gr.Textbox(
                            label="Group name",
                            info="You'll get a share code others can use to join.",
                            placeholder="ML Study Group", max_lines=1,
                        )
                        create_btn = gr.Button("Create", size="sm", variant="primary")
                        create_msg = gr.Markdown()

                        gr.Markdown("---")
                        gr.Markdown("#### Join a friend's group")
                        join_input = gr.Textbox(
                            label="Share code",
                            info="Paste the 6-character code someone shared with you.",
                            placeholder="A1B2C3", max_lines=1,
                        )
                        join_btn = gr.Button("Join", size="sm", variant="secondary")
                        join_msg = gr.Markdown()

                    # Group detail panel
                    with gr.Column(scale=3, visible=False) as details_col:
                        group_header = gr.Markdown()

                        with gr.Tabs():
                            with gr.Tab("Chat"):
                                group_chatbot = gr.Chatbot(
                                    height=380,
                                    label="Group chat — visible to all members",
                                    type="tuples",  # ponytail: same tuple format as the personal chatbot
                                )
                                with gr.Row():
                                    g_msg_input = gr.Textbox(
                                        scale=4, show_label=False,
                                        placeholder="Ask about the group's documents…",
                                        max_lines=1, container=False,
                                    )
                                    g_submit_btn = gr.Button("Send", scale=1, variant="primary")
                                g_source_display = gr.Markdown()

                            with gr.Tab("Documents"):
                                gr.Markdown(
                                    "Files added here are searchable by **everyone** in the group's chat."
                                )
                                g_docs_table = gr.Dataframe(
                                    headers=["Filename", "Uploaded By", "Size", "Chunks"],
                                    interactive=False,
                                )
                                g_file_upload = gr.File(
                                    label="Add documents to share with the group",
                                    file_count="multiple",
                                )
                                g_upload_btn = gr.Button("Upload to Group", variant="primary")
                                g_upload_msg = gr.Markdown()

                            with gr.Tab("Members"):
                                g_members_table = gr.Dataframe(
                                    headers=["Name", "Role", "Docs Contributed", "Joined"],
                                    interactive=False,
                                )

    # ── Personal workspace events ─────────────────────────────────────────
    demo.load(refresh_collections, inputs=[], outputs=[collection_dropdown])

    process_btn.click(
        process_documents, inputs=[file_input],
        outputs=[upload_status, collection_dropdown],
    )
    refresh_btn.click(refresh_collections, inputs=[], outputs=[collection_dropdown])
    eval_btn.click(run_evaluation, inputs=[collection_dropdown], outputs=[eval_status])

    msg_input.submit(
        chat_logic,
        inputs=[msg_input, chatbot, collection_dropdown, personal_session_id],
        outputs=[msg_input, chatbot, source_display],
    )
    submit_btn.click(
        chat_logic,
        inputs=[msg_input, chatbot, collection_dropdown, personal_session_id],
        outputs=[msg_input, chatbot, source_display],
    )
    clear_btn.click(
        clear_session_memory, inputs=[personal_session_id],
        outputs=[chatbot, source_display],
    )

    # ── Group events ──────────────────────────────────────────────────────
    login_btn.click(
        login_user_simple,
        inputs=[name_input, pass_input],
        outputs=[session_id, display_name_state, login_col, workspace_col, login_err, workspace_banner],
    ).then(refresh_group_list, inputs=[session_id], outputs=[my_groups_drop])

    pass_input.submit(
        login_user_simple,
        inputs=[name_input, pass_input],
        outputs=[session_id, display_name_state, login_col, workspace_col, login_err, workspace_banner],
    ).then(refresh_group_list, inputs=[session_id], outputs=[my_groups_drop])

    refresh_mine_btn.click(refresh_group_list, inputs=[session_id], outputs=[my_groups_drop])
    create_btn.click(
        create_group_action, inputs=[session_id, create_input],
        outputs=[create_msg, my_groups_drop],
    )
    join_btn.click(
        join_group_action, inputs=[session_id, join_input],
        outputs=[join_msg, my_groups_drop],
    )

    my_groups_drop.change(
        load_group_details,
        inputs=[session_id, my_groups_drop],
        outputs=[details_col, g_members_table, g_docs_table, group_header, group_chatbot],
    )

    g_upload_btn.click(
        upload_group_docs,
        inputs=[session_id, my_groups_drop, g_file_upload],
        outputs=[g_upload_msg, g_file_upload],
    ).then(
        load_group_details,
        inputs=[session_id, my_groups_drop],
        outputs=[details_col, g_members_table, g_docs_table, group_header, group_chatbot],
    )

    g_msg_input.submit(
        group_chat_logic,
        inputs=[g_msg_input, group_chatbot, session_id, my_groups_drop],
        outputs=[g_msg_input, group_chatbot, g_source_display],
    )
    g_submit_btn.click(
        group_chat_logic,
        inputs=[g_msg_input, group_chatbot, session_id, my_groups_drop],
        outputs=[g_msg_input, group_chatbot, g_source_display],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, **LAUNCH_KWARGS)
