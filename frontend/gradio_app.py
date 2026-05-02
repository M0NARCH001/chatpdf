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
# Custom CSS — refined dark-accent editorial style
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --ink:        #0f1117;
    --ink-soft:   #4a4f5c;
    --bg:         #f7f6f2;
    --bg-card:    #ffffff;
    --accent:     #1a56e8;
    --accent-dim: #d6e4ff;
    --border:     #e2e0d8;
    --success:    #16a34a;
    --warn:       #b45309;
    --mono:       'JetBrains Mono', monospace;
    --serif:      'DM Serif Display', serif;
    --sans:       'DM Sans', sans-serif;
    --radius:     10px;
    --shadow:     0 2px 12px rgba(0,0,0,.07);
}

/* Global */
body, .gradio-container {
    font-family: var(--sans) !important;
    background: var(--bg) !important;
    color: var(--ink) !important;
}

/* ── Masthead ── */
#docchat-header {
    padding: 2.2rem 2rem 1.6rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0;
    background: var(--bg-card);
}
#docchat-header h1 {
    font-family: var(--serif);
    font-size: 2.4rem;
    letter-spacing: -.02em;
    color: var(--ink);
    margin: 0 0 .25rem;
}
#docchat-header p {
    font-size: .95rem;
    color: var(--ink-soft);
    margin: 0;
    font-weight: 300;
}

/* ── Tabs ── */
.tab-nav button {
    font-family: var(--sans) !important;
    font-weight: 500 !important;
    font-size: .9rem !important;
    letter-spacing: .01em !important;
    color: var(--ink-soft) !important;
    border-bottom: 2px solid transparent !important;
    padding: .55rem 1.1rem !important;
    transition: color .2s, border-color .2s !important;
}
.tab-nav button.selected {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

/* ── Sidebar panel ── */
.sidebar-panel {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem;
    box-shadow: var(--shadow);
}
.sidebar-panel label {
    font-weight: 500;
    font-size: .85rem;
    color: var(--ink-soft);
    text-transform: uppercase;
    letter-spacing: .06em;
}

/* ── Chatbot bubbles ── */
.message.user  { background: var(--accent-dim) !important; border-radius: 14px 14px 2px 14px !important; }
.message.bot   { background: var(--bg-card)    !important; border: 1px solid var(--border) !important; border-radius: 14px 14px 14px 2px !important; box-shadow: var(--shadow); }
.message p     { font-size: .92rem; line-height: 1.65; }

/* ── Buttons ── */
button.primary {
    background: var(--accent) !important;
    font-family: var(--sans) !important;
    font-weight: 600 !important;
    letter-spacing: .02em !important;
    border-radius: 8px !important;
    transition: opacity .15s !important;
}
button.primary:hover { opacity: .88 !important; }
button.secondary {
    border: 1.5px solid var(--border) !important;
    font-family: var(--sans) !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    color: var(--ink) !important;
    background: transparent !important;
}

/* ── Textboxes ── */
textarea, input[type=text], input[type=password] {
    font-family: var(--sans) !important;
    font-size: .93rem !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 8px !important;
    background: var(--bg-card) !important;
}
textarea:focus, input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(26,86,232,.12) !important;
}

/* ── Source cards ── */
.source-panel blockquote {
    border-left: 3px solid var(--accent-dim);
    margin: .4rem 0 0 .5rem;
    padding: .35rem .7rem;
    color: var(--ink-soft);
    font-size: .86rem;
    line-height: 1.55;
}

/* ── Accordion ── */
.accordion .label-wrap {
    font-weight: 500 !important;
    font-size: .9rem !important;
}

/* ── Dataframes ── */
table th {
    background: var(--bg) !important;
    font-weight: 600 !important;
    font-size: .82rem !important;
    text-transform: uppercase !important;
    letter-spacing: .05em !important;
    color: var(--ink-soft) !important;
}

/* ── Status / markdown badges ── */
.upload-status p { font-size: .9rem; }

/* ── Code / mono ── */
code, pre { font-family: var(--mono) !important; font-size: .85rem !important; }

/* ── Login card ── */
.login-card {
    max-width: 440px;
    margin: 2rem auto;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 2rem;
    box-shadow: var(--shadow);
}
"""

# ---------------------------------------------------------------------------
# Build the Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="DocChat AI", theme=gr.themes.Soft(), css=CUSTOM_CSS) as demo:

    # Per-browser session ID (generated once on page load, not shared)
    personal_session_id = gr.State(lambda: f"ps_{uuid.uuid4().hex[:12]}")

    # ── Masthead ──────────────────────────────────────────────────────────
    gr.HTML("""
    <div id="docchat-header">
      <h1>DocChat AI</h1>
      <p>Hybrid RAG &nbsp;·&nbsp; FastAPI &nbsp;·&nbsp; LangChain &nbsp;·&nbsp; ChromaDB &nbsp;·&nbsp; Gradio</p>
    </div>
    """)

    with gr.Tabs():

        # ── Personal Workspace ────────────────────────────────────────────
        with gr.Tab("Personal Workspace"):
            with gr.Row(equal_height=False):

                # Sidebar
                with gr.Column(scale=1, min_width=240, elem_classes="sidebar-panel"):
                    gr.Markdown("### Upload Documents")
                    file_input = gr.File(
                        label="PDF · TXT · DOCX (max 15 MB each)",
                        file_count="multiple",
                        file_types=[".pdf", ".txt", ".docx"],
                    )
                    process_btn = gr.Button("Index Documents", variant="primary")
                    upload_status = gr.Markdown("", elem_classes="upload-status")

                    gr.Markdown("---")
                    gr.Markdown("### Active Collection")
                    collection_dropdown = gr.Dropdown(
                        label="", choices=[], interactive=True,
                        show_label=False,
                    )
                    refresh_btn = gr.Button("Refresh list", size="sm", variant="secondary")

                    gr.Markdown("---")
                    gr.Markdown("### RAGAS Quality Score")
                    eval_btn = gr.Button("Evaluate collection", variant="secondary")
                    eval_status = gr.JSON(label="Scores")

                # Chat + sources
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="",
                        height=480,
                        show_label=False,
                        bubble_full_width=False,
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

                    with gr.Accordion("Source Citations", open=False, elem_classes="source-panel"):
                        source_display = gr.Markdown(
                            value="*Ask a question to see which passages were used.*"
                        )

        # ── Group Collaboration ───────────────────────────────────────────
        with gr.Tab("Group Collaboration"):
            session_id       = gr.State("")
            display_name_state = gr.State("")

            # Login screen
            with gr.Column(visible=True, elem_classes="login-card") as login_col:
                gr.Markdown("## Welcome")
                gr.Markdown(
                    "Enter your name and a password.  \n"
                    "**New?** — creates your account.  \n"
                    "**Returning?** — restores your groups and chats."
                )
                name_input = gr.Textbox(label="Display Name", placeholder="e.g. Alice", max_lines=1)
                pass_input = gr.Textbox(
                    label="Password (min 4 chars)",
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
                        gr.Markdown("#### Create Group")
                        create_input = gr.Textbox(
                            label="Group name", placeholder="ML Study Group", max_lines=1
                        )
                        create_btn = gr.Button("Create", size="sm", variant="primary")
                        create_msg = gr.Markdown()

                        gr.Markdown("---")
                        gr.Markdown("#### Join Group")
                        join_input = gr.Textbox(
                            label="Join Code", placeholder="A1B2C3", max_lines=1
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
                                    bubble_full_width=False,
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
                                g_docs_table = gr.Dataframe(
                                    headers=["Filename", "Uploaded By", "Size", "Chunks"],
                                    interactive=False,
                                )
                                g_file_upload = gr.File(
                                    label="Share documents with the group",
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
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
