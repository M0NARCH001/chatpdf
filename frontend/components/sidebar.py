import gradio as gr

def render_sidebar():
    with gr.Column(scale=1):
        gr.Markdown("### 📄 Upload Documents")
        file_input = gr.File(label="Upload PDF/TXT/DOCX", file_count="multiple", file_types=[".pdf", ".txt", ".docx"])
        process_btn = gr.Button("Process Documents", variant="primary")
        upload_status = gr.Markdown("")
        
        gr.Markdown("---")
        gr.Markdown("### ⚙️ Settings")
        collection_dropdown = gr.Dropdown(label="Select Collection", choices=[], interactive=True)
        refresh_btn = gr.Button("🔄 Refresh Collections", size="sm")
        
        gr.Markdown("---")
        gr.Markdown("### 📊 Evaluation (RAGAS)")
        eval_btn = gr.Button("Run Evaluation on Current Collection", variant="secondary")
        eval_status = gr.JSON(label="Evaluation Scores")
            
    return file_input, process_btn, upload_status, collection_dropdown, refresh_btn, eval_btn, eval_status
