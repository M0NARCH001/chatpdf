import gradio as gr

def render_chat():
    with gr.Column(scale=3):
        gr.Markdown("## DocChat AI 🤖")
        
        chatbot = gr.Chatbot(
            label="Chat History",
            height=500,
        )
        
        with gr.Row():
            msg_input = gr.Textbox(
                show_label=False, 
                placeholder="Ask a question about your documents...", 
                scale=8,
                container=False
            )
            submit_btn = gr.Button("Send", variant="primary", scale=1)
        
        with gr.Row():
            clear_btn = gr.Button("Clear Chat 🗑️", size="sm")
            
    return chatbot, msg_input, submit_btn, clear_btn
