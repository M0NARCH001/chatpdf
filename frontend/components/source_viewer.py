import gradio as gr


def render_source_viewer():
    """Render source citations as formatted markdown cards."""
    with gr.Accordion("📎 Source Citations", open=False):
        source_display = gr.Markdown(
            value="*Ask a question to see which document passages were used.*",
            label=None,
        )
    return source_display


def format_sources(sources: list) -> str:
    """Convert the list of source dicts into readable markdown cards."""
    if not sources:
        return "*No sources retrieved for this answer.*"

    lines = []
    seen = set()
    for s in sources:
        file = s.get("file", s.get("source_file", "Unknown"))
        page = s.get("page", s.get("page_number", "—"))
        content = s.get("content", "")[:240].strip()
        key = f"{file}:{page}"
        if key in seen:
            continue
        seen.add(key)
        lines.append(
            f"**📄 {file}** &nbsp;·&nbsp; page {page}\n"
            f"> {content}{'…' if len(content) >= 240 else ''}"
        )

    return "\n\n---\n\n".join(lines)
