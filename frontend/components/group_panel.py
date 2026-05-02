import gradio as gr
import httpx
import os

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000/api")


def api_call(method, endpoint, json_data=None, data=None, files=None, headers=None):
    url = f"{API_BASE_URL}{endpoint}"
    try:
        with httpx.Client(timeout=120.0) as client:
            if method == "GET":
                res = client.get(url, headers=headers)
            elif method == "POST":
                res = client.post(url, json=json_data, data=data, files=files, headers=headers)
            elif method == "PUT":
                res = client.put(url, json=json_data, headers=headers)
            elif method == "DELETE":
                res = client.delete(url, headers=headers)
            else:
                return None, f"Method {method} not supported"
            if res.status_code in [200, 201]:
                return res.json(), None
            else:
                return None, res.text
    except Exception as e:
        return None, str(e)


def login_user_simple(name, password):
    """Login or register → returns 6 outputs."""
    if not name or not name.strip():
        return "", "", gr.update(visible=True), gr.update(visible=False), "⚠️ Please enter a name.", ""
    if not password or len(password) < 3:
        return "", "", gr.update(visible=True), gr.update(visible=False), "⚠️ Password must be at least 3 characters.", ""

    data, err = api_call("POST", "/session/start",
                         json_data={"display_name": name.strip(), "password": password})
    if err:
        # Parse the error — could be "Wrong password" (401) or server error
        error_msg = err
        try:
            import json
            parsed = json.loads(err)
            error_msg = parsed.get("detail", err)
        except:
            pass
        return "", "", gr.update(visible=True), gr.update(visible=False), f"❌ {error_msg}", ""

    sess_id = data["session_id"]
    d_name = data["display_name"]
    banner = f"👤 Logged in as **{d_name}** · Your groups and chats are restored."
    return sess_id, d_name, gr.update(visible=False), gr.update(visible=True), "", banner


def get_my_group_choices(sess_id):
    if not sess_id:
        return []
    data, err = api_call("GET", "/groups/mine", headers={"X-Session-ID": sess_id})
    if err or not isinstance(data, list):
        return []
    choices = []
    for g in data:
        mc = g.get("member_count", "?")
        dc = g.get("doc_count", "?")
        choices.append((f"{g['name']} ({mc} members, {dc} docs)", g["group_id"]))
    return choices


def refresh_group_list(sess_id):
    choices = get_my_group_choices(sess_id)
    return gr.update(choices=choices, value=None)


def create_group_action(sess_id, group_name):
    if not group_name or not group_name.strip():
        return "⚠️ Enter a group name.", refresh_group_list(sess_id)
    data, err = api_call("POST", "/groups", json_data={"name": group_name.strip()}, headers={"X-Session-ID": sess_id})
    if err:
        return f"❌ {err}", gr.update()
    # Auto-select the newly created group
    new_gid = data["group_id"]
    choices = get_my_group_choices(sess_id)
    return f"✅ **{data['name']}** created! Code: **{data['join_code']}**", gr.update(choices=choices, value=new_gid)


def join_group_action(sess_id, join_code):
    if not join_code or not join_code.strip():
        return "⚠️ Enter a join code.", refresh_group_list(sess_id)
    data, err = api_call("POST", "/groups/join", json_data={"join_code": join_code.strip()}, headers={"X-Session-ID": sess_id})
    if err:
        return f"❌ {err}", gr.update()
    # Auto-select the joined group
    joined_gid = data.get("group_id", None)
    choices = get_my_group_choices(sess_id)
    return f"✅ Joined **{data['name']}**!", gr.update(choices=choices, value=joined_gid)


def load_group_details(sess_id, group_id):
    """Returns 5 outputs: details_col_vis, members, docs, header, chat_history."""
    if not group_id or not sess_id:
        return gr.update(visible=False), [], [], "", []

    data, err = api_call("GET", f"/groups/{group_id}", headers={"X-Session-ID": sess_id})
    if err:
        return gr.update(visible=False), [], [], f"❌ {err}", []

    m_list = []
    for m in data.get("members", []):
        joined = (m.get("joined_at") or "")[:10]
        m_list.append([m["display_name"], m["role"], m.get("docs_contributed", 0), joined])

    d_list = []
    for d in data.get("documents", []):
        d_list.append([d["filename"], d["uploader_name"], f"{d['file_size_kb']:.1f} KB", d["chunk_count"]])

    header = f"### 📂 {data['name']}\n**Join Code:** `{data['join_code']}` · **Members:** {len(m_list)} · **Docs:** {len(d_list)}"

    chat_data, _ = api_call("GET", f"/groups/{group_id}/chat/history", headers={"X-Session-ID": sess_id})
    chat_history = []
    if chat_data and "history" in chat_data:
        for pair in chat_data["history"]:
            if isinstance(pair, list) and len(pair) == 2:
                chat_history.append(pair)

    return gr.update(visible=True), m_list, d_list, header, chat_history


def upload_group_docs(sess_id, group_id, files):
    if not group_id:
        return "⚠️ Select a group.", gr.update()
    if not files:
        return "⚠️ No files.", gr.update()
    upload_files = [("files", (os.path.basename(f.name), open(f.name, "rb"), "application/octet-stream")) for f in files]
    data, err = api_call("POST", f"/groups/{group_id}/upload", files=upload_files, headers={"X-Session-ID": sess_id})
    if err:
        return f"❌ {err}", gr.update()
    return f"✅ {data.get('message', 'Done!')}", gr.update()


def group_chat_logic(message, history, sess_id, group_id):
    if not message or not message.strip():
        return "", history, ""
    if not group_id:
        history = history + [[message, "⚠️ Select a group first."]]
        return "", history, ""
    data, err = api_call("POST", f"/groups/{group_id}/chat",
                         json_data={"question": message, "session_id": sess_id},
                         headers={"X-Session-ID": sess_id})
    if err:
        history = history + [[message, f"❌ {err}"]]
        return "", history, ""

    ans = data.get("answer", "")
    srcs = data.get("sources", [])
    src_text = ""
    if srcs:
        src_text = "### 📎 Sources\n"
        for s in srcs:
            src_text += f"- **{s['filename']}** (p.{s['page_number']}) by {s['uploaded_by']}\n"
    history = history + [[message, ans]]
    return "", history, src_text
