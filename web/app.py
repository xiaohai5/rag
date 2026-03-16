import streamlit as st

from services.api_client import api_client
from utils.session import ensure_session_defaults, logout_user


VIEW_LABEL_TO_KEY = {
    "向量库": "vector_store",
    "大模型对话": "chat",
}
VIEW_KEY_TO_LABEL = {value: key for key, value in VIEW_LABEL_TO_KEY.items()}

st.set_page_config(page_title="RAG Workbench", page_icon="🧠", layout="centered")
ensure_session_defaults()


def _on_view_change() -> None:
    selected_label = st.session_state.get("nav_view_label", "向量库")
    st.session_state["active_view"] = VIEW_LABEL_TO_KEY.get(selected_label, "vector_store")


def render_login_screen() -> None:
    _, col_center, _ = st.columns([1, 2, 1])
    with col_center:
        st.title("RAG 工作台")
        st.caption("登录后可上传文档并进行检索问答。")

        login_tab, register_tab = st.tabs(["登录", "注册"])

        with login_tab:
            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("用户名", key="login_username")
                password = st.text_input("密码", type="password", key="login_password")
                submit_login = st.form_submit_button("登录")

        if submit_login:
            try:
                result = api_client.login(username=username, password=password)
                st.session_state["token"] = result["access_token"]
                st.session_state["username"] = result["username"]
                st.session_state["menu_view"] = ""
                st.rerun()
            except RuntimeError as exc:
                st.error(str(exc))

        with register_tab:
            with st.form("register_form", clear_on_submit=False):
                username = st.text_input("用户名", key="register_username")
                email = st.text_input("邮箱", key="register_email")
                password = st.text_input("密码", type="password", key="register_password")
                submit_register = st.form_submit_button("注册")

        if submit_register:
            try:
                result = api_client.register(username=username, email=email, password=password)
                st.session_state["token"] = result["access_token"]
                st.session_state["username"] = result["username"]
                st.session_state["menu_view"] = ""
                st.success(result["message"])
                st.rerun()
            except RuntimeError as exc:
                st.error(str(exc))


def render_sidebar() -> None:
    with st.sidebar:
        st.header("导航")
        st.caption(f"当前用户：{st.session_state['username']}")

        if "nav_view_label" not in st.session_state:
            st.session_state["nav_view_label"] = VIEW_KEY_TO_LABEL.get(
                st.session_state.get("active_view", "vector_store"),
                "向量库",
            )

        st.radio(
            "页面",
            options=list(VIEW_LABEL_TO_KEY.keys()),
            key="nav_view_label",
            on_change=_on_view_change,
        )
        st.session_state["active_view"] = VIEW_LABEL_TO_KEY.get(
            st.session_state.get("nav_view_label", "向量库"),
            "vector_store",
        )

        if st.session_state.get("active_view", "vector_store") == "chat":
            with st.expander("大模型参数", expanded=False):
                st.slider("Top K", min_value=1, max_value=10, key="top_k")
            if st.button("清除对话", use_container_width=True):
                st.session_state["chat_history"] = []
                st.success("对话已清空")
                st.rerun()


def render_user_menu() -> None:
    left_col, right_col = st.columns([6, 1])
    with left_col:
        st.markdown("<h1 style='text-align: center;'>RAG Workbench</h1>", unsafe_allow_html=True)
        st.markdown(
            "<p style='text-align: center; color: #666;'>文档记录与大模型检索问答</p>",
            unsafe_allow_html=True,
        )
    with right_col:
        with st.popover(st.session_state["username"], use_container_width=True):
            if st.button("个人资料", use_container_width=True):
                st.session_state["menu_view"] = "profile"
                st.rerun()
            if st.button("修改密码", use_container_width=True):
                st.session_state["menu_view"] = "change_password"
                st.rerun()
            if st.button("退出登录", use_container_width=True):
                logout_user()
                st.rerun()


def render_profile_view() -> None:
    if st.button("← 返回工作台"):
        st.session_state["menu_view"] = ""
        st.rerun()

    st.subheader("个人资料")
    try:
        profile = api_client.get_profile()
        st.session_state["email"] = profile.get("email", st.session_state.get("email", ""))
    except RuntimeError as exc:
        st.error(str(exc))
        profile = {
            "username": st.session_state.get("username", ""),
            "email": st.session_state.get("email", ""),
        }

    st.write(f"用户名：{profile.get('username', '')}")
    st.write(f"邮箱：{profile.get('email', '')}")


def render_change_password_view() -> None:
    if st.button("← 返回工作台"):
        st.session_state["menu_view"] = ""
        st.rerun()

    st.subheader("修改密码")
    with st.form("change_password_form"):
        username = st.text_input("用户名", value=st.session_state.get("username", ""), disabled=True)
        old_password = st.text_input("旧密码", type="password")
        new_password = st.text_input("新密码", type="password")
        confirm_password = st.text_input("确认新密码", type="password")
        submit_change = st.form_submit_button("提交修改")

    if submit_change:
        try:
            result = api_client.change_password(
                username=username,
                old_password=old_password,
                new_password=new_password,
                confirm_password=confirm_password,
            )
            st.success(result["message"])
            logout_user()
            st.rerun()
        except RuntimeError as exc:
            st.error(str(exc))


def render_vector_store_panel() -> None:
    st.markdown("<h3 style='text-align: center;'>文档记录</h3>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; color: #666;'>点击上传会先解析文档，再写入向量库并记录。</p>",
        unsafe_allow_html=True,
    )

    with st.form("upload_form"):
        uploaded_file = st.file_uploader("上传文档", type=["txt", "md", "pdf", "docx", "csv", "json", "jsonl", "html", "htm"])
        submit_upload = st.form_submit_button("上传并处理")

    if submit_upload:
        if uploaded_file is None:
            st.warning("请先选择一个文件。")
        else:
            try:
                result = api_client.upload_document(
                    file_name=uploaded_file.name,
                    file_bytes=uploaded_file.getvalue(),
                )
                st.success(result["message"])
            except RuntimeError as exc:
                st.error(str(exc))

    if st.button("刷新文档列表", use_container_width=True):
        try:
            st.session_state["documents"] = api_client.list_documents()
        except RuntimeError as exc:
            st.error(str(exc))

    documents = st.session_state.get("documents", [])
    if documents:
        st.dataframe(documents, use_container_width=True)
    else:
        st.caption("暂无文档记录。")


def render_chat_panel() -> None:
    top_k = int(st.session_state.get("top_k", 3))

    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("请输入问题")
    if prompt:
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("正在生成回答..."):
                try:
                    result = api_client.chat(
                        question=prompt,
                        top_k=top_k,
                        history=st.session_state["chat_history"][:-1],
                    )
                    st.markdown(result["answer"])
                    st.session_state["chat_history"] = result["history"]
                except RuntimeError as exc:
                    st.error(str(exc))


def render_workspace() -> None:
    if st.session_state.get("active_view", "vector_store") == "vector_store":
        render_vector_store_panel()
    else:
        render_chat_panel()


if not st.session_state.get("token"):
    render_login_screen()
else:
    render_sidebar()
    render_user_menu()

    menu_view = st.session_state.get("menu_view", "")
    if menu_view == "profile":
        render_profile_view()
    elif menu_view == "change_password":
        render_change_password_view()
    else:
        render_workspace()

