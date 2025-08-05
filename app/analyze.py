import streamlit as st
from app.utils import extract_text, extract_ids_from_text, run_rag_pipeline

def app():
    st.title("Analyze Medical Report")
    
    uploaded_file = st.file_uploader("Upload a report (PDF/Image)", type=["pdf", "png", "jpg", "jpeg"])
    if uploaded_file:
        text = extract_text(uploaded_file)
        st.success("Text Extracted")
        st.text_area("Extracted Text", text, height=200)

        subject_id, hadm_id = extract_ids_from_text(text)
        st.write(f"SUBJECT_ID: `{subject_id}` | HADM_ID: `{hadm_id}`")

        if st.button("Start Q&A"):
            st.session_state.chat_started = True
            st.session_state.text = text
            st.session_state.subject_id = subject_id
            st.session_state.hadm_id = hadm_id

    if st.session_state.get("chat_started"):
        st.subheader("💬 Ask Your Question")
        query = st.text_input("Your Question")
        if query:
            answer = run_rag_pipeline(query, st.session_state.text, st.session_state.subject_id, st.session_state.hadm_id)
            st.markdown(f"**Answer:** {answer}")
        if st.button("End Chat"):
            st.session_state.chat_started = False
            st.experimental_rerun()
