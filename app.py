import streamlit as st
from utils import get_pdf_text, get_text_chunks, get_vectorstore, queryLLM, LLM

st.title('Mr. Booker')

file = st.file_uploader("Upload your book here", accept_multiple_files=False, type='pdf')

@st.cache_resource(max_entries=3, ttl=3600, show_spinner="Loading...")
def load_vectorstore(file):
    texts = get_pdf_text(file)
    chunks = get_text_chunks(texts)
    return get_vectorstore(chunks)

if file is not None:

    vectorstore = load_vectorstore(file)

    user_question = st.text_input("Ask your question:")

    if st.button('GO!'):
        if user_question == "":
            st.warning("Please ask a question")
        else:
            with st.info("Processing..."):
                res = queryLLM(LLM, vectorstore, question=user_question)
                st.write(res)
