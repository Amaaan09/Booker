import streamlit as st
from utils import extract_pdf_content, split_content_into_chunks, create_vector_store, queryLLM, LLM

st.title('Mr. Jarvis - The PDF Master')


file = st.file_uploader("Upload your PDF here", accept_multiple_files=False, type='pdf')

@st.cache_resource(max_entries=3, ttl=3600, show_spinner="Loading...")
def load_vectorstore(file):
    texts = extract_pdf_content(file)
    chunks = split_content_into_chunks(texts)
    return create_vector_store(chunks)

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

# footer
footer="""<style>
a:link , a:visited{
color: white;
background-color: transparent;
text-decoration: none;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
color: white;
text-decoration: none;
text-align: center;
}
</style>
<div class="footer">
<p>Do check out <a style='text-align: center;' href="https://mr-jarvis.streamlit.app/" target="_blank">Jarvis 2.0</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)