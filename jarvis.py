import streamlit as st
from utils import LLM, invokeResult

st.title('Mr. Jarvis 2.0')

user_question = st.text_input("Ask your question:")
if st.button('GO!'):
    if user_question == "":
        st.warning("Please ask a question")
    else:
        with st.info("Processing..."):
            res = invokeResult(LLM, question=user_question)
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
<p>Do check out <a style='text-align: center;' href="https://jarvis-pdf-master.streamlit.app/" target="_blank">Jarvis PDF Master</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)