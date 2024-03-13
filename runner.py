from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms.huggingface_hub import HuggingFaceHub


llm = HuggingFaceHub(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    model_kwargs={"temperature": 0.1, "max_length": 64,"max_new_tokens":512}
)

pdf_path = "sample_pdf/reAct.pdf"
loader = PyPDFLoader(file_path=pdf_path)
documents = loader.load()
text_splitter = CharacterTextSplitter(
    chunk_size=1000, chunk_overlap=30, separator="\n"
)
docs = text_splitter.split_documents(documents=documents)

embeddings = HuggingFaceEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("faiss_index_react")

new_vectorstore = FAISS.load_local("faiss_index_react", embeddings)
qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=new_vectorstore.as_retriever()
)
res = qa.run("Give me the gist of ReAct in 3 sentences")
print(res)
