# Mr. PDF
This Gen AI model utilizes [Retrieval Augmented Generation](https://research.ibm.com/blog/retrieval-augmented-generation-RAG) which extracts content from PDF documents, large language model to answer questions based on the content.

## Prerequisites
The project is written in Python. You need to have Python installed on your machine. You can use the requirements.txt for the dependencies

## Installing
To install the project, clone the repository to your local machine:

```bash
git clone https://github.com/Amaaan09/Booker.git
```

```bash
pip install -r requirements.txt
```


## Usage
The project contains a file named utils.py which contains several functions:

```python
extract_pdf_content(pdf_document)
```
This function takes a PDF document as input and extracts the text content from it.

```python
split_content_into_chunks(content)
```
This function takes a string of text content and splits it into chunks of a specified size.

```python
create_vector_store(content_chunks)
```
This function takes a list of text chunks and creates a vector store from them.

```python
queryLLM(llm, vectorstore, question)
```
This function takes a language model, a vector store, and a question as input. It uses the language model to answer the question based on the content in the vector store.


## License
This project is licensed under the MIT License 

## Acknowledgments
- HuggingFace
- PyPDF2
- langchain and langchain_community
