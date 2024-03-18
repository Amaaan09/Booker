# Mr. Jarvis
This Gen AI model utilizes:
1. [Retrieval Augmented Generation](https://research.ibm.com/blog/retrieval-augmented-generation-RAG) which extracts content from PDF documents, large language model to answer questions based on the content.
2. [LORA](https://huggingface.co/docs/diffusers/main/en/training/lora) for FineTuning the Gemma Model on a curated dataset of my favorite quotes for a personalized touch 
3. [Agents](https://python.langchain.com/docs/modules/agents/) to grant the system web access to otherwise unvailable data

## Installing
To install the project, clone the repository to your local machine:

```bash
git clone https://github.com/Amaaan09/Mr-Jarvis.git
```

```bash
pip install -r requirements.txt
```

## Usage

For PDF Master
```python
streamlit run pdf.py
```

For Jarvis 2.0
```python
streamlit run jarvis.py
```


## Explaination
Utils.py: 

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

```python
invokeResult(llm, question)
```
This function takes a language model and a question as input. It uses the language model to answer the question.


## License
This project is licensed under the MIT License 

## Acknowledgments
- HuggingFace
- PyPDF2
- langchain and langchain_community
