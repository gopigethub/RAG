import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.title("📄 RAG PDF Chat")

# Load and process PDF once
@st.cache_resource
def setup_rag():
    loader = PyPDFLoader("RAG_Implementation_Guide.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_db = FAISS.from_documents(chunks, embeddings)

    return vector_db.as_retriever(search_kwargs={"k": 4})

retriever = setup_rag()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_template("""
Answer using ONLY the provided context.

Context:
{context}

Question:
{question}

Answer:
""")

parser = StrOutputParser()

query = st.text_input("Ask a question")

if query:
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    chain = prompt | llm | parser
    answer = chain.invoke({"context": context, "question": query})

    st.write(answer)
