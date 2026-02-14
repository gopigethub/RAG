from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def main():

    # 1️⃣ Load PDF
    loader = PyPDFLoader("RAG_Implementation_Guide.pdf")
    documents = loader.load()

    # 2️⃣ Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    # 3️⃣ Use Ollama Embeddings (NO HuggingFace)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Make sure you run:
    # ollama pull nomic-embed-text

    # 4️⃣ Create FAISS vector store
    vector_db = FAISS.from_documents(chunks, embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 4})

    # 5️⃣ LLM via Ollama
    llm = Ollama(model="llama3")

    # 6️⃣ Prompt
    prompt = ChatPromptTemplate.from_template("""
    Answer the question using ONLY the context below.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """)

    parser = StrOutputParser()

    print("\n✅ RAG Ready (No HuggingFace). Type 'exit' to quit.\n")

    while True:
        query = input("Ask a question: ")

        if query.lower() == "exit":
            break

        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])

        chain = prompt | llm | parser
        answer = chain.invoke({"context": context, "question": query})

        print("\nAnswer:\n", answer)
        print("-" * 60)


if __name__ == "__main__":
    main()
