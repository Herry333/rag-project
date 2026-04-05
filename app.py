from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

loader = TextLoader("data/docs.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)

embeddings=HuggingFaceEmbeddings()
vector_db=FAISS.from_documents(chunks,embeddings)
retriever=vector_db.as_retriever()

llm = OllamaLLM(model="llama3")

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:
{context}

Question: {question}
""")

# 7. Build LCEL chain
chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

while True:
    query = input("\nAsk a question (or 'exit' to quit): ")

    if query.lower() == "exit":
        break
    answer = chain.invoke(query)
    print("\nAnswer:", answer)
