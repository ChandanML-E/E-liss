from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import tool
from langchain_huggingface import HuggingFaceEmbeddings
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

@tool
def pdf_query_tool(query: str, pdf_path: str, db_path: str) -> str:
    """
    Returns a related answer from a specified PDF using semantic search based on the input query.
    Args:
        query (str): The user's query.
        pdf_path (str): Path to the target PDF file.
        db_path (str): Path to save or load the FAISS index.
    Returns:
        str: Resulting answer based on the query.
    """
    llm = ChatGroq(model="llama3-8b-8192")
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    try:
        db = FAISS.load_local(
            db_path, embeddings_model, allow_dangerous_deserialization=True
        )
    except Exception as e:
        # Fallback: Create a new index if not available
        reader = PdfReader(pdf_path)
        raw_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                raw_text += text

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=400,
        )
        texts = text_splitter.split_text(raw_text)
        db = FAISS.from_texts(texts, embeddings_model)
        db.save_local(db_path)

    retriever = db.as_retriever(k=4)
    result = retriever.invoke(query)
    return result

@tool
def constitution_query(query: str) -> str:
    """
    Returns a related answer from the Indian Constitution PDF using semantic search.
    """
    pdf_path = "tools/data/constitution.pdf"
    db_path = "db/faiss_index_constitution"
    return pdf_query_tool(query, pdf_path, db_path)

@tool
def laws_query(query: str) -> str:
    """
    Returns a related answer from the Bharatiya Nyaya Sanhita PDF using semantic search.
    """
    pdf_path = "tools/data/BNS.pdf"
    db_path = "db/faiss_index_bns"
    return pdf_query_tool(query, pdf_path, db_path)
