"""
DefScholar RAG Assistant
A local AI assistant that answers questions based on defense documents
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
import gradio as gr

load_dotenv()

# Configuration
DATA_PATH = "../data"
DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
LLM_MODEL = "phi3:mini" # model "mistral" is better but for storage purposes "phi3:mini" is used.

# ============ INDEX DOCUMENTS ============

def load_documents():
    """Load all PDFs from the data folder"""
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"📁 Folder '{DATA_PATH}' created. Place your PDFs here.")
        return []
    
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    print(f"✅ {len(documents)} documents loaded")
    return documents

def split_documents(documents):
    """Split documents into smaller chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✂️ {len(chunks)} chunks created")
    return chunks

def create_vector_store(chunks):
    """Create a Chroma vector database"""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    if os.path.exists(DB_PATH):
        print("📂 Existing vector database found")
        vector_store = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embeddings
        )
        vector_store.add_documents(chunks)
    else:
        print("🆕 Creating new vector database")
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=DB_PATH
        )
    
    vector_store.persist()
    print("💾 Vector database saved")
    return vector_store

def setup_qa_chain(vector_store):
    """Set up the RetrievalQA chain"""
    
    prompt_template = """You are an AI assistant for Defence (DefScholar).
Answer the question ONLY based on the context below.
If the answer is not in the context, say: "I cannot find this in the available documents."
ALWAYS mention at the end which sources you used (with document name).

Context:
{context}

Question: {question}

Answer (with source attribution):"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    llm = OllamaLLM(
        model=LLM_MODEL,
        temperature=0.2,
        num_predict=2048
    )
    
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 5}
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    
    return qa_chain

# ============ GRADIO INTERFACE ============

qa_chain = None

def ask_question(question):
    """Ask a question and get answer with sources"""
    global qa_chain
    
    if qa_chain is None:
        return "⚠️ Please index documents first using the button below."
    
    result = qa_chain.invoke({"query": question})
    answer = result["result"]
    sources = result["source_documents"]
    
    sources_text = "\n\n---\n**📚 Sources:**\n"
    for i, doc in enumerate(sources, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        sources_text += f"{i}. {source} (page {page})\n"
    
    return answer + sources_text

def index_documents():
    """Complete indexing pipeline"""
    global qa_chain
    print("🚀 Starting indexing...")
    
    documents = load_documents()
    if not documents:
        return "⚠️ No documents found in ./data folder"
    
    chunks = split_documents(documents)
    vector_store = create_vector_store(chunks)
    qa_chain = setup_qa_chain(vector_store)
    
    print("✅ Indexing complete!")
    return "✅ Indexing complete! You can now ask questions."

# ============ START ============

with gr.Blocks(title="DefScholar AI Assistant") as demo:
    gr.Markdown("""
    # 📚 DefScholar AI Research Assistant
    
    Ask questions about defense research documents. The system provides answers **with source attribution**.
    **All data stays local - no cloud.**
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            question_input = gr.Textbox(
                label="📝 Your question",
                placeholder="e.g., What are the main requirements for reconnaissance drones?",
                lines=3
            )
            
            with gr.Row():
                ask_btn = gr.Button("🔍 Ask question", variant="primary")
                index_btn = gr.Button("📂 Reindex documents", variant="secondary")
            
            output_text = gr.Textbox(
                label="💬 Answer",
                lines=15
            )
        
        with gr.Column(scale=1):
            gr.Markdown("""
            ### ℹ️ Info
            - **LLM:** phi3:mini (local Ollama model)
            - **Embedding:** Multilingual E5
            - **Database:** ChromaDB
            
            ### 📁 Documents
            Place PDFs in the `data/` folder and click 'Reindex documents'.
            """)
    
    ask_btn.click(ask_question, inputs=question_input, outputs=output_text)
    index_btn.click(index_documents, outputs=output_text)

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║           📚 DEFSCHOLAR RAG ASSISTANT                         ║
    ║                                                               ║
    ║   1. Make sure Ollama is running: 'ollama serve' in terminal  ║
    ║   2. Place PDFs in the 'data' folder                          ║
    ║   3. Click 'Reindex documents' in the web interface           ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    demo.launch(server_name="127.0.0.1", server_port=7860)
