"""
DefScholar RAG Assistant
Een lokale AI-assistent die vragen beantwoordt op basis van defensiedocumenten
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

# Configuratie
DATA_PATH = "./data"
DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
LLM_MODEL = "mistral"

# ============ DOCUMENTEN INDEXEREN ============

def load_documents():
    """Laad alle PDF's uit de data folder"""
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"📁 Map '{DATA_PATH}' aangemaakt. Plaats hier je PDF's.")
        return []
    
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    print(f"✅ {len(documents)} documenten geladen")
    return documents

def split_documents(documents):
    """Splits documenten in kleinere chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✂️ {len(chunks)} chunks gemaakt")
    return chunks

def create_vector_store(chunks):
    """Maak een Chroma vector database"""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    if os.path.exists(DB_PATH):
        print("📂 Bestaande vector database gevonden")
        vector_store = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embeddings
        )
        vector_store.add_documents(chunks)
    else:
        print("🆕 Nieuwe vector database wordt aangemaakt")
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=DB_PATH
        )
    
    vector_store.persist()
    print("💾 Vector database opgeslagen")
    return vector_store

def setup_qa_chain(vector_store):
    """Zet de RetrievalQA chain op"""
    
    prompt_template = """Je bent een AI-assistent voor Defensie (DefScholar).
Beantwoord de vraag ALLEEN op basis van de onderstaande context.
Als het antwoord niet in de context staat, zeg dan: "Ik kan dit niet vinden in de beschikbare documenten."
Vermeld ALTIJD aan het einde van je antwoord welke bronnen je hebt gebruikt (met documentnaam).

Context:
{context}

Vraag: {question}

Antwoord (met bronvermelding):"""

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
    """Stel een vraag en krijg antwoord met bronnen"""
    global qa_chain
    
    if qa_chain is None:
        return "⚠️ Indexeer eerst documenten met de knop hieronder."
    
    result = qa_chain.invoke({"query": question})
    antwoord = result["result"]
    bronnen = result["source_documents"]
    
    bronnen_text = "\n\n---\n**📚 Bronnen:**\n"
    for i, doc in enumerate(bronnen, 1):
        bron = doc.metadata.get("source", "Onbekend")
        page = doc.metadata.get("page", "?")
        bronnen_text += f"{i}. {bron} (pagina {page})\n"
    
    return antwoord + bronnen_text

def index_documents():
    """Complete indexeer pipeline"""
    global qa_chain
    print("🚀 Start indexeren...")
    
    documents = load_documents()
    if not documents:
        return "⚠️ Geen documenten gevonden in ./data map"
    
    chunks = split_documents(documents)
    vector_store = create_vector_store(chunks)
    qa_chain = setup_qa_chain(vector_store)
    
    print("✅ Indexeren compleet!")
    return "✅ Indexeren compleet! Je kunt nu vragen stellen."

# ============ START ============

with gr.Blocks(title="DefScholar AI Assistant") as demo:
    gr.Markdown("""
    # 📚 DefScholar AI Research Assistant
    
    Stel vragen over defensie-onderzoeksdocumenten. Het systeem geeft antwoord **met bronvermelding**.
    **Alle data blijft lokaal - geen cloud.**
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            question_input = gr.Textbox(
                label="📝 Jouw vraag",
                placeholder="Bijv: Wat zijn de belangrijkste eisen voor verkenningsdrones?",
                lines=3
            )
            
            with gr.Row():
                ask_btn = gr.Button("🔍 Stel vraag", variant="primary")
                index_btn = gr.Button("📂 Herindexeer documenten", variant="secondary")
            
            output_text = gr.Textbox(
                label="💬 Antwoord",
                lines=15
            )
        
        with gr.Column(scale=1):
            gr.Markdown("""
            ### ℹ️ Info
            - **LLM:** Mistral 7B (lokaal)
            - **Embedding:** Multilingual E5
            - **Database:** ChromaDB
            
            ### 📁 Documenten
            Plaats PDF's in de `data/` map en klik op 'Herindexeer documenten'.
            """)
    
    ask_btn.click(ask_question, inputs=question_input, outputs=output_text)
    index_btn.click(index_documents, outputs=output_text)

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║           📚 DEFSCHOLAR RAG ASSISTANT                         ║
    ║                                                               ║
    ║   1. Zorg dat Ollama draait: 'ollama serve' in aparte terminal║
    ║   2. Plaats PDF's in de 'data' map                            ║
    ║   3. Klik 'Herindexeer documenten' in de web interface        ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    demo.launch(server_name="127.0.0.1", server_port=7860)

