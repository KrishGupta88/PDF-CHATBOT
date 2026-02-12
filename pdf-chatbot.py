import json
from typing import List
from langchain_community.chat_models import ChatOllama
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import subprocess

load_dotenv()

def check_ollama_model(model_name: str) -> bool:
    try:
        result = subprocess.run(
            ['ollama', 'list'], 
            capture_output=True, 
            text=True, 
            check=True
        )
        if model_name in result.stdout:
            print(f"Model '{model_name}' is available")
            return True
        else:
            print(f"Model '{model_name}' not found. Pulling it now...")
            print(f"   This may take a few minutes...")
            pull_result = subprocess.run(
                ['ollama', 'pull', model_name],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"Model '{model_name}' pulled successfully")
            return True
    except subprocess.CalledProcessError as e:
        print(f"Error checking/pulling Ollama model: {e}")
        return False
    except FileNotFoundError:
        print(f"Ollama is not installed or not in PATH")
        return False

def partition_document(file_path: str):
    """Extract elements from PDF using unstructured"""
    print(f"Partitioning document: {file_path}")
    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True
    )
    print(f"Extracted {len(elements)} elements")
    return elements

# Test with your PDF file
file_path = "NIPS-2017-attention-is-all-you-need-Paper (2).pdf"
elements = partition_document(file_path)

print("\n Sample element (36th):")
if len(elements) > 36:
    print(elements[36].to_dict())

# Gather all images
images = [element for element in elements if element.category == 'Image']
print(f"\n Found {len(images)} images")
if images:
    print("First image metadata keys:", list(images[0].to_dict().keys()))

# Gather all tables
tables = [element for element in elements if element.category == 'Table']
print(f"\n Found {len(tables)} tables")
if tables:
    print("First table metadata keys:", list(tables[0].to_dict().keys()))

def create_chunks_by_title(elements):
    """Create intelligent chunks using title-based strategy"""
    print("\n Creating smart chunks...")
    chunks = chunk_by_title(
        elements,
        max_characters=3000,
        new_after_n_chars=2400,
        combine_text_under_n_chars=500
    )
    print(f" Created {len(chunks)} chunks")
    return chunks

# Create chunks
chunks = create_chunks_by_title(elements)

def separate_content_types(chunk):
    """Analyze what types of content are in a chunk"""
    content_data = {
        'text': chunk.text,
        'tables': [],
        'images': [],
        'types': ['text']
    }
    
    # Check for tables and images in original elements
    if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__
            
            # Handle tables
            if element_type == 'Table':
                content_data['types'].append('table')
                table_html = getattr(element.metadata, 'text_as_html', element.text)
                content_data['tables'].append(table_html)
            
            # Handle images
            elif element_type == 'Image':
                if hasattr(element, 'metadata') and hasattr(element.metadata, 'image_base64'):
                    content_data['types'].append('image')
                    content_data['images'].append(element.metadata.image_base64)
    
    content_data['types'] = list(set(content_data['types']))
    return content_data

def create_ai_enhanced_summary(text: str, tables: List[str], images: List[str]) -> str:
    """Create AI-enhanced summary for mixed content"""
    try:
        # Check if llava model is available
        if not check_ollama_model("llava"):
            raise Exception("llava model not available")
        
        # Initialize LLM (needs vision model for images)
        llm = ChatOllama(
            model="llava",
            temperature=0
        )
        
        # Build the text prompt
        prompt_text = f"""You are creating a searchable description for document content retrieval.

CONTENT TO ANALYZE:

TEXT CONTENT:
{text}

"""
        
        # Add tables if present
        if tables:
            prompt_text += "TABLES:\n"
            for i, table in enumerate(tables):
                prompt_text += f"Table {i+1}:\n{table}\n\n"
        
        prompt_text += """
YOUR TASK:
Generate a comprehensive, searchable description that covers:
1. Key facts, numbers, and data points from text and tables
2. Main topics and concepts discussed
3. Questions this content could answer
4. Visual content analysis (charts, diagrams, patterns in images)
5. Alternative search terms users might use

Make it detailed and searchable - prioritize findability over brevity.

SEARCHABLE DESCRIPTION:"""
        
        # Build message content starting with text
        message_content = [{"type": "text", "text": prompt_text}]
        
        # Add images to the message
        for image_base64 in images:
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            })
        
        # Send to AI and get response
        message = HumanMessage(content=message_content)
        response = llm.invoke([message])
        return response.content
        
    except Exception as e:
        print(f" AI summary failed: {e}")
        # Fallback to simple summary
        summary = f"{text[:300]}..."
        if tables:
            summary += f" [Contains {len(tables)} table(s)]"
        if images:
            summary += f" [Contains {len(images)} image(s)]"
        return summary

def summarise_chunks(chunks):
    """Process chunks with AI Summaries"""
    print("\n Processing chunks with AI Summaries...")
    langchain_documents = []
    total_chunks = len(chunks)
    
    for i, chunk in enumerate(chunks):
        current_chunk = i + 1
        print(f"\n Processing chunk {current_chunk}/{total_chunks}")
        
        # Analyze chunk content
        content_data = separate_content_types(chunk)
        
        # Debug prints
        print(f"   Types found: {content_data['types']}")
        print(f"   Tables: {len(content_data['tables'])}, Images: {len(content_data['images'])}")
        
        # Create AI-enhanced summary if chunk has tables/images
        if content_data['tables'] or content_data['images']:
            print(f"   → Creating AI summary for mixed content...")
            try:
                enhanced_content = create_ai_enhanced_summary(
                    content_data['text'],
                    content_data['tables'],
                    content_data['images']
                )
                print(f"AI summary created successfully")
                print(f"   Preview: {enhanced_content[:150]}...")
            except Exception as e:
                print(f"AI summary failed: {e}")
                enhanced_content = content_data['text']
        else:
            print(f"Using raw text (no tables/images)")
            enhanced_content = content_data['text']
        
        # Create LangChain Document with rich metadata
        doc = Document(
            page_content=enhanced_content,
            metadata={
                "original_content": json.dumps({
                    "raw_text": content_data['text'],
                    "tables_html": content_data['tables'],
                    "images_base64": content_data['images']
                })
            }
        )
        langchain_documents.append(doc)
    
    print(f"\n Processed {len(langchain_documents)} chunks")
    return langchain_documents

# Process chunks with AI
processed_chunks = summarise_chunks(chunks)

def export_chunks_to_json(chunks, filename="chunks_export.json"):
    """Export processed chunks to clean JSON format"""
    export_data = []
    
    for i, doc in enumerate(chunks):
        chunk_data = {
            "chunk_id": i + 1,
            "enhanced_content": doc.page_content,
            "metadata": {
                "original_content": json.loads(doc.metadata.get("original_content", "{}"))
            }
        }
        export_data.append(chunk_data)
    
    # Save to file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n Exported {len(export_data)} chunks to {filename}")
    return export_data

# Export your chunks
json_data = export_chunks_to_json(processed_chunks)

# Initialize embedding model
print("\n Initializing embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Test embedding
vector = embedding_model.embed_query("Attention is all you need")
print(f"Embedding dimension: {len(vector)}")

def create_vector_store(documents, persist_directory="dbv1/chroma_db"):
    """Create and persist ChromaDB vector store"""
    print(f"\n Creating embeddings and storing in ChromaDB...")
    
    # Create ChromaDB vector store
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore

# Create the vector store
db = create_vector_store(processed_chunks)

# Query 1
print("\n" + "="*80)
print("QUERY 1: What are the two main components of the Transformer architecture?")
print("="*80)
query1 = "What are the two main components of the Transformer architecture?"
retriever = db.as_retriever(search_kwargs={"k": 3})
chunks1 = retriever.invoke(query1)

# Export to JSON
export_chunks_to_json(chunks1, "rag_results_query1.json")

# Query 2
print("\n" + "="*80)
print("QUERY 2: How many attention heads does the Transformer use?")
print("="*80)
query2 = "How many attention heads does the Transformer use, and what is the dimension of each head?"
chunks2 = retriever.invoke(query2)

def generate_final_answer(chunks, query, model_name="llama3.2"):
    """Generate final answer using multimodal content"""
    try:
        # Check if model is available
        if not check_ollama_model(model_name):
            raise Exception(f"{model_name} model not available")
        
        # Initialize LLM
        llm = ChatOllama(
            model=model_name,
            temperature=0,
            num_ctx=4096  # Increase context window
        )
        
        # Build the text prompt
        prompt_text = f"""Based on the following documents, please answer this question:

QUESTION: {query}

RETRIEVED DOCUMENTS:

"""
        
        for i, chunk in enumerate(chunks):
            prompt_text += f"--- Document {i+1} ---\n"
            
            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata["original_content"])
                
                # Add raw text
                raw_text = original_data.get("raw_text", "")
                if raw_text:
                    prompt_text += f"TEXT:\n{raw_text}\n\n"
                
                # Add tables as HTML
                tables_html = original_data.get("tables_html", [])
                if tables_html:
                    prompt_text += "TABLES:\n"
                    for j, table in enumerate(tables_html):
                        prompt_text += f"Table {j+1}:\n{table}\n\n"
            
            prompt_text += "\n"
        
        prompt_text += """
Please provide a clear, comprehensive answer using the text and tables above.
If the documents don't contain sufficient information to answer the question, say "I don't have enough information to answer that question based on the provided documents."

ANSWER:"""
        
        # For text-only models, just use text content
        message = HumanMessage(content=prompt_text)
        response = llm.invoke([message])
        return response.content
        
    except Exception as e:
        print(f"Answer generation failed: {e}")
        return f"Sorry, I encountered an error while generating the answer: {str(e)}"

# Generate final answer
print("\nGenerating final answer...")
final_answer = generate_final_answer(chunks2, query2)
print("FINAL ANSWER:")
print(final_answer)
