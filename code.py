import fitz  # PyMuPDF
from PIL import Image
import io
import base64
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import os
import sys

try:
    from IPython.display import display
except ImportError:
    print("IPython.display not found. Images will not be displayed inline.")
    # Define a dummy display function if IPython is not available
    def display(image):
        print("--- [Image would be displayed here in a notebook environment] ---")
        image.show() # Fallback to default viewer if not in notebook

# --- 1. Configuration & Initialization (Hierarchical Hybrid Model) ---

# Elasticsearch connection details
ES_HOST = "http://localhost:9200"
ES_INDEX_NAME = "hybrid_multimodal_search" # Using a new index name for the new architecture

# Model 1: CLIP for Image Embeddings and cross-modal search
print("Loading CLIP model (for images)...")
IMAGE_MODEL = SentenceTransformer('clip-ViT-B-32')
IMAGE_EMBEDDING_DIM = 512
print("CLIP model loaded successfully.")

# Model 2: A powerful text model for deep text understanding
print("Loading Text Embedding model (for text and tables)...")
TEXT_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
TEXT_EMBEDDING_DIM = 384
print("Text model loaded successfully.")

# Initialize Elasticsearch client
try:
    es_client = Elasticsearch(hosts=[ES_HOST])
    if not es_client.ping():
        raise ConnectionError("Could not connect to Elasticsearch")
    print("Connected to Elasticsearch successfully.")
except Exception as e:
    print(f"Error connecting to Elasticsearch: {e}")
    sys.exit()

# PDF file path (Using raw string for Windows path compatibility)
PDF_PATH = r"D:\CDAC\task4\cat.pdf"
if not os.path.exists(PDF_PATH):
    print(f"Error: PDF file not found at '{PDF_PATH}'. Please create it.")
    sys.exit()

# --- 2. Elasticsearch Index Setup (Hybrid Mapping) ---

def create_index():
    """
    Creates the Elasticsearch index with a new 'parent_text' field for full context.
    """
    if es_client.indices.exists(index=ES_INDEX_NAME):
        print(f"Index '{ES_INDEX_NAME}' already exists. Deleting and recreating.")
        es_client.indices.delete(index=ES_INDEX_NAME)

    index_mapping = {
        "properties": {
            "content_type": {"type": "keyword"},
            "text_content": {"type": "text"},     # The small chunk for embedding
            # --- NEW FIELD ---
            "parent_text": {
                "type": "text", 
                "index": "false" # We don't need to keyword search this, just retrieve it.
            },
            "image_content": {"type": "binary"},
            "source_page": {"type": "integer"},
            "image_embedding": {
                "type": "dense_vector", "dims": IMAGE_EMBEDDING_DIM,
                "index": "true", "similarity": "cosine"
            },
            "text_embedding": {
                "type": "dense_vector", "dims": TEXT_EMBEDDING_DIM,
                "index": "true", "similarity": "cosine"
            }
        }
    }
    es_client.indices.create(index=ES_INDEX_NAME, mappings=index_mapping)
    print(f"Index '{ES_INDEX_NAME}' created successfully with 'parent_text' field.")

# --- 3. PDF Parsing and Data Preparation (Hybrid Logic) ---

def create_text_chunks(text, chunk_size=256, overlap=50):
    """A more robust chunker suitable for models with larger token limits."""
    if not text: return []
    words = text.split()
    if len(words) < chunk_size: return [" ".join(words)]
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

def serialize_table_to_markdown(table_data, table_title=""):
    """Converts extracted table data into a descriptive Markdown string."""
    if not table_data: return ""
    markdown_str = f"### {table_title}\n\n" if table_title else "The following is a data table:\n\n"
    header = [str(h).strip() if h is not None else "" for h in table_data[0]]
    markdown_str += "| " + " | ".join(header) + " |\n"
    markdown_str += "| " + " | ".join(["---"] * len(header)) + " |\n"
    for row in table_data[1:]:
        row = [str(r).strip() if r is not None else "" for r in row]
        markdown_str += "| " + " | ".join(row) + " |\n"
    return markdown_str

def parse_pdf_and_generate_data(pdf_path):
    """
    Parses PDF, generating specialized embeddings for text/tables vs. images.
    This is the corrected implementation of the hybrid architecture.
    """
    print(f"Parsing PDF with Hybrid Embedding Logic: {pdf_path}")
    doc = fitz.open(pdf_path)
    documents_to_index = []
    doc_id = 1

    for page_num, page in enumerate(doc):
        print(f"  - Processing page {page_num + 1}/{len(doc)}")
        for block in page.get_text("blocks"):
            parent_paragraph = block[4].strip()
            if parent_paragraph:
                # Create smaller chunks from the parent paragraph
                text_chunks = create_text_chunks(parent_paragraph)
                for chunk in text_chunks:
                    if not chunk.strip(): continue
                    # For each chunk, create a document that contains BOTH the chunk and its parent
                    documents_to_index.append({
                        "_id": doc_id, "_index": ES_INDEX_NAME,
                        "_source": {
                            "content_type": "text",
                            "text_content": chunk,             # The small chunk for precise search
                            "parent_text": parent_paragraph,  # The full paragraph for context
                            "source_page": page_num + 1,
                            "text_embedding": TEXT_MODEL.encode(chunk) # Embedding is of the small chunk
                        }
                    })
                    doc_id += 1

        # B. Extract and embed images using the IMAGE_MODEL (CLIP)
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            pil_image = Image.open(io.BytesIO(image_bytes))
            documents_to_index.append({
                "_id": doc_id, "_index": ES_INDEX_NAME,
                "_source": {
                    "content_type": "image", "source_page": page_num + 1,
                    "image_content": base64.b64encode(image_bytes).decode('utf-8'),
                    "image_embedding": IMAGE_MODEL.encode(pil_image)
                }
            })
            doc_id += 1
            
        # C. Extract, serialize, and embed tables using the TEXT_MODEL
        tables = page.find_tables()
        if tables.tables: print(f"    Found {len(tables.tables)} tables on page {page_num + 1}.")
        for table in tables:
            table_data = table.extract()
            if not table_data: continue
            table_markdown = serialize_table_to_markdown(table_data, f"Table from page {page_num+1}")
            documents_to_index.append({
                "_id": doc_id, "_index": ES_INDEX_NAME,
                "_source": {
                    "content_type": "table", "text_content": table_markdown, "source_page": page_num + 1,
                    "text_embedding": TEXT_MODEL.encode(table_markdown)
                }
            })
            doc_id += 1

    print(f"PDF parsing complete. Found {len(documents_to_index)} items to index.")
    return documents_to_index

# --- 4. Search Functions (Rewritten for Hybrid Search) ---

def search_text_by_text(query_text, k=3):
    """Performs semantic search for TEXT and TABLES using the powerful TEXT_MODEL."""
    print(f"\n--- Searching TEXT/TABLES with Text Model: '{query_text}' ---")
    query_embedding = TEXT_MODEL.encode(query_text)
    knn_query = {"field": "text_embedding", "query_vector": query_embedding, "k": k, "num_candidates": 100}
    query = {"bool": {"filter": [{"terms": {"content_type": ["text", "table"]}}]}}
    response = es_client.search(index=ES_INDEX_NAME, knn=knn_query, query=query, _source=["content_type", "source_page", "text_content"])
    print(f"Found {len(response['hits']['hits'])} relevant text/table chunks:")
    for hit in response['hits']['hits']:
        print(f"  - Score: {hit['_score']:.4f}, Type: {hit['_source']['content_type']}, Page: {hit['_source']['source_page']}")
        print(f"    Content: \"{hit['_source']['text_content'][:300]}...\"")

def search_images_by_text(query_text, k=3,min_score=0.5):
    """Performs cross-modal search for IMAGES using the IMAGE_MODEL (CLIP)."""
    print(f"\n--- Cross-Modal Search: IMAGES matching text '{query_text}' ---")
    query_embedding = IMAGE_MODEL.encode(query_text)
    knn_query = {"field": "image_embedding", "query_vector": query_embedding, "k": k, "num_candidates": 100}
    query = {"bool": {"filter": [{"term": {"content_type": "image"}}]}}
    response = es_client.search(index=ES_INDEX_NAME, knn=knn_query, query=query, _source=["source_page", "image_content"])
    print(f"Found {len(response['hits']['hits'])} images:")
    for hit in response['hits']['hits']:
        if hit['_score'] < min_score:
            continue
        print(f"\n  - Score: {hit['_score']:.4f}, Page: {hit['_source']['source_page']}")
        img = Image.open(io.BytesIO(base64.b64decode(hit['_source']['image_content'])))
        display(img)

def search_text_by_text(query_text, k=5, min_score=0.70):
    """
    Performs semantic search, filters by a minimum score, and returns the full parent paragraph for context.
    """
    print(f"\n--- Searching TEXT/TABLES with Text Model (min_score={min_score}): '{query_text}' ---")
    query_embedding = TEXT_MODEL.encode(query_text)
    
    knn_query = {"field": "text_embedding", "query_vector": query_embedding, "k": k, "num_candidates": 100}
    query = {"bool": {"filter": [{"terms": {"content_type": ["text", "table"]}}]}}
    
    # Request the new 'parent_text' field from the search
    response = es_client.search(
        index=ES_INDEX_NAME, knn=knn_query, query=query, 
        _source=["content_type", "source_page", "parent_text", "text_content"] # Request all for safety
    )
    
    print(f"Found {len(response['hits']['hits'])} potential matches. Filtering by score...")
    
    count = 0
    for hit in response['hits']['hits']:
        if hit['_score'] < min_score:
            continue
            
        count += 1
        source = hit['_source'] # Use a variable for cleaner access
        print(f"\n  - Result {count}: Score: {hit['_score']:.4f}, Type: {source.get('content_type')}, Page: {source.get('source_page')}")
        
        # --- ROBUST FIX: Use .get() to avoid KeyError ---
        # Display parent_text if it exists, otherwise fall back to text_content
        context = source.get('parent_text', source.get('text_content', 'No text content found.'))
        print(f"    Full Context: \"{context}\"")
        
    if count == 0:
        print("  No relevant results found above the minimum score.")

# APPLY THE SAME ROBUST FIX TO search_text_by_image
def search_text_by_image(image_path, k=5, min_score=0.50):
    # ... (start of function is the same) ...
    try:
        response = es_client.search(
            index=ES_INDEX_NAME, knn=knn_query, query=query, 
            _source=["content_type", "source_page", "parent_text", "text_content"] # Request all for safety
        )
        
        print(f"Found {len(response['hits']['hits'])} potential matches. Filtering by score...")
        
        count = 0
        for hit in response['hits']['hits']:
            if hit['_score'] < min_score:
                continue
            
            count += 1
            source = hit['_source'] # Use a variable
            print(f"\n  - Result {count}: Score: {hit['_score']:.4f}, Type: {source.get('content_type')}, Page: {source.get('source_page')}")
            
            # --- ROBUST FIX: Use .get() to avoid KeyError ---
            context = source.get('parent_text', source.get('text_content', 'No text content found.'))
            print(f"    Full Context: \"{context}\"")

        if count == 0:
            print("  No relevant results found above the minimum score.")

    except Exception as e:
        print(f"Error during image-to-text search. Error: {e}")


def search_text_by_image(image_path, k=5, min_score=0.50):
    """
    Performs cross-modal search, filters by score, and returns full parent context.
    """
    print(f"\n--- Cross-Modal Search: TEXT/TABLES matching image (min_score={min_score}): '{image_path}' ---")
    if not os.path.exists(image_path): print(f"Error: Query image not found at '{image_path}'"); return
    
    query_embedding = IMAGE_MODEL.encode(Image.open(image_path))
    knn_query = {"field": "text_embedding", "query_vector": query_embedding, "k": k, "num_candidates": 100}
    query = {"bool": {"filter": [{"terms": {"content_type": ["text", "table"]}}]}}
    
    try:
        # Request the new 'parent_text' field
        response = es_client.search(
            index=ES_INDEX_NAME, knn=knn_query, query=query, 
            _source=["content_type", "source_page", "parent_text"]
        )
        
        print(f"Found {len(response['hits']['hits'])} potential matches. Filtering by score...")
        
        count = 0
        for hit in response['hits']['hits']:
            # --- NEW: Relevance Score Filtering ---
            if hit['_score'] < min_score:
                continue
            
            count += 1
            print(f"\n  - Result {count}: Score: {hit['_score']:.4f}, Type: {hit['_source']['content_type']}, Page: {hit['_source']['source_page']}")
            
            # --- NEW: Display the full parent paragraph for context ---
            print(f"    Full Context: \"{hit['_source']['parent_text']}\"")

        if count == 0:
            print("  No relevant results found above the minimum score.")

    except Exception as e:
        print(f"Error during image-to-text search. Error: {e}")

def search_images_by_image(image_path, k=3):
    """Performs image-to-image search using the IMAGE_MODEL (CLIP)."""
    print(f"\n--- Searching for IMAGES similar to image: '{image_path}' ---")
    if not os.path.exists(image_path): print(f"Error: Query image not found at '{image_path}'"); return

    query_embedding = IMAGE_MODEL.encode(Image.open(image_path))
    knn_query = {"field": "image_embedding", "query_vector": query_embedding, "k": k + 1, "num_candidates": 100}
    query = {"bool": {"filter": [{"term": {"content_type": "image"}}]}}
    
    response = es_client.search(index=ES_INDEX_NAME, knn=knn_query, query=query, _source=["source_page", "image_content"])
    
    with open(image_path, "rb") as f: query_image_base64 = base64.b64encode(f.read()).decode('utf-8')
    final_hits = [h for h in response['hits']['hits'] if h['_source']['image_content'] != query_image_base64][:k]

    print(f"Found {len(final_hits)} similar images:")
    for i, hit in enumerate(final_hits):
        print(f"\n  - Result {i+1}: Score: {hit['_score']:.4f}, Page: {hit['_source']['source_page']}")
        img = Image.open(io.BytesIO(base64.b64decode(hit['_source']['image_content'])))
        display(img)

# --- 5. Main Execution Block ---
if __name__ == "__main__":
    create_index()
    documents = parse_pdf_and_generate_data(PDF_PATH)
    
    if documents:
        print("\nIndexing documents into Elasticsearch...")
        try:
            success, failed = bulk(es_client, documents)
            print(f"Successfully indexed {success} documents.")
            if failed: print(f"Failed to index {len(failed)} documents.")
        except Exception as e:
            print(f"Error during bulk indexing: {e}")
    else:
        print("No documents found to index.")

    es_client.indices.refresh(index=ES_INDEX_NAME)

    print("\n" + "="*50 + "\n--- RUNNING EXAMPLE SEARCHES ---\n" + "="*50)
    
    # Example 2: Ask a question about text searchtable in the document (uses Text Model)
    search_text_by_text("White cats belong to")

    # Example 3: Text-to-image search (uses CLIP)
    search_images_by_text("a photo of a white cat")
    
    # Example 4: Image-to-image search (uses CLIP)
    query_image_path = "image.png" # You need to create this file in your directory
    if os.path.exists(query_image_path):
        search_images_by_image(query_image_path, k=2)
    else:
        print(f"\nSkipping image-to-image search: '{query_image_path}' not found.")
        
    # Example 5: Image-to-text search (uses CLIP to query Text Embeddings)
    # if os.path.exists(query_image_path):
    #     search_text_by_image(query_image_path, k=2)
    # else:
    #     print(f"\nSkipping image-to-text search: '{query_image_path}' not found.")
