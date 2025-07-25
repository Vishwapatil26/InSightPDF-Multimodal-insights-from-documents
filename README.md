# InSightPDF-Multimodal insights from documents
**InSightPDF** is a hybrid multimodal semantic search system that allows you to semantically search PDF documents by both **text and images**. It combines powerful **text embeddings** with **image embeddings** to enable intelligent cross-modal search (text-to-text, text-to-image, image-to-image, and image-to-text).

---

## 🚀 Features

- **Semantic Text Search:** Search document content using deep contextual embeddings.
- **Cross-modal Search:** Search relevant images from text and vice versa.
- **Table Extraction:** Extract and semantically index tables in markdown format.
- **Contextual Understanding:** Returns both chunks and their parent paragraphs for better context.
- **PDF Parser:** Parses pages, blocks, images, and tables from academic/research PDFs.
- **Elasticsearch Powered:** Fast and scalable KNN-based retrieval using dense vector indexing.

---

## 🧰 Tech Stack

| Component           | Description                                    |
|---------------------|------------------------------------------------|
| `PyMuPDF (fitz)`     | PDF parsing, block and image extraction       |
| `PIL (Pillow)`       | Image processing                              |
| `Elasticsearch`      | Vector indexing and KNN search                |
| `SentenceTransformers` | Embeddings for both text and image data    |
| `CLIP (ViT-B/32)`    | Vision-language model for image-based search  |
| `MiniLM-L6-v2`       | Lightweight transformer for text embeddings   |
| `Base64`             | Storing and retrieving images in ES           |

---

## 🧠 Models Used

### 📝 Textual Embedding Model
- **Model:** `all-MiniLM-L6-v2`
- **Library:** `sentence-transformers`
- **Use:** Generates 384-dimensional embeddings for all text and tables extracted from the PDF.

### 🖼️ Image Embedding Model
- **Model:** `clip-ViT-B-32`
- **Library:** `sentence-transformers`
- **Use:** Generates 512-dimensional embeddings for images and for text used in cross-modal search.

---

## 📂 Index Structure (Elasticsearch)

Each document indexed in Elasticsearch contains:

- `content_type` – `text`, `image`, or `table`
- `text_content` – Text chunk or serialized table (for text-based items)
- `parent_text` – Full paragraph from which the chunk was taken
- `image_content` – Base64-encoded image
- `text_embedding` – 384-dim vector for text/table
- `image_embedding` – 512-dim vector for images
- `source_page` – Page number in the original PDF

---

## ⚙️ Functionality Overview

### ✅ `create_index()`
Creates a custom Elasticsearch index with separate vector fields for text and image embeddings, along with other metadata fields.

### 📖 `parse_pdf_and_generate_data(pdf_path)`
Parses each page of the PDF to:
- Chunk text into smaller parts
- Serialize tables to markdown
- Extract and encode images
- Create dense embeddings and prepare documents for indexing

### 📥 `bulk(es_client, documents)`
Bulk indexes all processed documents (text, tables, images) into Elasticsearch.

---

## 🔍 Search Functions

### 🔎 `search_text_by_text(query, k=5)`
- Uses `MiniLM` to embed query text.
- Retrieves matching text and table chunks based on cosine similarity.
- Also returns `parent_text` for richer context.

### 🖼️ `search_images_by_text(query, k=3)`
- Uses `CLIP` to encode text query into vision embedding space.
- Retrieves similar image embeddings from Elasticsearch.

### 🧠 `search_text_by_image(image_path, k=5)`
- Uses `CLIP` to encode the image and retrieve related text/table embeddings.
- Useful for understanding images through surrounding textual context.

### 🖼️ `search_images_by_image(image_path, k=3)`
- Uses `CLIP` to find visually similar images across the PDF.

---

## 🧪 Example Usage

```bash
python main.py
