# SimpleRAG: Local Ollama-powered RAG App using Streamlit

## Overview

SimpleRAG is an educational project that demonstrates the implementation of a Retrieval-Augmented Generation (RAG) system using Streamlit and Ollama. It allows users to upload documents, process them, and then engage in a chat interface to ask questions about the content of these documents.

## Features

- Document upload and processing (PDF, DOCX, XLSX, and plain text files)
- Text chunking and embedding generation
- Hybrid search combining semantic similarity and keyword matching
- Interactive chat interface with streaming responses
- Customizable settings for model selection, processing parameters, and chat behavior

## How It Works

1. **Document Processing:**
   - Users upload documents through the Streamlit interface.
   - The app reads and extracts text from various file formats.
   - Text is split into smaller chunks with optional overlap.
   - Each chunk is embedded using the selected Ollama model.
   - Embeddings are stored in a ChromaDB collection for efficient retrieval.

2. **Chat Interface:**
   - Users input questions about the processed documents.
   - The app performs a hybrid search to find relevant document chunks:
     - Semantic search using the embedded vectors
     - Keyword-based search for additional relevance
   - Retrieved chunks are combined with the chat history and document summaries.
   - A prompt is constructed and sent to the Ollama chat model.
   - The model's response is streamed back to the user interface.

3. **Key Components:**
   - Ollama: Provides both embedding and chat models.
   - ChromaDB: Vector database for storing and querying embeddings.
   - Streamlit: Powers the web interface and user interactions.

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/NeoVand/SimpleRAG.git
   cd SimpleRAG
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Install and set up Ollama:
   - Follow the instructions at [Ollama's official website](https://ollama.ai/) to install Ollama on your system.
   - Pull an embedding model.
   ```
   ollama pull paraphrase-multilingual
   ```
   - Pull a chat model.
   ```
   ollama pull llama3.1
   ```

5. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

6. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

## Usage

1. In the sidebar, select the embedding and chat models from the available Ollama models.
2. Adjust processing and chat settings as needed.
3. Upload documents using the file uploader.
4. Click "Process Documents" to embed and index the uploaded files.
5. Once processing is complete, use the chat interface to ask questions about the documents.
6. View relevant document chunks by expanding the "View Relevant Document Chunks" section below each response.

## Customization

- Modify the `DEFAULT_PROMPT_TEMPLATE` in the code to change how the chat model interprets the context and generates responses.
- Adjust the hybrid search algorithm in the `hybrid_search` function to fine-tune retrieval performance.
- Experiment with different Ollama models for embedding and chat to optimize for your specific use case.

## Limitations and Considerations

- The app's performance depends on the quality and capabilities of the Ollama models used.
- Large documents or a high number of uploads may require significant processing time and memory.
- The hybrid search method is a simple implementation and may not be optimal for all types of queries or documents.

## Contributing

This project is for educational purposes. Feel free to fork, modify, and experiment with the code to learn more about RAG systems and Streamlit development.

