import streamlit as st
from streamlit_chat import message
import os
import time
from pathlib import Path
import chromadb
import requests
import json
import PyPDF2
import docx
import openpyxl
import chardet
from typing import List, Dict, Union
import io
from collections import Counter
import re
import hashlib

# Set page config at the very beginning
st.set_page_config(page_title="SimpleRAG", page_icon="💬", layout="wide")

# Constants for initiating the interface
DEFAULT_OLLAMA_URL = 'http://localhost:11434'
DEFAULT_CHUNK_SIZE = 1000  # the number of characters in each chunk 
DEFAULT_CHUNK_OVERLAP = 200  # the number of overlapping characters between two consecutive chunks
DEFAULT_NUM_CHUNKS = 4  # the number of chunks that are added to the chat context
DEFAULT_TEMPERATURE = 0.1  # the degree of model creativity (less = more strict)
DEFAULT_MEMORY_SIZE = 3  # the number of recent Q&A pairs to keep in memory and add to the chat context

# Default prompt template for the chat model.
DEFAULT_PROMPT_TEMPLATE = """Here are the chunks retrieved based on similarity search from user's question. They might or might not be directly related to the question:
{context}

Document Summaries:
{summaries}

Chat History:
{memory}

User Question: {question}

Please provide a comprehensive answer to the user's question based on the given context, document summaries, and chat history. If the information is not available in the provided context, please state that you don't have enough information to answer the question.

Your Answer:"""

# Ollama API functions

@st.cache_data(show_spinner=False)
def check_ollama_connection(ollama_url: str) -> bool:
    """
    Check if the Ollama server is accessible and responding.

    This function attempts to connect to the Ollama server using the provided URL.
    It sends a GET request to the '/api/tags' endpoint and checks if the response is successful.

    Args:
        ollama_url (str): The URL of the Ollama server.

    Returns:
        bool: True if the connection is successful, False otherwise.

    Note:
        This function is cached using st.cache_data to improve performance.    """
    try:
        requests.get(f'{ollama_url}/api/tags', timeout=5).raise_for_status()
        return True
    except requests.RequestException:
        return False

@st.cache_data(show_spinner=False)
def get_ollama_models(ollama_url: str) -> List[str]:
    """
    Retrieve a list of available models from the Ollama server.

    This function sends a GET request to the '/api/tags' endpoint of the Ollama server
    to fetch the list of available models.

    Args:
        ollama_url (str): The URL of the Ollama server.

    Returns:
        List[str]: A list of model names available on the Ollama server.

    Note:
        This function is cached using st.cache_data to improve performance.
        If there's an error connecting to the server, it will display an error message
        in the Streamlit sidebar and return an empty list.
    """
    try:
        response = requests.get(f'{ollama_url}/api/tags', timeout=5)
        response.raise_for_status()
        return [model['name'] for model in response.json()['models']]
    except requests.RequestException as e:
        st.sidebar.error(f"Error connecting to Ollama: {str(e)}")
        return []

def embed_documents(ollama_url: str, model: str, texts: List[str]) -> Union[List[List[float]], None]:
    """
    Generate embeddings for a list of text documents using the specified Ollama model.

    This function sends POST requests to the Ollama server to generate embeddings
    for each text document in the input list.

    Args:
        ollama_url (str): The URL of the Ollama server.
        model (str): The name of the Ollama model to use for generating embeddings.
        texts (List[str]): A list of text documents to embed.

    Returns:
        Union[List[List[float]], None]: A list of embeddings (each embedding is a list of floats)
                                        if successful, or None if there's an error.

    Note:
        If there's an error generating embeddings for any document, the function will
        display an error message in the Streamlit sidebar and return None.
    """
    embeddings = []
    for text in texts:
        try:
            response = requests.post(
                f'{ollama_url}/api/embeddings',
                json={'model': model, 'prompt': text},
                timeout=30
            )
            response.raise_for_status()
            embedding = response.json()['embedding']
            embeddings.append(embedding)
        except requests.RequestException as e:
            st.sidebar.error(f"Error getting embedding: {str(e)}")
            return None
    return embeddings

def chat_with_documents(ollama_url: str, model: str, prompt: str, temperature: float):
    """
    Generate a response from the Ollama model based on the given prompt.

    This function sends a POST request to the Ollama server to generate a response
    using the specified model and prompt.

    Args:
        ollama_url (str): The URL of the Ollama server.
        model (str): The name of the Ollama model to use for generating the response.
        prompt (str): The input prompt for the model.
        temperature (float): The temperature parameter for controlling response randomness.

    Returns:
        requests.Response: The streaming response from the Ollama server if successful,
                           or None if there's an error.

    Note:
        This function returns a streaming response, which should be processed
        iteratively to get the full generated text.
        If there's an error, it will display an error message in the Streamlit sidebar.
    """
    try:
        response = requests.post(
            f'{ollama_url}/api/generate',
            json={'model': model, 'prompt': prompt, 'temperature': temperature},
            stream=True
        )
        response.raise_for_status()
        return response
    except requests.RequestException as e:
        st.sidebar.error(f"Error generating response: {str(e)}")
        return None

# File processing functions

def detect_encoding(file_content: bytes) -> str:
    """
    Detect the encoding of a given byte string.

    This function uses the chardet library to guess the encoding of the input bytes.

    Args:
        file_content (bytes): The byte string content of a file.

    Returns:
        str: The detected encoding of the file content.
    """
    return chardet.detect(file_content)['encoding']

def read_file(file: io.BytesIO, filename: str) -> str:
    """
    Read and extract text content from various file types.

    This function supports reading PDF, DOCX, XLSX, and plain text files.
    It attempts to detect the file type based on the file extension and uses
    appropriate libraries to extract the text content.

    Args:
        file (io.BytesIO): A file-like object containing the file content.
        filename (str): The name of the file, used to determine the file type.

    Returns:
        str: The extracted text content from the file.

    Note:
        If there's an error reading the file, it will display an error message
        in the Streamlit sidebar and return an empty string.
    """
    _, file_extension = os.path.splitext(filename)
    try:
        if file_extension.lower() == '.pdf':
            pdf_reader = PyPDF2.PdfReader(file)
            return ' '.join([page.extract_text() for page in pdf_reader.pages])
        elif file_extension.lower() == '.docx':
            doc = docx.Document(file)
            return ' '.join([para.text for para in doc.paragraphs])
        elif file_extension.lower() in ['.xlsx', '.xls']:
            workbook = openpyxl.load_workbook(file)
            return ' '.join([str(cell.value) for sheet in workbook.worksheets for row in sheet.iter_rows() for cell in row if cell.value])
        else:  # Assume it's a text file
            content = file.read()
            encoding = detect_encoding(content)
            return content.decode(encoding)
    except Exception as e:
        st.sidebar.error(f"Error reading file {filename}: {str(e)}")
        return ""

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Split a long text into smaller overlapping chunks.

    This function divides the input text into chunks of specified size,
    with a defined overlap between consecutive chunks.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The size of each chunk in characters.
        chunk_overlap (int): The number of overlapping characters between chunks.

    Returns:
        List[str]: A list of text chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks

def summarize_document(ollama_url: str, model: str, text: str) -> str:
    """
    Generate a concise summary of a given document using the Ollama model.

    This function sends a request to the Ollama server to summarize the input text
    using the specified model.

    Args:
        ollama_url (str): The URL of the Ollama server.
        model (str): The name of the Ollama model to use for summarization.
        text (str): The input text to be summarized.

    Returns:
        str: The generated summary of the input text.

    Note:
        If the summarization fails, it returns a failure message.
    """
    prompt = f"Please provide a concise summary of the following document:\n\n{text}\n\nSummary:"
    response = chat_with_documents(ollama_url, model, prompt, 0.1)
    if response:
        summary = ""
        for line in response.iter_lines():
            if line:
                try:
                    json_response = json.loads(line)
                    if 'response' in json_response:
                        summary += json_response['response']
                except json.JSONDecodeError:
                    continue
        return summary
    return "Failed to generate summary."

def process_documents(ollama_url: str, model: str, files: List[io.BytesIO], chunk_size: int, chunk_overlap: int) -> List[Dict[str, Union[str, Dict[str, Union[str, int]]]]]:
    """
    Process a list of document files, extracting content, generating summaries, and chunking text.

    This function performs the following steps for each file:
    1. Reads the file content
    2. Generates a summary of the document
    3. Chunks the document text into smaller segments
    4. Creates a dictionary for each chunk with content and metadata

    Args:
        ollama_url (str): The URL of the Ollama server.
        model (str): The name of the Ollama model to use for summarization.
        files (List[io.BytesIO]): A list of file-like objects to process.
        chunk_size (int): The size of each text chunk in characters.
        chunk_overlap (int): The number of overlapping characters between chunks.

    Returns:
        Tuple[List[Dict[str, Union[str, Dict[str, Union[str, int]]]]], Dict[str, str]]:
            - A list of dictionaries, each containing a text chunk and its metadata.
            - A dictionary mapping file names to their summaries.

    Note:
        This function uses helper functions like read_file, summarize_document, and chunk_text.
        It handles various file types and creates a structured representation of the documents
        suitable for further processing in the RAG system.
    """
    documents = []
    summaries = {}
    for file in files:
        content = read_file(file, file.name)
        if content:
            summary = summarize_document(ollama_url, model, content)
            summaries[file.name] = summary
            chunks = chunk_text(content, chunk_size, chunk_overlap)
            for i, chunk in enumerate(chunks):
                documents.append({
                    "content": chunk,
                    "metadata": {
                        "source": file.name,
                        "chunk": i,
                        "summary": summary
                    }
                })
    return documents, summaries

def get_chat_history(messages: List[Dict[str, str]], memory_size: int) -> str:
    """
    Format the recent chat history into a string representation.

    This function takes the list of chat messages and formats the most recent
    question-answer pairs into a string, based on the specified memory size.

    Args:
        messages (List[Dict[str, str]]): The list of chat messages.
        memory_size (int): The number of recent question-answer pairs to include.

    Returns:
        str: A formatted string representation of the recent chat history.
    """
    history = messages[-memory_size*2:]  # Get last n question-answer pairs
    formatted_history = []
    for msg in history:
        role = "Human" if msg["role"] == "user" else "AI"
        formatted_history.append(f"{role}: {msg['content']}")
    return "\n".join(formatted_history)

def keyword_search(query: str, documents: List[Dict[str, Union[str, Dict[str, Union[str, int]]]]]) -> Dict[str, float]:
    """
    Perform a simple keyword-based search over the documents.

    This function calculates a relevance score for each document based on
    the overlap between query words and document words.

    Args:
        query (str): The search query.
        documents (List[Dict]): The list of document dictionaries.

    Returns:
        Dict[str, float]: A dictionary mapping document IDs to their relevance scores.
    """
    query_words = set(query.lower().split())
    scores = {}
    for i, doc in enumerate(documents):
        doc_words = set(doc['content'].lower().split())
        score = len(query_words.intersection(doc_words)) / len(query_words) if len(query_words) > 0 else 0
        doc_id = str(i)  # Assuming IDs are strings of their index
        scores[doc_id] = score
    return scores

def hybrid_search(query: str, collection, documents: List[Dict[str, Union[str, Dict[str, Union[str, int]]]]], num_chunks: int):
    """
    Perform a hybrid search combining semantic and keyword search.

    This function combines the results of semantic search (using embeddings)
    and keyword search to provide a more robust search mechanism.

    Args:
        query (str): The search query.
        collection: The ChromaDB collection containing document embeddings.
        documents (List[Dict]): The list of document dictionaries.
        num_chunks (int): The number of top chunks to retrieve.

    Returns:
        List[Tuple]: A list of tuples containing the top document chunks,
                     their metadata, and relevance scores.
    """
    # Embed the query
    question_embedding = embed_documents(st.session_state.ollama_url, st.session_state.embedding_model, [query])

    # Perform semantic search over all documents
    total_docs = len(documents)
    semantic_results = collection.query(
        query_embeddings=question_embedding,
        n_results=total_docs
    )

    # Build a mapping from document IDs to semantic similarities
    semantic_similarities = {}
    for i in range(total_docs):
        doc_id = semantic_results['ids'][0][i]
        # Convert distance to similarity
        distance = semantic_results['distances'][0][i]
        similarity = 1 / (1 + distance)
        semantic_similarities[doc_id] = similarity

    # Perform keyword search and get scores
    keyword_scores = keyword_search(query, documents)

    # Normalize both sets of scores
    semantic_values = list(semantic_similarities.values())
    keyword_values = list(keyword_scores.values())

    # Normalize semantic similarities
    max_semantic = max(semantic_values) if semantic_values else 1
    min_semantic = min(semantic_values) if semantic_values else 0
    for doc_id in semantic_similarities:
        semantic_similarities[doc_id] = (semantic_similarities[doc_id] - min_semantic) / (max_semantic - min_semantic) if max_semantic != min_semantic else 0

    # Normalize keyword scores
    max_keyword = max(keyword_values) if keyword_values else 1
    min_keyword = min(keyword_values) if keyword_values else 0
    for doc_id in keyword_scores:
        keyword_scores[doc_id] = (keyword_scores[doc_id] - min_keyword) / (max_keyword - min_keyword) if max_keyword != min_keyword else 0

    # Combine the scores
    combined_scores = {}
    for doc_id in semantic_similarities.keys():
        combined_scores[doc_id] = (semantic_similarities[doc_id] + keyword_scores.get(doc_id, 0)) / 2  # Average

    # Sort documents based on combined score
    sorted_docs = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)

    # Retrieve the top documents
    top_docs = []
    for doc_id, score in sorted_docs[:num_chunks]:
        index = int(doc_id)
        doc = documents[index]
        top_docs.append((doc['content'], doc['metadata'], score))

    return top_docs

def sanitize_collection_name(name: str) -> str:
    """
    Sanitize the collection name to meet ChromaDB's requirements.

    This function ensures that the collection name only contains valid characters,
    starts and ends with an alphanumeric character, and is within the allowed length.

    Args:
        name (str): The original collection name.

    Returns:
        str: The sanitized collection name.
    """
    # Remove any character that isn't alphanumeric, underscore, or hyphen
    sanitized = re.sub(r'[^\w-]', '_', name)
    # Ensure it starts and ends with an alphanumeric character
    sanitized = re.sub(r'^[^a-zA-Z0-9]+', '', sanitized)
    sanitized = re.sub(r'[^a-zA-Z0-9]+$', '', sanitized)
    # Limit to 63 characters
    sanitized = sanitized[:63]
    # Ensure it's at least 3 characters long
    while len(sanitized) < 3:
        sanitized += '_'
    return sanitized

def get_file_hash(file):
    """
    Generate a hash for the file content.

    This function creates an MD5 hash of the file content, which can be used
    to identify unique files and avoid reprocessing duplicate content.

    Args:
        file: A file-like object.

    Returns:
        str: The MD5 hash of the file content.
    """
    file.seek(0)
    data = file.read()
    file.seek(0)
    return hashlib.md5(data).hexdigest()

def main():
    """
    The main function that sets up the Streamlit interface and handles the chat application flow.

    This function performs the following tasks:
    1. Initializes session state variables
    2. Sets up the sidebar with various settings and controls
    3. Handles document processing when the user uploads files
    4. Manages the chat interface, including displaying messages and handling user input
    5. Processes user queries using hybrid search and the selected chat model
    6. Displays relevant document chunks used in generating the response

    The function uses several helper functions to perform specific tasks such as
    embedding documents, performing hybrid search, and generating chat responses.

    Note:
        This function is the entry point of the Streamlit application and controls
        the overall flow and user interaction of the SimpleRAG system.
    """
    # Initialize session state variables
    if 'ollama_url' not in st.session_state:
        st.session_state.ollama_url = DEFAULT_OLLAMA_URL
    if 'models' not in st.session_state:
        st.session_state.models = []
    if 'connection_status' not in st.session_state:
        st.session_state.connection_status = False
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'collection' not in st.session_state:
        st.session_state.collection = None
    if 'data_processed' not in st.session_state:
        st.session_state.data_processed = False
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    if 'summaries' not in st.session_state:
        st.session_state.summaries = {}
    if 'processed_file_hashes' not in st.session_state:
        st.session_state.processed_file_hashes = {}
    if 'embedding_model' not in st.session_state:
        st.session_state.embedding_model = ''
    if 'chat_model' not in st.session_state:
        st.session_state.chat_model = ''
    if 'previous_embedding_model' not in st.session_state:
        st.session_state.previous_embedding_model = ''

    with st.sidebar:
        st.title("💬 SimpleRAG")
        st.header("Settings")
        
        with st.expander("🤖 Model Settings", expanded=False):
            ollama_url_input = st.text_input("Ollama Server URL:", value=st.session_state.get('ollama_url', DEFAULT_OLLAMA_URL), key="ollama_url_input")
            
            previous_url = st.session_state.get('ollama_url', DEFAULT_OLLAMA_URL)
            st.session_state.ollama_url = ollama_url_input

            # If the URL has changed or connection_status is False, check the connection
            if (ollama_url_input != previous_url) or not st.session_state.connection_status:
                if check_ollama_connection(st.session_state.ollama_url):
                    st.session_state.connection_status = True
                    st.session_state.models = get_ollama_models(st.session_state.ollama_url)
                else:
                    st.session_state.connection_status = False
                    st.sidebar.error(f"Cannot connect to Ollama server at {st.session_state.ollama_url}. Please make sure it's running.")
            
            # Now proceed
            if not st.session_state.connection_status:
                st.error(f"Cannot connect to Ollama server at {st.session_state.ollama_url}. Please make sure it's running.")
            elif not st.session_state.models:
                st.error("No Ollama models available. Please check your Ollama installation.")
            else:
                embedding_models = st.session_state.models
                selected_embedding_model = st.selectbox("Select the embedding model", embedding_models, key="embedding_model_select")

                # Check if embedding model has changed
                if st.session_state.embedding_model != selected_embedding_model:
                    st.session_state.embedding_model = selected_embedding_model
                    st.session_state.data_processed = False
                    st.session_state.collection = None
                    st.session_state.documents = []
                    st.session_state.summaries = {}
                    st.session_state.processed_file_hashes = {}
                    st.warning("Embedding model changed. Please reprocess your documents.")

                chat_models = st.session_state.models
                selected_chat_model = st.selectbox("Select the chat model", chat_models, key="chat_model_select")

                if st.session_state.chat_model != selected_chat_model:
                    st.session_state.chat_model = selected_chat_model

                temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=DEFAULT_TEMPERATURE, step=0.1, key="temperature_slider")

        with st.expander("📄 Process Settings", expanded=False):
            chunk_size = st.number_input("Chunk size", min_value=100, max_value=2000, value=DEFAULT_CHUNK_SIZE, key="chunk_size_input")
            chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=500, value=DEFAULT_CHUNK_OVERLAP, key="chunk_overlap_input")
            num_chunks = st.number_input("Number of chunks to retrieve", min_value=1, max_value=10, value=DEFAULT_NUM_CHUNKS, key="num_chunks_input")

        with st.expander("💬 Chat Settings", expanded=False):
            memory_size = st.number_input("Number of messages to keep in memory", min_value=1, max_value=10, value=DEFAULT_MEMORY_SIZE, key="memory_size_input")
            prompt_template = st.text_area("Prompt template", value=DEFAULT_PROMPT_TEMPLATE, key="prompt_template_input")

        uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True, key="file_uploader")
        process_button = st.button("🔄 Process Documents", key="process_button")

        # Clear chat button in sidebar
        if st.button("🧹 Clear Chat", key="clear_chat"):
            st.session_state.messages = []
            st.experimental_rerun()

        if process_button and uploaded_files:
            with st.spinner("Processing documents..."):
                start_time = time.time()
                
                # Create a progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Process uploaded documents
                status_text.text("Reading and chunking documents...")
                documents = []
                summaries = {}
                for idx, file in enumerate(uploaded_files):
                    file_hash = get_file_hash(file)
                    if file_hash in st.session_state.processed_file_hashes:
                        # Use cached content
                        content = st.session_state.processed_file_hashes[file_hash]['content']
                        summary = st.session_state.processed_file_hashes[file_hash]['summary']
                    else:
                        content = read_file(file, file.name)
                        if content:
                            summary = summarize_document(st.session_state.ollama_url, st.session_state.chat_model, content)
                            # Cache the content and summary
                            st.session_state.processed_file_hashes[file_hash] = {'content': content, 'summary': summary}
                        else:
                            summary = ""
                    summaries[file.name] = summary
                    chunks = chunk_text(content, chunk_size, chunk_overlap)
                    for i, chunk in enumerate(chunks):
                        documents.append({
                            "content": chunk,
                            "metadata": {
                                "source": file.name,
                                "chunk": i,
                                "summary": summary
                            }
                        })
                    progress_bar.progress(int(25 + (idx / len(uploaded_files)) * 25))

                # Display processed files
                st.subheader("Processed Files:")
                for file in uploaded_files:
                    st.write(file.name)
                
                # Embed documents
                status_text.text("Generating embeddings...")
                embeddings = embed_documents(st.session_state.ollama_url, st.session_state.embedding_model, [doc["content"] for doc in documents])
                progress_bar.progress(75)
                
                if embeddings:
                    # Initialize ChromaDB for vector storage
                    status_text.text("Initializing vector database...")
                    chroma_client = chromadb.Client()
                    progress_bar.progress(85)
                    
                    # Create a new collection with a sanitized name based on the embedding model
                    base_name = f"document_collection_{st.session_state.embedding_model}"
                    collection_name = sanitize_collection_name(base_name)
                    
                    # Delete existing collection if it exists
                    try:
                        existing_collection = chroma_client.get_collection(name=collection_name)
                        if existing_collection:
                            chroma_client.delete_collection(name=collection_name)
                            st.warning(f"Deleted existing collection: {collection_name}")
                    except ValueError:
                        # Collection doesn't exist, so we can proceed
                        pass
                    
                    # Create a new collection
                    collection = chroma_client.create_collection(name=collection_name)
                    
                    # Add documents to ChromaDB
                    status_text.text("Indexing documents...")
                    collection.add(
                        embeddings=embeddings,
                        documents=[doc["content"] for doc in documents],
                        metadatas=[doc["metadata"] for doc in documents],
                        ids=[str(i) for i in range(len(documents))]
                    )
                    progress_bar.progress(100)
                    
                    end_time = time.time()
                    processing_time = end_time - start_time
                    status_text.text(f"Processing complete in {processing_time:.2f} seconds.")
                    
                    # Store processed data in session state
                    st.session_state.collection = collection
                    st.session_state.documents = documents
                    st.session_state.summaries = summaries
                    st.session_state.data_processed = True
                else:
                    status_text.text("Failed to embed documents.")
                    st.error("Failed to embed documents.")

    # Chat interface
    st.header("Chat")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if st.session_state.data_processed:
        if prompt := st.chat_input("Ask a question about the documents"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            if st.session_state.collection:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        # Hybrid search
                        results = hybrid_search(prompt, st.session_state.collection, st.session_state.documents, num_chunks)
                        
                        # Prepare context for the chat model
                        context = "\n\n".join([doc[0] for doc in results])
                        summaries_text = "\n\n".join([f"{file}: {summary}" for file, summary in st.session_state.summaries.items()])
                        chat_history = get_chat_history(st.session_state.messages, memory_size)
                        full_prompt = prompt_template.format(context=context, summaries=summaries_text, memory=chat_history, question=prompt)
                        
                        # Generate response using the chat model
                        response = chat_with_documents(st.session_state.ollama_url, st.session_state.chat_model, full_prompt, temperature)
                        
                        if response:
                            # Stream the response
                            full_response = ""
                            message_placeholder = st.empty()
                            for line in response.iter_lines():
                                if line:
                                    try:
                                        json_response = json.loads(line)
                                        if 'response' in json_response:
                                            full_response += json_response['response']
                                            message_placeholder.markdown(full_response + "▌")
                                    except json.JSONDecodeError:
                                        continue
                            message_placeholder.markdown(full_response)
                            st.session_state.messages.append({"role": "assistant", "content": full_response})

                            # Display used chunks in an expander
                            with st.expander("📚 View Relevant Document Chunks", expanded=False):
                                for i, (doc, metadata, score) in enumerate(results):
                                    st.markdown(f"**Chunk {i + 1}:**")
                                    st.text(doc)
                                    st.write(f"Source: {metadata['source']}")
                                    st.write(f"Chunk number: {metadata['chunk']}")
                                    st.write(f"Relevance score: {score:.4f}")
                                    st.markdown("---")
                        else:
                            st.error("Failed to generate response from the chat model.")
            else:
                st.error("Please process some documents first.")
    else:
        st.info("💡 Please select an embedding and chat model and then upload and process documents to start chatting.")

if __name__ == "__main__":
    main()
