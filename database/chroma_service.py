import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from typing import List, Dict, Any

# --- Configuration ---
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME_PAPERS = "papers"
COLLECTION_NAME_SUMMARIES = "summaries"
SUMMARY_ID = "full_summary_document"
# ---------------------

# Initialize objects globally to avoid re-loading on every function call
try:

    # 1. Initialize Chroma Client
    CLIENT = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # 2. Get the existing collection, using the model for embedding
    EMBEDDING_FUNCTION = DefaultEmbeddingFunction()
    COLLECTION = CLIENT.get_collection(
        name=COLLECTION_NAME_PAPERS,
        embedding_function=EMBEDDING_FUNCTION
    )

    print("ChromaDB and SentenceTransformer initialized successfully.")

except Exception as e:
    print(f"⚠️ ERROR: Could not initialize ChromaDB components. Ensure the ingestion script ran successfully.")
    print(f"Details: {e}")
    # Set to None to handle errors gracefully in the main application
    COLLECTION = None


def get_all_publications_for_dashboard():
    """
    Fetches all documents and metadata to populate the Gradio dashboard dropdown 
    and detailed view.
    """
    if not COLLECTION:
        return {}
    try:
        # Fetch all document IDs first
        all_ids = COLLECTION.get(include=[])['ids']

        # Fetch all documents, metadatas, and texts using the IDs
        results = COLLECTION.get(
            ids=all_ids,
            include=['metadatas', 'documents']
        )

        titles = []
        title_to_text_map = {}

        # Populate the lists and map
        for metadata, document in zip(results['metadatas'], results['documents']):
            title = metadata.get('title')
            if title:
                titles.append(title)
                title_to_text_map[title] = document

        return titles, title_to_text_map

    except Exception as e:
        print(f"Error fetching all publications: {e}")
        return {}


def get_paper_text_by_title(title, title_to_text_map):
    """
    Retrieves the full text for a given paper title from the map.
    """
    if not title:
        return "Please select a paper title."

    text = title_to_text_map.get(title)
    if text:
        return text
    else:
        return f"Could not find document text for '{title}'. Map keys: {list(title_to_text_map.keys())}"


def get_summaries_for_dashboard():
        """
        Retrieves a single document by its specific ID from a ChromaDB collection and returns it as a string.

        Returns:
            The document text as a string if found, otherwise None.
        """
        print(f"--- Attempting to retrieve document with ID '{SUMMARY_ID}' from collection '{COLLECTION_NAME_SUMMARIES}' ---")
        try:
            client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            collection = client.get_collection(name=COLLECTION_NAME_SUMMARIES)

            # Retrieve the specific item by its ID
            result = collection.get(
                ids=[SUMMARY_ID],
                include=["documents"]
            )

            # The result 'documents' list will contain the document if the ID was found
            if result and result['documents']:
                print("--- Document found. Returning as string. ---")
                return result['documents'][0]
            else:
                print(f"--- Document with ID '{SUMMARY_ID}' not found in collection. ---")
                return None

        except ValueError:
            print(f"ERROR: Collection '{COLLECTION_NAME_SUMMARIES}' not found at path '{CHROMA_DB_PATH}'.")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None



def query_papers(query_text: str, n_results: int = 3) -> List[Dict[str, Any]]:
    """
    Performs a vector similarity search (RAG retrieval step) on the paper collection.
    Returns a list of dictionaries with document, metadata, and distance.
    """
    if not COLLECTION:
        return []

    try:
        results = COLLECTION.query(
            query_texts=[query_text],
            n_results=n_results,
            include=['documents', 'metadatas', 'embeddings']
        )

        # Restructure results for easier processing in the chatbot function
        output = []
        if results['documents'] and results['documents'][0]:
            for doc, meta, embed in zip(results['documents'][0], results['metadatas'][0], results['embeddings'][0]):
                output.append({
                    "document": doc,
                    "metadata": meta,
                    "embeddings": embed
                })

        return output
    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        return []
