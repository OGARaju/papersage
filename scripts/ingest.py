import sys
import os
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

# --- Ensure API Key is available ---
# We check here to provide a quick error message if the key is still missing
if not os.environ.get("GEMINI_API_KEY"):
    print("FATAL ERROR: GEMINI_API_KEY is not set in the .env file at the project root.")

# Add the parent directory to the Python path so it can find the 'modules' package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.call_gemini_api import call_gemini_api
from modules.fetch_papers import fetch_paper_texts

CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME_PAPERS = "papers"
COLLECTION_NAME_SUMMARIES = "summaries"
# Initialize Chroma client
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
# Instantiate the embedding function used during creation
EMBEDDING_FUNCTION = DefaultEmbeddingFunction()
# Use get_or_create_collection to prevent errors on subsequent runs
collection_papers = client.get_or_create_collection(
    name=COLLECTION_NAME_PAPERS,
    embedding_function=EMBEDDING_FUNCTION
)
# Get or create a collection specifically for the single corpus document
collection_summaries = client.get_or_create_collection(
    name=COLLECTION_NAME_SUMMARIES,
    embedding_function=EMBEDDING_FUNCTION
)


def create_and_store_paper_texts():
    """
    Fetches paper texts, and stores them in ChromaDB.
    Run this function once to set up the database.
    """
    # Check if collection is already populated
    if collection_papers.count() > 0:
        print(
            f"Collection '{COLLECTION_NAME_PAPERS}' already contains {collection_papers.count()} documents. Skipping ingestion.")
        return

    print("Starting ingestion: Fetching papers...")
    papers = fetch_paper_texts()
    titles = list(papers.keys())
    texts = [papers[title] for title in titles]

    if not titles:
        print("No papers fetched. Ingestion skipped.")
        return

    # Store in ChromaDB
    ids = [f"id_{i}" for i in range(len(titles))]
    metadatas = [{"title": title} for title in titles]

    collection_papers.add(
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )

    print(f"Ingestion complete. Added {len(titles)} papers to ChromaDB.")


# Assume these are defined elsewhere in your code
# from your_utils import get_all_publications_for_dashboard, _generate_summary_for_single_paper
# CHROMA_DB_PATH = "./chroma_db"
# COLLECTION_NAME_CORPUS = "corpus_collection" # Using a different name for clarity

def create_and_store_summaries():
    """
    Generates a single corpus text by concatenating all paper summaries
    and stores that single text block in ChromaDB.
    """
    TITLES, TITLE_TO_TEXT_MAP = get_all_publications()

    # Define a unique ID for our single corpus document
    SUMMARY_ID = "full_summary_document"

    # Check if the summary document already exists
    # .get() returns a dict with empty lists if the id is not found
    if collection_summaries.get(ids=[SUMMARY_ID])['ids']:
        print(f"Document with ID '{SUMMARY_ID}' already exists in '{COLLECTION_NAME_SUMMARIES}'. Skipping ingestion.")
        return

    if TITLES:
        print(f"\n--- Starting Summarization ({len(TITLES)} papers) ---")
        summary_snippets = []

        for i, title in enumerate(TITLES):
            text = TITLE_TO_TEXT_MAP.get(title, "")
            if text:
                summary = generate_summary_for_single_paper(title, text)
                if summary.startswith(
                        ("HTTP Error:", "An error occurred:", "Failed to get", "AI returned no content.")):
                    print(f"[{i + 1}/{len(TITLES)}] ERROR: Skipping '{title}' due to API failure. '{summary}")
                else:
                    print(f"[{i + 1}/{len(TITLES)}] Successfully summarized: '{title}'")
                    # Format and append the snippet
                    snippet = f"--- PAPER START ---\nTITLE: {title}\nSUMMARY: {summary}\n--- PAPER END ---"
                    summary_snippets.append(snippet)
            else:
                print(f"[{i + 1}/{len(TITLES)}] WARNING: Skipping '{title}' due to empty text.")

        if not summary_snippets:
            print("\n--- No summaries were generated. Nothing to add to the database. ---")
            return

        # Join all the individual summaries into one large text block
        SUMMARY = "\n\n".join(summary_snippets)  # Use double newline for better separation
        print("\n--- Summarization Complete ---")

        # Now, add this single text block to ChromaDB
        print(f"--- Adding concatenated corpus to ChromaDB collection '{COLLECTION_NAME_SUMMARIES}' ---")
        collection_summaries.add(
            documents=[SUMMARY],
            metadatas=[{"paper_count": len(summary_snippets), "type": "concatenated_corpus"}],
            ids=[SUMMARY_ID]
        )
        print("--- Successfully stored the summary document. ---")


def generate_summary_for_single_paper(title: str, text: str) -> str:
    """
    Generates a comprehensive summary for a single paper, focusing on
    structure (Abstract, Results, Conclusions, etc.) for corpus analysis.
    """
    # Use only the first 5000 characters to save tokens/latency
    paper_sample = text

    # Custom prompt to enforce the requested section coverage
    user_prompt = f"""
    Generate a detailed summary for the following research paper. 
    The summary must contain the key information about Abstract, Introduction, Materials and Methods (briefly), Results and Discussion, General Outcomes, and Conclusions.
    
    Format the summary as a single, coherent narrative paragraph.

    Paper Title: {title}
    Paper Text:\n\n{paper_sample}
    """

    system_prompt = "You are a highly efficient scientific summarizer. Your goal is to combine the most critical information from a paper's various sections into one dense, informative paragraph."

    # Use a simpler helper function that doesn't use the Google Search tool for internal summarization
    summary = call_gemini_api(user_prompt, system_prompt, tools=None)

    return summary.strip()


def get_all_publications():
    """
    Fetches all documents and metadata to generate summaries
    """
    if not collection_papers:
        return {}
    try:
        # Fetch all document IDs first
        all_ids = collection_papers.get(include=[])['ids']
        first_100_ids = all_ids[:10]

        # Fetch all documents, metadatas, and texts using the IDs
        results = collection_papers.get(
            ids=first_100_ids,
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


if __name__ == "__main__":
    create_and_store_paper_texts()
    create_and_store_summaries()
