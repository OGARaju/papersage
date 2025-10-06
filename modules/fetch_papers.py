import pandas as pd
import requests
import sys
import os
from modules.scrub_paper import scrub_paper

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def fetch_publication_links():
    # construct path relative to the current script's directory
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # up one directory (..) and into the 'data' directory
    csv_path = os.path.join(base_dir, '..', 'data', 'SB_publication_PMC.csv')

    # Check if the file exists before attempting to read
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at expected location: {csv_path}")

    df = pd.read_csv(csv_path)
    publication_map = dict(zip(df['Title'], df['Link']))
    return publication_map


def fetch_paper_texts():
    publication_map = fetch_publication_links()
    paper_texts = {}
    for title, link in publication_map.items():
        try:
            paper_text = scrub_paper(link)
            paper_texts[title] = paper_text
        except requests.exceptions.RequestException as e:
            paper_texts[title] = str(e)
    print("Fetched all paper texts.")
    return paper_texts
