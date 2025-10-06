# Project Title
### Cosmic Analyst: NASA Space Biology Knowledge AI Dashboard

## Description

This dashboard serves as an advanced analytical tool for processing and querying a comprehensive corpus of NASA's Biological and Physical Sciences (BPS) research papers. Leveraging the Gemini API for sophisticated language tasks, the application provides three key capabilities:

Strategic Analysis: Performs a multi-step, high-level analysis (Progress, Gaps, and Actionable Next Steps) across the entire research portfolio, adopting the persona of a Chief Scientific Officer to deliver strategic reports.

Individual Paper Exploration: Allows users to select any specific publication, view its full text, and generate a detailed, on-demand summary highlighting its key findings and impact using AI.

Research Chat (Codename: Godzilla): Provides an interactive, grounded chat interface that answers user questions exclusively by synthesizing information extracted from the portfolio's summarized documents.

The goal is to transform a large volume of scientific research and publications into actionable intelligence for resource allocation and future experimental planning.

## Getting Started

### Dependencies
*  sys
*  os
*  requests
*  pandas
*  BeautifulSoup
*  openai
*  gradio 
*  base64
*  struct
*  json
*  time
*  List, Dict, Tuple
*  chromadb


### Installing

#### Clone the repository to your local machine using:
```
git clone https://github.com/OGARaju/papersage.git
```
#### Generate your Gemini API key. 
1. Follow the instructions at https://developers.generativeai.google/products/gemini to create a project and generate an API key.
2. Add the API key to GEMINI_API_KEY in the .nv file located root directory of the project.

### Executing program

#### Install the required dependencies using pip:
```
pip install -r requirements.txt
```
#### Setup ChromaDB:
```
python scripts/ingest.py
```

#### To run the program, use the following command:
```
python app.py
```
#### Open your web browser and navigate to http://127.0.0.1:7860 to access the application.

## Authors
Aparna Raju - https://github.com/OGARaju

## Version History
Initial Release

## License

This project is licensed under the MIT License - see the LICENSE.md file for details

## Acknowledgments
https://www.spaceappschallenge.org/
https://github.com/jgalazka/SB_publications/tree/main A list of 608 full-text open-access Space Biology publications: This resource provides links to access 608 full-text space biology publications. Open the .csv file to see titles