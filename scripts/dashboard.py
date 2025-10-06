import gradio as gr
import sys
import os
import requests
import json
import time
import base64
import io
import struct
import tempfile
from typing import List, Dict, Tuple

# --- Environment Variable Setup ---
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

# --- Ensure API Key is available ---
# We check here to provide a quick error message if the key is still missing
if not os.environ.get("GEMINI_API_KEY"):
    print("FATAL ERROR: GEMINI_API_KEY is not set in the .env file at the project root.")

# NOTE: The API_KEY is intentionally left blank; it is assumed to be provided
# by the hosting environment (Canvas) when the HTTP request is executed.
API_KEY = ""
TTS_MODEL = "gemini-2.5-flash-preview-tts"
TTS_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# Import the existing function for LLM calls
from modules.call_gemini_api import call_gemini_api

sys.path.append(os.path.join(os.path.dirname(__file__), 'database'))

try:
    from database.chroma_service import get_all_publications_for_dashboard, get_summaries_for_dashboard

    print("Successfully imported chroma_service.")

except ImportError as e:

    print(f"FATAL ERROR: Could not import chroma_service. Check your file structure and pathing. Details: {e}")

    # # Define placeholder functions/variables to allow the Gradio UI to load
    # def get_all_publications_for_dashboard():
    #     # Using a small mock corpus to ensure the UI functions even if the DB is missing
    #     MOCK_TITLES = ["Mock Paper A: TGN Role", "Mock Paper B: SNARE Redundancy"]
    #     MOCK_MAP = {
    #         "Mock Paper A: TGN Role": "TGN-localized proteins are essential for root direction.",
    #         "Mock Paper B: SNARE Redundancy": "SNARE proteins show functional overlap in trafficking."
    #     }
    #     return MOCK_TITLES, MOCK_MAP
    #
    #
    # def get_summaries_for_dashboard():
    #     return ["Summary: TGN is key for root growth.",
    #             "Summary: SNAREs are robust due to redundancy."], "Summary: TGN is key for root growth. SNAREs are robust due to redundancy."


# --- TTS UTILITY FUNCTIONS (Required for TTS output) ---
# NOTE: These functions are currently unused as TTS is temporarily disabled.

def pcm_to_wav_bytes(pcm16_bytes: bytes, sample_rate: int, num_channels: int = 1) -> bytes:
    """Converts raw 16-bit PCM bytes to a WAV file format."""
    buffer = io.BytesIO()
    data_size = len(pcm16_bytes)

    # 1. RIFF chunk
    buffer.write(b'RIFF')
    buffer.write(struct.pack('<I', 36 + data_size))  # File size
    buffer.write(b'WAVE')

    # 2. FMT chunk
    buffer.write(b'fmt ')
    buffer.write(struct.pack('<I', 16))  # Sub-chunk size
    buffer.write(struct.pack('<H', 1))  # Audio format (1=PCM)
    buffer.write(struct.pack('<H', num_channels))
    buffer.write(struct.pack('<I', sample_rate))
    byte_rate = sample_rate * num_channels * 2  # 2 bytes/sample
    buffer.write(struct.pack('<I', byte_rate))
    block_align = num_channels * 2
    buffer.write(struct.pack('<H', block_align))
    buffer.write(struct.pack('<H', 16))  # Bits per sample

    # 3. DATA chunk
    buffer.write(b'data')
    buffer.write(struct.pack('<I', data_size))  # Data size

    # 4. PCM data
    buffer.write(pcm16_bytes)

    return buffer.getvalue()


def call_tts_api_and_save(text: str) -> str:
    """Calls the TTS API, converts PCM to WAV, and saves to a temporary file."""
    url = f"{TTS_API_BASE_URL}/{TTS_MODEL}:generateContent?key={API_KEY}"

    # Voice: Orus (Firm)
    payload = {
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {"voiceName": "Orus"}
                }
            }
        }
    }

    max_retries = 4
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
            response.raise_for_status()
            result = response.json()

            part = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0]
            audio_data_b64 = part.get('inlineData', {}).get('data')
            mime_type = part.get('inlineData', {}).get('mimeType')

            if audio_data_b64 and mime_type and mime_type.startswith("audio/L16"):
                pcm_bytes = base64.b64decode(audio_data_b64)

                import re
                match = re.search(r'rate=(\d+)', mime_type)
                if not match:
                    raise ValueError(f"Could not parse sample rate from MIME type: {mime_type}")
                sample_rate = int(match.group(1))

                wav_bytes = pcm_to_wav_bytes(pcm_bytes, sample_rate)

                temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temp_file.write(wav_bytes)
                temp_file.close()
                return temp_file.name

            raise Exception("TTS API did not return valid audio data.")

        except requests.exceptions.HTTPError as http_err:
            if response.status_code == 429 and attempt < max_retries - 1:
                sleep_time = 2 ** attempt
                time.sleep(sleep_time)
            else:
                print(f"TTS HTTP Error: {http_err} - {response.text}")
                return "Error: Failed to generate audio."
        except Exception as err:
            print(f"TTS Error: {err}")
            return "Error: Failed to process audio."

    return "Error: TTS failed after retries."


def clean_temp_file(audio_path):
    """Utility to clean up the temporary audio file after playback."""
    if audio_path and os.path.exists(audio_path):
        try:
            os.remove(audio_path)
        except Exception as e:
            print(f"Error cleaning up temp file {audio_path}: {e}")
    return None


# --- Global Data Setup ---
# NOTE: The synchronous API calls for TITLE_TO_SUMMARY_MAP were removed here
# to prevent the application from hanging at startup.
TITLES, TITLE_TO_TEXT_MAP = get_all_publications_for_dashboard()
CORPUS_SUMMARY_SAMPLE = get_summaries_for_dashboard()

# Initialize the summary map as empty. Summaries will be generated on demand
# when the user clicks 'Summarize Paper'. (FIX for the "stuck" issue)
TITLE_TO_SUMMARY_MAP = {}


def _generate_summary_for_single_paper(title: str, text: str) -> str:
    """
    Generates a comprehensive summary for a single paper, focusing on
    structure (Abstract, Results, Conclusions, etc.) for corpus analysis.
    """
    # Use only the first 5000 characters to save tokens/latency
    paper_sample = text

    # Custom prompt to enforce the requested section coverage
    user_prompt = f"""
    Generate a detailed summary for the following research paper. The summary must specifically address the key information typically found in the Abstract, Introduction, Materials and Methods (briefly), Results and Discussion, General Outcomes, and Conclusions.

    Format the summary as a single, coherent narrative paragraph.

    Paper Title: {title}
    Paper Text:\n\n{paper_sample}
    """

    system_prompt = "You are a highly efficient scientific summarizer. Your goal is to combine the most critical information from a paper's various sections into one dense, informative paragraph."

    # Use a simpler helper function that doesn't use the Google Search tool for internal summarization
    summary = call_gemini_api(user_prompt, system_prompt, tools=None)

    return summary.strip()


# --- Gradio Action Functions (Existing) ---

def display_paper_text(selected_title: str) -> str:
    """Retrieves the full text for the selected title."""
    if not TITLES:
        return "⚠️ Database not loaded. Please check service configuration and ingestion status."
    return TITLE_TO_TEXT_MAP.get(selected_title, f"Could not find document text for '{selected_title}'.")


def summarize_paper(selected_title: str) -> str:
    """
    Retrieves the pre-computed summary from the global map. If a summary
    was not computed (e.g., startup failure or lazy load), it generates a new one.
    """
    # Check if the summary exists in the map (will be empty on startup)
    summary = TITLE_TO_SUMMARY_MAP.get(selected_title)

    if summary and not summary.startswith("API ERROR:"):
        return summary

    # Fallback/On-demand generation
    paper_text = TITLE_TO_TEXT_MAP.get(selected_title)
    if not paper_text:
        return "Error: Please select a valid paper to summarize."

    gr.Info(f"Generating summary for '{selected_title}' (on-demand fallback)...")

    # Use the function to generate the summary
    re_summary = _generate_summary_for_single_paper(selected_title, paper_text)

    if re_summary.startswith("API ERROR:") or re_summary.startswith("HTTP Error:"):
        return f"Error during on-demand summarization: {re_summary}"

    # Cache the result for future requests (optional, but good practice)
    TITLE_TO_SUMMARY_MAP[selected_title] = re_summary

    return re_summary


def analyze_full_corpus() -> str:
    """
    Analyzes the entire corpus using a multi-step, sequential process
    (mimicking a LangGraph workflow) to generate a strategic report.
    """
    # Use the globally-sampled summary corpus
    if not CORPUS_SUMMARY_SAMPLE:
        return "Error: Corpus summary sample is empty. Cannot perform analysis."

    gr.Info("Starting multi-step analysis (Progress, Gaps, Actions). This will take a moment...")

    # Pass the concatenated summaries to the model
    corpus_sample = CORPUS_SUMMARY_SAMPLE

    system_prompt = "Act as the Chief Scientific Officer for NASA's Biological and Physical Sciences (BPS) program. Your analysis must be strategic, concise, and focused on future resource allocation."

    # --- Step 1 (Node 1): Research Progress Summary ---
    gr.Info("Step 1/3: Summarizing research progress...")
    progress_prompt = f"""
    Analyze the following corpus sample, which contains the TITLE and a DETAILED SUMMARY for every paper in the portfolio. Provide a concise summary of the overall research status, major successes, and key findings observed across the entire set of documents.

    Corpus Sample Text:\n\n{corpus_sample}
    """
    progress_summary = call_gemini_api(progress_prompt, system_prompt, tools=[{"google_search": {}}])

    if progress_summary.startswith(("HTTP Error:", "An error occurred:", "Failed to get")):
        return f"Analysis failed during Step 1 (Progress Summary): {progress_summary}"

    # --- Step 2 (Node 2): Critical Research Gaps (uses progress context) ---
    gr.Info("Step 2/3: Identifying critical research gaps...")
    gaps_prompt = f"""
    Based on the following Progress Summary, identify and detail 3-4 Critical Research Gaps (areas with insufficient data, conflicting results, or open questions) that require immediate BPS focus.

    Progress Summary:\n{progress_summary}
    Corpus Sample Text (for context):\n\n{corpus_sample}
    """
    gaps_analysis = call_gemini_api(gaps_prompt, system_prompt, tools=[{"google_search": {}}])

    if gaps_analysis.startswith(("HTTP Error:", "An error occurred:", "Failed to get")):
        return f"Analysis failed during Step 2 (Gaps Identification): {gaps_analysis}"

    # --- Step 3 (Node 3): Actionable Next Steps (uses progress and gaps context) ---
    gr.Info("Step 3/3: Generating 3-5 Actionable Next Steps...")
    actions_prompt = f"""
    Based on the Progress Summary and the identified Critical Research Gaps, provide 3-5 highly Actionable Next Steps (specific, strategic recommendations for resource allocation and future experiments) for the BPS program.

    Progress Summary:\n{progress_summary}
    Gaps Analysis:\n{gaps_analysis}
    """
    actions_steps = call_gemini_api(actions_prompt, system_prompt, tools=[{"google_search": {}}])

    if actions_steps.startswith(("HTTP Error:", "An error occurred:", "Failed to get")):
        return f"Analysis failed during Step 3 (Actionable Steps): {actions_steps}"

    # --- Aggregation (Final Output) ---
    final_report = f"""
# 🚀 AI Strategic Research Report 

This comprehensive report was generated in three sequential, stateful steps (Progress, Gaps, Actions) using the Gemini API, ensuring highly focused analysis across the entire paper portfolio.

---

## 1. Research Progress Summary (Overall Status)

{progress_summary}

---

## 2. Critical Research Gaps

{gaps_analysis}

---

## 3. Actionable Next Steps

{actions_steps}
"""
    gr.Info("Analysis complete!")
    return final_report


# --- CHAT FUNCTION (Updated to remove unused 'user_audio' parameter) ---

def research_chat(user_text: str, chat_history: List[Tuple[str, str]]) -> Tuple[
    List[Tuple[str, str]], str]:
    """
    Processes user input (text), generates a text response grounded in
    the corpus.
    """
    # 1. Determine User Prompt
    if not user_text.strip():
        return chat_history, "Please provide a question via text."

    user_prompt = user_text.strip()

    # 2. Build Grounding System Prompt
    system_prompt = f"""You are an expert Research Analyst AI specializing in NASA's BPS portfolio. Your goal is to answer user questions concisely and accurately by synthesizing information from the provided research documents. Focus on key strategic insights.
---
ALL CORPUS SUMMARIES:
{CORPUS_SUMMARY_SAMPLE}
---
If the information is not in these summaries, state clearly that you cannot answer based on the current portfolio."""

    # 3. Generate Text Response
    gr.Info("Generating text response...")
    ai_response_text = call_gemini_api(user_prompt, system_prompt,
                                       tools=None)  # No Google Search for internal chat grounding

    if ai_response_text.startswith(("HTTP Error:", "An error occurred:", "Failed to get")):
        error_msg = f"LLM Error: {ai_response_text}. Cannot continue."
        chat_history.append((user_prompt, error_msg))
        return chat_history, ""

    # 4. Update Chat History
    chat_history.append((user_prompt, ai_response_text))

    # 5. Return updated history and clear the text input
    return chat_history, ""  # Only returning chat history and an empty string to clear text_input


# --- Gradio Interface ---

# Custom CSS for pastel theme
custom_css = """
/* Pastel Background for the main interface */
.gradio-container {
    background-color: #f0f4f7 !important; /* Very light blue-gray pastel */
}

/* --- Gradio Theme Variable Overrides for Pastel Primary Color --- */
:root {
    --primary-50: #f0f7f8 !important;   /* Lighter shade */
    --primary-100: #d4e8ec !important;  /* Light shade */
    --primary-200: #aed9e0 !important;  /* Soft Teal (Base Button Color) */
    --primary-300: #92c4cd !important;  /* Hover Color */
    --primary-400: #71a9b4 !important;
    --primary-500: #1a4b52 !important;  /* Dark Teal (Text Color) */
    --primary-600: #143a41 !important;
}

/* Pastel Button Styles - Unified Soft Teal */
.gr-button, 
.gr-button.primary, 
.gr-button:not(.primary) {
    background-color: var(--primary-200) !important; /* Soft Teal (Using variable override) */
    color: var(--primary-500) !important; /* Dark teal text (Using variable override) */
    border: 1px solid var(--primary-200) !important;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    transition: all 0.2s;
}

.gr-button:hover, 
.gr-button.primary:hover, 
.gr-button:not(.primary):hover {
    background-color: var(--primary-300) !important; /* Slightly darker hover */
    border: 1px solid var(--primary-300) !important;
}

/* Muted Violet-Gray for main headings/text to match pastel theme */
h1, h2, h3 {
    color: #797d9a !important; 
}

/* --- TAB VISIBILITY FIX --- */
/* Target the span element that holds the tab title text */
/* Sets the color for ALL tab labels (active and inactive) */
.gradio-tabs > div:first-child .gr-tab-button span {
    color: var(--primary-500) !important; /* Force dark teal text always visible */
}

/* Optional: Ensure the selected tab is highlighted with background */
.gradio-tabs > div:first-child .gr-tab-button.selected {
    background-color: var(--primary-100) !important; /* Light teal background for active tab */
}
"""

with gr.Blocks(theme=gr.themes.Soft(), title="NASA Space Biology Knowledge AI Research Dashboard") as app:
    # --- Logo and Main Title Section ---
    gr.Markdown(
        f"""
        <div style='text-align: center; padding: 15px; background-color: #d1e2e9; border-radius: 12px; margin-bottom: 20px;'>
            <h1 style="color: #1a4b52; margin: 0;">NASA Space Biology Knowledge AI Research Dashboard</h1>
            <p style="color: #6e3224; margin: 5px 0 0 0;">Leveraging AI for Strategic Analysis of Bioscience Publications</p>
        </div>
        """
    )

    # -------------------------------------------------
    # --- GLOBAL CORPUS ANALYSIS (AI Section) ---
    # -------------------------------------------------
    with gr.Tab("Strategic Analysis"):
        gr.Markdown("### Full Research Portfolio Analysis")
        gr.Markdown(
            """
            Click the button below to analyze the first 50 publications using AI to generate high-level strategic insights. 
            """
        )

        # 1. Button (Will take up the full width since it's not constrained by gr.Row)
        analyze_button = gr.Button("🧠 Generate AI Strategic Report", variant="primary", scale=1)

        # 2. Output (Will appear below the button, taking up the full width)
        analysis_output = gr.Markdown(
            label="AI Strategic Report",
            value="Click 'Analyze Publications' to generate the report...",
        )

        # Define interaction for Corpus Analysis (This remains correct)
        analyze_button.click(
            fn=analyze_full_corpus,
            inputs=None,
            outputs=analysis_output
        )

    # -------------------------------------------------
    # --- INDIVIDUAL PAPER EXPLORATION ---
    # -------------------------------------------------
    with gr.Tab("Individual Paper Exploration"):
        gr.Markdown("### Explore and Summarize Specific Papers")

        if not TITLES:
            gr.Markdown(
                """
                ### ❌ Database Initialization Failed
                The application could not load the paper titles from ChromaDB. Please ensure the following:
                1. You have run `python ingest.py` successfully.
                2. The `CHROMA_DB_PATH` in `papersage/database/chroma_service.py` is correctly set to `../scripts/chroma_db`.
                """
            )

        # Dropdown element, initialized with titles
        title_dropdown = gr.Dropdown(
            label="Select Paper Title",
            info="Choose a document to see its full content and generate an AI summary.",
            value=TITLES[0] if TITLES else None,
            choices=TITLES,
            interactive=True,
        )

        # Paper Summary and Full Text Display
        with gr.Row():
            # AI Summary Output
            summary_output = gr.Textbox(
                label="AI Summary (Key Findings & Impact)",
                lines=5,
                interactive=False,
                # Updated value to reflect lazy loading status
                value="Select a paper and click 'Summarize Paper' to generate an AI summary.",
                scale=2
            )

            # Summary Trigger Button
            summarize_button = gr.Button("✨ Summarize Paper (AI)", scale=1)

        with gr.Row():
            # Full Paper Content Output
            paper_content = gr.Textbox(
                label="Full Paper Content",
                lines=15,
                interactive=False,
                value=display_paper_text(TITLES[0]) if TITLES else "",
            )

        # Define interaction: when a title is selected
        # The .then(fn=summarize_paper) step will now trigger the API call only when a title is selected
        title_dropdown.change(
            fn=display_paper_text,
            inputs=title_dropdown,
            outputs=paper_content,
            queue=False
        ).then(
            fn=summarize_paper,  # Also update the summary when the paper changes
            inputs=title_dropdown,
            outputs=summary_output
        )

        # Define interaction: when the Summarize button is clicked (fallback/manual trigger)
        summarize_button.click(
            fn=summarize_paper,
            inputs=title_dropdown,
            outputs=summary_output
        )

    # -------------------------------------------------
    # --- VOICE ANALYSIS CHAT ---
    # -------------------------------------------------
    with gr.Tab("AI Assistant"):
        gr.Markdown("### I am your Research AI (Codename: Godzilla)")
        gr.Markdown(
            """
            Ask any question about the research portfolio. I will respond with text, grounded in the publication summaries.
            """
        )

        # Chatbot to display history
        chatbot = gr.Chatbot(
            label="Analyst Chat History",
            height=300
        )

        # Hidden state to manage the chat history list
        chat_history_state = gr.State([])

        # Input Area (Voice and Text)
        with gr.Row(variant="panel"):

            # Text input (for typing the question)
            text_input = gr.Textbox(
                placeholder="Type your question (e.g., 'What are the key gaps in SNARE research?')",
                label="Text Query",
                lines=1,
                scale=3
            )

            # Submit button
            submit_btn = gr.Button("Ask Godzilla", variant="primary", scale=1)  # Updated button text

        # Define interaction
        # The inputs are now correctly matched to the updated research_chat function
        submit_btn.click(
            fn=research_chat,
            inputs=[text_input, chat_history_state],
            outputs=[chatbot, text_input],
            queue=False
        )

        # Link the text input's Enter key to the submit button's action
        text_input.submit(
            fn=research_chat,
            inputs=[text_input, chat_history_state],
            outputs=[chatbot, text_input],
            queue=False
        )

if __name__ == "__main__":
    if not TITLES:
        print("Launching Gradio app with data loading error.")

    # To run this app, execute: python app.py
    app.launch()