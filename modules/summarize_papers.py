import openai


def summarize(text):
    response = openai.ChatCompletion.create(
        model='gpt-4',
        messages=[
            {"role": "system", "content": "Summarize the following technical paper."},
            {"role": "user", "content": text}
        ]
    )
    return response['choices'][0]['message']['content']


def summarize_all_papers(paper_texts):
    summaries = {}
    for title, text in paper_texts.items():
        if text:  # Ensure text is not None or empty
            summaries[title] = summarize(text)
        else:
            summaries[title] = None

    print("Summarization complete.")
    for title, summary in summaries.items():
        print(f"Title: {title}\nSummary: {summary}\n{'-' * 40}")

    return summaries
