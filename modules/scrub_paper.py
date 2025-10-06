import requests
from bs4 import BeautifulSoup

url = "https://pmc.ncbi.nlm.nih.gov/articles/PMC4136787/"


def get_html(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/119.0.0.0 Safari/537.36"
    }
    r = requests.get(url, headers=headers, timeout=10)
    r.raise_for_status()
    return r.text


def extract_authors(soup):
    # Authors are usually in <meta name="citation_author" ...> or in a visible <div> at the top
    authors = []
    for meta in soup.find_all("meta", attrs={"name": "citation_author"}):
        authors.append(meta["content"])
    # Fallback if not present
    if not authors:
        author_div = soup.find("div", class_="authors")
        if author_div:
            authors = [x.get_text(strip=True) for x in author_div.find_all("a")]
    return authors


def extract_section_by_heading(soup, heading_text):
    # Find a heading (h2, h3, etc.) whose text matches heading_text
    heading = soup.find(
        lambda tag: tag.name in ["h2", "h3"] and heading_text.lower() in tag.get_text(strip=True).lower())
    if not heading:
        return ""
    # Get all the text until the next heading of the same level
    contents = []
    for sib in heading.next_siblings:
        if sib.name in ["h2", "h3"]:
            break
        if hasattr(sib, "get_text"):
            contents.append(sib.get_text(strip=True))
        elif isinstance(sib, str):
            contents.append(sib.strip())
    return "\n".join([c for c in contents if c])


def extract_abstract(soup):
    abs_div = soup.find("div", class_="abstract")
    if abs_div:
        return abs_div.get_text("\n", strip=True)
    # Fallback: look for heading 'Abstract'
    return extract_section_by_heading(soup, "Abstract")


def extract_references(soup):
    refs = []
    ref_section = soup.find("ol", class_="references")
    if ref_section:
        for li in ref_section.find_all("li"):
            refs.append(li.get_text(" ", strip=True))
    else:
        # Fallback: look for heading
        refs_text = extract_section_by_heading(soup, "References")
        if refs_text:
            refs = refs_text.split("\n")
    return refs


def scrub_paper(url):
    html = get_html(url)
    soup = BeautifulSoup(html, "html.parser")
    result = []

    # Authors
    authors = ", ".join(extract_authors(soup))
    result.append(f"Authors: {authors}\n")

    # Abstract
    abstract = extract_abstract(soup)
    result.append(f"Abstract:\n{abstract}\n")

    # Sections
    sections = [
        "Introduction",
        "Materials and Methods",
        "Results and Discussion",
        "General Outcomes",
        "Conclusions",
        "Acknowledgments",
        "Funding Statement"
    ]
    for section in sections:
        section_text = extract_section_by_heading(soup, section)
        result.append(f"{section}:\n{section_text}\n")

    # References
    references = extract_references(soup)
    result.append("References:\n")
    for i, ref in enumerate(references, 1):
        result.append(f"{i}. {ref}\n")

    # Join all parts into a single string
    return "".join(result)
