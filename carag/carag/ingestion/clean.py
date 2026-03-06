from bs4 import BeautifulSoup


def clean_html(html):
    """
    Clean HTML by removing scripts, styles, navigation, and extracting main text.
    
    Args:
        html: Raw HTML content as string
    
    Returns:
        Cleaned text content
    """
    if not html:
        return ""
    
    soup = BeautifulSoup(html, "html.parser")
    
    # Remove unwanted tags
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()
    
    # Extract text
    text = soup.get_text(separator=" ")
    
    # Clean up whitespace
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = " ".join(chunk for chunk in chunks if chunk)
    
    return text

