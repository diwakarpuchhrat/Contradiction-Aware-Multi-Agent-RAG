import requests


def fetch_page(url, timeout=10):
    """
    Fetch HTML content from a URL.
    
    Args:
        url: URL to fetch
        timeout: Request timeout in seconds (default: 10)
    
    Returns:
        HTML content as string, or None if fetch fails
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        if response.status_code == 200:
            return response.text
    except Exception as e:
        print(f"Error fetching {url}: {e}")
    return None

