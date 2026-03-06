from ddgs import DDGS
import time


def duckduckgo_search(query, max_results=10, retries=3):
    """
    Search DuckDuckGo for a query and return top results.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return (default: 10)
        retries: Number of retry attempts on failure (default: 3)
    
    Returns:
        List of dictionaries with 'title', 'url', and 'snippet' keys
    """
    results = []
    
    for attempt in range(retries):
        try:
            # Use timeout and better error handling
            with DDGS(timeout=20) as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", "")
                    })
                # If we got results, break out of retry loop
                if results:
                    break
        except Exception as e:
            error_msg = str(e)
            print(f"Error during search (attempt {attempt + 1}/{retries}): {error_msg}")
            
            # If it's a protocol error, try with a delay
            if "protocol" in error_msg.lower() or "0x304" in error_msg:
                if attempt < retries - 1:
                    print(f"  Retrying in {2 ** attempt} seconds...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    print("  Protocol error persists. This may be due to:")
                    print("  - Network/TLS configuration issues")
                    print("  - DuckDuckGo API changes")
                    print("  - Try updating: pip install --upgrade ddgs")
            else:
                # For other errors, don't retry
                break
    
    return results

