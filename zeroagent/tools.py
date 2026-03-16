import httpx
from httpx import Timeout
from rich import print


def search_gutenberg_books(search_terms):
    search_query = " ".join(search_terms)
    url = "https://gutendex.com/books/"
    response = httpx.get(url, params={"search": search_query}, timeout=Timeout(timeout=10.0))
    simplified_results = []
    for book in response.json().get("results", []):
        simplified_results.append({
            "id": book.get("id"),
            "title": book.get("title"),
            "authors": book.get("authors")
        })

    return simplified_results

getGutenbergBooksTool = {
    "type": "function",
    "function": {
        "name": "search_gutenberg_books",
        "description": "Search for books in the Project Gutenberg library based on specified search terms",
        "parameters": {
            "type": "object",
            "properties": {
                "search_terms": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "List of search terms to find books in the Gutenberg library (e.g. ['dickens', 'great'] to search for books by Dickens with 'great' in the title)"
                }
            },
            "required": ["search_terms"]
        }
    }
}

