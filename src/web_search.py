from typing import List, Dict


class WebSearchClient:
    """Client for performing live web searches using ddgs."""

    def __init__(self, max_results: int = 5):
        self.max_results = max_results

    def search(self, query: str) -> List[Dict[str, str]]:
        results = []
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=self.max_results):
                    results.append({
                        "title": r.get("title", "").strip(),
                        "url": r.get("href", "").strip(),
                        "snippet": r.get("body", "").strip()
                    })
        except ImportError:
            # Fallback to old package name if ddgs not installed
            try:
                from duckduckgo_search import DDGS
                with DDGS() as ddgs:
                    for r in ddgs.text(query, max_results=self.max_results):
                        results.append({
                            "title": r.get("title", "").strip(),
                            "url": r.get("href", "").strip(),
                            "snippet": r.get("body", "").strip()
                        })
            except Exception as e:
                print(f"Web search error (fallback): {e}")
        except Exception as e:
            print(f"Web search error: {e}")

        return results

    def format_results_as_context(self, results: List[Dict[str, str]]) -> str:
        if not results:
            return ""
        formatted = []
        for r in results:
            formatted.append(
                f"Source: {r['url']}\nTitle: {r['title']}\nSnippet: {r['snippet']}"
            )
        return "\n\n".join(formatted)