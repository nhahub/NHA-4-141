import requests
from bs4 import BeautifulSoup
from typing import Optional
import re


class WebScraper:
    """Web scraper with timeout protection and content size limit."""

    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.timeout = 15
        self.max_chars = 12000  # Limit to avoid memory/embedding issues

    def scrape_url(self, url: str) -> Optional[str]:
        try:
            if not self._is_valid_url(url):
                raise ValueError("Invalid URL format")

            response = requests.get(
                url, headers=self.headers,
                timeout=self.timeout,
                stream=True  # Stream to avoid downloading huge pages
            )
            response.raise_for_status()

            # Read only first 500KB to avoid huge pages hanging
            content_bytes = b""
            for chunk in response.iter_content(chunk_size=8192):
                content_bytes += chunk
                if len(content_bytes) > 500_000:
                    break

            soup = BeautifulSoup(content_bytes, 'html.parser')
            content = self._extract_content(soup, url)

            # Truncate large pages
            if content and len(content) > self.max_chars:
                content = content[:self.max_chars] + "\n\n[Content truncated for processing...]"

            return content

        except requests.exceptions.Timeout:
            print(f"Timeout scraping {url}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Error fetching URL {url}: {e}")
            return None
        except Exception as e:
            print(f"Error processing URL {url}: {e}")
            return None

    def _is_valid_url(self, url: str) -> bool:
        return url.startswith(('http://', 'https://'))

    def _extract_content(self, soup: BeautifulSoup, url: str) -> str:
        # Remove noise tags
        for tag in soup(['script', 'style', 'nav', 'footer', 'header',
                         'aside', 'advertisement', 'form', 'iframe']):
            tag.decompose()

        # Try to find main content
        main = (
            soup.find('main') or
            soup.find('article') or
            soup.find(id=re.compile(r'content|main|article', re.I)) or
            soup.find(class_=re.compile(r'content|main|article', re.I)) or
            soup.find('body') or
            soup
        )

        text = main.get_text(separator='\n', strip=True)

        # Clean up excessive whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return '\n'.join(lines)