#!/usr/bin/env python3
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def collect_all_book_urls(base_url):
    """Crawl all catalogue pages and return a list of full book-detail URLs."""
    urls = []
    # Start at page-1
    next_page = urljoin(base_url, "catalogue/page-1.html")
    while True:
        print(f"Scraping catalogue page: {next_page}")
        resp = requests.get(next_page, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # Extract each book link
        for a in soup.select("article.product_pod h3 a"):
            full_url = urljoin(next_page, a["href"])
            urls.append(full_url)
        # Find “next” page link
        nxt = soup.select_one("li.next a")
        if not nxt:
            break
        next_page = urljoin(next_page, nxt["href"])
    return urls

def main():
    base_url = "http://books.toscrape.com/"
    urls = collect_all_book_urls(base_url)
    print(f"Collected {len(urls)} book URLs.")
    # Write out to JSON array
    with open("urls.json", "w") as f:
        json.dump(urls, f, indent=2)
    print("Wrote urls.json")

if __name__ == "__main__":
    main()