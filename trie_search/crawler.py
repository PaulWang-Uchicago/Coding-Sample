from .utils import get_links, get_text, fetch_html, ALLOWED_DOMAINS
from .trie import Trie


def crawl_site(start_url: str, max_depth: int) -> dict[str, list[str]]:
    """
    Given a starting URL, return a mapping of URLs mapped to words that appeared on that page.

    Important: In addition to following max_depth rule, pages must not be visited twice
    from a single call to crawl_site.

    Parameters:

        start_url - URL of page to start crawl on.
        max_depth - Maximum link depth into site to visit.
                    Links from the start page would be depth=1, links from those depth=2, and so on.

    Returns:
        Dictionary mapping strings to lists of strings.

        Dictionary keys: URLs of pages visited.
        Dictionary values: lists of all words that appeared on a given page.
    """

    # Initialize results, visited set, and queue
    results = {}
    visited = set()
    queue = [(start_url, 0)]

    while queue:
        # Pop the next URL to visit
        url, depth = queue.pop(0)
        # Skip if already visited
        if url in visited:
            continue
        visited.add(url)
        if depth > max_depth:
            continue
        # Fetch the HTML and extract text
        # Raise an exception if there is an error fetching the HTML
        try:
            html = fetch_html(url)
        except Exception:
            continue
        # Extract text and links
        text = get_text(html)
        words = text.split()
        results[url] = words
        links = get_links(html, url)
        # Add links to the queue
        for link in links:
            if link in visited:
                continue
            if not any(link.startswith(domain) for domain in ALLOWED_DOMAINS):
                continue
            queue.append((link, depth + 1))

    return results


def build_index(site_url: str, max_depth: int) -> Trie:
    """
    Given a starting URL, build a `Trie` of all words seen mapped to
    the page(s) they appeared upon.

    Parameters:

        start_url - URL of page to start crawl on.
        max_depth - Maximum link depth into site to visit.

    Returns:
        `Trie` where the keys are words seen on the crawl, and the
        value associated with each key is a set of URLs that word
        appeared on.
    """

    t = Trie()
    # Crawl the site
    results = crawl_site(site_url, max_depth)

    # Build the index
    for url, words in results.items():
        for word in words:
            if word.lower() in t:
                t[word.lower()].add(url)
            else:
                t[word.lower()] = {url}

    return t
