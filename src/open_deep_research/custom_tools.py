"""Custom tool implementations for Open Deep Research."""

import json
import os
import re
import urllib.parse
import urllib.request

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool


@tool
def google_cse_search(query: str, num: int = 5, start: int = 1) -> str:
    """Search via Google Custom Search JSON API and return top results.

    This tool uses Google's Custom Search Engine to find relevant web pages.
    Requires GOOGLE_SEARCH_JSON_API_KEY and GOOGLE_CSE_ID environment variables.
    
    Args:
        query: Search query string
        num: Number of results to return (1-10, default: 5)
        start: Starting index for results (default: 1)
    
    Returns:
        JSON string with search results containing title, link, snippet, and displayLink
    """
    api_key = os.getenv("GOOGLE_SEARCH_JSON_API_KEY")
    cx = os.getenv("GOOGLE_CSE_ID")
    
    if not api_key or not cx:
        return json.dumps({
            "error": "Missing environment variables: GOOGLE_SEARCH_JSON_API_KEY or GOOGLE_CSE_ID"
        })

    # Validate and clamp parameters
    num = max(1, min(10, int(num)))
    start = max(1, int(start))

    params = {
        "key": api_key,
        "cx": cx,
        "q": query,
        "num": str(num),
        "start": str(start),
    }
    url = f"https://www.googleapis.com/customsearch/v1?{urllib.parse.urlencode(params)}"

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read()
        
        data = json.loads(body)
        items = data.get("items", []) or []
        
        # Format results consistently
        formatted = [
            {
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet"),
                "displayLink": item.get("displayLink"),
            }
            for item in items
        ]
        
        return json.dumps({"items": formatted}, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Search failed: {str(e)}"})


@tool
def fetch_url_content(url: str, max_chars: int = 8000) -> str:
    """Fetch URL and extract readable text content.

    Uses requests and BeautifulSoup to retrieve webpage content,
    strips scripts/styles, and returns clean text.
    
    Args:
        url: URL to fetch
        max_chars: Maximum characters to return (default: 8000)
    
    Returns:
        JSON string with URL and extracted content or error message
    """
    try:
        response = requests.get(
            url,
            timeout=20,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                             "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
            }
        )
        response.raise_for_status()
        
    except Exception as e:
        return json.dumps({"error": f"Request failed: {str(e)}"})

    try:
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove script, style, and noscript tags
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        
        # Extract text with newline separation
        text = soup.get_text(separator="\n")
        
        # Clean up whitespace
        text = "\n".join(
            line.strip() 
            for line in text.splitlines() 
            if line.strip()
        )
        
        # Truncate if needed
        if len(text) > max_chars:
            text = text[:max_chars] + "\n..."
        
        return json.dumps({
            "url": url,
            "content": text
        }, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Parse failed: {str(e)}"})


@tool
def wikipedia_summary(topic: str, sentences: int = 3) -> str:
    """Get a summary of a Wikipedia article.
    
    Fetches the summary/introduction section of a Wikipedia article.
    Use this tool when you need authoritative, encyclopedic information
    about people, places, concepts, historical events, or general knowledge topics.
    
    Args:
        topic: The Wikipedia article topic/title to look up
        sentences: Number of sentences to return in summary (default: 3, max: 10)
    
    Returns:
        JSON string with article title, summary, and URL, or error message
    """
    sentences = max(1, min(10, int(sentences)))
    
    try:
        # Wikipedia API endpoint
        api_url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(topic)
        
        response = requests.get(
            api_url,
            timeout=15,
            headers={"User-Agent": "OpenDeepResearch/1.0"}
        )
        response.raise_for_status()
        
        data = response.json()
        
        # Extract summary and limit to requested sentences
        extract = data.get("extract", "")
        if extract:
            # Split into sentences (basic approach)
            sentence_list = re.split(r'(?<=[.!?])\s+', extract)
            extract = " ".join(sentence_list[:sentences])
        
        result = {
            "title": data.get("title", topic),
            "summary": extract,
            "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
            "thumbnail": data.get("thumbnail", {}).get("source", "") if "thumbnail" in data else None
        }
        
        return json.dumps(result, indent=2)
        
    except requests.HTTPError as e:
        if e.response.status_code == 404:
            return json.dumps({"error": f"Wikipedia article not found for topic: {topic}"})
        return json.dumps({"error": f"Wikipedia API error: {str(e)}"})
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch Wikipedia summary: {str(e)}"})
        return json.dumps({"error": f"Failed to fetch Wikipedia summary: {str(e)}"})


@tool
def arxiv_search(query: str, max_results: int = 5) -> str:
    """Search arXiv for academic papers and preprints.
    
    Searches the arXiv repository for scientific papers across physics, mathematics,
    computer science, quantitative biology, quantitative finance, statistics, electrical
    engineering, and economics. Returns paper metadata including title, authors, abstract,
    and PDF link.
    
    Use this tool when researching:
    - Academic and scientific topics
    - Latest research papers and preprints
    - Technical topics in physics, CS, math, etc.
    - Author publications
    
    Args:
        query: Search query (can include keywords, author names, or arXiv IDs)
        max_results: Maximum number of papers to return (default: 5, max: 20)
    
    Returns:
        JSON string with paper metadata or error message
    """
    max_results = max(1, min(20, int(max_results)))
    
    try:
        # arXiv API endpoint
        base_url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }
        
        response = requests.get(
            base_url,
            params=params,
            timeout=20,
            headers={"User-Agent": "OpenDeepResearch/1.0"}
        )
        response.raise_for_status()
        
        # Parse XML response
        from xml.etree import ElementTree as ET
        root = ET.fromstring(response.content)
        
        # Define namespaces
        ns = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }
        
        papers = []
        for entry in root.findall('atom:entry', ns):
            # Extract paper metadata
            title = entry.find('atom:title', ns)
            summary = entry.find('atom:summary', ns)
            published = entry.find('atom:published', ns)
            updated = entry.find('atom:updated', ns)
            
            # Extract authors
            authors = []
            for author in entry.findall('atom:author', ns):
                name = author.find('atom:name', ns)
                if name is not None:
                    authors.append(name.text)
            
            # Extract links (PDF and abstract page)
            pdf_link = None
            abs_link = None
            for link in entry.findall('atom:link', ns):
                if link.get('title') == 'pdf':
                    pdf_link = link.get('href')
                elif link.get('rel') == 'alternate':
                    abs_link = link.get('href')
            
            # Extract arXiv ID from abstract link
            arxiv_id = abs_link.split('/')[-1] if abs_link else None
            
            # Extract categories
            categories = []
            for category in entry.findall('atom:category', ns):
                term = category.get('term')
                if term:
                    categories.append(term)
            
            paper = {
                "arxiv_id": arxiv_id,
                "title": title.text.strip() if title is not None else "",
                "authors": authors,
                "summary": summary.text.strip() if summary is not None else "",
                "published": published.text if published is not None else "",
                "updated": updated.text if updated is not None else "",
                "categories": categories,
                "abstract_url": abs_link,
                "pdf_url": pdf_link
            }
            papers.append(paper)
        
        return json.dumps({
            "query": query,
            "total_results": len(papers),
            "papers": papers
        }, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"arXiv search failed: {str(e)}"})


def get_custom_tools():
    """
    Get all custom tools for the research agent.
    Conditionally includes Google CSE only if API credentials are available.
    
    Returns:
        List of custom tool instances
    """
    tools = [fetch_url_content, wikipedia_summary, arxiv_search]
    
    # Only add Google CSE if credentials are available
    google_api_key = os.getenv("GOOGLE_SEARCH_JSON_API_KEY")
    google_cse_id = os.getenv("GOOGLE_CSE_ID")
    if google_api_key and google_cse_id:
        tools.insert(0, google_cse_search)
    
    return tools