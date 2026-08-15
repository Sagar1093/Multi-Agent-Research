from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import os
from dotenv import load_dotenv
from rich import print
load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def web_search(query:str) -> str:
    """Search the web for recent and reliable information on a given topic.Returns Titles,URLs and snippets"""
    results = tavily.search(query = query,max_results=5)
    out = []

    for r in results["results"]:
        title = r["title"]
        url = r["url"]
        snippet = r["snippet"][:300]
        out.append(f"Title: {title}\nURL: {url}\nSnippet: {snippet}\n")

    return "\n---\n".join(out)

@tool
def scrape_url(url:str)->str:
    """"Scrape and return clean text content from a given URL for deeper reading"""
    try:
        resp = requests.get(url,timeout = 8,headers = {'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(resp.text, 'html.parser')
        for tag in soup(["script","style","nav","footer"]):
            tag.decompose()
        return soup.get_text(separator=" ",strip=True)[:2000]
    except Exception as e:
        return f"Error scraping URL: {str(e)}"

print(scrape_url.invoke("https://www.comicbasics.com/spider-man-brand-new-day-just-dethroned-no-way-home-and-its-officially-the-biggest-spider-man-movie-ever/"))