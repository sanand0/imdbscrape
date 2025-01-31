import json
import httpx
from datetime import datetime, UTC
from lxml import html
from typing import List, Dict


def scrape_imdb() -> List[Dict[str, str]]:
    """Scrape IMDb Top 250 movies using httpx and lxml.

    Returns:
        List of dictionaries containing movie title, year, and rating.
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; IMDbBot/1.0)"}
    response = httpx.get("https://www.imdb.com/chart/top/", headers=headers)
    response.raise_for_status()

    tree = html.fromstring(response.text)
    movies = []

    for item in tree.cssselect(".ipc-metadata-list-summary-item"):
        title = (
            item.cssselect(".ipc-title__text")[0].text_content()
            if item.cssselect(".ipc-title__text")
            else None
        )
        year = (
            item.cssselect(".cli-title-metadata span")[0].text_content()
            if item.cssselect(".cli-title-metadata span")
            else None
        )
        rating = (
            item.cssselect(".ipc-rating-star")[0].text_content()
            if item.cssselect(".ipc-rating-star")
            else None
        )

        if title and year and rating:
            movies.append({"title": title, "year": year, "rating": rating})

    return movies


# Scrape data and save with timestamp
now = datetime.now(UTC)
with open(f'imdb-top250-{now.strftime("%Y-%m-%d")}.json', "a") as f:
    f.write(json.dumps({"timestamp": now.isoformat(), "movies": scrape_imdb()}) + "\n")
