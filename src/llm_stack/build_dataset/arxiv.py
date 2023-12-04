from datetime import datetime

import feedparser
import pandas as pd
import requests

from dateutil import parser


class ArxivAPI:
    """Class for interacting with arXiv API."""

    @classmethod
    def search_and_parse_papers(
        cls,
        query: str,
        start_date: int,
        end_date: int,
        first_result: int = 0,
        max_results: int = 50,
    ) -> pd.DataFrame:
        """Search arXiv API for papers matching query and parse results into a dataframe.

        Parameters
        ----------
        query
            Query to search for. Must be a string of query words separated by commas.

        start_date
            Start date of search in format YYYYMMDD.

        end_date
            End date of search in format YYYYMMDD.

        first_result
            Index of first result to return, by default 0.

        max_results
            Maximum number of results to return, by default 50.

        Returns
        -------
        Dataframe of parsed results from arXiv API.

        """
        response = cls.search(query, start_date, end_date, first_result, max_results)
        feed = cls._get_feed(response)
        entries = cls._get_entries(feed)

        # This will be slow for millions of entries but it's fine for our tiny dataset
        parsed_entries = [cls._parse_entry(entry) for entry in entries]
        return pd.DataFrame(parsed_entries)

    @classmethod
    def search(
        cls,
        query: str,
        start_date: int,
        end_date: int,
        first_result: int = 0,
        max_results: int = 50,
        timeout: int = 300,
    ) -> requests.Response:
        """Search arXiv API for papers matching query.

        Parameters
        ----------
        query
            Query to search for. Must be a string of query words separated by commas.

        start_date
            Start date of search in format YYYYMMDD.

        end_date
            End date of search in format YYYYMMDD.

        first_result
            Index of first result to return, by default 0.

        max_results
            Maximum number of results to return, by default 50.

        timeout
            Timeout for request in seconds, by default 300.

        Returns
        -------
        Response from arXiv API.

        """
        # Keeping things simple, only an OR query is supported
        query = cls._construct_query(query)

        url = "http://export.arxiv.org/api/query?"
        url += f"""search_query={query}&start={first_result}&max_results={max_results}&sortBy=submittedDate&sortOrder=descending&date-range={start_date}TO{end_date}"""

        response = requests.get(url, timeout=timeout)
        response.raise_for_status()

        return response

    @staticmethod
    def _construct_query(query: str, fields: list[str] = None) -> str:
        """Construct query string for arXiv API."""
        if fields is None:
            fields = ["all"]
        # Split the query string into individual terms
        terms = query.split(",")

        # Create a part of the query string for each field
        field_queries = []
        for field in fields:
            field_query = "+OR+".join([f'{field}:"{term.replace(" ", "+")}"' for term in terms])
            field_queries.append(f"({field_query})")

        # Combine the field queries with the OR operator
        combined_query = "+OR+".join(field_queries)

        return combined_query

    @staticmethod
    def _get_feed(response: requests.Response) -> feedparser.FeedParserDict:
        """Get feed from arXiv API response."""
        return feedparser.parse(response.content)

    @staticmethod
    def _get_entries(feed: feedparser.FeedParserDict) -> list:
        """Get entries from arXiv API feed."""
        try:
            return feed["entries"]
        except KeyError as e:
            raise ValueError("No entries found in feed.") from e

    @staticmethod
    def _parse_entry(entry: feedparser.util.FeedParserDict) -> dict[str, str]:
        """Parse entry from arXiv API feed."""
        return {
            "arxiv_url": entry["id"],
            "title": entry["title"],
            "summary": entry["summary"],
            "published": datetime.strftime(parser.parse(entry["published"]), "%Y-%m-%d"),
            "pdf_url": [item["href"] for item in entry["links"] if all(w in item["href"] for w in ["arxiv", "pdf"])][
                0
            ],
            "categories": [d["term"] for d in entry["tags"]],
        }
