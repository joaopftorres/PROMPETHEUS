import arxiv
import re
import time

from cleaner import clean_text

def search_arxiv(query, max_retries=10, retry_delay=5, target_num_papers=200):
    """Search Arxiv and return a list of articles, retrying on failure."""

    num_papers_found = 0
    retries = 0
    articles = []
    ids = []

    while retries < max_retries and num_papers_found < target_num_papers:

        try:
            # Perform the search
            search = arxiv.Search(
                query=query,
                max_results=3000,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )

            # Process results as they are fetched
            for result in search.results():
                paper_id = extract_paper_id(result.entry_id)
                if paper_id and paper_id not in ids:
                    ids.append(paper_id)
                    article = {
                        "id": paper_id,
                        "title": result.title,
                        "abstract": result.summary,
                        "clean_abstract": clean_text(result.summary)
                    }
                    articles.append(article)
                    num_papers_found += 1

        except arxiv.UnexpectedEmptyPageError:
            print("No more results found. Retrying...")

        except Exception as e:
            print(f"Error during search: {e}")

        # Retry if not enough papers are found
        retries += 1
        time.sleep(retry_delay)

    print(f"Max retries reached querying arxiv.")
    return articles

def extract_paper_id(entry_id):
    """Extract the paper ID from the Arxiv entry ID."""
    match = re.search(r'(\d+\.\d+)(v\d+)?', entry_id)
    if match:
        return match.group(1)
    return None
