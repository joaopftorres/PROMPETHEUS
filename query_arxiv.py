import arxiv
import pprint
import re
import sys
import time
from cleaner import clean

def search_arxiv(query, max_retries=3, retry_delay=5):
    search = arxiv.Search(
        query=query,
        max_results=3000,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending 
    )

    articles = []
    ids=[]
    retries = 0

    while retries < max_retries:
        try:
            for result in search.results():
                paper_id = result.entry_id

                match = re.search(r'(\d+\.\d+)(v\d+)?', paper_id)
                if match:
                  paper_id = match.group(1)
                  if paper_id not in ids:
                    ids.append(paper_id)
                    article = {}
                    article["id"] = paper_id
                    article["title"] = result.title
                    article["abstract"] = result.summary
                    article["clean_abstract"] = clean(result.summary)
                    articles.append(article)
                else:
                    print("Invalid paper ID:", paper_id)

            # If we reached here without encountering an exception, return articles
            return articles

        except arxiv.UnexpectedEmptyPageError:
            print("No more results found. Retrying...")
            retries += 1
            time.sleep(retry_delay)

    print("Max retries reached. Exiting search.")
    return articles


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python arxiv_search.py <query>")
        sys.exit(1)

    query = ' '.join(sys.argv[1:])
    articles = search_arxiv(query)
    print("\n\nRESULT\n")
    pprint.pprint(articles)