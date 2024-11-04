import os
import json
import arxiv # https://colab.research.google.com/github/EPS-Libraries-Berkeley/volt/blob/main/Search/arxiv_api.ipynb
from dateutil.parser import parse
from datetime import datetime

# Construct the default API client.
# client = Client()
# https://lukasschwab.me/arxiv.py/arxiv.html
# Perform the search using arxiv.Search


def create_author_str(authors):
    # Join authors with ", " and handle the last author differently
    if len(authors) > 1:
        authors_str = ", ".join(authors[:-1]) + ", and " + authors[-1]
    else:
        authors_str = authors[0] if authors else ""

    return authors_str

search = arxiv.Search(
    query="(cat:cs.CR) AND (model stealing OR model extraction OR high-fidelity)",
    max_results=500,
    sort_by=arxiv.SortCriterion.SubmittedDate,
    sort_order=arxiv.SortOrder.Descending
)

papers_data = []

# Iterate over the results from search
for result in search.results():
    # breakpoint()
    formatted_date = result.published.strftime("%Y-%m-%d")
    authors = [author.name for author in result.authors]
    
    # papers_data.append({'id': result.entry_id, 'title': result.title, 'authors': ', '.join(authors)})
    papers_data.append({
        'date': formatted_date, 
        'title': result.title,
        'author': create_author_str(authors), 
        'link':result.entry_id,
        'abstract': result.summary
        })
    
    # papers_data.append({
    #     'date': formatted_date, 
    #     'paper': f"{result.title}" + "\n " + create_author_str(authors), 
    #     'link':result.entry_id})

# Save to JSON file
with open(os.path.join('assets/json', 'model_stealing_papers.json'), 'w') as f:
    json.dump(papers_data, f, indent=4)