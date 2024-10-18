import os
import json
import arxiv # https://colab.research.google.com/github/EPS-Libraries-Berkeley/volt/blob/main/Search/arxiv_api.ipynb


# Construct the default API client.
# client = Client()
# https://lukasschwab.me/arxiv.py/arxiv.html
# Perform the search using arxiv.Search


search = arxiv.Search(
    query="(cat:cs.CR OR cat:cs.LG OR cat:cs.AI) AND (model stealing OR model extraction or high-fidelity)",
    max_results=100,
    sort_by=arxiv.SortCriterion.SubmittedDate,
    sort_order=arxiv.SortOrder.Descending
)

papers_data = []

# Iterate over the results from search
for result in search.results():
    # breakpoint()
    authors = [author.name for author in result.authors]
    papers_data.append({'id': result.entry_id, 'title': result.title, 'authors': ', '.join(authors)})

# Save to JSON file
with open(os.path.join('assets/json', 'model_stealing_papers.json'), 'w') as f:
    json.dump(papers_data, f, indent=4)