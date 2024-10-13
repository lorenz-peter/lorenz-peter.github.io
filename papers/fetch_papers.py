import json
import arxiv # https://colab.research.google.com/github/EPS-Libraries-Berkeley/volt/blob/main/Search/arxiv_api.ipynb


# Construct the default API client.
# client = Client()
# https://lukasschwab.me/arxiv.py/arxiv.html
# Perform the search using arxiv.Search


search = arxiv.Search(
    query="machine learning",  # Modify this to your desired keywords
    max_results=10,
    sort_by=arxiv.SortCriterion.SubmittedDate
)

papers_data = []

# Iterate over the results from search
for result in search.results():
    papers_data.append({'id': result.entry_id, 'title': result.title})

# Save to JSON file
with open('papers.json', 'w') as f:
    json.dump(papers_data, f, indent=4)