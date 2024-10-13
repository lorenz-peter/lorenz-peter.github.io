import arxiv
import json

# Define your search parameters
search = arxiv.Search(
    query="machine learning",  # Modify this to your desired keywords
    max_results=10,
    sort_by=arxiv.SortCriterion.SubmittedDate
)

papers_data = []

for result in search.results():
    papers_data.append({'id': result.entry_id, 'title': result.title})

# Save to JSON file
with open('papers.json', 'w') as f:
    json.dump(papers_data, f)