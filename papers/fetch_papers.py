import os
import json
import arxiv # https://colab.research.google.com/github/EPS-Libraries-Berkeley/volt/blob/main/Search/arxiv_api.ipynb
from dateutil.parser import parse
from datetime import datetime

# https://info.arxiv.org/help/api/user-manual.html
# https://lukasschwab.me/arxiv.py/arxiv.html

def create_author_str(authors):
    # Join authors with ", " and handle the last author differently
    if len(authors) > 1:
        authors_str = ", ".join(authors[:-1]) + ", and " + authors[-1]
    else:
        authors_str = authors[0] if authors else ""

    return authors_str

# https://info.arxiv.org/help/api/user-manual.html#query_details
# https://arxiv.org/category_taxonomy

# text_search = "ti:model+stealing OR ti:model+extraction OR ti:high-fidelity OR abs:reverse-engineering"
# query=f"(cat:cs.CR OR cat:cs.AI or cat:cs.CV or or cat:cs.LG) AND ({text_search} OR au:carlini ANDNOT ti:malware)",

text_search = "ti:cryptanalytical OR (ti:model AND ti:stealing) OR (ti:model AND ti:extraction) OR all:high-fidelity OR abs:reverse-engineering"
query = f"(cat:cs.CR OR cat:cs.AI OR cat:cs.CV OR cat:cs.LG) AND (({text_search}) OR (au:carlini AND ({text_search}))) AND NOT ti:malware"

# Construct the default API client.
client = arxiv.Client()

search = arxiv.Search(
    query=query,
    max_results=200,
#   sort_by=arxiv.SortCriterion.LastUpdatedDate,
    # id_list = ["2406.10011"],
    sort_by=arxiv.SortCriterion.SubmittedDate,
    sort_order=arxiv.SortOrder.Descending
)

print(list(client.results(search)))
results_generator = client.results(search)

papers_data = []

# Iterate over the results from search
for result in results_generator:
    # breakpoint()
    formatted_date = result.published.strftime("%Y-%m")
    authors = [author.name for author in result.authors]
    
    # papers_data.append({'id': result.entry_id, 'title': result.title, 'authors': ', '.join(authors)})
    papers_data.append({
        'date': formatted_date, 
        'title': result.title,
        'author': create_author_str(authors), 
        'link':result.entry_id,
        'abstract': result.summary
        })

# Save to JSON file
with open(os.path.join('assets/json', 'model_stealing_papers.json'), 'w') as f:
    json.dump(papers_data, f, indent=4)