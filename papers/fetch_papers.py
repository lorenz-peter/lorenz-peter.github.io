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


submittedDate = f"submittedDate:[2017 TO {datetime.now().year}]"
query=f"{submittedDate} AND (cat:cs.CR) AND (model steal* OR model extract* OR high-fidelity)",
query="(cat:cs.CR) AND (model stealing OR model extract OR high-fidelity)",

# query='"quantum dots"'

# id_list = [240610011]

results_generator = arxiv.Client(
  page_size=1000,
  delay_seconds=3,
  num_retries=3
).results(arxiv.Search(
  query=query,
  id_list=[],
  sort_by=arxiv.SortCriterion.SubmittedDate,
  sort_order=arxiv.SortOrder.Descending
))


# search = arxiv.Search(
#     query=f"{submittedDate} AND (cat:cs.CR) AND (model stealing OR model extraction OR high-fidelity)",
#     # max_results=500,
#     sort_by=arxiv.SortCriterion.SubmittedDate,
#     sort_order=arxiv.SortOrder.Descending
# )

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
    
    # papers_data.append({
    #     'date': formatted_date, 
    #     'paper': f"{result.title}" + "\n " + create_author_str(authors), 
    #     'link':result.entry_id})

# Save to JSON file
with open(os.path.join('assets/json', 'model_stealing_papers.json'), 'w') as f:
    json.dump(papers_data, f, indent=4)