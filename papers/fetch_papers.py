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


# submittedDate = f"submittedDate:[2017-01-01 TO {datetime.now().strftime('%Y-%m-%d')}]"
# query=f"{submittedDate} AND (cat:cs.CR) AND (model steal* OR model extract* OR high-fidelity)",
query="(cat:cs.CR) AND (model steal* OR model extract* OR high-fidelity)",
# Construct the default API client.
client = arxiv.Client()

search = arxiv.Search(
  query=query,
  max_results=50,
#   sort_by=arxiv.SortCriterion.LastUpdatedDate,
# id_list = [240610011]
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