import os
import subprocess
import requests
import json

def save_text(text, filename, title, doc_type=None, info=""):
  path_name = title+info

  if doc_type:
    if doc_type=="bib":
      directory="SLR"
    else:
      directory = doc_type
  else:
    directory = "results"

  # Create 'output' directory if it doesn't exist
  if not os.path.exists('output'):
      os.makedirs('output')

  if not os.path.exists(f'output/{path_name}'):
      os.makedirs(f'output/{path_name}')
 
  if not os.path.exists(f'output/{path_name}/{directory}'):
      os.makedirs(f'output/{path_name}/{directory}')

  if doc_type == "SLR":
    with open(f'output/{path_name}/{directory}/{filename}.tex', 'w') as f:
      f.write(text)
  elif doc_type == "bib":
    with open(f'output/{path_name}/{directory}/{filename}.bib', 'a') as f:
      f.write(text + '\n')
  elif doc_type == "metrics":
    with open(f'output/{path_name}/{directory}/{filename}.txt', 'w') as f:
      f.write(json.dumps(text, indent=4))
  else:
    with open(f'output/{path_name}/{directory}/{filename}.txt', 'w') as f:
        f.write(text)

        

def get_arxiv_bibtex(arxiv_id):
    """Fetches the BibTeX entry for an arXiv paper and adds the URL."""

    base_url = "https://arxiv.org/bibtex/"
    url = base_url + arxiv_id

    response = requests.get(url)
    response.raise_for_status()

    bibtex_entry = response.text

    # Remove any existing `url` line
    lines = bibtex_entry.split('\n')
    lines = [line for line in lines if not line.strip().startswith('url')]

    # Ensure there is a comma at the end of the second to last line
    if not lines[-2].strip().endswith(','):
        lines[-2] += ','

    # Add the URL field with proper indentation before the closing brace
    lines.insert(-1, f"      url = {{https://arxiv.org/abs/{arxiv_id}}},")

    return "\n".join(lines) + "\n"  # Add newline at the end for better formatting