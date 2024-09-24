import os
import subprocess
import requests

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


def save_to_pdf(latex_doc, title, info=""):
  save_to_tex(latex_doc, title)
  
  # Compile LaTeX to PDF within the 'output' directory

  command='!pdflatex -output-directory="output/$title" "output/$title/$title-literature_review.tex"'
  subprocess.check_output(command)

  # Delete files that are not .tex or .pdf within the directory
  for filename in os.listdir(f"output/{title}"):
      if not filename.endswith((".tex", ".pdf")):
          os.remove(os.path.join(f"output/{title}", filename))

  # (Optional) Display a confirmation message
  print(f"PDF saved and unnecessary files removed in 'output/{title}/'")

