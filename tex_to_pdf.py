import os
import subprocess
import requests

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


def save_bibtex(bibtex_entry, title, joint_title, info=""):
    """Saves the BibTeX entry to a file, appending if the file exists."""

    path_name = title+info

    # Create 'summaries' directory if it doesn't exist
    if not os.path.exists('summaries'):
        os.makedirs('summaries')

    # Create directory for the title if it doesn't exist
    title_dir = f'summaries/{path_name}'
    if not os.path.exists(title_dir):
        os.makedirs(title_dir)

    # Generate filename based on the title
    filename = f"{joint_title}.bib"

    # Open the file in append mode and write the BibTeX entry
    with open(f'{title_dir}/{filename}', 'a') as f:
        f.write(bibtex_entry + '\n')


def save_to_tex(latex_doc, title, info=""):
  path_name = title+info
  # Create 'summaries' directory if it doesn't exist
  if not os.path.exists('summaries'):
      os.makedirs('summaries')

  if not os.path.exists(f'summaries/{path_name}'):
      os.makedirs(f'summaries/{path_name}')

  with open(f'summaries/{path_name}/{title}-literature_review.tex', 'w') as f:
      f.write(latex_doc)


def save_to_pdf(latex_doc, title, info=""):
  save_to_tex(latex_doc, title)
  
  # Compile LaTeX to PDF within the 'summaries' directory

  command='!pdflatex -output-directory="summaries/$title" "summaries/$title/$title-literature_review.tex"'
  subprocess.check_output(command)

  # Delete files that are not .tex or .pdf within the directory
  for filename in os.listdir(f"summaries/{title}"):
      if not filename.endswith((".tex", ".pdf")):
          os.remove(os.path.join(f"summaries/{title}", filename))

  # (Optional) Display a confirmation message
  print(f"PDF saved and unnecessary files removed in 'summaries/{title}/'")

