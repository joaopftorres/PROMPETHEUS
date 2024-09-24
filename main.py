import re
import time
import sys
import pprint
import torch
import numpy as np
import matplotlib.pyplot as plt
import readability
import argparse

from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from query_arxiv import search_arxiv
from cleaner import clean_text, cut_papers
from topic_model import topic_model_pipeline
from summarizer_pipeline import summarize, improve_summary
from tex_to_pdf import save_to_tex, get_arxiv_bibtex, save_bibtex
from metrics import get_rouge, get_coherence, tokenit, gen_sentence, save_text
from prompts import create_arxiv_query, expand_title, create_latex_document, de_latex, create_template
from transformers import logging as transformers_logging

# Suppress specific transformers warnings
transformers_logging.set_verbosity_error()

# Entry Point
def main(title, gpt_model="gpt-3.5-turbo", max_papers=200):

    info = "_" + str(gpt_model) + "_" + str(max_papers) 

    start_time = time.time()

    # Process Title and Perform Arxiv Query
    expanded_title, arxiv_query, articles = process_title_and_query_arxiv(title, gpt_model)

    # Filter Articles Based on Embeddings
    selected_index, excluded_index, selected_sim, excluded_sim = filter_articles(expanded_title, articles, max_papers)

    # Save Article Reports
    save_article_reports(title, articles, selected_index, excluded_index, selected_sim, excluded_sim, info)

    # Process Topic Modeling and Summarization
    improved_summaries, topic_titles, topic_summaries, final_summary = topic_model_and_summarization(selected_index, articles, title, gpt_model, info)

    # Generate LaTeX Document
    latex_doc = generate_latex_document(title, topic_titles, improved_summaries, final_summary, gpt_model)

    # Run Metrics (ROUGE, Readability, Similarity)
    run_metrics(title, articles, selected_index, improved_summaries, topic_summaries, final_summary, topic_titles, latex_doc, start_time, info)


def process_title_and_query_arxiv(title, gpt_model):
    """Expand the title and perform an Arxiv query."""
    expanded_title = expand_title(title, gpt_model)
    print(f"Title: {title}\nExpanded Title: {expanded_title}")

    arxiv_query = create_arxiv_query(expanded_title, gpt_model)
    articles = search_arxiv(arxiv_query)
    print(f"Number of articles found: {len(articles)}")
    
    return expanded_title, arxiv_query, articles


def filter_articles(expanded_title, articles, max_papers):
    """Filter articles based on embeddings and similarity."""
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    input_embedding = model.encode([expanded_title], convert_to_tensor=True)

    selected_index, selected_sim, excluded_index, excluded_sim = [], [], [], []
    similarity_threshold = 0.4

    for i, article in enumerate(articles):
        article_embedding = model.encode(article.get("clean_abstract"), convert_to_tensor=True)
        sim = util.pytorch_cos_sim(article_embedding, input_embedding)
        if sim > similarity_threshold:
            selected_index.append(i)
            selected_sim.append(sim.item())
        else:
            excluded_index.append(i)
            excluded_sim.append(sim.item())

    if len(selected_index) > max_papers:
        selected_index, _ = cut_papers(max_papers, selected_sim, selected_index)

    print(f"Filtered {len(selected_index)} articles")
    return selected_index, excluded_index, selected_sim, excluded_sim


def save_article_reports(title, articles, selected_index, excluded_index, selected_sim, excluded_sim, info):
    """Save the reports of included and excluded articles."""
    
    # Included articles report
    included_report = create_article_report("included", articles, selected_index, selected_sim)
    save_text(included_report, "included_report", title, info)

    # Excluded articles report
    excluded_report = create_article_report("excluded", articles, excluded_index, excluded_sim)
    save_text(excluded_report, "excluded_report", title, info)


def create_article_report(report_type, articles, indices, sims):
    """Create a report for the included or excluded articles."""
    report = f"Papers {report_type}:\n"
    for i in indices:
        article = articles[i]
        report += f"\tid: {article.get('id')}\n\tname: {article.get('title')}\n\tsim: {sims[indices.index(i)]}\n\n"
    return report


def topic_model_and_summarization(selected_index, articles, title, gpt_model, info):
    """Perform topic modeling and summarization."""
    abstracts = []
    ids=[]
    paper_names=[]
  
    for i in range(len(selected_index)):
      abstracts.append(articles[selected_index[i]].get("clean_abstract"))
      ids.append(articles[selected_index[i]].get("id"))
      paper_names.append(articles[selected_index[i]].get("title"))
      #references.append(add_authors(articles[i].get("id")))

    joint_title=title.replace(" ", "_")
    bib_tex_keys=[]

    for i in range(len(ids)):
      arxiv_id = ids[i]
      bibtex_entry = get_arxiv_bibtex(arxiv_id)
      # Extract the BibTeX key
      key_start = bibtex_entry.find("{") + 1
      key_end = bibtex_entry.find(",")
      bibtex_key = bibtex_entry[key_start:key_end]
      bib_tex_keys.append(bibtex_key)
      #filename = str(bibtex_key) + ".bib"
      save_bibtex(bibtex_entry, title, joint_title, info)


    topic_model, dfs_by_topic, topic_titles, visualize_documents, topic_report = topic_model_pipeline(abstracts, SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'), gpt_model)
    save_text(topic_report, "topic_report", title, info)

    summarizer = pipeline("summarization", model="Falconsai/text_summarization")
    topic_summaries = summarize(topic_model, dfs_by_topic, summarizer, bib_tex_keys, paper_names)

    improved_summaries = improve_summary(topic_summaries, topic_titles, title, gpt_model)

    final_summary=title+": Literature Review\n"
    for i in range(1,len(improved_summaries)):
      final_summary+=f"\n\n{topic_titles[i]}\n{improved_summaries[i]}"

    final_summary+=f"\n\n{topic_titles[0]}\n{improved_summaries[0]}"

    return improved_summaries, topic_titles, topic_summaries, final_summary


def generate_latex_document(title, topic_titles, summaries, final_summary, gpt_model):
    """Generate LaTeX document for the summaries."""
    joint_title = title.replace(" ", "_")
    latex_template = create_template(joint_title)
    
    ordered_titles = topic_titles[1:] + [topic_titles[0]]
    latex_doc = create_latex_document(title, len(summaries), final_summary, ordered_titles, latex_template, gpt_model)
    
    save_to_tex(latex_doc, title, f"_{gpt_model}_{len(summaries)}")

    return latex_doc


def run_metrics(title, articles, selected_index, improve_summaries, topic_summaries, final_summary, topic_titles, latex_doc, start_time, info):
    """Run ROUGE, readability, and similarity metrics."""
    raw_abstracts = "\n".join([articles[i].get("abstract") for i in selected_index])
    de_latexed_doc = de_latex(latex_doc)
    raw_summaries = " \n".join(topic_summaries)

    cleaned_final_summary = re.sub(r'\\citep\{[^}]+\}', '', final_summary)
    cleaned_raw_summaries = re.sub(r'\\citep\{[^}]+\}', '', raw_summaries)

    save_text(raw_abstracts, "raw_abstracts", title, info)
    save_text(cleaned_raw_summaries, "raw_summaries", title, info)
    save_text(cleaned_final_summary, "final_summary", title, info)
    save_text(latex_doc, "latex_doc", title, info)
    save_text(de_latexed_doc, "de_latex_doc", title, info)

    

    # ROUGE Metrics
    rouge_scores = {
        "raw_summaries": get_rouge(raw_abstracts, raw_summaries),
        "final_summary": get_rouge(raw_abstracts, final_summary),
        "de_latexed": get_rouge(raw_abstracts, de_latexed_doc)
    }
    save_text(str(rouge_scores), "rouge_metrics", title, f"_{len(selected_index)}")

    # Readability Metrics
    readability_scores = {
        "raw_abstracts": readability.getmeasures(tokenit(raw_abstracts), lang='en')['readability grades'],
        "final_summary": readability.getmeasures(tokenit(final_summary), lang='en')['readability grades'],
        "de_latexed": readability.getmeasures(tokenit(de_latexed_doc), lang='en')['readability grades']
    }
    save_text(str(readability_scores), "readability_metrics", title, f"_{len(selected_index)}")

    # Similarity Metrics
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    input_embedding = model.encode([title], convert_to_tensor=True)
    abstract_embedding = model.encode(raw_abstracts, convert_to_tensor=True)
    raw_summaries_embedding =  model.encode(raw_summaries, convert_to_tensor=True)
    final_summary_embedding = model.encode(final_summary, convert_to_tensor=True)
    de_latexed_embedding = model.encode(de_latexed_doc, convert_to_tensor=True)

    bert_scores = [
        util.pytorch_cos_sim(abstract_embedding, input_embedding).tolist()[0],
        util.pytorch_cos_sim(final_summary_embedding, input_embedding).tolist()[0],
        util.pytorch_cos_sim(de_latexed_embedding, input_embedding).tolist()[0]
    ]

    # Calculate multiple random embeddings
    num_random_embeddings = 4
    random_scores = []
    for _ in range(num_random_embeddings):
        random_embedding = model.encode(gen_sentence(), convert_to_tensor=True)
        random_sim = util.pytorch_cos_sim(random_embedding, input_embedding)
        random_scores.append(random_sim.tolist()[0])

    avg_random_score = np.mean(random_scores)

    path_name = title+info

    plot_similarity_scores(bert_scores, avg_random_score, title, path_name)
    print(f"\nFull computation time: {time.time() - start_time} seconds")


def plot_similarity_scores(bert_scores, avg_random_score, title, path_name):
  """Plot and save the similarity scores as a bar chart."""
  labels = ['Abstracts', 'Summary', 'LaTeX Document', 'Random']
  
  values = [score[0] for score in bert_scores] + [avg_random_score]
  colors = ['skyblue'] * len(bert_scores) + ['lightcoral']  # Color for Random separately

  plt.figure(figsize=(10, 6))
  plt.bar(labels, values, color=colors)  # Use the updated colors list
  plt.ylabel('Cosine Similarity')
  plt.title('Input vs. Document Similarity')

  for i, val in enumerate(values):
      plt.text(i, val, f'{val:.3f}', ha='center', va='bottom', fontsize=10)

  plt.xticks(range(len(labels)), labels, rotation=45)
  plt.gca().set_alpha(0.8)
  plt.tight_layout()

  plt.savefig(f"summaries/{path_name}/results/similarity_scores_bar.png")
  plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI-driven tool designed to automate the systematic literature review (SLR) process.")
    parser.add_argument("title", help="Title of the literature review")
    parser.add_argument("--gpt_model", default="gpt-3.5-turbo", help="GPT model to use (default: gpt-3.5-turbo, recommended model: gpt-4o)")
    parser.add_argument("--max_papers", type=int, default=200, help="Maximum number of papers to process (default: 200)")

    args = parser.parse_args()

    main(args.title, args.gpt_model, args.max_papers)

