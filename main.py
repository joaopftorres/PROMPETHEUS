from query_arxiv import *
from cleaner import *
from topic_model import *
from summarizer_pipeline import *
from tex_to_pdf import *
from references import *
from gpt import *
from metrics import *
import matplotlib.pyplot as plt
import numpy as np

#import tensorflow as tf
import json
import pickle
import random
import torch
import time
import re
import sys
import pprint

from torch.nn.functional import max_pool1d
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

#import settings

title = sys.argv[1]

gpt_model = sys.argv[2]

max_papers = int(sys.argv[3])

info = "_" + str(gpt_model) + "_" + str(max_papers)

time_report = ""

start_time = time.time()

print(f"GPT model: {gpt_model}")

expanded_title = expand_title_gpt(title, gpt_model)
print(f"******\nTitle\n{title}\n\nExpanded Title\n{expanded_title}\n******\n")

arxiv_query = create_arxiv_query(title, gpt_model)

print(f"----\nNew arxiv query:\n{arxiv_query}\n----")

title_report=f"Title: {title}\nExpanded title: {expanded_title}\nArxiv query: {arxiv_query}"

save_text(title_report, "title_report", title, info)

articles = search_arxiv(arxiv_query)

number_of_papers_report = ""

n_articles_arxiv = len(articles)
number_of_papers_report+=f"Number of articles found on arxiv: {n_articles_arxiv}\n"
print(f"Number of articles found on arxiv: {n_articles_arxiv}")

arxiv_time = time.time() - start_time
time_report += f"Arxiv search time: {arxiv_time} seconds\n"
print(f"Arxiv search time: {arxiv_time} seconds")


input_raw = []
input_raw.append(expanded_title)
#print(settings.title)


similarity = 0.4

section_similarity = 0.3


####----------------Sentence Transformer Models-----------------#####
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embedding_model = "All MiniLM: Small and Fast Pre-trained Models for Language Understanding and Generation.\n All-round model tuned for many use-cases. Trained on a large and diverse dataset of over 1 billion training pairs. https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2"
#####################################################################


####------------------ Summarization Models-----------------#####
summarizer = pipeline("summarization", model="Falconsai/text_summarization")
summarization_model = "T5 is an encoder-decoder model pre-trained on a multi-task mixture of unsupervised and supervised tasks and for which each task is converted into a text-to-text format."
#################################################################

#Compute embedding for both lists

input_embedding = model.encode(input_raw, convert_to_tensor=True)


selected_index = []

selected_sim = []

excluded_index = []

excluded_sim=[]

for i in range(len(articles)):
  embedding = model.encode(articles[i].get("clean_abstract"), convert_to_tensor=True)
  sim = util.pytorch_cos_sim(embedding, input_embedding)
  if (sim>similarity):
    selected_index.append(i)
    selected_sim.append(sim.item())
  else:
    excluded_index.append(i)
    excluded_sim.append(sim.item())

n_articles_filtered = len(selected_index)
print(f"Number of articles filtered: {n_articles_filtered}")
number_of_papers_report+=f"Number of articles filtered: {n_articles_filtered}\n"


if(len(selected_index))>max_papers:
  selected_index, newly_excluded = cut_papers(max_papers, selected_sim, selected_index)


excluded_ids = []
excluded_paper_names=[]
excluded_report = "Papers excluded: \n"
for i in range(len(excluded_index)):
  excluded_ids.append(articles[excluded_index[i]].get("id"))
  excluded_paper_names.append(articles[excluded_index[i]].get("title"))
  excluded_report += "\tid: " + str(articles[excluded_index[i]].get("id")) + "\n\tname: " + str(articles[excluded_index[i]].get("title")) + "\n\tsim: " + str(excluded_sim[i]) + "\n\n"



n_articles_filtered_top_50 = len(selected_index)
print(f"Number of articles filtered top 50: {n_articles_filtered_top_50}")
number_of_papers_report+=f"Number of articles filtered top 50: {n_articles_filtered_top_50}\n"

save_text(number_of_papers_report, "number_of_papers_report", title, info)

filtering_papers = time.time() - arxiv_time
time_report += f"Filtering papers time: {filtering_papers} seconds\n"
print(f"Filtering papers time: {filtering_papers} seconds")

abstracts = []
ids=[]
paper_names=[]
#references=[]
included_report = "Papers included: \n"
for i in range(len(selected_index)):
  abstracts.append(articles[selected_index[i]].get("clean_abstract"))
  ids.append(articles[selected_index[i]].get("id"))
  paper_names.append(articles[selected_index[i]].get("title"))
  #references.append(add_authors(articles[i].get("id")))
  included_report += "\tid: " + str(articles[selected_index[i]].get("id")) + "\n\tname: " + str(articles[selected_index[i]].get("title")) + "\n\tsim: " + str(selected_sim[i]) + "\n\n"


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
   
  
#print(selected_index)
print(ids)
print(bib_tex_keys)

for i in selected_index:
  print(articles[i].get("title"))


save_text(included_report, "included_report", title, info)
save_text(excluded_report, "excluded_report", title, info)


topic_model, dfs_by_topic, topic_titles, visualize_documents, topic_report = topic_model_pipeline(abstracts, model, gpt_model)

save_text(topic_report, "topic_report", title, info)

path_name = title+info
fig = topic_model.visualize_heatmap()
fig.write_html(f"summaries/{path_name}/results/heatmap.html")
visualize_documents.write_html(f"summaries/{path_name}/results/topic model docs.html") 

topic_modelling_time = time.time() - filtering_papers
time_report += f"Topic modelling time: {topic_modelling_time} seconds\n"
print(f"Topic modelling time: {topic_modelling_time} seconds")

topic_summaries = summarize(topic_model, dfs_by_topic, summarizer, bib_tex_keys, paper_names)


summarizer_time = time.time() - topic_modelling_time
time_report += f"Summarizer time: {summarizer_time} seconds\n"
print(f"Summarizer time: {summarizer_time} seconds")


summary = improve_summary(topic_summaries, topic_titles, title, gpt_model)


improve_summary_time = time.time() - summarizer_time
time_report += f"Improve summary time: {improve_summary_time} seconds\n"
print(f"Improve summary time: {improve_summary_time} seconds")


print("#######\nT5 Summary:")
for i in range(len(topic_summaries)):
  print(f"{topic_titles[i]}\n{summary[i]}")

final_summary=title+": Literature Review\n"
for i in range(1,len(summary)):
  final_summary+=f"\n\n{topic_titles[i]}\n{summary[i]}"

final_summary+=f"\n\n{topic_titles[0]}\n{summary[0]}"


latex_template = create_template(joint_title)

ordered_topic_titles = topic_titles[1:]
ordered_topic_titles.append(topic_titles[0])

latex_doc = create_latex_document(title, len(summary), final_summary, ordered_topic_titles, latex_template, gpt_model)

latex_doc_time = time.time() - improve_summary_time
time_report += f"Latex doc time: {latex_doc_time} seconds\n"
print(f"Latex doc time: {latex_doc_time} seconds")

final_time = time.time() - start_time
time_report += f"\nFull computation time: {final_time} seconds"
print(f"\nFull computation time: {final_time} seconds")

save_text(time_report, "time_report", title, info)


print("#######\nLatex summary:")
print(latex_doc)

save_to_tex(latex_doc, title, info)


de_latexed=de_latex(latex_doc)

raw_abstracts=""
for i in selected_index:
  raw_abstracts += articles[i].get("abstract") + " \n"

raw_summaries = " \n".join(topic_summaries)

cleaned_final_summary = re.sub(r'\\citep\{[^}]+\}', '', final_summary)
cleaned_raw_summaries = re.sub(r'\\citep\{[^}]+\}', '', raw_summaries)

save_text(raw_abstracts, "raw_abstracts", title, info)
save_text(cleaned_raw_summaries, "raw_summaries", title, info)
save_text(cleaned_final_summary, "final_summary", title, info)
save_text(latex_doc, "latex_doc", title, info)
save_text(de_latexed, "de_latex_doc", title, info)


#ROUGE
print("\nROUGE score for raw summaries")
raw_sum_rouge = get_rouge(raw_abstracts, raw_summaries)
pprint.pprint(raw_sum_rouge)

print("\n\nROUGE score for gpt edited summaries")
final_summary_rouge = get_rouge(raw_abstracts, final_summary)
pprint.pprint(final_summary_rouge)

print("\n\nROUGE score for de latexed document")
de_latexed_rouge = get_rouge(raw_abstracts, de_latexed)
pprint.pprint(de_latexed_rouge)

rouge_report = "Summarized documents: " + str(raw_sum_rouge) + "\nPost-Edited Summarized Document: " + str(final_summary_rouge) + "\n final LaTeX Document: " + str(de_latexed_rouge) 
save_text(rouge_report, "rouge_metrics", title, info)


# READABILITY
print("\nREADABILITY score for raw abstracts ")
raw_abstracts_readability = readability.getmeasures(tokenit(raw_abstracts), lang='en')['readability grades']
pprint.pprint(raw_abstracts_readability)

print("\nREADABILITY score for raw summaries")
raw_summaries_readability = readability.getmeasures(tokenit(raw_summaries), lang='en')['readability grades']
pprint.pprint(raw_summaries_readability)

print("\nREADABILITY score for gpt edited summaries")
final_summary_readability = readability.getmeasures(tokenit(final_summary), lang='en')['readability grades']
pprint.pprint(final_summary_readability)

print("\nREADABILITY score for de latexed document")
de_latexed_readability = readability.getmeasures(tokenit(de_latexed), lang='en')['readability grades']
pprint.pprint(de_latexed_readability)

readability_report = "Raw abstracts (baseline): " + str(raw_abstracts_readability) + "\nSummarized documents: " + str(raw_summaries_readability) + "\nPost-Edited Summarized Document: " + str(final_summary_readability) + "\n final LaTeX Document: " + str(de_latexed_readability) 
save_text(readability_report, "readability_metrics", title, info)


coherence = get_coherence(abstracts, topic_model)
print(f"Coherence Score: {coherence}")
save_text(str(coherence), "coherence_score", title, info)


abstract_embedding =  model.encode(raw_abstracts, convert_to_tensor=True)
abstract__sim = util.pytorch_cos_sim(abstract_embedding, input_embedding)
print(f"\nAbstract against input sim: {abstract__sim.tolist()[0]}")

raw_summaries_embedding =  model.encode(raw_summaries, convert_to_tensor=True)
raw_summaries_sim = util.pytorch_cos_sim(raw_summaries_embedding, input_embedding)
print(f"\nRaw summaries against input sim: {raw_summaries_sim.tolist()[0]}")

final_summary_embedding =  model.encode(final_summary, convert_to_tensor=True)
final_summary_sim = util.pytorch_cos_sim(final_summary_embedding, input_embedding)
print(f"\nFinal summary against input sim: {final_summary_sim.tolist()[0]}")

de_latexed_embedding =  model.encode(de_latexed, convert_to_tensor=True)
de_latexed_sim = util.pytorch_cos_sim(de_latexed_embedding, input_embedding)
print(f"\nFinal doc against input sim: {de_latexed_sim.tolist()[0]}")


# Calculate multiple random embeddings
num_random_embeddings = 4
random_scores = []
for _ in range(num_random_embeddings):
    random_embedding = model.encode(gen_sentence(), convert_to_tensor=True)
    random_sim = util.pytorch_cos_sim(random_embedding, input_embedding)
    random_scores.append(random_sim.tolist()[0])
    
# Organize similarity scores into lists (update bert_scores)
bert_scores = [abstract__sim.tolist()[0], raw_summaries_sim.tolist()[0], final_summary_sim.tolist()[0], de_latexed_sim.tolist()[0]]
avg_random_score = np.mean(random_scores)

labels = ['Abstracts', 'Summary', 'Post-Edited Summary', 'Literature Review', 'Random']

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

embedding_report = "Raw abstracts (baseline): " + str(avg_random_score) + "\nSummarized documents: " + str(raw_summaries_sim.tolist()[0]) + "\nPost-Edited Summarized Document: " + str(final_summary_sim.tolist()[0]) + "\nFinal LaTeX Document: " + str(de_latexed_sim.tolist()[0]) 
save_text(embedding_report, "embedding_report", title, info)