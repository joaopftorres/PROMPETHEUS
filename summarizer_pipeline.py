from prompts import post_edit

def summarize(topic_model, dfs_by_topic, summarizer, bib_tex_keys, paper_names):
  topic_summaries=[]
  for i, df_topic in enumerate(dfs_by_topic):
    topic_summary=""
    documents_on_topic=df_topic.Document.tolist()
    ids_on_topic=df_topic.index.tolist()
    for i in range(len(documents_on_topic)):
      doc=documents_on_topic[i]
      index=ids_on_topic[i]
      bib_tex_key = bib_tex_keys[index]
      topic_summary += summarizer(doc, max_length=100, min_length=20, do_sample=False)[0].get('summary_text') + ("\n") +  f"\\citep{{{bib_tex_key}}}\n" + ("\n")
    topic_summaries.append(topic_summary)
  return topic_summaries


def improve_summary(topic_summaries, topic_titles, title, model = "gpt-3.5-turbo"):
  sums = []
  for i in range(len(topic_summaries)):
    sum = post_edit(topic_summaries[i], topic_titles[i], title, model)
    sums.append(sum)
  return sums