from bertopic.representation import KeyBERTInspired
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from gpt import *


def create_topic_model(docs, model):
  min_topic_size=10
  n_topics=5
  emb_model = model
  doc_embeddings = emb_model.encode(docs, show_progress_bar=False)
  topic_model = BERTopic(min_topic_size=min_topic_size).fit(docs, doc_embeddings)
  while (len(topic_model.get_topics())<n_topics):
    if min_topic_size == 4 and (3 <= n_topics <= 5):
      n_topics-=1
    elif (1 < min_topic_size <= 4) and n_topics == 2:
      min_topic_size-=1
    elif min_topic_size == 1 and n_topics == 2:
      print("Topic model: finding best parameters for topic model...")
    else:
      min_topic_size-=1

    topic_model = BERTopic(min_topic_size=min_topic_size).fit(docs, doc_embeddings)

  return topic_model, doc_embeddings


def topic_model_pipeline(docs, model, gpt_model="gpt-3.5-turbo"):
  topic_model, doc_embeddings= create_topic_model(docs, model)

  topic_report =f"Topics found:\n{topic_model.get_topic_info()}\n"

  topics_list=topic_model.get_document_info(docs)

  dfs_by_topic = []
  for topic, group_df in topics_list.groupby('Topic'):
      dfs_by_topic.append(group_df)

  for i, df_topic in enumerate(dfs_by_topic):
    topic_info_report=f"Documents on Topic {i-1}: {df_topic.shape[0]}"
    topic_report += topic_info_report +"\n"

  topic_titles=[]
  topic_report += "\nKeywords:\n"
  for i in range(-1,len(dfs_by_topic)-1):
    topic_n = [item[0] for item in topic_model.get_topic(i)]
    topic_keywords = ", ".join(topic_n)
    topic_report += topic_keywords+"\n"
    
    topic_title=get_title_gpt(topic_keywords, gpt_model)
    topic_titles.append(topic_title)
  gen_titles=f"Generated topic titles:\n{topic_titles}"
  topic_report += "\n" +f"Generated topic titles:\n{topic_titles}"

  visualize_documents=topic_model.visualize_documents(docs, embeddings=doc_embeddings)

  return topic_model, dfs_by_topic, topic_titles, visualize_documents, topic_report

  