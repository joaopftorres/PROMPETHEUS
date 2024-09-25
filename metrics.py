from rouge_score import rouge_scorer
import os
import nltk
import re
import spacy
from nltk.tokenize import word_tokenize
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from wonderwords import RandomWord
import random

sp = spacy.load('en_core_web_sm')

def get_rouge(original, summarized):
  scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

  # Compute ROUGE scores
  scores = scorer.score(original, summarized)
  return {"precision": scores['rouge1'].precision, "recall": scores['rouge1'].recall, "f1": scores['rouge1'].fmeasure }
    
      
def tokenit(working_txt):
  # clean text by removing successive white space and line breaks
  clean_txt = re.sub(r"\n", " ", working_txt)
  clean_txt = re.sub(r"\s+", " ", clean_txt)
  clean_txt = clean_txt.strip()
  tokens = word_tokenize(clean_txt)

  filtered_tokens_alpha = [word for word in tokens if word.isalpha()]
  
  # replace non-alphabetic characters with single whitespace
  reg_txt = re.sub(r'[^a-zA-Z\s]', ' ', clean_txt)
  # remove any whitespace that appears in sequence
  reg_txt = re.sub(r"\s+", " ", reg_txt)
  # remove any new leading and trailing whitespace
  reg_txt = reg_txt.strip()
  # tokenize regularized text
  reg_tokens = word_tokenize(reg_txt)
  #print(reg_tokens)

  return reg_tokens

def get_coherence(docs, topic_model):
  
  # Preprocess the documents using the topic model's preprocessing method
  cleaned_docs = topic_model._preprocess_text(docs)

  # Build the analyzer to tokenize the documents
  vectorizer = topic_model.vectorizer_model
  analyzer = vectorizer.build_analyzer()
  tokens = [analyzer(doc) for doc in cleaned_docs]

  # Create a dictionary and corpus needed for coherence calculation
  dictionary = corpora.Dictionary(tokens)
  corpus = [dictionary.doc2bow(token) for token in tokens]

  # Extract topics and get the words for each topic
  topics = topic_model.get_topics()

  # Ensure topics are a list of topic indices if not already
  if not isinstance(topics, list):
      topics = list(topics)

  # Remove any invalid topics (if any)
  topics = [topic for topic in topics if isinstance(topic, int)]

  # Get topic words
  topic_words = [
      [word for word, _ in topic_model.get_topic(topic_id)] for topic_id in topics
  ]

  # Evaluate coherence using CoherenceModel from Gensim
  coherence_model = CoherenceModel(
      topics=topic_words,
      texts=tokens,
      corpus=corpus,
      dictionary=dictionary,
      coherence='c_v'
  )
  coherence = coherence_model.get_coherence()

  return coherence

def gen_sentence(lenght=100):
  r = RandomWord()
  sentence=""
  for i in range(lenght):
    sentence+=r.word() + " "
  return sentence


  
