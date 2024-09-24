import openai
import os
from dotenv import load_dotenv

load_dotenv()

client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def expand_title(title, model="gpt-3.5-turbo"):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": """You are a knowledgeable AI specializing in generating expanded titles for research topics.  Your expanded titles should be concise and focus on capturing the core semantic meaning of a topic, suitable for creating informative embeddings for tasks like similarity comparisons."""},
            {"role": "user", "content": f"""
                Task: Generate a slightly expanded title for the following research topic, keeping the core focus while potentially adding 1-2 highly relevant terms for improved semantic representation.

                Topic: {title}

                Guidelines:
                * Include essential keywords directly related to the topic.
                * If necessary, add 1-2 closely related terms to capture variations of the topic.
                * Avoid introducing new concepts or significantly altering the original title's meaning.
                * Keep the expanded title concise and focused on the core meaning.

                Examples:

                | Topic                                     | Expanded Title                                                                                                                               |
                |-----------------------|----------------------------------------------------------------------------------------------------------------------------------------|
                | Explainable AI         | Explainable Artificial Intelligence (XAI)                                                                                              |
                | Graph Neural Networks  | Graph Neural Networks (GNNs), Graph Representation Learning                                                                  |
                | Transformers (NLP)     | Transformer Models for Natural Language Processing (NLP)                                                                           |
                | Reinforcement Learning | Deep Reinforcement Learning, Reward-based Learning                                                                               |
                | Federated Learning             | Federated Learning, Privacy-Preserving Machine Learning                                                                         |
                | Generative Adversarial Networks (GANs) | Generative Adversarial Networks (GANs), Deep Generative Models |
                | Natural Language Processing (NLP)     | Natural Language Processing (NLP), Language Models, Transformers |
                | Climate Change Impacts on Coastal Ecosystems | Coastal Ecosystem Vulnerability, Climate Change Adaptation, Sea Level Rise                                                      |
                | Renewable Energy Integration in Power Grids  | Grid Integration of Renewable Energy, Smart Grids, Energy Storage Systems                                                             |
                | The Impact of Social Media on Political Discourse | Political Communication, Social Media Influence, Online Public Opinion                                                             |

                Output format:
                * Provide the expanded title only. Do not include any additional explanations or commentary.
                """}
        ]
    )
    # expanded topic
    return completion.choices[0].message.content



def create_arxiv_query(topic, model="gpt-3.5-turbo"):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": """You are a skilled research assistant specializing in crafting precise and effective search queries for the arXiv scientific paper repository."""},
            {"role": "user", "content": f"""
                Task: Craft an effective search query tailored for the arXiv database, specifically designed to retrieve research papers pertaining to the following topic:

                Topic: {topic}

                Guidelines:
                1.  Concise & Precise: The query should be succinct yet accurately represent the core concept of the topic.
                2.  Key Terms: Incorporate the most relevant keywords or phrases directly associated with the topic.
                3.  Synonyms & Variants (Optional): If applicable, include synonyms or alternative terms to broaden the search scope and capture nuanced variations of the topic.
                4.  Specificity:  Prioritize terms that are specific to the field or subfield to minimize irrelevant results.
                5.  arXiv Compatibility: Utilize operators like `ti:` (title) and `abs:` (abstract) to target specific fields within the arXiv entries.

                Example Outputs:

                | Topic                                      | Query                                                                                                                                          |
                | ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
                | Reinforcement Learning for Robotics        | (ti:"reinforcement learning" OR abs:("reinforcement learning" OR "RL")) AND (ti:"robotics" OR abs:"robotics")                                    |
                | Explainable AI in Medical Imaging          | (ti:"explainable AI" OR abs:("explainable AI" OR "XAI")) AND (ti:"medical imaging" OR abs:"medical imaging")                                     |
                | Graph Neural Networks for Drug Discovery   | (ti:"graph neural network*" OR abs:("graph neural network*" OR "GNN")) AND (ti:"drug discovery" OR abs:"drug discovery")                       |
                | Self-Supervised Learning for Computer Vision| (ti:"self-supervised learning" OR abs:"self-supervised learning") AND (ti:"computer vision" OR abs:("computer vision" OR "image recognition"))  |
                | Quantum Machine Learning                   | (ti:"quantum machine learning" OR abs:"quantum machine learning") AND (ti:"quantum algorithms" OR abs:"quantum algorithms")                    |

                Output format:
                * Provide the arxiv query only. Do not include any additional explanations or commentary.
                """}
        ]
    )
    # arxiv query
    return completion.choices[0].message.content






def get_title(topic_keywords, model = "gpt-3.5-turbo"):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": """You are an experienced researcher specializing in literature reviews. You are adept at crafting concise, informative, and engaging topic names for subsections that accurately reflect the content and guide the reader."""},
            {"role": "user", "content": f"""
                Task: Create a clear and concise topic name for a subsection in a literature review. The subsection covers the following keywords: {topic_keywords}

                Guidelines:
                * Length: Aim for 1-5 words.
                * Accuracy: Ensure the topic name precisely reflects the keywords' meaning.
                * Relevance: The name should fit within the broader context of a literature review.
                * Informativeness: Clearly indicate the subsection's focus to the reader.
                * Engagement: Make the topic name interesting and inviting to read.

                Optional: If the keywords are too broad or ambiguous, suggest a more specific or narrowed-down focus within the topic.

                Output format:
                * Provide the topic title only. Do not include any additional explanations or commentary.
                """}
        ]
    )
    #topic name
    return completion.choices[0].message.content.strip()  # Remove extra whitespace




def create_template(joint_title):
  latex_template = r"""\documentclass[12pt]{article}

  \usepackage{amsmath}
  \usepackage{amssymb}
  \usepackage{graphicx}
  \usepackage{hyperref}
  \usepackage{natbib}

  \begin{document}

  \title{Literature Review}
  \date{\today}

  \maketitle

  \begin{abstract}
  This literature review provides an overview of the research on [topic]. The review covers the following areas: [list of areas]. The review concludes by discussing the implications of the research for [topic].
  \end{abstract}

  \section{Introduction}

  [Introduction to the topic of the literature review.]

  \section{Background}

  [Background information on the topic of the literature review.]

  \section{Literature Review}

  [Review of the literature on the topic of the literature review.]

  \section{Discussion}

  [Discussion of the implications of the research for the topic of the literature review.]

  \section{Conclusion}

  [Conclusion of the literature review.]

  \bibliographystyle{plainnat}

  """


  # Construct the LaTeX References section
  reference_section = f"\n\\bibliography{{{joint_title}.bib}}\n\n"

  latex_template+=reference_section +  "\\end{document}"

  return latex_template


def post_edit(summary, section_name, title, model = "gpt-3.5-turbo"):
  completion = client.chat.completions.create(
      model=model,
      messages=[
          {"role": "system", "content": f"""
              You are an expert researcher specializing in literature reviews in the field of {title}.
              Your task is to meticulously refine and enhance machine-generated summaries of multiple research papers.
          """},
          {"role": "user", "content": f"""
              Refine the following machine-generated summary for the section "{section_name}" in a literature review titled "{title}".

              The original summary is a compilation from various papers. Please focus on retaining the most relevant information for this section of the literature review.

              Crucially, ensure the inclusion of in-text citations (e.g., \\citep{{kadir2024revealing}}) for all information directly sourced from the referenced documents. Feel free to shorten the section summary if it enhances clarity and conciseness, but prioritize keeping essential details and all relevant citations.

              Original Summary:
              ```
              {summary}
              ```

              Output format:
              * Provide only the revised summary. Do not include any additional explanations or commentary.
              """}
      ]
  )

  return completion.choices[0].message.content.strip()



def create_latex_document(title, number_sections, final_summary, section_names, latex_template, model = "gpt-3.5-turbo"):
    # Remove citations from final_summary

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": """You are a highly experienced researcher specializing in writing comprehensive and well-structured literature reviews in LaTeX. Your expertise includes summarizing key findings, identifying research gaps, and highlighting the implications of research."""},
            {"role": "user", "content": f"""
                Task: Create a well-organized and informative Literature Review in LaTeX format based on the provided information.

                Title: {title}
                Sections: {number_sections} with names: {section_names}
                Summary Content: {final_summary}

                LaTeX Template:
                ```
                {latex_template}
                ```

                Instructions:

                1. Structure: Follow the LaTeX template, filling in each section with content from the summary.
                2. References: Ensure the inclusion of the in-text citations (e.g., \\citep{{kadir2024revealing}}) in the literature review, as presented in the provided summary, in each relevant part of the section where the document is referenced.
                3. Abstract: Write a concise abstract summarizing the literature review's main themes, key findings, and implications.
                4. Introduction: Provide an overview of the topic, its significance, and the review's objectives.
                5. Background: Give necessary context or foundational information for the reader.
                6. Literature Review: Critically analyze the literature for each section, highlighting key contributions, methodologies, strengths, and limitations.
                7. Discussion: Synthesize findings across sections, identify trends, gaps, and controversies, and discuss the implications of the research.
                8. Conclusion: Summarize the key takeaways from the literature review and offer potential directions for future research.

                Additional Notes:
                * Ensure the Literature Review is comprehensive, well-organized, and adheres to academic writing standards.
                * If a section seems irrelevant to the overall topic, omit it or briefly mention its limited relevance.
                * Do not shorten the summary content excessively; prioritize maintaining a thorough and informative review.
                * Ensure the inclusion of the in-text citations (e.g., \\citep{{kadir2024revealing}}) in the literature review, as presented in the provided summary, in each relevant part of the section where the document is referenced.

                 Output format:
                * Provide only the final Literature review latex document only. Do not include any additional explanations or commentary.
                """}
        ]
    )
    return completion.choices[0].message.content



def de_latex(latex_doc, model = "gpt-3.5-turbo"):
    # Remove citations from final_summary

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": """You are a highly experienced researcher specializing in transforming latex documents into plain text."""},
            {"role": "user", "content": f"""
                Task: Transform the following latex document into plain text, without the latex format. Remove all citations.

                LaTeX document:
                ```
                {latex_doc}
                ```
                 Output format:
                * Return only the final plain text document. Do not include any additional explanations or commentary.
                """}
        ]
    )
    return completion.choices[0].message.content


