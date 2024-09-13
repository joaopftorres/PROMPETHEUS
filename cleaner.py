import re
from unicode_tr.extras import slugify
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Set of English stopwords for filtering
STOP_WORDS = set(stopwords.words('english'))
PREFIXES = ('xcite', 'xmath', 'fig')


def remove_stopwords(text):
    """Remove stopwords from the text."""
    word_tokens = word_tokenize(text)
    filtered_words = [word for word in word_tokens if word.lower() not in STOP_WORDS]
    return " ".join(filtered_words)


def clean_text(line):
    """Clean the input text by removing stopwords, unwanted phrases, and non-ASCII characters."""
    line = remove_stopwords(line)
    
    # Replace specific phrases
    replacements = {
        'we ': ' ', 'We ': ' ', 'our ': ' ', 'Our ': ' ', 
        'In this study': 'studies', 'in this study': 'studies',
        'In this paper': 'studies', 'in this paper': 'studies',
        'This work': 'studies', 'this work': 'studies',
        'In this section': 'studies', 'in this section': 'studies',
        'The paper': 'studies', 'the paper': 'studies'
    }
    for old, new in replacements.items():
        line = line.replace(old, new)
    
    # Replace non-word characters with spaces
    line = re.sub(r'[^\w]', ' ', line)
    line = line.replace('_', ' ')  # Replace underscores with spaces

    # Handle non-ASCII words by slugifying them
    if not line.isascii():
        words = line.split(" ")
        words = [slugify(word) if not word.isascii() else word for word in words]
        line = ' '.join(words)
    
    # Remove extra spaces
    line = re.sub(' +', ' ', line)

    # Remove words starting with prefixes
    line = ' '.join(word for word in line.split() if not word.startswith(PREFIXES))

    return line


def cut_papers(max_selection, selected_sim, selected_index):
    """Reduce the number of selected papers to `max_selection` based on similarity scores."""
    selected_sim, selected_index = zip(*sorted(zip(selected_sim, selected_index), reverse=True))
    selected_sim = selected_sim[:max_selection]
    selected_index = list(selected_index[:max_selection])

    excluded = [index for sim, index in zip(selected_sim, selected_index) if sim not in selected_sim]

    return selected_index, excluded
