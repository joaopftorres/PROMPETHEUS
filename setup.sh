#!/bin/bash

# Install Python packages
python3 -m pip install -r requirements.txt

# Download NLTK stopwords and punkt
python3 -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"

# Download spaCy model
python3 -m spacy download en_core_web_sm
