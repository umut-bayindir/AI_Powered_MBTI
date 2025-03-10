# MBTI Personality Classifier

## Overview
This project is an MBTI (Myers-Briggs Type Indicator) Personality Classifier that uses Natural Language Processing (NLP) techniques, including NLTK, spaCy, and a pre-trained BERT model, to analyze user responses and predict their MBTI personality type based on cognitive functions and sentiment analysis.

## Features
- **NLP Processing:** Utilizes `nltk` for tokenization and `spaCy` for text processing.
- **BERT Sentiment Analysis:** Uses a pre-trained BERT model (`nlptown/bert-base-multilingual-uncased-sentiment`) to analyze the sentiment of responses.
- **Cognitive Function Matching:** Analyzes user responses against predefined cognitive function keywords to determine MBTI type.
- **Interactive User Input:** Asks users a series of questions to gather responses for MBTI classification.

## Installation
To run this project, install the required dependencies:

```sh
pip install nltk spacy torch transformers
