import nltk
import spacy
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from collections import defaultdict

def setup_nlp_models():
    nltk.download("punkt")
    spacy.cli.download("en_core_web_sm")
    return spacy.load("en_core_web_sm")

# Load pre-trained BERT model and tokenizer
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME)

# Cognitive function keywords
function_keywords = {
    'Ni': ['insight', 'deep', 'vision', 'future', 'possibility', 'analysis'],
    'Ne': ['explore', 'connections', 'new', 'possibilities', 'innovative', 'excited'],
    'Fi': ['value', 'ethical', 'inner', 'personal', 'authentic'],
    'Te': ['goal', 'practical', 'plan', 'logical', 'objective', 'data'],
    'Ti': ['analyze', 'logic', 'reason', 'systematic', 'internally'],
    'Fe': ['empathetic', 'social', 'feelings', 'hearts', 'relationship'],
    'Si': ['past', 'experience', 'familiar', 'routine', 'tradition'],
    'Se': ['immediate', 'sensory', 'action', 'observe', 'environment']
}

def analyze_response(response):
    scores = defaultdict(int)
    tokens = [token.lower() for token in nltk.word_tokenize(response)]
    for func, keywords in function_keywords.items():
        for word in tokens:
            if word in keywords:
                scores[func] += 1
    return scores

def get_bert_sentiment(response):
    inputs = tokenizer(response, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    scores = torch.softmax(outputs.logits, dim=1).tolist()[0]
    return scores

def determine_mbti_type(user_responses):
    total_scores = defaultdict(int)
    for question, response in user_responses.items():
        func_scores = analyze_response(response)
        for func, score in func_scores.items():
            total_scores[func] += score
    
    introverted_score = total_scores['Ni'] + total_scores['Ti'] + total_scores['Fi'] + total_scores['Si']
    extraverted_score = total_scores['Ne'] + total_scores['Te'] + total_scores['Fe'] + total_scores['Se']
    ie = 'I' if introverted_score >= extraverted_score else 'E'
    
    ns = 'N' if (total_scores['Ni'] + total_scores['Ne']) >= (total_scores['Si'] + total_scores['Se']) else 'S'
    tf = 'T' if (total_scores['Ti'] + total_scores['Te']) >= (total_scores['Fi'] + total_scores['Fe']) else 'F'
    dominant_function = max(total_scores, key=total_scores.get)
    jp = 'J' if dominant_function in ['Te', 'Fe'] else 'P'
    
    return ie + ns + tf + jp

def interactive_mbti_demo():
    print("Welcome to the MBTI Personality Classifier!")
    user_responses = {}
    
    questions = [
        "How do you prefer to interact in social situations?",
        "Do you focus more on facts and details, or on abstract ideas?",
        "When making decisions, do you prioritize logic or emotions?",
        "Do you prefer structure and planning, or flexibility?"
    ]
    
    for q in questions:
        response = input(q + "\n")
        user_responses[q] = response
    
    mbti_type = determine_mbti_type(user_responses)
    print(f"Your MBTI type is likely: {mbti_type}")

if __name__ == "__main__":
    setup_nlp_models()
    interactive_mbti_demo()
