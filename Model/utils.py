from numpy import log

import string
import spacy

from spacy.lang.en.stop_words import STOP_WORDS
from collections import Counter


nlp = spacy.load('en_core_web_sm')
def tokenize_sentence(sentence):
    # Error handling for non-string inputs
    if not isinstance(sentence, str):
        return []
    
    # Tokenise the sentence
    tokenized = nlp(sentence)
    
    # Lowercase and lemmatise the words
    tokens = [token.lemma_.lower().strip() if token.pos_ != "PRON" else token.lower_ for token in tokenized]
    
    # Remove empty stings
    tokens = [token for token in tokens if token != '']

    # Remove stop words and special characters
    tokens = [token for token in tokens if token not in STOP_WORDS and token not in string.punctuation]

    return tokens

def term_frequency(documents):

    token_counts = list()
    for document in documents:
        if not document:
            token_counts.append({})
            continue
        document = document[1:]

        token_count = Counter(document).items()
        
        total_tokens = len(document)
        
        token_count_normalized = {token: count/ total_tokens for token, count in token_count}
        
        token_counts.append(token_count_normalized)
    
    return token_counts

def inverse_document_frequency(documents):

    total_documents = len(documents)

    token_counts = Counter()
    for document in documents:
        document = document[1:]

        token_counts.update(set(document))

    idf_values = {term: log(total_documents / (count + 1)) + 1e-10 for term, count in token_counts.items() if count > 1} 
    return idf_values

def tfidf_calculator(tf, idf):

    scores = list()
    for document in tf:
        document_score ={token: frequency * idf[token] for token, frequency in document.items() if token in idf}

        scores.append(document_score)

    return scores