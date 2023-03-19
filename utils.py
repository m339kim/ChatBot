import nltk
# init:
#   nltk.download('punkt')
import numpy as np

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

# Workflow
# 1. Tokenize
# 2. Lowercase, stem
# 3. Elim punctuation chars
# 4. Convert to Bag of words

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    # place a '1' at bow[idx] if words[idx] exists in sentence[].
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bow = np.zeros(len(all_words), dtype = np.float32)
    for idx, w, in enumerate(all_words):
        if w in tokenized_sentence:
            bow[idx] = 1.0
    return bow