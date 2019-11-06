import json
import string


def read_corpus(file_path, source='src', verbose=True):
    # return list of list of words
    corpus=json.load(open(file_path))
    review_texts=[]
    for review in corpus:
        try:
            text=review["text"].lower()
        except KeyError: continue
        text=text.translate(str.maketrans('', '', string.punctuation))
        review_texts.append(text.split())
    return review_texts
