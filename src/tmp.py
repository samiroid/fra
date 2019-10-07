from collections import Counter
import lda
from math import log
import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import os
import pandas as pd
import pickle
import sys

from gensim.models import KeyedVectors

ASMAT_PATH="/Users/samir/Dev/projects/ASMAT2"
sys.path.append(ASMAT_PATH)
sys.path.append("..")
from ASMAT import vectorizer, embeddings, features


HOME="/Users/samir/Dev/projects/comment_feedback_aligner/"
FEEDBACK_REQUESTS_PATH = HOME+"DATA/raw/regulations_proposed_rules_feedback.csv"
COMMENTS_PATH=HOME+"DATA/raw/filtered_final_dockets_ecig.obj"
WORD2VEC=HOME+"DATA/embeddings/skip_50.txt"

OUTPUT_TXT = HOME+"DATA/processed/txt/"
OUTPUT_PKL = HOME+"DATA/processed/pkl/"
OUTPUT_VECTORS = HOME+"DATA/processed/vectors/"

CORPUS=OUTPUT_TXT+"all_text.txt"
VOCABULARY_PATH=OUTPUT_PKL+"vocabulary.pkl"
IDF_ESTIMATE_PATH=OUTPUT_PKL+"IDF.pkl"


with open(VOCABULARY_PATH,"rb") as f:
    vocab = pickle.load(f)
# with open(CORPUS,"r") as f:
#     all_text = f.readlines()

# wv_from_text = KeyedVectors.load_word2vec_format(WORD2VEC, binary=False)
# print(wv_from_text)

embeddings.filter_embeddings(WORD2VEC, OUTPUT_PKL+"word2vec.txt", vocab)

# print("reading")
# all_idxs, _ = vectorizer.docs2idx(all_text[:1000], vocab)
# print(len(all_idxs))
# n_topics=100
# n_iter=200
# print("extracting features")
# X = features.BOW_freq(all_idxs, vocab,sparse=True)
# print("creating model")
# topic_model = lda.LDA(n_topics=n_topics, n_iter=n_iter)
# X = X.astype('int32')
# print("training model")
# topic_model.fit(X)
# #save model
# with open(OUTPUT_PKL+"/lda.pkl","wb") as f:
#     pickle.dump([topic_model, vocab], f)
# Xt = topic_model.transform(X)
# print(Xt)