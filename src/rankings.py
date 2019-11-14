from collections import Counter, defaultdict
import csv
import lda
from math import log
import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import os
import json
import pandas as pd
from pdb import set_trace
import pickle
import random
import re
import string
import sys
import scipy as sp
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import BertTokenizer, BertModel

#paths
HOME="/Users/samir/Dev/projects/feedback_request_aligner/fra/"
OUTPUT_TXT = HOME+"DATA/processed/txt/"
OUTPUT_VECTORS = HOME+"DATA/processed/vectors/"
OUTPUT_RANKINGS = HOME+"DATA/processed/rankings/"

def similarity_ranks(Q, C, queries, comments, method, top_k):
    assert len(queries) == Q.shape[0]
    assert len(comments) == C.shape[0]
    S = cosine_similarity(Q,C)
    ranks = np.argsort(S, axis=1)[:,::-1]
    top_ranks = ranks[:,:top_k]
    results = []
    for i in range(top_ranks.shape[0]):
        rank_i  = top_ranks[i]
        sims = S[i, rank_i]  
        sims = [str(x.round(3)) for x in sims]
        sentence_ids = comments["sentenceId"].iloc[rank_i].values.tolist()   
        sentences = comments["text"].iloc[rank_i].values.tolist()
        query_id = queries.iloc[i]["queryId"]
        results+=[[method,query_id,sid,sim,txt.replace("\n"," ").replace("\t"," ")] 
                    for sid,sim,txt in zip(sentence_ids, sims, sentences)]
    return results


def rank_docket(docket, methods, top_k=5):    
    #load docket data    
    print("[reading docket: {}]".format(docket))
    df_queries = pd.read_csv(OUTPUT_TXT+"{}_queries.csv".format(docket))
    df_comments = pd.read_csv(OUTPUT_TXT+"{}_comments.csv".format(docket))
    df_queries = df_queries.dropna(subset=['clean_text'])
    df_comments = df_comments.dropna(subset=['clean_text'])

    #ranks for each method
    ranks = []
    for m in methods:
        print("[ranking method: {}]".format(m))
        if m == "tf-idf":
            Q = sp.sparse.load_npz(OUTPUT_VECTORS+"{}_queries_tf-idf.npz".format(docket))
            C = sp.sparse.load_npz(OUTPUT_VECTORS+"{}_comments_tf-idf.npz".format(docket))
        else:
            with open(OUTPUT_VECTORS+"{}_queries_{}.np".format(docket,m),"rb") as f:
                Q = np.load(f)
            with open(OUTPUT_VECTORS+"{}_comments_{}.np".format(docket,m),"rb") as f:
                C = np.load(f)      
        rank = similarity_ranks(Q, C, df_queries, df_comments, m, top_k)
        ranks+=rank
    
    with open(OUTPUT_RANKINGS+"{}_rank.csv".format(docket),"w") as fo:
        for r in ranks:
            fo.write("\t".join([docket]+r)+"\n")
    return ranks    

dockets = ['FDA-2014-N-0189', 'FDA-2013-N-0521', 
            'FDA-2015-N-1514', 'FDA-2012-N-1148', 
            'FDA-2011-N-0467'][::-1]

for docket in dockets:
    rank_docket(docket,["boe","boe_tuned","tf-idf","bert_cls","bert_pool","lda"])