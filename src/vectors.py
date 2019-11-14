from collections import Counter
import csv
import lda
from math import log
import numpy as np
# from nltk.tokenize import sent_tokenize
# from nltk.corpus import stopwords
import os
# import json
import pandas as pd
from pdb import set_trace
import pickle
# import random
import re
import string
import sys
import scipy as sp
# from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import BertTokenizer, BertModel


#add ASMAT toolkit
ASMAT_PATH="/Users/samir/Dev/projects/ASMAT2"
sys.path.append(ASMAT_PATH)
sys.path.append("..")
from ASMAT import vectorizer, embeddings, features
from ASMAT.toolkit import gensimer

#paths
HOME="/Users/samir/Dev/projects/feedback_request_aligner/fra/"
WORD2VEC_INPUT=HOME+"DATA/embeddings/glove.42B.300d.txt"
OUTPUT_TXT = HOME+"DATA/processed/txt/"
OUTPUT_PKL = HOME+"DATA/processed/pkl/"
OUTPUT_VECTORS = HOME+"DATA/processed/vectors/"
WORD2VEC=OUTPUT_PKL+"/word2vec.txt"
WORD2VEC_TUNED=OUTPUT_PKL+"/word2vec_tuned.txt"
CORPUS=OUTPUT_TXT+"all_text.txt"
VOCABULARY_PATH=OUTPUT_PKL+"vocabulary.pkl"
IDF_ESTIMATE_PATH=OUTPUT_PKL+"IDF.pkl"

if not os.path.exists(OUTPUT_PKL):
    os.makedirs(OUTPUT_PKL)
if not os.path.exists(OUTPUT_VECTORS):
    os.makedirs(OUTPUT_VECTORS)

def getIDF(N, t):
    return log(float(N)/float(t))

MIN_WORD_FREQ=5
N_TOPICS = 100
LDA_EPOCHS = 100
VECTOR_DIM=300
NEGATIVE_SAMPLES=10
W2V_EPOCHS=5
BERT_MAX_INPUT=510
BERT_BATCH_SIZE = 100

def compute_vocabulary():
    with open(CORPUS,"r") as f:
        all_text =   f.readlines()
    #get vocabulary
    vocab = vectorizer.build_vocabulary(all_text, min_freq=MIN_WORD_FREQ)
    print("vocabulary size: {}".format(len(vocab)))
    #save vocabulary
    with open(VOCABULARY_PATH,"wb") as f:
        pickle.dump(vocab,f)

def compute_IDF():
    with open(VOCABULARY_PATH,"rb") as f:
        vocab = pickle.load(f)
    with open(CORPUS,"r") as f:
        all_text =   f.readlines()
    #compute document frequencies
    all_idxs, _ = vectorizer.docs2idx(all_text, vocab)
    ndocs = len(all_idxs)
    docfreq = Counter(str(x) for xs in all_idxs for x in set(xs))
    #inverse document frequencies
    idfs = {w: getIDF(ndocs, docfreq[w]) for w in docfreq}
    #get an IDF vector 
    idfvec = np.zeros(len(idfs))
    for w, v in idfs.items(): idfvec[int(w)] = v
    with open(OUTPUT_PKL+"/IDF.pkl","wb") as f:
        pickle.dump(idfvec,f)

#extract word embeddings
def get_word_embeddings():
    with open(VOCABULARY_PATH,"rb") as f:
        vocab = pickle.load(f)
    embeddings.extract_embeddings(WORD2VEC_INPUT, WORD2VEC, vocab)

def update_word_embeddings():
    #update word embeddings 
    train_seq = gensimer.Word2VecReader([CORPUS])
    w2v = gensimer.get_skipgram(dim=VECTOR_DIM,negative_samples=NEGATIVE_SAMPLES, min_freq=MIN_WORD_FREQ)
    w2v_trained = gensimer.train_skipgram(w2v, train_seq, epochs=W2V_EPOCHS,
                                        path_out=WORD2VEC_TUNED,
                                        pretrained_weights_path=WORD2VEC)

def train_topic_model():
    with open(VOCABULARY_PATH,"rb") as f:
        vocab = pickle.load(f)
    with open(CORPUS,"r") as f:
        all_text =   f.readlines()
    all_idxs, _ = vectorizer.docs2idx(all_text, vocab)
    X = features.BOW_freq(all_idxs, len(vocab), sparse=True)
    X = X.astype('int32')
    topic_model = lda.LDA(n_topics=N_TOPICS, n_iter=LDA_EPOCHS)
    topic_model.fit(X)
    #save model
    with open(OUTPUT_PKL+"/lda.pkl","wb") as f:
        pickle.dump([topic_model, vocab], f)

def transformer_encoder(tokenizer, encoder, D):    
    tokens_tensors = []
    segments_tensors = []
    tokenized_texts = []
    
    bertify = "[CLS] {} [SEP]"  
    tokenized_texts = [tokenizer.tokenize(bertify.format(doc)) for doc in D] 

    #count the document lengths  
    max_len = max([len(d) for d in tokenized_texts]) 
    #document cannot exceed BERT input matrix size 
    max_len = min(BERT_MAX_INPUT, max_len)
    # print("[max len: {}]".format(max_len))
    for tokens in tokenized_texts:   
        # Convert tokens to vocabulary indices
        idxs = tokenizer.convert_tokens_to_ids(tokens)        
        #truncate sentences longer than what BERT supports
        if len(idxs) > BERT_MAX_INPUT: idxs = idxs[:BERT_MAX_INPUT]
        pad_size = max_len - len(idxs)
        #add padding to indexed tokens
        idxs+=[0] * pad_size
        segments_ids = [0] * len(idxs) 
        tokens_tensors.append(torch.tensor([idxs]))
        segments_tensors.append(torch.tensor([segments_ids]))
    
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.cat(tokens_tensors)
    segments_tensor = torch.cat(segments_tensors)
    
    #set encoder to eval mode
    encoder.eval()
    with torch.no_grad():        
        pool_features, cls_features = encoder(tokens_tensor, token_type_ids=segments_tensor)    
        pool_features = pool_features.sum(axis=1)
    return cls_features.numpy(), pool_features.numpy()

def BERT_vectors(docket, tokenizer, model):
    df_queries = pd.read_csv(OUTPUT_TXT+"{}_queries.csv".format(docket))
    df_comments = pd.read_csv(OUTPUT_TXT+"{}_comments.csv".format(docket))
    df_queries = df_queries.dropna(subset=['clean_text'])
    df_comments = df_comments.dropna(subset=['clean_text'])
    #BERT
    query_pool_vectors = []
    query_cls_vectors = []
    all_queries = df_queries["clean_text"]
    n_batches = int(len(all_queries)/BERT_BATCH_SIZE)+1
    for j in range(n_batches):
        query_batch = all_queries[BERT_BATCH_SIZE*j:BERT_BATCH_SIZE*(j+1)]
        if len(query_batch) > 0:
            sys.stdout.write("\rquery batch:{}\{} (size: {})".format(j+1,n_batches, str(len(query_batch))))
            sys.stdout.flush()
            Q_cls, Q_pool = transformer_encoder(tokenizer, model, query_batch)
            # set_trace()
            query_cls_vectors.append(Q_cls)
            query_pool_vectors.append(Q_pool)            
    query_pool_vectors = np.vstack(query_pool_vectors)
    query_cls_vectors = np.vstack(query_cls_vectors)
    print()
    comment_cls_vectors = []
    comment_pool_vectors = []
    all_comments = df_comments["clean_text"]
    n_batches = int(len(all_comments)/BERT_BATCH_SIZE)+1
    for j in range(n_batches):
        comment_batch = all_comments[BERT_BATCH_SIZE*j:BERT_BATCH_SIZE*(j+1)]
        if len(comment_batch) > 0:
            sys.stdout.write("\rcomment batch:{}\{} (size: {})".format(j+1,n_batches, str(len(comment_batch))))
            sys.stdout.flush()
            C_cls, C_pool = transformer_encoder(tokenizer, model, comment_batch)
            comment_cls_vectors.append(C_cls)
            comment_pool_vectors.append(C_pool)
    comment_pool_vectors = np.vstack(comment_pool_vectors)
    comment_cls_vectors = np.vstack(comment_cls_vectors)
    print()
    return query_cls_vectors, comment_cls_vectors, query_pool_vectors, comment_pool_vectors


def vectorize_docket(docket, vocab):
    df_queries = pd.read_csv(OUTPUT_TXT+"{}_queries.csv".format(docket))
    df_comments = pd.read_csv(OUTPUT_TXT+"{}_comments.csv".format(docket))
    df_queries = df_queries.dropna(subset=['clean_text'])
    df_comments = df_comments.dropna(subset=['clean_text'])
    qidxs, _  = vectorizer.docs2idx(df_queries["clean_text"], vocab)
    cidxs, _  = vectorizer.docs2idx(df_comments["clean_text"], vocab)
    return qidxs, cidxs



def vectorize_TFIDF(target_dockets):
    with open(VOCABULARY_PATH,"rb") as f:
        vocab = pickle.load(f)
    with open(OUTPUT_PKL+"/IDF.pkl","rb") as f:
        idfvec = pickle.load(f)
    print("[building TF-IDF vectors]")
    for docket in target_dockets:
        print("[vectorizing docket: {}]".format(docket))
        # Q,C = TFIDF_vectors(docket, vocab, idfvec)
        queries, comments = vectorize_docket(docket, vocab)
        Q = features.BOW_freq(queries, len(vocab), sparse=True)
        C = features.BOW_freq(comments, len(vocab), sparse=True)
        #sparse matrices    
        sp.sparse.save_npz(OUTPUT_VECTORS+"{}_queries_tf-idf".format(docket), Q.tocsc())
        sp.sparse.save_npz(OUTPUT_VECTORS+"{}_comments_tf-idf".format(docket), C.tocsc())
    print("[done]")


def vectorize_BERT(target_dockets):
    #BERT
    BERT_MODEL = 'bert-base-uncased'
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    # Load pre-trained model (weights)
    model = BertModel.from_pretrained(BERT_MODEL, output_hidden_states=False)

    print("[building BERT vectors]")
    for docket in target_dockets:
        print("> docket: {}".format(docket))
        Q_cls, C_cls, Q_pool, C_pool = BERT_vectors(docket, tokenizer, model)            
        with open(OUTPUT_VECTORS+"{}_queries_bert_cls.np".format(docket),"wb") as f:
            np.save(f, Q_cls)
        with open(OUTPUT_VECTORS+"{}_comments_bert_cls.np".format(docket),"wb") as f:
            np.save(f, C_cls)
        with open(OUTPUT_VECTORS+"{}_queries_bert_pool.np".format(docket),"wb") as f:
            np.save(f, Q_pool)
        with open(OUTPUT_VECTORS+"{}_comments_bert_pool.np".format(docket),"wb") as f:
            np.save(f,  C_pool)
    print("[done]")

#word2vec
def vectorize_BOE(target_dockets):
    with open(VOCABULARY_PATH,"rb") as f:
        vocab = pickle.load(f)
    agg="sum"
    E, _ = embeddings.read_embeddings(WORD2VEC, vocab)
    E_tuned, _ = embeddings.read_embeddings(WORD2VEC_TUNED, vocab)

    print("[building BOE vectors]")
    for docket in target_dockets:
        print("[vectorizing docket: {}]".format(docket))
        queries, comments = vectorize_docket(docket, vocab)
        Q = features.BOE(queries, E, agg)
        C = features.BOE(comments, E, agg)
        Qt = features.BOE(queries, E_tuned, agg)
        Ct = features.BOE(comments, E_tuned, agg)
        with open(OUTPUT_VECTORS+"{}_queries_boe.np".format(docket),"wb") as f:
            np.save(f, Q)
        with open(OUTPUT_VECTORS+"{}_comments_boe.np".format(docket),"wb") as f:
            np.save(f, C)
        with open(OUTPUT_VECTORS+"{}_queries_boe_tuned.np".format(docket),"wb") as f:
            np.save(f, Qt)
        with open(OUTPUT_VECTORS+"{}_comments_boe_tuned.np".format(docket),"wb") as f:
            np.save(f, Ct)
    print("[done]")
    
def vectorize_LDA(target_dockets):
    with open(VOCABULARY_PATH,"rb") as f:
        vocab = pickle.load(f)
    #topics
    with open(OUTPUT_PKL+"/lda.pkl","rb") as f:
        topic_model, _ = pickle.load(f)
    print("[building LDA vectors]")
    for docket in target_dockets:
        print("[vectorizing docket: {}]".format(docket))
        queries, comments = vectorize_docket(docket, vocab)
        Q = features.BOW_freq(queries, len(vocab), sparse=True)
        C = features.BOW_freq(comments, len(vocab), sparse=True)
        Qt = topic_model.transform(Q.astype('int32'))
        Ct = topic_model.transform(C.astype('int32'))
        with open(OUTPUT_VECTORS+"{}_queries_lda.np".format(docket),"wb") as f:
            np.save(f, Qt)
        with open(OUTPUT_VECTORS+"{}_comments_lda.np".format(docket),"wb") as f:
            np.save(f, Ct)
    print("[done]")

print("vectorizing")
dockets = ['FDA-2014-N-0189', 'FDA-2013-N-0521', 
            'FDA-2015-N-1514', 'FDA-2012-N-1148', 
            'FDA-2011-N-0467'][::-1]

vectorize_TFIDF(dockets)
vectorize_LDA(dockets)
vectorize_BOE(dockets)
vectorize_BERT(dockets)
