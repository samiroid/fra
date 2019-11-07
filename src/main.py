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
import torch
from transformers import BertTokenizer, BertModel


#add ASMAT toolkit
ASMAT_PATH="/Users/samir/Dev/projects/ASMAT2"
sys.path.append(ASMAT_PATH)
sys.path.append("..")
from ASMAT import vectorizer, embeddings, features
from ASMAT.toolkit import gensimer


# In[ ]:


#paths
HOME="/Users/samir/Dev/projects/comment_feedback_aligner/fra/"
FEEDBACK_REQUESTS_PATH = HOME+"DATA/raw/regulations_proposed_rules_feedback.csv"
QUERIES_PATH = HOME+"DATA/raw/regulations_proposed_rules_feedback_queries.tsv"

CIGARRETES_COMMENTS_PATH=HOME+"DATA/raw/all_data_cigarettes"
TOBACCO_COMMENTS_PATH=HOME+"DATA/raw/all_data_tobacco"



WORD2VEC_INPUT=HOME+"DATA/embeddings/skip_50.txt"
WORD2VEC_INPUT=HOME+"DATA/embeddings/glove.42B.300d.txt"


OUTPUT_TXT = HOME+"DATA/processed/txt/"
OUTPUT_PKL = HOME+"DATA/processed/pkl/"
OUTPUT_VECTORS = HOME+"DATA/processed/vectors/"
OUTPUT_RANKINGS = HOME+"DATA/processed/rankings/"

WORD2VEC=OUTPUT_PKL+"/word2vec.txt"
WORD2VEC_TUNED=OUTPUT_PKL+"/word2vec_tuned.txt"
COMMENTS_PATH=OUTPUT_TXT+"/all_comments.txt"
CORPUS=OUTPUT_TXT+"all_text.txt"
FILTERED_COMMENTS=OUTPUT_TXT+"/filtered_comments.txt"
QUERIES=OUTPUT_TXT+"/queries.txt"
VOCABULARY_PATH=OUTPUT_PKL+"vocabulary.pkl"
IDF_ESTIMATE_PATH=OUTPUT_PKL+"IDF.pkl"


if not os.path.exists(OUTPUT_TXT):
    os.makedirs(OUTPUT_TXT)
if not os.path.exists(OUTPUT_PKL):
    os.makedirs(OUTPUT_PKL)
if not os.path.exists(OUTPUT_VECTORS):
    os.makedirs(OUTPUT_VECTORS)
if not os.path.exists(OUTPUT_RANKINGS):
    os.makedirs(OUTPUT_RANKINGS)


# ## Read RGOV data

# In[ ]:


MIN_Q_LEN = 100

stop_wordz = set(stopwords.words('english'))
remove_punctuation = str.maketrans('', '', string.punctuation+"”“")
QUOTES_REGEX=r'[\"“](.+?)[\"”]'


def read_raw_rgov(path):
    df = pd.DataFrame([])
    for f in os.listdir(path):
        fname=os.path.join(path,f)
        with open(fname,"r") as jf:
            try:
                raw_data = json.load(jf)
            except UnicodeDecodeError:
                print("\n\nCould not read file {}\n\n".format(fname))
                continue
            try:
                df = df.append(pd.DataFrame(raw_data["documents"]))
            except KeyError:
                print("\n\nCould not find any documents in file {}\n\n".format(fname))
                continue
            print(fname)

    return df

def count_dockets(df):
    aggz=df.groupby("docketId").size()
    target_dockets = ['FDA-2014-N-0189', 'FDA-2017-N-6565', 'FDA-2017-N-6189', 'FDA-2017-N-6107', 'FDA-2013-N-0521', 'FDA-2015-N-1514', 'FDA-2012-N-1148', 'FDA-2011-N-0467', 'FDA-2017-N-6529', 'FDA-2011-N-0493', 'FDA-2017-N-5095']
    for docket in target_dockets:
        try:
            print("{} {}".format(docket,aggz[docket]))
        except KeyError:
            print("{} {}".format(docket,"NULL"))

            
def mask_quotes(text):      
    matches=re.sub(QUOTES_REGEX,"00quote00",text)
    return matches

def preprocess(d):
    d = d.lower()
    d = d.replace("\n", "\t")
    #remove stop words and punctuation
    d = " ".join([w.translate(remove_punctuation) for w in d.split() if w not in stop_wordz])
    return d
     
def process_comments(df):
    #filter for comments
    comments = []
    doc_counters=defaultdict(int)
    for _, comment in df.iterrows():    
        #segment comment into sentences            
        txt = comment["commentText"]     
        doc_counters[comment["docketId"]]+=1
        #document id: docketID#C(id)
        docid = comment["docketId"]+"#C"+str(doc_counters[comment["docketId"]])
        try:
            sentences = sent_tokenize(txt)
        except TypeError:
            print("failed tokenizer")
            continue
        c = [[ comment["docketId"], docid,             docid+"#S"+str(i), s, preprocess(mask_quotes(s)), len(s.split())]             for i,s in enumerate(sentences)]
        comments += c
    df = pd.DataFrame(comments,columns=["docketId", "commentId", "sentenceId","text", "clean_text", "len"])
    return df

def process_queries(path):
    queries = defaultdict(list)
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')    
        next(csv_reader)
        for row in csv_reader:
            docket_id, query = row
            queries[docket_id].append(query)
    ordered_queries = []
    for did, qs in queries.items():
        for i, q in enumerate(qs):
            x = [did,did+"#Q"+str(i),q]
            ordered_queries.append(x)
    df = pd.DataFrame(ordered_queries, columns=["docketId","queryId","text"])
    #preprocess text
    df["clean_text"] = df["text"].map(preprocess)
    return df

def extract_comments():
    #read documents
    df_tob = read_raw_rgov(TOBACCO_COMMENTS_PATH)
    df_cig= read_raw_rgov(CIGARRETES_COMMENTS_PATH)
    df_all = df_tob.append(df_cig)
    #filter comments
    count_dockets(df_all)
    df_all = df_all[df_all["documentType"] == "Public Submission"] 
    print("just comments")
    count_dockets(df_all)
    #remove empty comments
    df_all = df_all.dropna(subset=['commentText'])
    print("no empty comments")
    count_dockets(df_all)
    #remove entries with attachments
    print("no attachments")
    df_all = df_all[df_all["attachmentCount"] == 0] 
    count_dockets(df_all)
    #process comments
    df_comments = process_comments(df_all)
    df_comments.to_csv(COMMENTS_PATH, header=True, index=False)
    return df_comments

# extract_comments()


# ## Generate Background Corpus

# In[ ]:


def generate_background_corpus():
    #read comments
    df_comments = pd.read_csv(COMMENTS_PATH)
    #read feedback requests
    df = pd.read_csv(FEEDBACK_REQUESTS_PATH)

    #extract all the text 
    requests_text = df["docket_title"].values.tolist() + df["summary"].values.tolist() + df["feedback_asked"].values.tolist()
    #preprocess text
    clean_requests_text = [preprocess(str(w)) for w in requests_text]
    clean_comments_text = [str(d) for d in df_comments["clean_text"].values.tolist()]
    all_text = clean_requests_text + clean_comments_text
    #shuffle text 
    random.shuffle(all_text)
    with open(CORPUS,"w") as f:
        f.write("\n".join(all_text))

# generate_background_corpus()


# ##  Prepare Queries

# In[ ]:


def prepare_queries():
    #read queries
    df_queries =  process_queries(QUERIES_PATH)
    #extract target dockets
    dockets = df_queries.docketId.unique().tolist()
    target_dockets = []
    #read comments
    df_comments = pd.read_csv(COMMENTS_PATH)
    for docket in dockets:
        queries = df_queries[df_queries["docketId"] == docket]
        comments = df_comments[df_comments["docketId"] == docket]
        if len(comments)>0:
            print(docket + " " + str(len(comments)))
            target_dockets.append(docket)
            #save queries and commments
            queries.to_csv(OUTPUT_TXT+"{}_queries.csv".format(docket), header=True, index=False)
            comments.to_csv(OUTPUT_TXT+"{}_comments.csv".format(docket), header=True, index=False)
        print("target dockets: {}".format(repr(target_dockets)))
#target dockets = ['FDA-2014-N-0189', 'FDA-2017-N-6565', 'FDA-2017-N-6189', 'FDA-2017-N-6107', 'FDA-2013-N-0521', 'FDA-2015-N-1514', 'FDA-2012-N-1148', 'FDA-2011-N-0467', 'FDA-2017-N-6529', 'FDA-2011-N-0493', 'FDA-2017-N-5095']
# prepare_queries()


# ## Generate Embeddings

# In[ ]:


#inverse document frequency
def getIDF(N, t):
    return log(float(N)/float(t))


# In[ ]:


MIN_WORD_FREQ=5
def compute_vocabulary():
    with open(CORPUS,"r") as f:
        all_text =   f.readlines()
    #get vocabulary
    vocab = vectorizer.build_vocabulary(all_text, min_freq=MIN_WORD_FREQ)
    print("vocabulary size: {}".format(len(vocab)))
    #save vocabulary
    with open(VOCABULARY_PATH,"wb") as f:
        pickle.dump(vocab,f)
# compute_vocabulary()


# In[ ]:


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

# compute_IDF()


# In[ ]:


VECTOR_DIM=300
NEGATIVE_SAMPLES=10
EPOCHS=5
#extract word embeddings
def get_word_embeddings():
    with open(VOCABULARY_PATH,"rb") as f:
        vocab = pickle.load(f)
    embeddings.extract_embeddings(WORD2VEC_INPUT, WORD2VEC, vocab)

def update_word_embeddings():
    #update word embeddings 
    train_seq = gensimer.Word2VecReader([CORPUS])
    w2v = gensimer.get_skipgram(dim=VECTOR_DIM,negative_samples=NEGATIVE_SAMPLES, min_freq=MIN_WORD_FREQ)
    w2v_trained = gensimer.train_skipgram(w2v, train_seq, epochs=EPOCHS,
                                          path_out=WORD2VEC_TUNED,
                                          pretrained_weights_path=WORD2VEC)

# get_word_embeddings()
# update_word_embeddings()


# In[ ]:


#train topic model
def train_topic_model():
    with open(VOCABULARY_PATH,"rb") as f:
        vocab = pickle.load(f)
    with open(CORPUS,"r") as f:
        all_text =   f.readlines()
    all_idxs, _ = vectorizer.docs2idx(all_text, vocab)
    n_topics=100
    n_iter=100
    X = features.BOW_freq(all_idxs, len(vocab), sparse=True)
    X = X.astype('int32')
    topic_model = lda.LDA(n_topics=n_topics, n_iter=n_iter)
    topic_model.fit(X)
    #save model
    with open(OUTPUT_PKL+"/lda.pkl","wb") as f:
        pickle.dump([topic_model, vocab], f)

# train_topic_model()


# ##  Vectorize

# In[ ]:


BERT_MAX_INPUT=510
BERT_BATCH_SIZE = 100

def TFIDF_vectors(docket, vocab, idfs):
    queries, comments = vectorize_docket(docket, vocab)
    Q = features.BOW_freq(queries, len(vocab), sparse=False)
    C = features.BOW_freq(comments, len(vocab), sparse=False)
    Q*=idfs
    C*=idfs
    return Q,C

def BOE_vectors(docket, vocab, E, agg):
    queries, comments = vectorize_docket(docket, vocab)
    Q = features.BOE(queries, E, agg)
    C = features.BOE(comments, E, agg)
    return Q,C

def LDA_vectors(docket, vocab, topic_model):
    queries, comments = vectorize_docket(docket, vocab)
    Q = features.BOW_freq(queries, len(vocab), sparse=True)
    C = features.BOW_freq(comments, len(vocab), sparse=True)
    Q = Q.astype('int32')
    C = C.astype('int32')
    Qt = topic_model.transform(Q)
    Ct = topic_model.transform(C)
    return Qt, Ct

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


# In[ ]:


target_dockets = ['FDA-2014-N-0189', 'FDA-2013-N-0521', 
                  'FDA-2015-N-1514', 'FDA-2012-N-1148', 
                  'FDA-2011-N-0467']


# ## BERT

# In[ ]:


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
        with open(OUTPUT_VECTORS+"{}_bert_cls.pkl".format(docket),"wb") as f:
            pickle.dump([docket, Q_cls, C_cls], f)
        with open(OUTPUT_VECTORS+"{}_bert_pool.pkl".format(docket),"wb") as f:
            pickle.dump([docket, Q_pool, C_pool], f)
    print("[done]")

# vectorize_BERT(target_dockets)


# ## TF-IDF

# In[ ]:


def vectorize_TFIDF(target_dockets):
    with open(VOCABULARY_PATH,"rb") as f:
        vocab = pickle.load(f)
    with open(OUTPUT_PKL+"/IDF.pkl","rb") as f:
        idfvec = pickle.load(f)
    
    print("[building TF-IDF vectors]")
    for docket in target_dockets:
        print("[vectorizing docket: {}]".format(docket))
        Q,C = TFIDF_vectors(docket, vocab, idfvec)
        # vectors.append()    
        with open(OUTPUT_VECTORS+"{}_tf-idf.pkl".format(docket),"wb") as f:
            pickle.dump([docket, Q, C], f)
    print("[done]")
    
# vectorize_TFIDF(target_dockets)


# ## BOE

# In[ ]:


#word2vec
def vectorize_BOE(target_dockets):
    with open(VOCABULARY_PATH,"rb") as f:
        vocab = pickle.load(f)
    agg="sum"
    E, _ = embeddings.read_embeddings(WORD2VEC, vocab)
    
    print("[building BOE vectors]")
    for docket in target_dockets:
        print("[vectorizing docket: {}]".format(docket))
        Q,C = BOE_vectors(docket, vocab, E, agg)    
        with open(OUTPUT_VECTORS+"{}_boe.pkl".format(docket),"wb") as f:
            pickle.dump([docket, Q, C], f)
    print("[done]")
    
# vectorize_BOE(target_dockets)


# In[ ]:


def vectorize_BOE_tuned(target_dockets):
    agg="sum"
    with open(VOCABULARY_PATH,"rb") as f:
        vocab = pickle.load(f)
    E_tuned, _ = embeddings.read_embeddings(WORD2VEC_TUNED, vocab)
    
    print("[building fine-tuned BOE vectors]")
    for docket in target_dockets:
        print("[vectorizing docket: {}]".format(docket))
        Q,C = BOE_vectors(docket, vocab, E_tuned, agg)
        with open(OUTPUT_VECTORS+"{}_boe-tuned.pkl".format(docket),"wb") as f:
            pickle.dump([docket, Q, C], f, protocol=-1)
    print("[done]")

# vectorize_BOE_tuned(target_dockets)


# ## LDA

# In[ ]:


def vectorize_LDA(target_dockets):
    #topics
    with open(OUTPUT_PKL+"/lda.pkl","rb") as f:
        topic_model, _ = pickle.load(f)
    vectors = []
    print("[building LDA vectors]")
    for docket in target_dockets:
        print("[vectorizing docket: {}]".format(docket))
        Q,C = LDA_vectors(docket, vocab, topic_model)
        vectors.append([docket, Q, C])    
    with open(OUTPUT_VECTORS+"lda.pkl","wb") as f:
        pickle.dump(vectors, f)
    print("[done]")
# vectorize_LDA(target_dockets)


# ##  Rank

# In[ ]:


def similarity_rank(q, D):
    simz = np.dot(D,q)/(np.linalg.norm(D)*np.linalg.norm(q))
    rank = np.argsort(simz)[::-1]
    ranked_simz = simz[rank]
    return rank, ranked_simz

def similarity_ranks(Q, D, queries, comments, method, top_k):
    results = []
    for i in range(Q.shape[0]):
        qid = queries.iloc[i]["queryId"]
        r,s = similarity_rank(Q[i], D)
        top_ranked = r[:top_k]
        top_similarities = s[:top_k]
        sentence_ids = comments["sentenceId"].iloc[top_ranked].values.tolist()
        sims = [str(x) for x in top_similarities.round(5).tolist()]
        res = [[method,qid,r,s] for r,s in zip(sentence_ids, sims)]
        results+=res
    return results

def rank_dockets(dockets, methods, top_k=10):
    docket_data = {}
    #load docket data
    all_results = []
    print("[reading dockets data]")
    for docket in dockets:
        df_queries = pd.read_csv(OUTPUT_TXT+"{}_queries.csv".format(docket))
        df_comments = pd.read_csv(OUTPUT_TXT+"{}_comments.csv".format(docket))
        df_queries = df_queries.dropna(subset=['clean_text'])
        df_comments = df_comments.dropna(subset=['clean_text'])
        docket_data[docket] = {"queries": df_queries, "comments":df_comments, "ranks":[]}
    #ranks for each method
    for m in methods:
        print("[ranking method: {}]".format(m))
        with open(OUTPUT_VECTORS+m+".pkl","rb") as f:
            vectors = pickle.load(f)
        for docket, Q, C in vectors:
            results = similarity_ranks(Q, C, docket_data[docket]["queries"], 
                                       docket_data[docket]["comments"], m, top_k)
#             set_trace()
            docket_data[docket]["ranks"]+=results
    #write rankings per docket
    for docket in dockets:
        if len(docket_data[docket]["ranks"]) > 0:                      
            with open(OUTPUT_RANKINGS+"{}_rank.csv".format(docket),"w") as fo:
                top_sentences = []
                for r in docket_data[docket]["ranks"]:
                    top_sentences+=[r[2]]
                    fo.write("\t".join(r)+"\n")
            df_queries = docket_data[docket]["queries"][["queryId","text"]]
            df_queries.to_csv(OUTPUT_RANKINGS+"{}_queries.csv".format(docket), header=True, index=False, sep="\t")
            df_comments = docket_data[docket]["comments"][["sentenceId","text"]]    
#             from pdb import set_trace; set_trace()
            df_comments = df_comments[df_comments["sentenceId"].isin(set(top_sentences))]            
            df_comments["text"] = [w.replace("\n"," ").replace("\t"," ") for w in df_comments["text"]] 
            df_comments.to_csv(OUTPUT_RANKINGS+"{}_comments.csv".format(docket), header=True, index=False, sep="\t")
    return docket_data


# In[ ]:


#get data
print("Processing Data")
extract_comments()
generate_background_corpus()
prepare_queries()
#compute embeddings
print("computing embeddings")
compute_vocabulary()
compute_IDF()
get_word_embeddings()
update_word_embeddings()
train_topic_model()
#vectorize
print("vectorizing")
target_dockets = ['FDA-2014-N-0189', 'FDA-2013-N-0521', 
                  'FDA-2015-N-1514', 'FDA-2012-N-1148', 
                  'FDA-2011-N-0467'][::-1]
vectorize_BERT(target_dockets)
vectorize_LDA(target_dockets)
vectorize_BOE(target_dockets)
vectorize_TFIDF(target_dockets)
vectorize_BOE_tuned(target_dockets)
#rank

