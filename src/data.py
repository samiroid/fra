from collections import Counter, defaultdict
import csv
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import os
import json
import pandas as pd
from pdb import set_trace
import random
import re
import string

#paths
HOME="/Users/samir/Dev/projects/feedback_request_aligner/fra/"
FEEDBACK_REQUESTS_PATH = HOME+"DATA/raw/regulations_proposed_rules_feedback.csv"
QUERIES_PATH = HOME+"DATA/raw/regulations_proposed_rules_feedback_queries.tsv"
CIGARRETES_COMMENTS_PATH=HOME+"DATA/raw/all_data_cigarettes"
TOBACCO_COMMENTS_PATH=HOME+"DATA/raw/all_data_tobacco"
OUTPUT_TXT = HOME+"DATA/processed/txt/"
COMMENTS_PATH=OUTPUT_TXT+"/all_comments.txt"
CORPUS=OUTPUT_TXT+"all_text.txt"
QUERIES=OUTPUT_TXT+"/queries.txt"

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
#             print(fname)

    return df
            
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
        c = [[ comment["docketId"], docid, docid+"#S"+str(i), s, 
                        preprocess(mask_quotes(s)), len(s.split())] \
                            for i,s in enumerate(sentences)]
        comments += c
    df = pd.DataFrame(comments,columns=["docketId", "commentId", 
                                        "sentenceId","text", "clean_text", "len"])
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
    df_all = df_all[df_all["documentType"] == "Public Submission"] 
    #remove empty comments
    df_all = df_all.dropna(subset=['commentText'])
    #remove entries with attachments
    print("no attachments")
    df_all = df_all[df_all["attachmentCount"] == 0] 
    #process comments
    df_comments = process_comments(df_all)
    df_comments.to_csv(COMMENTS_PATH, header=True, index=False)
    return df_comments

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


def prepare_queries():
    #read queries
    df_queries =  process_queries(QUERIES_PATH)
    #extract target dockets
    dockets = df_queries.docketId.unique().tolist()
    target_dockets = []
    #read comments
    df_comments = pd.read_csv(COMMENTS_PATH)
    print("all comments: {}".format(str(len(df_comments))))    
    df_comments = df_comments.drop_duplicates(subset=["docketId","clean_text"])
    print("no duplicates: {}".format(str(len(df_comments))))
    df_comments = df_comments[df_comments["len"] > 5]
    print("no shorties: {}".format(str(len(df_comments))))
    for docket in dockets:
        queries = df_queries[df_queries["docketId"] == docket]
        comments = df_comments[df_comments["docketId"] == docket]
        if len(comments)>0:
            print(docket + " " + str(len(comments)))
            target_dockets.append(docket)
            #save queries and commments
            queries.to_csv(OUTPUT_TXT+"{}_queries.csv".format(docket), header=True, index=False)
            comments.to_csv(OUTPUT_TXT+"{}_comments.csv".format(docket), header=True, index=False)

if not os.path.exists(OUTPUT_TXT):
    os.makedirs(OUTPUT_TXT)
print("Processing Data")
extract_comments()
generate_background_corpus()
prepare_queries()