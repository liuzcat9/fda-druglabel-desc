import time
import os
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import spacy

# custom files
import main

# Takes in a list
# Removes the word "purpose" and "use" and punctuation
# Lemmatizes tokens
# Returns cleaned list
def clean_list(purpose_str, nlp):
    temp_doc = nlp(purpose_str.lower()) # set lowercase to not confuse lemmatization
    # blacklist all field names
    additional = ["purpose", "use", "active", "inactive", "ingredient", "warning", "treatment", "indication"]

    # tokenize excluding punctuation
    cleaned_list = [token.lemma_ for token in temp_doc
                    if not(token.is_stop or token.is_punct or token.is_space
                           or token.lemma_ in additional)]
    return cleaned_list

# Uses spacy to clean all words in all relevant columns
def tokenize_columns(drug_df, columns_to_tokenize):
    tokenize_t0 = time.time()
    nlp = spacy.load("en_core_web_sm", disable=['parser','ner','textcat'])

    for column in columns_to_tokenize:
        new_col = []
        old_col = drug_df[column].tolist()

        for entry in old_col:
            new_col.append(" ".join(clean_list(entry, nlp)) if entry else None) # keep null values null

        drug_df[column] = new_col

        print("Tokenized", column)

    print("Tokenization:", str(time.time() - tokenize_t0))

    return drug_df

# write cleaned dataset
def write_preprocessed_to_pkl(drug_df, filename):
    write_t0 = time.time()
    drug_df.to_pickle("pkl/" + filename + ".pkl", compression="zip")
    print("Write preprocessed:", time.time() - write_t0)

# read cleaned dataset into dataframe
def read_preprocessed_to_pkl(filename):
    read_t0 = time.time()
    drug_df = pd.read_pickle("pkl/" + filename + ".pkl", compression="zip")
    print("Read preprocessed:", time.time() - read_t0)
    return drug_df

# call all workflows to preprocess and write data to dataframe pkl
def preprocess_and_write_df(raw_drug_df, filename):
    pd.set_option('display.max_rows', 50)
    pd.set_option('display.max_columns', 50)

    print(raw_drug_df.head())  # verify read
    print(str(len(raw_drug_df)), "rows")

    # drop invalid user-related fields
    drug_df = raw_drug_df.copy()
    drug_df = drug_df.dropna(
        subset=["id", "brand_name", "route", "product_type"])  # exclude all rows with columns of null
    print("Truncated df: ", str(len(drug_df)))

    # clean lists first
    columns_to_tokenize = ["purpose", "active_ingredient", "inactive_ingredient", "warnings", "mechanism_of_action",
                           "dosage_and_administration", "indications_and_usage", "contraindications"]
    drug_df = tokenize_columns(drug_df, columns_to_tokenize)
    print(drug_df["purpose"])
    print(drug_df["indications_and_usage"])

    # perform purpose clustering using cleaned lists
    # use two fields for purpose: purpose & indications and usage
    drug_df["combined_purpose"] = drug_df["purpose"].fillna("") + " " + drug_df["indications_and_usage"].fillna("")
    drug_df = cluster_purpose(drug_df)

    # alphabetize brand names for future display
    drug_df = drug_df.sort_values(by=["brand_name"])

    write_preprocessed_to_pkl(drug_df, filename)

## sklearn
# sklearn purpose cluster: top terms per cluster
def obtain_top_cluster_terms(tfidfv, lsa, km, n_cluster):
    print("Top terms per cluster:")
    top_t0 = time.time()
    original_centroids = lsa.inverse_transform(km.cluster_centers_)
    order_centroids = original_centroids.argsort()[:, ::-1]

    terms = tfidfv.get_feature_names()
    cluster_list = []
    for i in range(n_cluster):
        top_words = [terms[ind] for ind in order_centroids[i, :5]]
        print("Cluster {}: {}".format(i, " ".join(top_words)))
        cluster_list.append(" ".join(top_words))
    print("List top terms per cluster: ", str(time.time() - top_t0))

    return cluster_list

# sklearn purpose cluster: all names per cluster
def cluster_groups_to_df(drug_df, km, cluster_list):
    cluster_df_t0 = time.time()
    cluster_col = []
    for i, label in enumerate(km.labels_):
        cluster_col.append(cluster_list[label])

    drug_df["purpose_cluster"] = cluster_col
    print(len(drug_df), len(cluster_col))
    print("Add cluster column to drug_df: " , str(time.time() - cluster_df_t0))

# purpose clustering
def cluster_purpose(drug_df):
    # 5. See what TF-IDF can do with respect to clustering purposes
    # TF-IDF
    tfidfv_t0 = time.time()
    purpose_list = drug_df["combined_purpose"].tolist()
    tfidfv = TfidfVectorizer()
    X = tfidfv.fit_transform(purpose_list)
    print("TF-IDF: ", str(time.time() - tfidfv_t0))

    lsa, transformedX = main.perform_LSA(X)

    # KMeans
    km_t0 = time.time()
    km = KMeans(n_clusters=50)
    km.fit(transformedX)
    print("Fit KMeans: ", str(time.time() - km_t0))

    print("Num KMeans labels: ", str(len(km.labels_)))

    cluster_list = obtain_top_cluster_terms(tfidfv, lsa, km, 50)

    cluster_groups_to_df(drug_df, km, cluster_list)

    print(drug_df["purpose_cluster"])

    return drug_df

# return a list of all purpose clusters
def find_unique_purposes(drug_df):
    return list(drug_df.purpose_cluster.unique())

# function to write individual (smaller purpose chunks of dataframe to disk)
def write_purpose_clusters_to_df(drug_df):
    purposes = find_unique_purposes(drug_df)
    for purpose in purposes:
        purpose_key = "_".join(purpose.split())
        if not os.path.isfile("pkl/purpose/" + purpose_key + ".pkl"):
            purpose_df = purpose_df.loc[purpose_df["purpose_cluster"].str.contains(purpose)]
            print(purpose_df.head())
            purpose_df.to_pickle("pkl/purpose/" + purpose_key + ".pkl")