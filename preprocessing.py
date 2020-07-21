import time
from spacy.lang.en import English
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# custom files
import parse_json

# Takes in a list
# Removes the word "Purpose" and punctuation
# Returns cleaned list
def clean_list(purpose_str, nlp):
    temp_doc = nlp(purpose_str)

    # tokenize excluding punctuation
    cleaned_list = [token.text.lower() for token in temp_doc
                    if not(token.is_stop or token.is_punct or token.is_space or token.text.lower() == "purpose")]
    return cleaned_list

# Uses spacy to clean all words in all relevant columns
def tokenize_columns(drug_df):
    tokenize_t0 = time.time()
    nlp = English()

    columns_to_tokenize = ["purpose", "active_ingredient", "inactive_ingredient", "warnings", "mechanism_of_action",
                           "dosage_and_administration", "indications_and_usage", "contraindications"]
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
    drug_df.to_pickle(filename + ".pkl", compression="zip")
    print("Write preprocessed:", time.time() - write_t0)

# read cleaned dataset into dataframe
def read_preprocessed_to_pkl(filename):
    read_t0 = time.time()
    drug_df = pd.read_pickle(filename + ".pkl", compression="zip")
    print("Read preprocessed:", time.time() - read_t0)
    return drug_df

# call all workflows to preprocess and write data to dataframe pkl
def preprocess_and_write_df(raw_drug_df, filename):
    pd.set_option('display.max_rows', 50)
    pd.set_option('display.max_columns', 50)

    print(raw_drug_df.head())  # verify read
    print(str(len(raw_drug_df)), "rows")

    # clean lists first
    drug_df = tokenize_columns(raw_drug_df)
    print(drug_df["purpose"])
    print(drug_df["indications_and_usage"])

    # perform purpose clustering using cleaned lists
    drug_df = cluster_purpose(drug_df)
    write_preprocessed_to_pkl(drug_df, filename)

## sklearn
# sklearn purpose cluster: top terms per cluster
def obtain_top_cluster_terms(tfidfv, km, n_cluster):
    print("Top terms per cluster:")
    top_t0 = time.time()
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
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
    purpose_list = drug_df["purpose"].tolist()
    tfidfv = TfidfVectorizer()
    X = tfidfv.fit_transform(purpose_list)
    print("TF-IDF: ", str(time.time() - tfidfv_t0))

    # KMeans
    km_t0 = time.time()
    km = KMeans(n_clusters=50)
    km.fit(X)
    print("Fit KMeans: ", str(time.time() - km_t0))

    print("Num KMeans labels: ", str(len(km.labels_)))

    cluster_list = obtain_top_cluster_terms(tfidfv, km, 50)

    cluster_groups_to_df(drug_df, km, cluster_list)

    print(drug_df["purpose_cluster"])

    return drug_df
