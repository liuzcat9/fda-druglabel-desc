import time
from os import walk

import json, ijson
import string
import math
from collections import OrderedDict
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib_venn
import itertools

import numpy as np
import networkx as nx
from bokeh import events
from bokeh.io import output_file, show
from bokeh.models import (BoxZoomTool, Circle, HoverTool, TapTool,
                          MultiLine, Plot, Range1d, ResetTool,
                          NodesAndLinkedEdges, EdgesAndLinkedNodes, ColumnDataSource,
                          CustomJS, Div)
from bokeh.plotting import figure, from_networkx, output_file, save
from bokeh.layouts import column, row
from bokeh.embed import components, file_html
from bokeh.resources import Resources

from jinja2 import Environment, FileSystemLoader

from wordcloud import WordCloud

import spacy

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances, linear_kernel
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.cluster import DBSCAN, SpectralClustering, Birch, KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV

# custom files
import parse_json, preprocessing

# 1. Finds top most searched purposes
def rank_purpose(drug_df, n_rank): # all entries in read file have drugs
    rank_t0 = time.time()

    master_list = []

    # in light of processing memory, perform cleaning separately
    # compile all purpose field words
    for purpose in drug_df["purpose"].tolist():
        purpose_list = purpose.split()
        master_list.extend(purpose_list)

    purposes = dict()
    for word in master_list:
        # add keywords into purpose dict
        if (word not in purposes):
            purposes[word] = 0
        else:
            purposes[word] += 1

    # sort extracted purposes by frequency
    purposes_df = pd.DataFrame(purposes.items(), columns=["Keyword", "Frequency"])
    purposes_df = purposes_df.sort_values(by="Frequency", ascending=False)
    purposes_trunc = purposes_df.head(n_rank)  # shorten to top n

    print("Rank: ", str(time.time() - rank_t0))

    plot_rank_purpose(purposes_trunc, n_rank)

# plot 1. most searched purposes
def plot_rank_purpose(purposes_trunc, n_rank):
    # plot truncated top purposes
    font = {'weight': 'bold',
            'size': 6}
    matplotlib.rc('font', **font)
    top_purpose = purposes_trunc.plot(kind='bar', x="Keyword", y="Frequency")
    plt.title("Top " + str(n_rank) + " Purpose Keywords for FDA Drugs", fontsize=10)
    plt.xlabel("Keyword", fontsize=7)
    plt.ylabel("Frequency", fontsize=7)
    top_purpose.get_legend().remove()
    plt.show()

# 2. Venn Diagram similarity between most similar drugs based on purpose keyword/indication matches
def draw_venn(drug_df, original, field):
    drug_venn_t0 = time.time()
    # find top purpose keyword
    original_purpose_list = drug_df.loc[drug_df["id"] == original]["purpose"].values[0].split()

    # find similar products by purpose
    similar_products = find_similar_products_from_product_purpose(drug_df, original_purpose_list, original, "indications_and_usage")

    # within similar products, find ratio of matching indications for same-purpose medications to other product
    top_indic_df = find_matching_field_for_product(drug_df, similar_products, original, field)

    original_indic_set = set(drug_df.loc[drug_df["id"] == original, field].values[0].split())

    print(top_indic_df)

    print("Drug Venn: ", str(time.time() - drug_venn_t0))

    # plot top 3 by matching indications to original product
    plt.figure()
    original_name = drug_df.loc[drug_df["id"] == original]["brand_name"].values[0]
    venn = matplotlib_venn.venn3([original_indic_set,
                                  set(drug_df.loc[drug_df["id"] == top_indic_df.iloc[0][
                                      'ProductID'], field].values[0].split()),
                                  set(drug_df.loc[drug_df["id"] == top_indic_df.iloc[1][
                                      'ProductID'], field].values[0].split())],
                                 set_labels=[original_name, top_indic_df.iloc[0]['ProductName'],
                                             top_indic_df.iloc[1]['ProductName']])
    plt.title("Indication Word Matches in Common Purpose Product Labels", fontsize=10)
    plt.show()

# helper function to find relevant drug information for a common purpose
def find_df_fitting_purpose(drug_df, purpose, field):
    purpose_df = drug_df.dropna(
        subset=["id", "brand_name", "route", "product_type", field])  # exclude all rows with columns of null

    purpose_df = purpose_df.loc[purpose_df["purpose_cluster"].str.contains(purpose)]
    purpose_df = purpose_df.reset_index()

    print("Has rows: ", str(purpose_df.count(axis=0)))
    return purpose_df

# helper function to find product names of a similar purpose to a product
# returns: list of similar product names
def find_similar_products_from_product_purpose(drug_df, original_purpose_list, original, field):
    valid_df = drug_df.dropna(subset=["id", "purpose", "brand_name", field]) # exclude all rows with columns of null
    valid_df = valid_df.loc[valid_df["id"] != original] # exclude self
    print("brand_name, indications_and_usage:", len(valid_df))

    similar_products = set()
    for purpose in original_purpose_list:
        similar_products.update(valid_df.loc[valid_df["purpose"].str.contains(purpose), "id"].tolist())

    return list(similar_products)

# helper function to count and record matching words/total words in indications and usage field
# returns: dictionary of product names and counts of matches
def find_matching_field_for_product(drug_df, similar_products, original, field):
    valid_df = drug_df.dropna(subset=["id", "purpose", "brand_name", field])  # exclude all rows with relevant columns of null
    valid_df = valid_df.loc[valid_df["id"] != original]  # exclude self

    # set original indications
    original_indic_set = set(drug_df.loc[drug_df["id"] == original, field].values[0].split())
    # within similar purpose products, find the ones with top similarities in indications and usage
    top_indic = []

    for similar_product in similar_products:
        # data frame already wiped of invalid null rows
        similar_indic_set = set(drug_df.loc[drug_df["id"] == similar_product, field].values[0].split())

        # create columns of ID, product, and matching words/total words
        top_indic.append([similar_product,
                         valid_df.loc[valid_df["id"] == similar_product, "brand_name"].values[0],
                          len(original_indic_set.intersection(similar_indic_set)) * 1.0 /
                          len(similar_indic_set)])

    top_indic_df = pd.DataFrame(top_indic, columns=["ProductID", "ProductName", "Match_Words"])
    top_indic_df = top_indic_df.sort_values(by="Match_Words", ascending=False)

    return top_indic_df

# Generate tdidf method for cluster similarity
def generate_similarity_matching_field_of_purpose(purpose_df, field, topics=True):
    field_list = purpose_df[field].tolist()
    print(field_list[0:10])

    # TD-IDF matrix will be used for cluster choice and for matrix similarity
    fieldX = create_tfidf(field_list)
    print(fieldX.shape)
    print(purpose_df.head())

    if topics:
        # preprocess for LDA
        count_data, words = fit_count_vectorizer(field_list)

        best_lda, lda_prob = fit_LDA(count_data)
        print(lda_prob)

        topics_dict = get_topics_word_dict(best_lda, words)

        # create documents (in order) clustered into topics
        labels = create_doc_topic_cluster(best_lda, lda_prob)
        print(labels)
    else:
        # run density model
        density_cluster = run_density_cluster(fieldX)
        labels = density_cluster.labels_

    # build ordered dictionary of cluster:corresponding drug numbers
    cluster_ref = build_ordered_cluster_dict(labels)

    if topics:
        # some documents are not assigned to all topics, so reset index
        cluster_ref, topics_dict = reset_cluster_index(cluster_ref, topics_dict)

    # calculate adjacency matrix of supernodes by averaging cosine similarity of edges in cluster
    full_mat = compute_tfidf_cos(fieldX) # each individual pairwise cosine similarity
    adj_mat = calculate_super_adj_mat(cluster_ref, full_mat)

    if topics:
        # create mapping dictionaries for cluster-based graph
        num_to_cluster, attr_dict = generate_super_attr_mappings(purpose_df, cluster_ref, topics_dict)

        # create full comprehensive html descriptions based on topics
        num_to_html = generate_cluster_num_to_html(purpose_df, cluster_ref, field)

    else:
        # create mapping dictionaries for cluster-based graph
        num_to_cluster, attr_dict = generate_super_attr_mappings(purpose_df, cluster_ref)

    # adj_mat = weight_and_process_adj_mat(adj_mat)

    # assign attributes
    # num_to_name, attr_dict = generate_attr_mappings(purpose_df)

    # generate graph with top n edges per node weighted by similarity
    # create sparse adjacency matrix, removing edges
    max_n = 8
    sparse_mat = restrict_adjacency_matrix(adj_mat, max_n if adj_mat.shape[0] >= max_n else adj_mat.shape[0])

    print(adj_mat, adj_mat.shape)
    print(sparse_mat)

    return (adj_mat, sparse_mat, attr_dict, num_to_cluster, num_to_html)

# create TF-IDF for any list of text entries
def create_tfidf(field_list):
    # method of representation: TF-IDF
    tfidfv_t0 = time.time()
    tfidfv = TfidfVectorizer()
    fieldX = tfidfv.fit_transform(field_list)
    print("TF-IDF Field: ", str(time.time() - tfidfv_t0))

    return fieldX

# compute cosine_similarity using faster linear_kernel for normalized data
def compute_tfidf_cos(fieldX):
    # compute cosine similarity
    # cos_sim_t0 = time.time()
    # cos_sim = cosine_similarity(fieldX, fieldX)
    # print("cosine_similarity:", str(time.time() - cos_sim_t0)) # slightly more inefficient
    lin_kern_t0 = time.time()
    adj_mat = linear_kernel(fieldX, fieldX)
    print("linear_kernel:", str(time.time() - lin_kern_t0))

    return adj_mat

# perform count vectorization and return count_data, words list
def fit_count_vectorizer(field_list):
    # find top 10 words by frequency
    count_vectorizer = CountVectorizer()
    count_data = count_vectorizer.fit_transform(field_list)

    words = count_vectorizer.get_feature_names()

    return count_data, words

# optimize and create LDA model from vectorization
def fit_LDA(count_data):
    # create and fit LDA
    # lda = LatentDirichletAllocation(n_components=100)

    # optimize topics
    # cv_t0 = time.time()
    # lda_cv = GridSearchCV(lda, param_grid = {"n_components": [n for n in range(35, 105, 5)]}, n_jobs=4)
    # lda_cv.fit(count_data)
    # print(lda_cv.best_params_)
    # print("GridSearch LDA Optimization:", str(time.time() - cv_t0))

    # pick best LDA model
    best_lda_t0 = time.time()
    # best_lda = lda_cv.best_estimator_
    # select a visually appealing number of "topics"
    num_comp = int(count_data.shape[0] / 100) + 5
    best_lda = LatentDirichletAllocation(n_components=num_comp, n_jobs=4)
    lda_prob = best_lda.fit_transform(count_data)
    print("Best LDA fit:", str(time.time() - best_lda_t0))

    return best_lda, lda_prob

# obtain a dictionary of top 10 count vectorization words
def get_top_words_by_freq(count_data, words):
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts += t.toarray()[0]

    count_dict = zip(words, total_counts)
    count_dict = sorted(count_dict, key=lambda k: k[1], reverse=True)[0:10]
    print(count_dict)

    return count_dict

# print all topics and top keywords for LDA
def get_topics_word_dict(best_lda, words):
    # print topics
    for index, topic in enumerate(best_lda.components_):
        print("Topic {}".format(index))
        print(" ".join([words[i] for i in topic.argsort()[:-10 - 1: -1]]))

    # create dictionary of topics
    topics_dict = {}
    for index, topic in enumerate(best_lda.components_):
        topics_dict[index] = [words[i] for i in topic.argsort()[:-10 - 1: -1]]

    print(topics_dict)

    return topics_dict

# align document topics to each document
def create_doc_topic_cluster(best_lda, lda_prob):
    topic_t0 = time.time()
    topics = ["Topic" + str(i) for i in range(best_lda.n_components)]
    docs = ["Doc" + str(i) for i in range(lda_prob.shape[0])]

    doc_topic_df = pd.DataFrame(np.round(lda_prob, 2), columns=topics, index=docs)

    # obtain most related topic to each
    dominant_topic = np.argmax(doc_topic_df.values, axis=1)
    dominant_probs = np.amax(doc_topic_df.values, axis=1)
    doc_topic_df["dominant_topic"] = dominant_topic
    doc_topic_df["max_probability"] = dominant_probs

    print(doc_topic_df)
    print(len(doc_topic_df))
    print("Assign topics to document:", str(time.time() - topic_t0))

    return dominant_topic

# provide Birch cluster results (mini-kbatch clusters)
def run_Birch_cluster(X):
    brc = Birch(branching_factor=100, n_clusters=263, threshold=0.02, compute_labels=True)
    brc.fit(X)

    birch_clusters = brc.predict(X)
    print(birch_clusters)
    print("Total samples:", len(birch_clusters))
    print(set(birch_clusters))
    print("Birch:", len(list(set(birch_clusters))))

    return birch_clusters

# provide spectral model results with k clusters (KMeans-based, euclidean)
def run_spectral_cluster(X):
    spectral_t0 = time.time()
    spectral_cluster = SpectralClustering(n_clusters=300).fit(X)
    print(spectral_cluster.labels_)

    print("Spectral:", str(time.time() - spectral_t0))

    return spectral_cluster

# provide DBSCAN model results using n_samples x n_features X
def run_density_cluster(X):
    n_cluster = 0
    para = .05
    for dist in np.arange(.02, .7, .02):
        # cluster into supernodes based on cosine distance, density/nearest neighbors
        density_cluster = DBSCAN(eps=dist, min_samples=5, metric="cosine").fit(X)
        curr_cluster = len(set(density_cluster.labels_)) - 1
        print("Number of clusters:", str(curr_cluster))
        if curr_cluster >= n_cluster:
            n_cluster = curr_cluster
            para = dist
        print(list(density_cluster.labels_).count(-1))

    print("Best params:", str(para), "producing", str(n_cluster), "clusters")
    density_cluster = DBSCAN(eps=para, min_samples=5, metric="cosine").fit(X)

    return density_cluster


# perform TruncatedSVD (LSA for TD-IDF) to see if dimensionality can be sufficiently reduced
def perform_LSA(drug_df, field="purpose"):
    # TF-IDF
    purposeX = create_tfidf(drug_df[field].tolist())

    # LSA
    lsa = TruncatedSVD(n_components=100)
    lsa.fit(purposeX)
    print(sum(lsa.explained_variance_ratio_))

    return lsa.fit_transform(purposeX)

# perform silhouette/inertia scoring for TF-IDF > KMeans clustering
def perform_kmeans_scoring(drug_df):
    # TF-IDF
    tfidfv_t0 = time.time()
    purpose_list = drug_df["combined_purpose"].tolist()
    tfidfv = TfidfVectorizer()
    X = tfidfv.fit_transform(purpose_list)
    print("TF-IDF: ", str(time.time() - tfidfv_t0))

    # Perform inertia/silhouette scoring for new data frame of 81,501
    # collect average scores
    avg_silhouettes = {}
    avg_inertias = {}
    # KMeans
    n_clusters_range = [10, 50]
    # find ideal number of clusters
    for n_clusters in n_clusters_range:
        n_trials = 5
        sum_sil = 0
        sum_inert = 0
        # conduct multiple trials per n_clusters
        for trial in range(n_trials):
            km = KMeans(n_clusters=n_clusters)
            km.fit(X)

            # Calculate the mean silhouette coefficient for the number of clusters chosen
            sum_sil += silhouette_score(X, km.predict(X), metric='euclidean')
            sum_inert += km.inertia_

        avg_silhouettes[n_clusters] = sum_sil / n_trials
        avg_inertias[n_clusters] = sum_inert / n_trials
        print("Calculated for", str(n_clusters), "clusters:", avg_silhouettes[n_clusters])

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(*zip(*list(avg_inertias.items())))
    ax[1].plot(*zip(*list(avg_silhouettes.items())))

    ax[0].title.set_text("Inertias")
    ax[0].set(xlabel="Clusters", ylabel="Inertia")
    ax[1].title.set_text("Silhouettes")
    ax[1].set(xlabel="Clusters", ylabel="Silhouette Score")

    plt.show()

# rewrite the keys because they are not numbered completely consecutively, not assigned
def reset_cluster_index(cluster_ref, topics_dict):
    new_cluster_ref = {}
    new_topics_dict = {}

    # only based on cluster dict; cluster is the topic index
    for i, cluster in enumerate(cluster_ref):
        new_cluster_ref[i] = cluster_ref[cluster]
        new_topics_dict[i] = topics_dict[cluster]

    print("Reset dictionaries")
    print(new_cluster_ref)
    print(new_topics_dict)

    return new_cluster_ref, new_topics_dict

# build ordered dictionary of cluster: drug number pairs
def build_ordered_cluster_dict(cluster_labels):
    dict_t0 = time.time()
    cluster_ref = {}
    non_outlier_n = len(set(cluster_labels)) - 1  # start from viable end list of labels

    for num, cluster in enumerate(cluster_labels):
        if cluster == -1:  # outliers are individual points
            if non_outlier_n not in cluster_ref:
                cluster_ref[non_outlier_n] = []
            cluster_ref[non_outlier_n].append(num)
            non_outlier_n += 1
        else:
            if cluster not in cluster_ref:
                cluster_ref[cluster] = []
            cluster_ref[cluster].append(num)

    cluster_ref = OrderedDict(sorted(cluster_ref.items()))
    print("Build dict:", str(time.time() - dict_t0))
    print([str(cluster) + ": " + str(len(cluster_ref[cluster])) for cluster in cluster_ref])

    return cluster_ref

# calculate adjacency matrix of supernodes (clusters) by averaging all connections between the two clusters
def calculate_super_adj_mat(cluster_ref, full_mat):
    adj_t0 = time.time()
    adj_mat = np.zeros((len(cluster_ref), len(cluster_ref)))

    cluster_combs = itertools.combinations(cluster_ref.keys(), 2)
    for cluster1, cluster2 in cluster_combs:
        drug_combs = list(itertools.product(cluster_ref[cluster1], cluster_ref[cluster2]))
        cos_sum = 0
        for drug1, drug2 in drug_combs:
            cos_sum += full_mat[drug1][drug2]
        avg = cos_sum / len(drug_combs)

        adj_mat[cluster1][cluster2] = avg

    print("Build super adjacency matrix:", str(time.time() - adj_t0))

    # add weights now
    adj_mat[adj_mat != 0] = (adj_mat[adj_mat != 0] + .1) * 4

    return adj_mat

# general function to add weights and reduce adjacency matrix for networkx
def weight_and_process_adj_mat(adj_mat):
    # add weights now
    adj_mat[adj_mat != 0] = (adj_mat[adj_mat != 0] + .1) * 2

    # avoid redundant information and send networkx only lower triangle
    adj_mat = np.tril(adj_mat)
    np.fill_diagonal(adj_mat, 0)  # mask 1's (field similarity to itself)

    return adj_mat

# create all attribute mappings for each cluster showing in graph
def generate_super_attr_mappings(purpose_df, cluster_ref, *topics_dict):
    attr_t0 = time.time()
    # generate helper dictionary for list of ids per cluster: ordered
    num_to_id = generate_cluster_num_to_field(purpose_df, cluster_ref)
    attr_dict = {}  # attributes of nodes to be added

    cutoff = 10
    # iterate through products and generate attributes/mappings
    for i, cluster in enumerate(num_to_id):
        names = purpose_df.loc[purpose_df["id"].isin(num_to_id[cluster]), "brand_name"].tolist()
        num_per_cluster = len(names)

        # shorten names if in a cluster
        # truncate or leave as is
        names = names[0:cutoff]
        routes = purpose_df.loc[purpose_df["id"].isin(num_to_id[cluster]), "route"].tolist()[0:cutoff]
        types = purpose_df.loc[purpose_df["id"].isin(num_to_id[cluster]), "product_type"].tolist()[0:cutoff]
        if num_per_cluster > cutoff:
            # add ellipses for more
            names.append("...")
            routes.append("...")
            types.append("...")

        attr_dict[cluster] = {"id": "Topic " + str(i) if num_per_cluster > 1 else "Individual Drug",
                              "num_drugs": num_per_cluster, "size_drugs": math.log(num_per_cluster) * 2 + 3,
                              "name": ", ".join(names), "route": ", ".join(routes), "product_type": ", ".join(types)}

        # if sorting by topics cluster, use dictionary in args tuple
        if topics_dict:
            attr_dict[cluster]["keywords"] = ", ".join(topics_dict[0][cluster])

    print(attr_dict)

    print("Assign attributes: ", str(time.time() - attr_t0))

    num_to_cluster = {cluster: info["id"] for cluster, info in attr_dict.items()}
    print("num_to_cluster:", num_to_cluster)

    return num_to_cluster, attr_dict

# generate dictionary of cluster: [ids of drugs] or [field of drugs] as specified
def generate_cluster_num_to_field(purpose_df, cluster_ref, field="id"):
    num_to_name = {}
    # create mapping dictionaries
    # post cluster: [drug ids] or [drug fields]
    for cluster, drug_list in cluster_ref.items():
        if cluster not in num_to_name:
            num_to_name[cluster] = []
        for drug in drug_list:
            num_to_name[cluster].append(purpose_df.iloc[drug][field])

    print(num_to_name)

    return num_to_name

# generate full html of each drug within a topic
def generate_cluster_num_to_html(purpose_df, cluster_ref, field):
    html_t0 = time.time()
    # pull original drug database columns for full text; at this point file should be read
    raw_df = preprocessing.read_preprocessed_to_pkl("indic_full_drug_df")

    # join relevant raw field to cleaned dataframe
    full_df = pd.merge(purpose_df, raw_df[["id", "purpose", "indications_and_usage", field]].add_suffix("_raw"),
                       left_on='id', right_on='id_raw',
                       how="left")

    print("New dataframe joined:", str(full_df.shape))

    num_to_html = {}

    # cluster: brand name, field
    for cluster, drug_list in cluster_ref.items():
        if cluster not in num_to_html:
            num_to_html[cluster] = []
        for drug in drug_list:
            row = full_df.iloc[drug]
            card_id = "heading_" + row["id"]
            blurb = ("<b>Brand name: </b>" + row["brand_name"] + "<br/>"
            + "<b>Route: </b>" + row["route"] + "<br/>"
            + "<b>Drug Type: </b>" + row["product_type"] + "<br/>"
            + "<b>Purpose/Indications and Usage: </b>" + (row["purpose_raw"] if (row["purpose_raw"] and row["purpose_raw"] != "")
                                                          else row["indications_and_usage_raw"]) + "<br/>"
            + "<b>" + " ".join([word.capitalize() for word in field.split("_")]) + ": </b>"
            + row[field + "_raw"])
            html_str = """
            <div class="card">
                <div class="card-header">
                        <a href="#%s" class="collapsed card-link" data-toggle="collapse" data-target="#%s" aria-expanded="true" aria-controls="%s">
                            %s
                        </a>
                </div>

                <div id="%s" class="collapse" data-parent="#accordion">
                    <div class="card-body">
                        %s
                    </div>
                </div>
            </div>""" % (card_id, card_id, card_id, row["brand_name"], card_id, blurb)

            num_to_html[cluster].append(html_str)

    print("Generated html:", str(time.time() - html_t0))
    return num_to_html

# generate all reference dictionaries to traits for graph
def generate_attr_mappings(purpose_df):
    attr_t0 = time.time()
    # create mapping dictionaries
    num_to_name = pd.Series(purpose_df.id, index=purpose_df.index).to_dict()
    attr_dict = {}  # attributes of nodes to be added

    # iterate through products and generate attributes/mappings
    product_list = purpose_df["id"].tolist()
    for i, product_id in enumerate(product_list):
        attr_dict[product_id] = {"id": product_id,
                                 "name": purpose_df.loc[purpose_df["id"] == product_id, "brand_name"].values[0],
                                 "purpose": purpose_df.loc[purpose_df["id"] == product_id, "purpose"].values[0],
                                 "route": purpose_df.loc[purpose_df["id"] == product_id, "route"].values[0]}

    print("Assign attributes: ", str(time.time() - attr_t0))

    return (num_to_name, attr_dict)

# restrict adjacency matrix to max n edges per node
def restrict_adjacency_matrix(adj_mat, n_max):
    narrow_t0 = time.time()
    # method 2: narrow adjacency matrix to max n edges per node (including previously determined max edges)
    sparse_mat = np.zeros((adj_mat.shape[0], adj_mat.shape[1]))

    for current_node_i in range(0, adj_mat.shape[0]):
        adj_node_indexes = adj_mat[current_node_i].argsort()[-n_max:][
                           ::-1]  # max n edges per node by weight (duplicate possible)
        for adj_node_i in adj_node_indexes:
            sparse_mat[current_node_i][adj_node_i] = adj_mat[current_node_i][adj_node_i]

    print("Narrow adjacency matrix: ", str(time.time() - narrow_t0))

    return sparse_mat

# create graph of drugs in a particular purpose
def generate_purpose_graph(sparse_mat, attr_dict, num_to_name):
    venn_G = nx.from_numpy_matrix(sparse_mat)  # create from adjacency matrix
    # venn_G = nx.relabel.relabel_nodes(venn_G, num_to_name)  # set node names to product ids
    nx.set_node_attributes(venn_G, attr_dict)  # add relevant attributes in

    print("Number of nodes: ", str(len(venn_G.nodes)))

    return venn_G

# plot graph of drugs in a particular purpose (bokeh)
def generate_graph_plot(venn_G, purpose, field, num_to_html, topics=False):
    # Display the network graph from the venn diagram interactions
    plot = figure(title="Network of Top Similar Drugs by " + " ".join([word.capitalize() for word in field.split("_")]),
                  x_range=(-10, 10), y_range=(-10, 10), plot_width=600, plot_height=500,
                  tools="wheel_zoom, pan, lasso_select, box_select", active_scroll="wheel_zoom",
                  toolbar_location="left")

    # create general layout
    drug_panel = Div(width=400, height=plot.plot_height, height_policy="fixed")
    layout = row(plot, drug_panel, height=500, height_policy="fixed")

    # generate extra field for topic keywords
    if topics:
        node_hover_tool = HoverTool(tooltips=[("id", "@id"), ("number of drugs", "@num_drugs"),
                                            ("names", "@name"), ("routes", "@route"), ("types", "@product_type"),
                                              ("keywords", "@keywords")], show_arrow=False)
    # generate hover capabilities
    else:
        node_hover_tool = HoverTool(tooltips=[("id", "@id"), ("number of drugs", "@num_drugs"),
                                              ("name", "@name"), ("route", "@route")], show_arrow=False)

    # create tool to list drug names when node for topic is clicked
    callback_tap_tool = CustomJS(args=dict(div=drug_panel), code="""
var names = %s;
var names_str = "";
var cluster_i = cb_data.source.selected.indices[0];
    for (var name_i = 0; name_i < names[cluster_i].length; name_i ++) {
        names_str += names[cluster_i][name_i];
    }
div.text = "<b>Products</b><br/>"
+ "<div id='accordion' style='height: 500px; display: block; overflow: auto;'>"
+ names_str + "</div>";
    """ % (list(num_to_html.values())))

    # Hide unnecessary plot attributes
    plot.axis.visible = False
    plot.xgrid.visible = False
    plot.ygrid.visible = False
    plot.outline_line_color = None

    # set additional toolbar attributes
    plot.add_tools(node_hover_tool, TapTool(callback=callback_tap_tool), ResetTool())

    graph = from_networkx(venn_G, nx.fruchterman_reingold_layout, scale=10, center=(0, 0))

    # vary thickness with node length
    # weight influence helped by https://stackoverflow.com/questions/49136867/networkx-plotting-in-bokeh-how-to-set-edge-width-based-on-graph-edge-weight
    graph.edge_renderer.glyph.line_width = {'field': 'weight'}

    # graph settings
    # change size of node with cluster
    graph.node_renderer.glyph = Circle(size="size_drugs", fill_color="pink", line_width=1.5)
    graph.node_renderer.selection_glyph = Circle(size="size_drugs", fill_color="green", line_width=1.5)
    graph.edge_renderer.selection_glyph = MultiLine(line_color="#226882", line_width={'field': 'weight'})
    graph.edge_renderer.hover_glyph = MultiLine(line_color="red", line_width={'field': 'weight'})
    graph.selection_policy = NodesAndLinkedEdges()
    graph.inspection_policy = NodesAndLinkedEdges()

    plot.renderers.append(graph)

    script, div = components(layout)

    # output_file("static/purpose-field/" + "_".join(purpose.split()) + "-" + field + ".html")
    # save(layout)
    # save file as json serial
    # show(layout)

    return script, div

# save bokeh embedding into a raw HTML to load Bootstrap/JS
def save_html_template(bokeh_script, bokeh_div, purpose, field):
    env = Environment(loader=FileSystemLoader('templates'))
    template = env.get_template('graph.html')
    output = template.render(bokeh_script=bokeh_script, bokeh_div=bokeh_div)
    print(output)

    # to save the results
    with open("static/purpose-field/" + "_".join(purpose.split()) + "-" + field + ".html", "w") as f:
        f.write(output)

# 4. Heatmap form of adjacency matrix, full and small
def plot_adj_mat_heatmap(adj_mat, attr_dict, product_list):
    fig, (ax0, ax1) = plt.subplots(1, 2)
    im = ax0.imshow(adj_mat)
    ax0.set_title("Heatmap of Indications and Usage")

    # 4a: plot heatmap using a small sample of adjacency matrix
    n_ticks = 10
    ax1.imshow(adj_mat[:n_ticks, :n_ticks])
    ax1.set_title("Heatmap of " + str(n_ticks) + " Indications and Usage")

    ax1.set_xticklabels([attr_dict[product_id]['name'] for product_id in product_list[:n_ticks]])
    ax1.set_yticklabels([attr_dict[product_id]['name'] for product_id in product_list[:n_ticks]])
    ax1.xaxis.set_major_locator(plt.MaxNLocator(n_ticks))
    ax1.yaxis.set_major_locator(plt.MaxNLocator(n_ticks))

    # rotate the tick labels and set their alignment
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # annotate center of heatmap boxes
    for i in range(n_ticks):
        for j in range(n_ticks):
            ax1.text(j, i, adj_mat[i, j], ha="center", va="center", color="w")

    fig.tight_layout()
    plt.show()

# save wordcloud of full purpose-field grouping
def save_wordcloud(purpose_df, purpose, field):
    test_str = " ".join(purpose_df[field].tolist())
    print(test_str)

    word_t0 = time.time()
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_color='steelblue', colormap="GnBu")
    wordcloud.generate(test_str)
    wordcloud.to_file("static/" + "_".join(purpose.split()) + "-" + field + ".png")

    print("WordCloud:", str(time.time() - word_t0))

# create web-app-friendly pathway to build network graph
def create_web_graph(purpose, field):
    purpose_df = pd.read_pickle("pkl/purpose/" + "_".join(purpose.split()) + ".pkl")

    full_graph_t0 = time.time()
    # 3. Generate a graph node network of top products and their similarity to each other
    adj_mat, sparse_mat, attr_dict, num_to_name, num_to_html = \
        generate_similarity_matching_field_of_purpose(purpose_df, field)

    # create graph
    venn_G = generate_purpose_graph(sparse_mat, attr_dict, num_to_name)
    print("Time to build graph:", str(time.time() - full_graph_t0))

    script, div = generate_graph_plot(venn_G, purpose, field, num_to_html, topics=True)
    print("Time to generate graph for", purpose, "-", field, ":", str(time.time() - full_graph_t0))

    return script, div

# obtain a dictionary containing all disabled fields for files generated (ex: a certain purpose may not have enough active ingredients, etc)
def obtain_disabled_fields_dict_from_files(dir):
    files = []
    for (dirpath, dirnames, filenames) in walk(dir):
        files.extend(filenames)

    files_df = pd.DataFrame(filenames, columns=["full_name"])
    files_df["purpose_cluster"], files_df["field"] = zip(*files_df["full_name"].str.split("-"))

    # clean entries to uniform purpose and field descriptors
    files_df["purpose_cluster"] = files_df["purpose_cluster"].str.split("_").str.join(" ")
    files_df["field"] = files_df["field"].str.split(".").str[0]

    valid_fields = ["active_ingredient", "inactive_ingredient", "dosage_and_administration", "warnings"]

    # create a dictionary of disabled fields (nonexistent fields) based on file names
    purpose_cluster_groups = files_df.groupby("purpose_cluster")
    field_dict = {}
    for name, purpose_cluster in purpose_cluster_groups:
        field_dict[name] = [field for field in valid_fields if field not in set(purpose_cluster["field"])]

    print(field_dict)

# main
def main():
    # file read and setup
    json_list = ["drug-label-0001-of-0009.json", "drug-label-0002-of-0009.json", "drug-label-0003-of-0009.json",
                 "drug-label-0004-of-0009.json", "drug-label-0005-of-0009.json", "drug-label-0006-of-0009.json",
                 "drug-label-0007-of-0009.json", "drug-label-0008-of-0009.json", "drug-label-0009-of-0009.json"]
    drug_df = parse_json.obtain_preprocessed_drugs(json_list, "purpose_indic_full_drug_df")

    print(drug_df[0:10]) # verify read
    print(drug_df.shape)

    # # 1: generate key purposes
    # rank_purpose(drug_df, 30)
    #
    # print("Full length: ", str(len(drug_df)))
    #
    # # 2.
    # draw_venn(drug_df, "97f91168-9f82-34bc-e053-2a95a90a33f8", "indications_and_usage")  # VERATRUM ALBUM

    # 7. Perform KMeans scoring on purpose/indication combination clustering sort
    # perform_kmeans_scoring(drug_df)

    # 3b. Automate process to obtain node network graphs of purpose_field combinations
    purposes = preprocessing.find_unique_purposes(drug_df)
    fields = ["active_ingredient", "inactive_ingredient", "warnings", "dosage_and_administration"]

    print(purposes)

    # purposes = ["indicate usage patient symptom tablet"]
    # fields = ["warnings"]

    for purpose in purposes:
        for field in fields:
            print("Parsing", purpose, "with", field)
            purpose_df = find_df_fitting_purpose(drug_df, purpose, field)

            # # 2b: Use purpose to find top products to compare
            # full_graph_t0 = time.time()
            # # 3. Generate a graph node network of top products and their similarity to each other
            # adj_mat, sparse_mat, attr_dict, num_to_name = \
            #     generate_similarity_matching_field_of_purpose(purpose_df, field, topics=False)
            #
            # # create graph
            # venn_G = generate_purpose_graph(sparse_mat, attr_dict, num_to_name)
            # print("Time to build graph:", str(time.time() - full_graph_t0))
            #
            # generate_graph_plot(venn_G, purpose, field)
            # print("Time to generate graph for", purpose, "-", field, ":", str(time.time() - full_graph_t0))

            # certain fields for certain clusters are None, or single so can't be weighted
            if len(purpose_df.index) > 1:
                full_graph_t0 = time.time()
                # 3. Generate a graph node network of top products and their similarity to each other
                adj_mat, sparse_mat, attr_dict, num_to_name, num_to_html = \
                    generate_similarity_matching_field_of_purpose(purpose_df, field)

                # create graph
                venn_G = generate_purpose_graph(sparse_mat, attr_dict, num_to_name)
                print("Time to build graph:", str(time.time() - full_graph_t0))

                bokeh_script, bokeh_div = generate_graph_plot(venn_G, purpose, field, num_to_html, topics=True)
                save_html_template(bokeh_script, bokeh_div, purpose, field)
                print("Time to generate graph for", purpose, "-", field, ":", str(time.time() - full_graph_t0))

                # 6: Plot word cloud
                # save word cloud for purpose cluster - field combo
                save_wordcloud(purpose_df, purpose, field)

    # obtain a dictionary containing all disabled fields for files generated (ex: a certain purpose may not have enough active ingredients, etc)
    obtain_disabled_fields_dict_from_files("static/purpose-field")

    # 4: plot heatmap using adjacency matrix for all matches
    # TODO: fix name reference
    # plot_adj_mat_heatmap(adj_mat, attr_dict, similar_products)

    # 5: Check dimensionality reduction (not plottable...)
    # perform_LSA(drug_df, field="purpose")

if __name__ == "__main__":
    main()
