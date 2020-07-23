import time

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
from bokeh.io import output_file, show
from bokeh.models import (BoxZoomTool, Circle, HoverTool,
                          MultiLine, Plot, Range1d, ResetTool,
                          NodesAndLinkedEdges, EdgesAndLinkedNodes, ColumnDataSource)
from bokeh.plotting import figure, from_networkx
from bokeh.embed import components, file_html

import spacy
from spacy.lang.en import English

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances, linear_kernel
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import DBSCAN

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
    nlp = English()
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
def generate_similarity_matching_field_of_purpose(drug_df, purpose, field):
    # obtain all fields of similar_products (only relevant products to purpose)
    purpose_df = find_df_fitting_purpose(drug_df, purpose, field)

    field_list = purpose_df[field].tolist()
    print(field_list[0:10])

    fieldX = create_tfidf(field_list)

    print(fieldX.shape)
    print(purpose_df)

    # run density model
    density_cluster = run_density_cluster(fieldX)

    # build ordered dictionary of cluster:corresponding drug numbers
    density_ref = build_ordered_cluster_dict(density_cluster.labels_)

    # calculate adjacency matrix of supernodes by averaging cosine similarity of edges in cluster
    full_mat = compute_tfidf_cos(fieldX) # each individual pairwise cosine similarity
    adj_mat = calculate_super_adj_mat(density_ref, full_mat)

    # add weights now
    adj_mat[adj_mat != 0] = (adj_mat[adj_mat != 0] + .1) * 2

    # create mapping dictionaries for cluster-based graph
    num_to_cluster, attr_dict = generate_super_attr_mappings(purpose_df, density_ref)

    # adj_mat = weight_and_process_adj_mat(adj_mat)

    # assign attributes
    # num_to_name, attr_dict = generate_attr_mappings(purpose_df)

    # generate graph with top n edges per node weighted by similarity
    # create sparse adjacency matrix, removing edges
    sparse_mat = restrict_adjacency_matrix(adj_mat, 8)

    print(adj_mat, adj_mat.shape)
    print(sparse_mat)

    return (adj_mat, sparse_mat, attr_dict, num_to_cluster)

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

# provide DBSCAN model results using n_samples x n_features X
def run_density_cluster(X):
    # cluster into supernodes based on cosine distance, density/nearest neighbors: eps=.32
    density_cluster = DBSCAN(eps=.32, min_samples=5, metric="cosine").fit(X)
    print("Number of clusters:", str(len(set(density_cluster.labels_)) - 1))
    print(list(density_cluster.labels_).count(-1))

    return density_cluster

# build ordered dictionary of cluster: drug number pairs
def build_ordered_cluster_dict(density_cluster_labels):
    dict_t0 = time.time()
    density_ref = {}
    non_outlier_n = len(set(density_cluster_labels)) - 1  # start from viable end list of labels

    for num, cluster in enumerate(density_cluster_labels):
        if cluster == -1:  # outliers are individual points
            if non_outlier_n not in density_ref:
                density_ref[non_outlier_n] = []
            density_ref[non_outlier_n].append(num)
            non_outlier_n += 1
        else:
            if cluster not in density_ref:
                density_ref[cluster] = []
            density_ref[cluster].append(num)

    density_ref = OrderedDict(sorted(density_ref.items()))
    print("Build dict:", str(time.time() - dict_t0))
    print([str(cluster) + ": " + str(len(density_ref[cluster])) for cluster in density_ref])

    return density_ref

# calculate adjacency matrix of supernodes (clusters) by averaging all connections between the two clusters
def calculate_super_adj_mat(density_ref, full_mat):
    adj_t0 = time.time()
    adj_mat = np.zeros((len(density_ref), len(density_ref)))

    cluster_combs = itertools.combinations(density_ref.keys(), 2)
    for cluster1, cluster2 in cluster_combs:
        drug_combs = list(itertools.product(density_ref[cluster1], density_ref[cluster2]))
        cos_sum = 0
        for drug1, drug2 in drug_combs:
            cos_sum += full_mat[drug1][drug2]
        avg = cos_sum / len(drug_combs)

        adj_mat[cluster1][cluster2] = avg

    print("Build super adjacency matrix:", str(time.time() - adj_t0))

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
def generate_super_attr_mappings(purpose_df, density_ref):
    attr_t0 = time.time()
    # generate helper dictionary for list of ids per cluster: ordered
    num_to_name = generate_cluster_num_to_name(purpose_df, density_ref)

    attr_dict = {}  # attributes of nodes to be added

    cutoff = 10
    # iterate through products and generate attributes/mappings
    for i, cluster in enumerate(num_to_name):
        names = purpose_df.loc[purpose_df["id"].isin(num_to_name[cluster]), "brand_name"].tolist()
        num_per_cluster = len(names)

        # shorten names if in a cluster
        # truncate or leave as is
        names = names[0:cutoff]
        routes = purpose_df.loc[purpose_df["id"].isin(num_to_name[cluster]), "route"].tolist()[0:cutoff]
        if num_per_cluster > cutoff:
            # add ellipses for more
            names.append("...")
            routes.append("...")

        attr_dict[cluster] = {"id": "Cluster " + str(i) if num_per_cluster > 1 else "Individual Drug",
                              "num_drugs": num_per_cluster, "size_drugs": math.log(num_per_cluster) + 5,
                              "name": ", ".join(names), "route": ", ".join(routes)}

    print(attr_dict)

    print("Assign attributes: ", str(time.time() - attr_t0))

    num_to_cluster = {cluster: info["id"] for cluster, info in attr_dict.items()}
    print("num_to_cluster:", num_to_cluster)

    return num_to_cluster, attr_dict

# generate dictionary of cluster: [ids of drugs]
def generate_cluster_num_to_name(purpose_df, density_ref):
    num_to_name = {}
    # create mapping dictionaries
    # post cluster: [drug ids]
    for cluster, drug_list in density_ref.items():
        if cluster not in num_to_name:
            num_to_name[cluster] = []
        for drug in drug_list:
            num_to_name[cluster].append(purpose_df.iloc[drug]["id"])

    print(num_to_name)

    return num_to_name

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
            sparse_mat[adj_node_i][current_node_i] = adj_mat[current_node_i][adj_node_i]

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
def generate_graph_plot(venn_G, purpose, field):
    # Display the network graph from the venn diagram interactions
    plot = figure(title="Network of Top Similar Drugs by Field", x_range=(-1000, 1000), y_range=(-1000, 1000),
                  tools="pan,lasso_select,box_select", toolbar_location="right")
    # generate hover capabilities
    node_hover_tool = HoverTool(tooltips=[("id", "@id"), ("number of drugs", "@num_drugs"),
                                          ("name", "@name"), ("route", "@route")], show_arrow=False)
    plot.add_tools(node_hover_tool, BoxZoomTool(), ResetTool())

    graph = from_networkx(venn_G, nx.fruchterman_reingold_layout, k=.25, scale=1000, center=(0, 0))

    # vary thickness with node length
    # weight influence helped by https://stackoverflow.com/questions/49136867/networkx-plotting-in-bokeh-how-to-set-edge-width-based-on-graph-edge-weight
    graph.edge_renderer.glyph.line_width = {'field': 'weight'}

    # graph settings
    # change size of node with cluster
    graph.node_renderer.glyph = Circle(size="size_drugs", fill_color="pink")
    graph.edge_renderer.hover_glyph = MultiLine(line_color="red", line_width={'field': 'weight'})
    graph.inspection_policy = NodesAndLinkedEdges()

    plot.renderers.append(graph)

    script, div = components(plot)

    output_file("static/" + purpose + "-" + field + ".html")
    show(plot)

    return script, div

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

# perform TruncatedSVD (LSA for TD-IDF) to see if dimensionality can be sufficiently reduced
def perform_LSA(drug_df, field="purpose"):
    # TF-IDF
    purposeX = create_tfidf(drug_df[field].tolist())

    # LSA
    lsa = TruncatedSVD(n_components=1500)
    lsa.fit(purposeX)
    print(sum(lsa.explained_variance_ratio_))

# main
def main():
    # file read and setup
    json_list = ["drug-label-0001-of-0009.json", "drug-label-0002-of-0009.json", "drug-label-0003-of-0009.json",
                 "drug-label-0004-of-0009.json", "drug-label-0005-of-0009.json", "drug-label-0006-of-0009.json",
                 "drug-label-0007-of-0009.json", "drug-label-0008-of-0009.json", "drug-label-0009-of-0009.json"]
    drug_df = parse_json.obtain_preprocessed_drugs(json_list, "purpose_full_drug_df")

    print(drug_df[0:10]) # verify read

    # # 1: generate key purposes
    # rank_purpose(drug_df, 30)
    #
    # print("Full length: ", str(len(drug_df)))
    #
    # # 2.
    # draw_venn(drug_df, "97f91168-9f82-34bc-e053-2a95a90a33f8", "indications_and_usage")  # VERATRUM ALBUM

    # 3b. Automate process to obtain node network graphs of purpose_field combinations
    purposes = ["sanitizer hand antiseptic antimicrobial skin"]
    fields = ["warnings"]

    for purpose in purposes:
        for field in fields:
            # 2b: Use purpose to find top products to compare
            full_graph_t0 = time.time()
            # 3. Generate a graph node network of top products and their similarity to each other
            adj_mat, sparse_mat, attr_dict, num_to_name = \
                generate_similarity_matching_field_of_purpose(drug_df, purpose, field)

            # create graph
            venn_G = generate_purpose_graph(sparse_mat, attr_dict, num_to_name)
            print("Time to build graph:", str(time.time() - full_graph_t0))

            generate_graph_plot(venn_G, purpose, field)
            print("Time to generate graph for", purpose, "-", field, ":", str(time.time() - full_graph_t0))

    # 4: plot heatmap using adjacency matrix for all matches
    # TODO: fix name reference
    # plot_adj_mat_heatmap(adj_mat, attr_dict, similar_products)

    # 5: Check dimensionality reduction (not plottable...)
    # perform_LSA(drug_df, field="purpose")

if __name__ == "__main__":
    main()
