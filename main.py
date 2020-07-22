import time

import json, ijson
import string
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
                          NodesAndLinkedEdges, EdgesAndLinkedNodes)
from bokeh.plotting import figure, from_networkx
from bokeh.embed import components, file_html

import spacy
from spacy.lang.en import English

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

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

# helper function to find product names of a common purpose
def find_similar_drugs_from_purpose(drug_df, purpose, field):
    purpose_t0 = time.time()
    # FUNC: find similar products by purpose as list
    purpose_df = drug_df.dropna(
        subset=["id", "purpose", "brand_name", field])  # exclude all rows with columns of null

    print("Has all non-null: ", str(purpose_df.count(axis=0)))

    similar_products = []
    similar_products.extend(purpose_df.loc[purpose_df["purpose_cluster"].str.contains(purpose),
                                "id"].tolist())  # append each one separately

    print("Length of similar products: ", str(len(similar_products)))
    print("Purpose finding and matching field: ", str(time.time() - purpose_t0))
    return similar_products

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

# 2b-3. Generate a graph using adjacency matrix of all drugs containing purpose based on TF-IDF
def generate_graph_matching_field_of_purpose(drug_df, similar_products, original_purpose, field):
    tfidfv_t0 = time.time()
    # obtain all fields of similar_products (only relevant products to purpose)
    purpose_df = drug_df.loc[drug_df["id"].isin(similar_products)]
    purpose_df = purpose_df.reset_index()

    field_list = purpose_df[field].tolist()
    print(field_list[0:10])

    # method of representation: TF-IDF
    tfidfv = TfidfVectorizer()
    fieldX = tfidfv.fit_transform(field_list)
    print("TF-IDF Field: ", str(time.time() - tfidfv_t0))
    print(fieldX.shape)

    print(purpose_df)

    # compute cosine similarity
    # cos_sim_t0 = time.time()
    # cos_sim = cosine_similarity(fieldX, fieldX)
    # print("cosine_similarity:", str(time.time() - cos_sim_t0)) # slightly more inefficient
    lin_kern_t0 = time.time()
    adj_mat = linear_kernel(fieldX, fieldX)

    # add weights now
    adj_mat[adj_mat != 0] = (adj_mat[adj_mat != 0] + .1) * 2

    # avoid redundant information and send networkx only lower triangle
    adj_mat = np.tril(adj_mat)
    np.fill_diagonal(adj_mat, 0) # mask 1's (field similarity to itself)
    print("linear_kernel:", str(time.time() - lin_kern_t0))

    # assign attributes
    num_to_name, attr_dict = generate_attr_mappings(purpose_df, similar_products)

    # generate graph with top n edges per node weighted by similarity
    # iterate through all combinations and find similarities

    # create sparse adjacency matrix, removing edges
    sparse_mat = restrict_adjacency_matrix(adj_mat, num_to_name, 8)

    print(adj_mat)
    print(sparse_mat)

    # create graph
    venn_G = generate_purpose_graph(sparse_mat, attr_dict, num_to_name)
    print("Number of nodes: ", str(len(venn_G.nodes)))

    return (venn_G, adj_mat, attr_dict, num_to_name)

# generate all reference dictionaries to traits for graph
def generate_attr_mappings(purpose_df, product_list):
    attr_t0 = time.time()
    # create mapping dictionaries
    num_to_name = pd.Series(purpose_df.id, index=purpose_df.index).to_dict()
    attr_dict = {}  # attributes of nodes to be added

    # iterate through products and generate attributes/mappings
    for i, product_id in enumerate(product_list):
        attr_dict[product_id] = {"id": product_id,
                                 "name": purpose_df.loc[purpose_df["id"] == product_id, "brand_name"].values[0],
                                 "purpose": purpose_df.loc[purpose_df["id"] == product_id, "purpose"].values[0],
                                 "route": purpose_df.loc[purpose_df["id"] == product_id, "route"].values[0]}

    print("Assign attributes: ", str(time.time() - attr_t0))

    return (num_to_name, attr_dict)

# restrict adjacency matrix to max n edges per node
def restrict_adjacency_matrix(adj_mat, num_to_name, n_max):
    narrow_t0 = time.time()
    # method 2: narrow adjacency matrix to max n edges per node (including previously determined max edges)
    sparse_mat = np.zeros((len(num_to_name), len(num_to_name)))

    for current_node_i in range(0, len(num_to_name)):
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
    venn_G = nx.relabel.relabel_nodes(venn_G, num_to_name)  # set node names to product ids
    nx.set_node_attributes(venn_G, attr_dict)  # add relevant attributes in

    return venn_G

# plot graph of drugs in a particular purpose (bokeh)
def generate_graph_plot(venn_G, purpose, field):
    # Display the network graph from the venn diagram interactions
    plot = figure(title="Network of Top Similar Drugs by Indications and Usage", x_range=(-50, 50), y_range=(-50, 50),
                  tools="pan,lasso_select,box_select", toolbar_location="right")
    # generate hover capabilities
    node_hover_tool = HoverTool(tooltips=[("id", "@id"), ("name", "@name"), ("purpose", "@purpose"), ("route", "@route")], show_arrow=False)
    plot.add_tools(node_hover_tool, BoxZoomTool(), ResetTool())

    graph = from_networkx(venn_G, nx.fruchterman_reingold_layout, k=.25, scale=1000, center=(0, 0))

    # vary thickness with node length
    # weight influence helped by https://stackoverflow.com/questions/49136867/networkx-plotting-in-bokeh-how-to-set-edge-width-based-on-graph-edge-weight
    graph.edge_renderer.data_source.data["line_width"] = [(venn_G.get_edge_data(a, b)['weight']) for a, b in
                                                          venn_G.edges()]
    graph.edge_renderer.glyph.line_width = {'field': 'line_width'}

    # graph settings
    graph.node_renderer.glyph = Circle(size=15, fill_color="pink")
    graph.edge_renderer.hover_glyph = MultiLine(line_color="red", line_width={'field': 'line_width'})
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

# main
def main():
    # file read and setup
    json_list = ["drug-label-0001-of-0009.json", "drug-label-0002-of-0009.json", "drug-label-0003-of-0009.json",
                 "drug-label-0004-of-0009.json", "drug-label-0005-of-0009.json", "drug-label-0006-of-0009.json",
                 "drug-label-0007-of-0009.json", "drug-label-0008-of-0009.json", "drug-label-0009-of-0009.json"]
    drug_df = parse_json.obtain_preprocessed_drugs(json_list, "purpose_full_drug_df")

    print(drug_df[0:10]) # verify read

    # 1: generate key purposes
    rank_purpose(drug_df, 30)

    print("Full length: ", str(len(drug_df)))

    # 2.
    draw_venn(drug_df, "97f91168-9f82-34bc-e053-2a95a90a33f8", "indications_and_usage")  # VERATRUM ALBUM

    # 3b. Automate process to obtain node network graphs of purpose_field combinations
    purposes = ["sunscreen purposes uses protectant skin"]
    fields = ["indications_and_usage", "warnings"]

    for purpose in purposes:
        for field in fields:
            # 2b: Use purpose to find top products to compare
            full_graph_t0 = time.time()
            similar_products = find_similar_drugs_from_purpose(drug_df, purpose, field)

            # 3. Generate a graph node network of top products and their similarity to each other
            venn_G, adj_mat, attr_dict, num_to_name = \
                generate_graph_matching_field_of_purpose(drug_df, similar_products, purpose, field)

            print("Time to build graph:", str(time.time() - full_graph_t0))
            generate_graph_plot(venn_G, purpose, field)
            print("Time to generate graph for", purpose, "-", field, ":", str(time.time() - full_graph_t0))

    # 4: plot heatmap using adjacency matrix for all matches
    # TODO: fix name reference
    plot_adj_mat_heatmap(adj_mat, attr_dict, similar_products)

if __name__ == "__main__":
    main()
