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
from bokeh.embed import components

import spacy
from spacy.lang.en import English

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

# 1. Finds top most searched purposes
def rank_purpose(drug_df, n_rank):
    nlp = English()
    drug_trunc = drug_df.dropna(subset = ["purpose"])
    master_list = []

    # in light of processing memory, perform cleaning separately
    # compile all purpose field words
    for purpose in drug_trunc["purpose"].tolist():
        purpose_list = clean_list(purpose, nlp)
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
    nlp = English()

    # find drug purpose
    original_purpose = drug_df.loc[drug_df["id"] == original]["purpose"].values[0]
    # find top purpose keyword
    original_purpose_list = clean_list(original_purpose, nlp)

    # find similar products by purpose
    similar_products = find_similar_products_from_product_purpose(drug_df, original_purpose_list, original, "indications_and_usage")

    # within similar products, find matching indications for same-purpose medications
    top_indic_df = find_matching_field_for_product(drug_df, similar_products, original, field)

    original_indic_set = set(
        clean_list(drug_df.loc[drug_df["id"] == original, field].values[0], nlp))

    print(top_indic_df)

    # plot top 3 by matching indications to original product
    plt.figure()
    original_name = drug_df.loc[drug_df["id"] == original]["brand_name"].values[0]
    venn = matplotlib_venn.venn3([original_indic_set,
                                  set(clean_list(drug_df.loc[drug_df["id"] == top_indic_df.iloc[0][
                                      'ProductID'], field].values[0], nlp)),
                                  set(clean_list(drug_df.loc[drug_df["id"] == top_indic_df.iloc[1][
                                      'ProductID'], field].values[0], nlp))],
                                 set_labels=[original_name, top_indic_df.iloc[0]['ProductName'],
                                             top_indic_df.iloc[1]['ProductName']])
    plt.title("Indication Word Matches in Common Purpose Product Labels", fontsize=10)
    plt.show()

# helper function to find product names of a common purpose
def find_similar_drugs_from_purpose(drug_df, purpose, field):
    # FUNC: find similar products by purpose as list
    purpose_df = drug_df.dropna(
        subset=["id", "purpose", "brand_name", field])  # exclude all rows with columns of null

    print("Has all non-null: ", str(purpose_df.count(axis=0)))

    similar_products = []
    similar_products.extend(purpose_df.loc[purpose_df["purpose"].str.contains(purpose)][
                                "id"].tolist())  # append each one separately

    print("Length of similar products: ", str(len(similar_products)))
    return similar_products

# helper function to find product names of a similar purpose to a product
# returns: list of similar product names
def find_similar_products_from_product_purpose(drug_df, original_purpose_list, original, field):
    valid_df = drug_df.dropna(subset=["id", "purpose", "brand_name", field]) # exclude all rows with columns of null
    valid_df = valid_df.loc[valid_df["id"] != original] # exclude self

    similar_products = set()
    for purpose in original_purpose_list:
        similar_products.update(valid_df.loc[valid_df["purpose"].str.contains(purpose), "id"].tolist())

    return list(similar_products)

# helper function to count and record matching words in indications and usage field
# returns: dictionary of product names and counts of matches
def find_matching_field_for_product(drug_df, similar_products, original, field):
    nlp = English()
    valid_df = drug_df.dropna(subset=["id", "purpose", "brand_name", field])  # exclude all rows with relevant columns of null
    valid_df = valid_df.loc[valid_df["id"] != original]  # exclude self

    # set original indications
    original_indic_set = set(clean_list(drug_df.loc[drug_df["id"] == original, field].values[0], nlp))
    # within similar purpose products, find the ones with top similarities in indications and usage
    top_indic = []

    for similar_product in similar_products:
        # data frame already wiped of invalid null rows
        similar_indic_set = set(clean_list(valid_df.loc[valid_df["id"] == similar_product, field].values[0], nlp))

        # create columns of ID, product, and matching words
        top_indic.append([similar_product,
                         valid_df.loc[valid_df["id"] == similar_product]["brand_name"].values[0],
                          len(original_indic_set.intersection(similar_indic_set))])

    top_indic_df = pd.DataFrame(top_indic, columns=["ProductID", "ProductName", "Match_Words"])
    top_indic_df = top_indic_df.sort_values(by="Match_Words", ascending=False)

    return top_indic_df

# 2b-3. Generate a graph using adjacency matrix of all drugs containing purpose based on matching words
def generate_graph_matching_field_of_purpose(drug_df, similar_products, original_purpose, field):
    permute_t0 = time.time()
    product_comb = itertools.combinations(similar_products, 2)  # convert permutations of product names into list
    print("Combinations: ", str(time.time() - permute_t0))

    attr_t0 = time.time()
    name_to_num, num_to_name, attr_dict = generate_attr_mappings(drug_df, similar_products)
    print("Assign attributes: ", str(time.time() - attr_t0))

    # generate graph with top n edges per node weighted by similarity
    # iterate through all combinations and find similarities

    # create adjacency matrix
    adj_mat_t0 = time.time()
    adj_mat = create_adjacency_matrix(drug_df, name_to_num, product_comb, field)
    print("Add to adjacency matrix: ", str(time.time() - adj_mat_t0))

    # create sparse adjacency matrix, removing edges
    narrow_t0 = time.time()
    sparse_mat = restrict_adjacency_matrix(adj_mat, name_to_num, 7)
    print("Narrow adjacency matrix: ", str(time.time() - narrow_t0))

    print(adj_mat)
    print(sparse_mat)

    # create graph
    venn_G = generate_purpose_graph(sparse_mat, attr_dict, num_to_name)
    print("Number of nodes: ", str(len(venn_G.nodes)))

    return (venn_G, adj_mat, attr_dict, name_to_num, num_to_name)

# generate all reference dictionaries to traits for graph
def generate_attr_mappings(drug_df, product_list):
    # create mapping dictionaries
    name_to_num = {}
    num_to_name = {}
    attr_dict = {}  # attributes of nodes to be added

    # iterate through products and generate attributes/mappings
    for i, product_id in enumerate(product_list):
        name_to_num[product_id] = i
        num_to_name[i] = product_id
        attr_dict[product_id] = {"id": product_id,
                                 "name": drug_df.loc[drug_df["id"] == product_id, "brand_name"].values[0],
                                 "purpose": drug_df.loc[drug_df["id"] == product_id, "purpose"].values[0],
                                 "route": drug_df.loc[drug_df["id"] == product_id, "route"].values[0]}

    return (name_to_num, num_to_name, attr_dict)

# create adjacency matrix of all drugs in graph
def create_adjacency_matrix(drug_df, name_to_num, product_comb, field):
    nlp = English()
    adj_mat = np.zeros((len(name_to_num), len(name_to_num)))

    # method 1
    # current_node = product_comb[0][0]   # the node for which we are calculating the top n edges

    # adj_node_weights = []  # the current node storage for edges
    # adj_node_list = []

    for comb in product_comb:
        set1 = set(clean_list(drug_df.loc[drug_df["id"] == comb[0], field].values[0], nlp))
        set2 = set(clean_list(drug_df.loc[drug_df["id"] == comb[1], field].values[0], nlp))

        # set weight of edges to reflect closeness of relationships (inverse)
        relation = len(set1.intersection(set2))

        # method 2: generate adjacency matrix
        adj_mat[name_to_num[comb[0]], name_to_num[comb[1]]] = relation
        adj_mat[name_to_num[comb[1]], name_to_num[comb[0]]] = relation

        # method 1
        # if comb[0] != current_node:
        #     # find the top 3 nodes by weight
        #     # make sortable data structure
        #     adj_node_df = pd.Series(data=adj_node_weights, index=adj_node_list)
        #     adj_node_df = adj_node_df.sort_values(ascending=False)
        #
        #     # set top 3 (inclusive of all currently) in edge adjacency matrix
        #     for i in range(0, 3) if adj_node_df.count() >= 3 else range(0, adj_node_df.count()):
        #         adj_mat[name_to_num[current_node], name_to_num[adj_node_df.index[i]]] = adj_node_df.iat[i]
        #         adj_mat[name_to_num[adj_node_df.index[i]], name_to_num[current_node]] = adj_node_df.iat[i]
        #
        #     # reset edge storage to current edge
        #     current_node = comb[0]
        #     adj_node_weights = []
        #     adj_node_list = []
        #
        # # continuation of edge accumulation
        # adj_node_list.append(comb[1])  # corresponding node to weight
        # adj_node_weights.append(relation)  # corresponding weight

        # method 1
        # # account for last combination
        # adj_node_df = pd.Series(data=adj_node_weights, index=adj_node_list)
        # adj_node_df = adj_node_df.sort_values(ascending=False)
        #
        # adj_mat[name_to_num[current_node], name_to_num[adj_node_df.index[0]]] = adj_node_df.iat[0]
        # adj_mat[name_to_num[adj_node_df.index[0]], name_to_num[current_node]] = adj_node_df.iat[0]

    return adj_mat

# restrict adjacency matrix to max n edges per node
def restrict_adjacency_matrix(adj_mat, name_to_num, n_max):
    # method 2: narrow adjacency matrix to max n edges per node (including previously determined max edges)
    sparse_mat = np.zeros((len(name_to_num), len(name_to_num)))

    for current_node_i in range(0, len(name_to_num)):
        adj_node_indexes = adj_mat[current_node_i].argsort()[-n_max:][
                           ::-1]  # max n edges per node by weight (duplicate possible)
        for adj_node_i in adj_node_indexes:
            sparse_mat[current_node_i][adj_node_i] = adj_mat[current_node_i][adj_node_i]
            sparse_mat[adj_node_i][current_node_i] = adj_mat[current_node_i][adj_node_i]

    return sparse_mat

# create graph of drugs in a particular purpose
def generate_purpose_graph(sparse_mat, attr_dict, num_to_name):
    venn_G = nx.from_numpy_matrix(sparse_mat)  # create from adjacency matrix
    venn_G = nx.relabel.relabel_nodes(venn_G, num_to_name)  # set node names to product ids
    nx.set_node_attributes(venn_G, attr_dict)  # add relevant attributes in

    return venn_G

# plot graph of drugs in a particular purpose (bokeh)
def plot_purpose_graph(venn_G):
    # Display the network graph from the venn diagram interactions
    plot = figure(title="Network of Top Similar Drugs by Indications and Usage", x_range=(-5, 5), y_range=(-5, 5),
                  tools="pan,lasso_select,box_select", toolbar_location="right")
    # generate hover capabilities
    node_hover_tool = HoverTool(tooltips=[("id", "@id"), ("name", "@name"), ("purpose", "@purpose"), ("route", "@route")], show_arrow=False)
    plot.add_tools(node_hover_tool, BoxZoomTool(), ResetTool())

    graph = from_networkx(venn_G, nx.spring_layout, scale=2, center=(0, 0))

    # vary thickness with node length
    # weight influence helped by https://stackoverflow.com/questions/49136867/networkx-plotting-in-bokeh-how-to-set-edge-width-based-on-graph-edge-weight
    graph.edge_renderer.data_source.data["line_width"] = [venn_G.get_edge_data(a, b)['weight'] / 10 for a, b in
                                                          venn_G.edges()]
    graph.edge_renderer.glyph.line_width = {'field': 'line_width'}

    # graph settings
    graph.node_renderer.glyph = Circle(size=15, fill_color="blue")
    graph.edge_renderer.hover_glyph = MultiLine(line_color="red", line_width={'field': 'line_width'})
    graph.inspection_policy = NodesAndLinkedEdges()

    plot.renderers.append(graph)

    script, div = components(plot)
    print(script)
    print(div)

    output_file("networkx_graph.html")
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
    json_list = ["drug-label-0001-of-0009.json", "drug-label-0002-of-0009.json"]
    drug_df = parse_json.parse_or_read_drugs(json_list, "drug_df", "pickle")

    print(drug_df) # verify read

    # 1: generate key purposes
    rank_t0 = time.time()
    rank_purpose(drug_df, 30)
    print("Rank: ", str(time.time() - rank_t0))

    print("Full length: ", str(len(drug_df)))

    # 2.
    drug_venn_t0 = time.time()
    draw_venn(drug_df, "85244a4b-c688-43f2-b3a4-f659943827a1", "indications_and_usage")  # BENZALKONIUM CHLORIDE
    print("Drug Venn: ", str(time.time() - drug_venn_t0))

    # 2b: Use purpose to find top products to compare
    purpose_t0 = time.time()
    similar_products = find_similar_drugs_from_purpose(drug_df, "analgesic", "indications_and_usage")
    print("Purpose finding and matching indications: ", str(time.time() - purpose_t0))

    # 3. Generate a graph node network of top products and their similarity to each other
    venn_G, adj_mat, attr_dict, name_to_num, num_to_name = generate_graph_matching_field_of_purpose(drug_df, similar_products, "analgesic", "indications_and_usage")
    plot_purpose_graph(venn_G)

    # 4: plot heatmap using adjacency matrix for all matches
    plot_adj_mat_heatmap(adj_mat, attr_dict, similar_products)

if __name__ == "__main__":
    main()
