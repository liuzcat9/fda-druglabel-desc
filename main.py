import time

import json, ijson
import string
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib_venn
import itertools

import networkx as nx
import bokeh
from bokeh.io import output_file, show
from bokeh.models import (BoxZoomTool, Circle, HoverTool,
                          MultiLine, Plot, Range1d, ResetTool,
                          NodesAndLinkedEdges, EdgesAndLinkedNodes)
from bokeh.plotting import figure, from_networkx
from bokeh.embed import components

import spacy
from spacy.lang.en import English

# Takes in a list
# Removes the word "Purpose" and punctuation
# Returns cleaned list
def clean_list(purpose_str, nlp):
    temp_doc = nlp(purpose_str)
    cleaned_list = [token.text.lower() for token in temp_doc if not(token.is_stop or token.is_punct or token.is_space or token.text.lower() == "purpose")]  # tokenize excluding punctuation
    return cleaned_list

# 1. Finds top most searched purposes
def rank_purpose(drug1_df):
    drug1_trunc = drug1_df.dropna(subset = ["purpose"])

    nlp = English()
    master_list = []

    # in light of processing memory, perform cleaning separately
    for purpose in drug1_trunc["purpose"].tolist():
        purpose_list = clean_list(purpose, nlp)
        master_list.extend(purpose_list)

    purposes = dict()
    for word in master_list:
        # add keywords into purpose dict
        if (word not in purposes.keys()):
            purposes[word] = 0
        else:
            purposes[word] += 1

    # sort extracted purposes by frequency
    num_wanted = 30
    purposes_df = pd.DataFrame(purposes.items(), columns=["Keyword", "Frequency"])
    purposes_df = purposes_df.sort_values(by="Frequency", ascending=False)
    purposes_trunc = purposes_df.head(num_wanted) # shorten to top n

    # plot truncated top purposes
    font = {'weight': 'bold',
            'size': 6}
    matplotlib.rc('font', **font)
    top_purpose = purposes_trunc.plot(kind='bar', x="Keyword", y="Frequency")
    plt.title("Top " + str(num_wanted) + " Purpose Keywords for FDA Drugs", fontsize=10)
    plt.xlabel("Keyword", fontsize=7)
    plt.ylabel("Frequency", fontsize=7)
    top_purpose.get_legend().remove()
    plt.show()

# helper function to find product names of a similar purpose
# returns: list of similar product names
def find_similar_products_purpose(drug1_df, original_purpose_list, original):
    valid_df = drug1_df.dropna() # exclude all rows with columns of null
    valid_df = valid_df.loc[valid_df["id"] != original] # exclude self

    similar_products = set()
    for purpose in original_purpose_list:
        similar_products.update(valid_df.loc[valid_df["purpose"].str.contains(purpose), "id"].tolist())

    return list(similar_products)

# helper function to count and record matching words in indications and usage field
# returns: dictionary of product names and counts of matches
def find_matching_indic(drug1_df, similar_products, original):
    nlp = English()
    valid_df = drug1_df.dropna()  # exclude all rows with columns of null
    valid_df = valid_df.loc[valid_df["id"] != original]  # exclude self

    # set original indications
    original_indic_set = set(clean_list(drug1_df.loc[drug1_df["id"] == original, "indications_and_usage"].values[0], nlp))
    # within similar purpose products, find the ones with top similarities in indications and usage
    top_indic = []

    for similar_product in similar_products:
        # data frame already wiped of invalid null rows
        similar_indic_set = set(clean_list(valid_df.loc[valid_df["id"] == similar_product, "indications_and_usage"].values[0], nlp))

        # create columns of ID, product, and matching words
        top_indic.append([similar_product,
                         valid_df.loc[valid_df["id"] == similar_product]["brand_name"].values[0],
                          len(original_indic_set.intersection(similar_indic_set))])

    top_indic_df = pd.DataFrame(top_indic, columns=["ProductID", "ProductName", "Match_Words"])
    top_indic_df = top_indic_df.sort_values(by="Match_Words", ascending=False)

    return top_indic_df

# main
def main():
    nlp = English()
    # reading using ijson
    t0 = time.time()

    drug1_df = pd.DataFrame(columns=["purpose", "id", "brand_name", "indications_and_usage"])

    # loop over and read all information
    file_list = ["drug-label-0001-of-0008.json", "drug-label-0002-of-0008.json"]

    for filename in file_list:
        current = []
        current_temp = []
        with open(filename) as file:
            objects = ijson.items(file, "results.item")

            # obtain information in a dataframe
            for o in objects:
                current_temp.append([o["purpose"][0] if "purpose" in o.keys() else None,
                                   o["id"] if "id" in o.keys() else None,
                                   o["openfda"]["brand_name"][0] if (
                                               "openfda" in o.keys() and "brand_name" in o["openfda"].keys()) else None,
                                   o["indications_and_usage"][0] if "indications_and_usage" in o.keys() else None])
                current.append(o)

        # compile master dataframe
        current_df = pd.DataFrame(current_temp, columns=["purpose", "id", "brand_name", "indications_and_usage"])
        drug1_df = pd.concat([drug1_df, current_df])

    t1 = time.time()
    print(t1 - t0)

    # drug1 is a list of all viable objects
    print(drug1_df) # how many drugs there are

    # use source: https://gist.github.com/deekayen/4148741 to eliminate mundane words
    with open("common_english.txt") as f:
        mundane = f.read().splitlines() # read without newline character

    # 1: generate key purposes
    rank_purpose(drug1_df)

    print("Full length: ", str(len(drug1_df)))

    # 2: Venn Diagram similarity between most similar drugs based on purpose keyword/indication matches
    # choose a drug
    original = "b3eebddf-53f5-4f30-a071-ad70152ee97a" # H. Pylori Plus
    # find drug purpose
    original_purpose = drug1_df.loc[drug1_df["id"] == original, "purpose"].values[0]
    # find top purpose keyword
    original_purpose_list = clean_list(original_purpose, nlp)

    # find similar products by purpose
    similar_products = find_similar_products_purpose(drug1_df, original_purpose_list, original)

    # within similar products, find matching indications for same-purpose medications
    top_indic_df = find_matching_indic(drug1_df, similar_products, original)

    original_indic_set = set(
        clean_list(drug1_df.loc[drug1_df["id"] == original, "indications_and_usage"].values[0], nlp))

    print(top_indic_df)

    # plot top 3 by matching indications to original product
    plt.figure()
    original_name = drug1_df.loc[drug1_df["id"] == original]["brand_name"].values[0]
    venn = matplotlib_venn.venn3([original_indic_set,
                                  set(clean_list(drug1_df.loc[drug1_df["id"] == top_indic_df.iloc[0]['ProductID'], "indications_and_usage"].values[0], nlp)),
                                  set(clean_list(drug1_df.loc[drug1_df["id"] == top_indic_df.iloc[1]['ProductID'], "indications_and_usage"].values[0], nlp))],
                                  set_labels = [original_name, top_indic_df.iloc[0]['ProductName'], top_indic_df.iloc[1]['ProductName']])
    plt.title("Indication Word Matches in Common Purpose Product Labels", fontsize=10)
    plt.show()

    # 2b: Use purpose to find top products to compare
    # set drug purpose
    original_purpose = "sunscreen"

    # FUNC: find similar products by purpose as list
    purpose_df = drug1_df.dropna()  # exclude all rows with columns of null

    print("Has all non-null: ", str(purpose_df.count(axis=0)))

    similar_products = []
    similar_products.extend(purpose_df.loc[purpose_df["purpose"].str.contains(original_purpose)]["id"].tolist()) # append each one separately

    print("Length of similar products: ", str(len(similar_products)))

    # using first product of selected purpose, find matching indications for same-purpose medications
    product_id = similar_products[0]
    top_indic_df = find_matching_indic(drug1_df, similar_products[1:], product_id)

    print(top_indic_df)

    # 3. Generate a graph node network of top products and their similarity to each other
    # permute through all combinations of original and top x products to numerically obtain the venn diagram similarities
    product_list = [product_id]
    product_list.extend(list(top_indic_df.iloc[:, 0]))
    itr_products = product_list

    product_comb = list(itertools.combinations(itr_products, 2)) # convert permutations of product names into list

    # Generate a network graph from the venn diagram interactions
    venn_G = nx.Graph()

    # iterate through products and generate nodes with attributes
    for product_id in itr_products:
        venn_G.add_node(product_id, name=drug1_df.loc[drug1_df["id"] == product_id]["brand_name"],
                        purpose=drug1_df.loc[drug1_df["id"] == product_id]["purpose"])

    # iterate through all combinations and find similarities,
    # generate graph with edges weighted by similarity
    for comb in product_comb:
        set1 = set(clean_list(drug1_df.loc[drug1_df["id"] == comb[0], "indications_and_usage"].values[0], nlp))
        set2 = set(clean_list(drug1_df.loc[drug1_df["id"] == comb[1], "indications_and_usage"].values[0], nlp))

        # set weight of edges to reflect closeness of relationships (inverse)
        relation = len(set1.intersection(set2))
        venn_G.add_edge(comb[0], comb[1], weight=relation)

    print("Number of edges: ", str(len(venn_G.edges("cde70a36-38fa-44b4-9036-d6a5b1ea7a48"))))

    # Display the network graph from the venn diagram interactions
    plot = figure(title="Network of Top Similar Drugs by Indications and Usage", x_range=(-5, 5), y_range=(-5, 5),
                  tools="pan,lasso_select,box_select", toolbar_location="right")
    # generate hover capabilities
    node_hover_tool = HoverTool(tooltips=[("name", "@name"), ("purpose", "@purpose")], show_arrow = False)
    plot.add_tools(node_hover_tool, BoxZoomTool(), ResetTool())

    graph = from_networkx(venn_G, nx.spring_layout, scale=2, center=(0, 0))

    # vary thickness with node length
    # weight influence helped by https://stackoverflow.com/questions/49136867/networkx-plotting-in-bokeh-how-to-set-edge-width-based-on-graph-edge-weight
    graph.edge_renderer.data_source.data["line_width"] = [venn_G.get_edge_data(a, b)['weight']/10 for a, b in
                                                                   venn_G.edges()]
    graph.edge_renderer.glyph.line_width = {'field': 'line_width'}

    # graph settings
    graph.node_renderer.glyph = Circle(size=15, fill_color="blue")
    graph.edge_renderer.hover_glyph = MultiLine(line_color="red", line_width={'field': 'line_width'})
    graph.inspection_policy = NodesAndLinkedEdges()

    plot.renderers.append(graph)

    # script, div = components(plot)
    # print(script)
    # print(div)

    output_file("networkx_graph.html")
    show(plot)

if __name__ == "__main__":
    main()
