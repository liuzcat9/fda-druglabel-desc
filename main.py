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
    # join all text into big string
    purpose_str = " ".join(drug1_trunc["purpose"])
    nlp = English()
    purpose_list = clean_list(purpose_str, nlp)

    purposes = dict()
    for word in purpose_list:
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
    valid_df = valid_df.loc[valid_df["brand_name"] != original] # exclude self

    similar_products = set()
    for purpose in original_purpose_list:
        similar_products.update(valid_df.loc[valid_df["purpose"].str.contains(purpose), "brand_name"].values.tolist())

    return list(similar_products)

# helper function to count and record matching words in indications and usage field
# returns: dictionary of product names and counts of matches
def find_matching_indic(drug1_df, similar_products, original):
    nlp = English()
    valid_df = drug1_df.dropna()  # exclude all rows with columns of null
    valid_df = valid_df.loc[valid_df["brand_name"] != original]  # exclude self

    # set original indications
    original_indic_set = set(clean_list(drug1_df.loc[drug1_df["brand_name"] == original, "indications_and_usage"].values[0], nlp))
    print(original_indic_set)
    # within similar purpose products, find the ones with top similarities in indications and usage
    top_indic = dict()

    for similar_product in similar_products:
        # data frame already wiped of invalid null rows
        similar_indic_list = set(clean_list(valid_df.loc[valid_df["brand_name"] == similar_product, "indications_and_usage"].values[0], nlp))
        top_indic[similar_product] = len(original_indic_set.intersection(similar_indic_list))

    top_indic_df = pd.DataFrame(top_indic.items(), columns=["Product", "Match_Words"])
    top_indic_df = top_indic_df.sort_values(by="Match_Words", ascending=False)

    return top_indic_df

# main
def main():
    nlp = English()
    # reading using ijson
    t0 = time.time()
    drug1 = []
    drug1_temp = []
    with open("drug-label-0001-of-0008.json") as file:
        objects = ijson.items(file, "results.item")

        # obtain information in a dataframe
        for o in objects:
            drug1_temp.append([o["purpose"][0] if "purpose" in o.keys() else None,
                          o["openfda"]["brand_name"][0] if ("openfda" in o.keys() and "brand_name" in o["openfda"].keys()) else None,
                          o["indications_and_usage"][0] if "indications_and_usage" in o.keys() else None])
            drug1.append(o)
    t1 = time.time()

    print(t1 - t0)

    # visualize dataframe of drug info
    drug1_df = pd.DataFrame(drug1_temp, columns = ["purpose", "brand_name", "indications_and_usage"])

    # drug1 is a list of all viable objects
    print(drug1_df) # how many drugs there are

    # # use source: https://gist.github.com/deekayen/4148741 to eliminate mundane words
    # with open("common_english.txt") as f:
    #     mundane = f.read().splitlines() # read without newline character

    # 1: generate key purposes
    rank_purpose(drug1_df)

    print(len(drug1_df))

    # 2: Venn Diagram similarity between most similar drugs based on purpose keyword/indication matches
    # choose a drug
    original = "H. Pylori Plus"
    # find drug purpose
    original_purpose = drug1_df.loc[drug1_df["brand_name"] == original, "purpose"].values[0]
    # find top purpose keyword
    original_purpose_list = clean_list(original_purpose, nlp)

    # find similar products by purpose
    similar_products = find_similar_products_purpose(drug1_df, original_purpose_list, original)

    # within similar products, find matching indications for same-purpose medications
    top_indic_df = find_matching_indic(drug1_df, similar_products, original)

    original_indic_set = set(
        clean_list(drug1_df.loc[drug1_df["brand_name"] == original, "indications_and_usage"].values[0], nlp))

    print(top_indic_df)

    # plot top 3 by matching indications to original product
    plt.figure()
    venn = matplotlib_venn.venn3([original_indic_set,
                                  set(clean_list(drug1_df.loc[drug1_df["brand_name"] == top_indic_df.iloc[0]['Product'], "indications_and_usage"].values[0], nlp)),
                                  set(clean_list(drug1_df.loc[drug1_df["brand_name"] == top_indic_df.iloc[1]['Product'], "indications_and_usage"].values[0], nlp))],
                                  set_labels = [original, top_indic_df.iloc[0]['Product'], top_indic_df.iloc[1]['Product']])
    plt.title("Indication Word Matches in Common Purpose Product Labels", fontsize=10)
    plt.show()

    # 3. Generate a graph node network of top products and their similarity to each other
    # permute through all combinations of original and top 2 products to numerically obtain the venn diagram similarities
    product_list = [original]
    product_list.extend(list(top_indic_df.iloc[0:5, 0]))
    itr_products = product_list
    product_comb = list(itertools.combinations(itr_products, 2)) # convert permutations of product names into list

    # Generate a network graph from the venn diagram interactions
    venn_G = nx.Graph()

    # iterate through products and generate nodes with attributes
    for product in itr_products:
        venn_G.add_node(product, name=product)

    # iterate through all combinations and find similarities,
    # generate graph with edges weighted by similarity
    for comb in product_comb:
        set1 = set(clean_list(drug1_df.loc[drug1_df["brand_name"] == comb[0], "indications_and_usage"].values[0], nlp))
        set2 = set(clean_list(drug1_df.loc[drug1_df["brand_name"] == comb[1], "indications_and_usage"].values[0], nlp))

        # set weight of edges to reflect closeness of relationships (inverse)
        relation = len(set1.intersection(set2))
        venn_G.add_edge(comb[0], comb[1], weight=relation)

    print(venn_G.edges(data=True))

    # Display the network graph from the venn diagram interactions
    plot = figure(title="Network of Top Similar Drugs by Indications and Usage", x_range=(-5, 5), y_range=(-5, 5),
                  tools="", toolbar_location=None)
    # generate hover capabilities
    node_hover_tool = HoverTool(tooltips=[("name", "@name")])
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

    output_file("networkx_graph.html")
    show(plot)

if __name__ == "__main__":
    main()
