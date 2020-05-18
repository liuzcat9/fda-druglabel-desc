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

# Takes in a list
# Removes the word "Purpose" and punctuation
# Returns cleaned list
def clean_list(purpose_list, mundane):
    cleaned_list = []
    for word in purpose_list:
        stripped_word = word.translate(str.maketrans('', '', string.punctuation)).lower()  # C strip punctuation
        if (stripped_word != "purpose" and stripped_word != "" and stripped_word not in mundane):
            cleaned_list.append(stripped_word)
    return cleaned_list

# 1. Finds top most searched purposes
def rank_purpose(drug1_trunc, mundane):
    purposes = dict()
    for item in drug1_trunc:
        if ("purpose" in item.keys()):
            phrase_purpose = clean_list(item["purpose"][0].split(), mundane)

            # add keywords into purpose
            for word in phrase_purpose:
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

# helper function to find product names of a similar purpose
# returns: list of similar product names
def find_similar_products_purpose(drug1_products, original_purpose_list, original, mundane):
    similar_products = []
    for purpose in original_purpose_list:

        # find same purpose in other products
        for product in drug1_products:
            if (product != original and "purpose" in drug1_products[product].keys()):
                new_purpose = clean_list(drug1_products[product]["purpose"][0].split(), mundane)

                for word in new_purpose:
                    if (word == purpose):
                        similar_products.append(product)

    return similar_products

# helper function to count and record matching words in indications and usage field
# returns: dictionary of product names and counts of matches
def find_matching_indic(drug1_products, similar_products, original_indic_set):
    # within similar purpose products, find the ones with top similarities in indications and usage
    top_indic = dict()

    for similar_product in similar_products:
        if ("indications_and_usage" in drug1_products[similar_product].keys()):  # check for existence of key
            similar_indic_list = set(drug1_products[similar_product]["indications_and_usage"][0].split())
            top_indic[similar_product] = len(original_indic_set.intersection(similar_indic_list))

    return top_indic

# main
def main():
    # # source: https://open.fda.gov/downloads/
    # # load full json file
    # t0 = time.time()
    # with open("drug-label-0001-of-0008.json") as file:
    #     drug1 = json.load(file)
    # t1 = time.time()
    # print(t1 - t0)
    #
    # print(len(drug1["results"])) # how many drugs there are
    #
    # # extract n drugs' information
    # n_trunc = len(drug1["results"])
    # drug1_trunc = drug1["results"][0:n_trunc - 1]
    #
    # # use source: https://gist.github.com/deekayen/4148741 to eliminate mundane words
    # with open("common_english.txt") as f:
    #     mundane = f.read().splitlines() # read without newline character
    #
    # # 1: generate key purposes
    # rank_purpose(drug1_trunc, mundane)
    #
    # # 2: Venn Diagram similarity between most similar drugs based on purpose keyword/indication matches
    # # reorganize dictionary such that brand names appear as the key
    # drug1_products = dict()
    # for item in drug1_trunc:
    #     if ("openfda" in item.keys() and "brand_name" in item["openfda"].keys() and "purpose" in item.keys()): # have to check for nonexistent data
    #         drug1_products[item["openfda"]["brand_name"][0]] = item
    #
    # # choose a drug
    # original = "H. Pylori Plus"
    # # find drug purpose
    # original_purpose = drug1_products[original]["purpose"]
    # # find top purpose keyword
    # original_purpose_list = clean_list(original_purpose[0].split(), mundane)
    #
    # # find similar products by purpose
    # similar_products = find_similar_products_purpose(drug1_products, original_purpose_list, original, mundane)
    #
    # # within similar products, find matching indications
    # original_indic_set = set(drug1_products[original]["indications_and_usage"][0].split())
    # top_indic = find_matching_indic(drug1_products, similar_products, original_indic_set)
    #
    # # find top "similar indications, same-purpose medications"
    # top_indic_df = pd.DataFrame(top_indic.items(), columns=["Product", "Match_Words"])
    # top_indic_df = top_indic_df.sort_values(by="Match_Words", ascending=False)
    #
    # print(top_indic_df)
    # print(original_indic_set)
    #
    # # plot top 3 by matching indications to original product
    # plt.figure()
    # venn = matplotlib_venn.venn3([original_indic_set,
    #                        set(drug1_products[top_indic_df.iloc[0]['Product']]["indications_and_usage"][0].split()),
    #                        set(drug1_products[top_indic_df.iloc[1]['Product']]["indications_and_usage"][0].split())],
    #                       set_labels=[original, top_indic_df.iloc[0]['Product'], top_indic_df.iloc[1]['Product']])
    # plt.title("Indication Word Matches in Common Purpose Product Labels", fontsize=10)
    # plt.show()
    #
    # # 3. Generate a graph node network of top products and their similarity to each other
    # # permute through all combinations of original and top 2 products to numerically obtain the venn diagram similarities
    # product_list = [original]
    # product_list.extend(list(top_indic_df.iloc[0:5, 0]))
    # itr_products = product_list
    # product_comb = list(itertools.combinations(itr_products, 2)) # convert permutations of product names into list
    #
    # # Generate a network graph from the venn diagram interactions
    # venn_G = nx.Graph()
    #
    # # iterate through products and generate nodes with attributes
    # for product in itr_products:
    #     venn_G.add_node(product, name=product)
    #
    # # iterate through all combinations and find similarities,
    # # generate graph with edges weighted by similarity
    # for comb in product_comb:
    #     set1 = set(drug1_products[comb[0]]["indications_and_usage"][0].split())
    #     set2 = set(drug1_products[comb[1]]["indications_and_usage"][0].split())
    #
    #     # set weight of edges to reflect closeness of relationships (inverse)
    #     relation = len(set1.intersection(set2))
    #     venn_G.add_edge(comb[0], comb[1], weight=relation)
    #
    # print(venn_G.edges(data=True))
    #
    # # Display the network graph from the venn diagram interactions
    # plot = figure(title="Network of Top Similar Drugs by Indications and Usage", x_range=(-5, 5), y_range=(-5, 5),
    #               tools="", toolbar_location=None)
    # # generate hover capabilities
    # node_hover_tool = HoverTool(tooltips=[("name", "@name")])
    # plot.add_tools(node_hover_tool, BoxZoomTool(), ResetTool())
    #
    # graph = from_networkx(venn_G, nx.spring_layout, scale=2, center=(0, 0))
    #
    # # vary thickness with node length
    # # weight influence helped by https://stackoverflow.com/questions/49136867/networkx-plotting-in-bokeh-how-to-set-edge-width-based-on-graph-edge-weight
    # graph.edge_renderer.data_source.data["line_width"] = [venn_G.get_edge_data(a, b)['weight']/10 for a, b in
    #                                                                venn_G.edges()]
    # graph.edge_renderer.glyph.line_width = {'field': 'line_width'}
    #
    # # graph settings
    # graph.node_renderer.glyph = Circle(size=15, fill_color="blue")
    # graph.edge_renderer.hover_glyph = MultiLine(line_color="red", line_width={'field': 'line_width'})
    # graph.inspection_policy = NodesAndLinkedEdges()
    #
    # plot.renderers.append(graph)
    #
    # output_file("networkx_graph.html")
    # show(plot)

    # reading using ijson
    t0 = time.time()
    drug1 = []
    drug1_trunc = []
    with open("drug-label-0001-of-0008.json") as file:
        objects = ijson.items(file, "results.item")
        drug1 = [o for o in objects]

    t1 = time.time()

    print(t1 - t0)

    # drug1 is a list of all viable objects
    print(len(drug1)) # how many drugs there are

    # use source: https://gist.github.com/deekayen/4148741 to eliminate mundane words
    with open("common_english.txt") as f:
        mundane = f.read().splitlines() # read without newline character

    # 1: generate key purposes
    rank_purpose(drug1, mundane)

    # 2: Venn Diagram similarity between most similar drugs based on purpose keyword/indication matches
    # reorganize dictionary such that brand names appear as the key
    drug1_products = dict()
    for item in drug1:
        if ("openfda" in item.keys() and "brand_name" in item["openfda"].keys() and "purpose" in item.keys()): # have to check for nonexistent data
            drug1_products[item["openfda"]["brand_name"][0]] = item

    # choose a drug
    original = "H. Pylori Plus"
    # find drug purpose
    original_purpose = drug1_products[original]["purpose"]
    # find top purpose keyword
    original_purpose_list = clean_list(original_purpose[0].split(), mundane)

    # find similar products by purpose
    similar_products = find_similar_products_purpose(drug1_products, original_purpose_list, original, mundane)

    # within similar products, find matching indications
    original_indic_set = set(drug1_products[original]["indications_and_usage"][0].split())
    top_indic = find_matching_indic(drug1_products, similar_products, original_indic_set)

    # find top "similar indications, same-purpose medications"
    top_indic_df = pd.DataFrame(top_indic.items(), columns=["Product", "Match_Words"])
    top_indic_df = top_indic_df.sort_values(by="Match_Words", ascending=False)

    print(top_indic_df)
    print(original_indic_set)

    # plot top 3 by matching indications to original product
    plt.figure()
    venn = matplotlib_venn.venn3([original_indic_set,
                           set(drug1_products[top_indic_df.iloc[0]['Product']]["indications_and_usage"][0].split()),
                           set(drug1_products[top_indic_df.iloc[1]['Product']]["indications_and_usage"][0].split())],
                          set_labels=[original, top_indic_df.iloc[0]['Product'], top_indic_df.iloc[1]['Product']])
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
        set1 = set(drug1_products[comb[0]]["indications_and_usage"][0].split())
        set2 = set(drug1_products[comb[1]]["indications_and_usage"][0].split())

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
