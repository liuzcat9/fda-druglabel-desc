import json
import string
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib_venn

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

# main
def main():
    # source: https://open.fda.gov/downloads/
    with open("drug-label-0001-of-0008.json") as file:
        drug1 = json.load(file)

    print(len(drug1["results"])) # how many drugs there are

    # extract n drugs' information
    n_trunc = len(drug1["results"])
    drug1_trunc = drug1["results"][0:n_trunc - 1]

    # 1: generate key purposes
    purposes = dict()

    # use source: https://gist.github.com/deekayen/4148741 to eliminate mundane words
    with open("common_english.txt") as f:
        mundane = f.read().splitlines() # read without newline character

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

    # 2: Venn Diagram similarity between most similar drugs based on purpose keyword/indication matches
    # reorganize dictionary such that product names appear as the key
    drug1_products = dict()
    for item in drug1_trunc:
        if ("package_label_principal_display_panel" in item.keys()): # have to check for nonexistent data
            drug1_products[item["package_label_principal_display_panel"][0]] = item

    # choose a drug
    original = "CARMEX COLD SORE treatment External Analgesic 2g (10210-0012-0) InnerLabel OuterLabel"

    # find drug purpose
    original_purpose = drug1_products[original]["purpose"]
    # find top purpose keyword
    original_purpose_list = clean_list(original_purpose[0].split(), mundane)

    for purpose in original_purpose_list:
        similar_products = []

        # find same purpose in other products
        for product in drug1_products:
            if (product != original and "purpose" in drug1_products[product].keys()):
                new_purpose = clean_list(drug1_products[product]["purpose"][0].split(), mundane)

                for word in new_purpose:
                    if (word == purpose):
                        similar_products.append(product)

    # within similar purpose products, find the ones with top similarities in indications and usage
    top_indic = dict()
    original_indic_list = set(drug1_products[original]["indications_and_usage"][0].split())
    for similar_product in similar_products:
        if ("indications_and_usage" in drug1_products[similar_product].keys()): # check for existence of key
            similar_indic_list = set(drug1_products[similar_product]["indications_and_usage"][0].split())
            top_indic[similar_product] = len(original_indic_list.intersection(similar_indic_list))

    # find top 5 "similar indications, same-purpose medications"
    top_indic_df = pd.DataFrame(top_indic.items(), columns=["Product", "Match_Words"])
    top_indic_df = top_indic_df.sort_values(by="Match_Words", ascending=False)

    print(top_indic_df)

    plt.figure()
    venn = matplotlib_venn.venn3([original_indic_list,
                           set(drug1_products[top_indic_df.iloc[0]['Product']]["indications_and_usage"][0].split()),
                           set(drug1_products[top_indic_df.iloc[1]['Product']]["indications_and_usage"][0].split())],
                          set_labels=[original, top_indic_df.iloc[0]['Product'], top_indic_df.iloc[1]['Product']])
    plt.title("Indication Word Matches in Common Purpose Product Labels", fontsize=10)
    plt.show()


if __name__ == "__main__":
    main()
