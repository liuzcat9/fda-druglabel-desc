import parse_json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_product_type(drug_df):
    # plot OTC/Prescription/NaN
    product_type_counts = drug_df["product_type"].value_counts()
    percentages = [num / len(drug_df.index) * 100.0 for index, num in product_type_counts.items()]
    percentages.append((len(drug_df.index) - product_type_counts.sum()) / len(drug_df.index) * 100.0)
    labels = [index[:-5] for index, num in product_type_counts.items()]
    labels.append("UNLISTED")

    plt.figure()
    plt.pie(percentages, labels=labels, autopct='%1.1f%%')
    plt.title("Distribution of OTC/Prescription/Unlisted Drugs in Subset")
    plt.show()

def main():
    json_list = ["drug-label-0001-of-0009.json", "drug-label-0002-of-0009.json", "drug-label-0003-of-0009.json",
                 "drug-label-0004-of-0009.json", "drug-label-0005-of-0009.json", "drug-label-0006-of-0009.json",
                 "drug-label-0007-of-0009.json", "drug-label-0008-of-0009.json", "drug-label-0009-of-0009.json"]
    drug_df = parse_json.parse_or_read_drugs(json_list, "full_drug_df.zip")

    plot_product_type(drug_df)

    # test output
    print(drug_df.isnull().any(axis=0))
    print(drug_df)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(drug_df.loc[drug_df["id"] == "1c0bf028-e240-4667-887b-7a7695196fe1", :])



if __name__ == "__main__":
    main()