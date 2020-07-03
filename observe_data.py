import parse_json
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    # parse and write
    if not os.path.isfile("drug_df.zip"):
        file_list = ["drug-label-0001-of-0008.json", "drug-label-0002-of-0008.json"]
        parse_json.parse_and_write_zip(file_list)

    # read stored csv file
    drug_df = parse_json.read_zip("drug_df.zip")

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

    # test output
    print(drug_df.isnull().any(axis=0))
    print(drug_df)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(drug_df.loc[drug_df["id"] == "1c0bf028-e240-4667-887b-7a7695196fe1", :])



if __name__ == "__main__":
    main()