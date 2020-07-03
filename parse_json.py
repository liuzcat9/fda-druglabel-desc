import time
import ijson
import pandas as pd
import numpy as np

# read stored csv file
def read_zip(path):
    read_t0 = time.time()
    read_df = pd.read_csv("drug_df.zip", compression="zip")
    read_t1 = time.time()
    print("Read:", str(read_t1 - read_t0))
    return read_df

def parse_and_write_zip(file_list):
    parse_t0 = time.time()

    df_cols = ["purpose", "id", "brand_name", "product_type", "route", "indications_and_usage", "contraindications"]
    drug_df = pd.DataFrame(columns=df_cols)

    for filename in file_list:
        current = []
        nested_current = []
        with open(filename) as file:
            objects = ijson.items(file, "results.item")

            # read wanted information into dataframe
            # openfda is in every entry of subset of data
            for o in objects:
                nested_current.append([o["purpose"][0] if "purpose" in o.keys() else None,
                                       o["id"] if "id" in o.keys() else None,
                                       o["openfda"]["brand_name"][0] if ("brand_name" in o["openfda"].keys()) else None,
                                       o["openfda"]["product_type"][0] if "product_type" in o[
                                           "openfda"].keys() else None,
                                       o["openfda"]["route"][0] if "route" in o["openfda"].keys() else None,
                                       o["indications_and_usage"][0] if "indications_and_usage" in o.keys() else None,
                                       o["contraindications"][0] if "contraindications" in o.keys() else None])

                current.append(o)

        # compile master dataframe
        current_df = pd.DataFrame(nested_current, columns=df_cols)
        drug_df = pd.concat([drug_df, current_df])

    parse_t1 = time.time()
    print("Parse:", str(parse_t1 - parse_t0))

    write_t0 = time.time()
    opts = dict(method="zip", archive_name="drug_df.csv")
    drug_df.to_csv("drug_df.zip", compression=opts)
    write_t1 = time.time()

    print("Write:", str(write_t1 - write_t0))