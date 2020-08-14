import time, os
import ijson
import pandas as pd

# custom files
import preprocessing

# read stored csv file
def read_zip(path):
    read_t0 = time.time()
    read_df = pd.read_csv(path + ".zip", compression="zip")
    read_t1 = time.time()
    print("Read:", str(read_t1 - read_t0))
    return read_df

# parse json into dataframe
def parse_zip(file_list):
    parse_t0 = time.time()

    df_cols = ["purpose", "id", "package_label_principal_display_panel",
               "active_ingredient", "inactive_ingredient", "warnings",
               "brand_name", "product_type", "route",
               "mechanism_of_action", "clinical_pharmacology",
               "dosage_and_administration", "indications_and_usage", "contraindications"]
    drug_df = pd.DataFrame(columns=df_cols)

    # extract wanted information from json files
    for filename in file_list:
        print("Parsing: ", filename)
        current = []
        nested_current = []
        with open(filename) as file:
            objects = ijson.items(file, "results.item")

            # read wanted information into dataframe
            # openfda is in every entry of subset of data
            for o in objects:
                if ("purpose" in o.keys() and o["purpose"][0] != ""): # all represented data should have a purpose field
                    nested_current.append([o["purpose"][0] if "purpose" in o.keys() else None,
                                       o["id"] if "id" in o.keys() else None,
                                       o["package_label_principal_display_panel"][0] if "package_label_principal_display_panel" in o.keys() else None,
                                       o["active_ingredient"][0] if "active_ingredient" in o.keys() else None,
                                       o["inactive_ingredient"][0] if "inactive_ingredient" in o.keys() else None,
                                       o["warnings"][0] if "warnings" in o.keys() else None,
                                        # grammatically correct brand name
                                       o["openfda"]["brand_name"][0].strip().title() if ("brand_name" in o["openfda"].keys()) else None,
                                       o["openfda"]["product_type"][0] if "product_type" in o[
                                           "openfda"].keys() else None,
                                       o["openfda"]["route"][0] if "route" in o["openfda"].keys() else None,
                                        o["mechanism_of_action"][0] if "mechanism_of_action" in o.keys() else None,
                                           o["clinical_pharmacology"][0] if "clinical_pharmacology" in o.keys() else None,
                                       o["dosage_and_administration"][0] if "dosage_and_administration" in o.keys() else None,
                                           o["indications_and_usage"][
                                               0] if "indications_and_usage" in o.keys() else None,
                                       o["contraindications"][0] if "contraindications" in o.keys() else None])

                current.append(o)

        # compile master dataframe
        current_df = pd.DataFrame(nested_current, columns=df_cols)
        drug_df = pd.concat([drug_df, current_df])

    parse_t1 = time.time()
    print("Parse:", str(parse_t1 - parse_t0))

    return drug_df

# write file to zip containing csv
def write_zip(drug_df, path):
    write_t0 = time.time()
    opts = dict(method="zip", archive_name=path + ".csv")
    drug_df.to_csv(path + ".zip", compression=opts)
    write_t1 = time.time()

    print("Write:", str(write_t1 - write_t0))

# combines parse and write
def parse_and_write_zip(file_list, path):
    drug_df = parse_zip(file_list)
    write_zip(drug_df, path)

# combines read, parse, and write depending on files
def parse_and_read_drugs(json_list, filename, filetype="pickle"):
    if filetype == "csv":
        # parse and write if no file to read
        if not os.path.isfile("pkl/" + filename + ".zip"):
            parse_and_write_zip(json_list, filename)

        # read stored csv file
        return read_zip(filename)
    else:
        if not os.path.isfile("pkl/" + filename + ".pkl"):
            drug_df = parse_zip(json_list)

            # serialize file
            write_t0 = time.time()
            drug_df.to_pickle("pkl/" + filename + ".pkl", compression="zip")
            print("Write: ", str(time.time() - write_t0))

        # read file
        read_t0 = time.time()
        read_df = pd.read_pickle("pkl/" + filename + ".pkl", compression="zip")
        print("Read: ", str(time.time() - read_t0))
        return read_df

# choose preprocessed or raw drug files to sift through, ultimately returning preprocessed drug_df
def obtain_preprocessed_drugs(json_list, filename):
    # no preprocessed drugs, start from raw
    if not os.path.isfile("pkl/" + filename + ".pkl"):
        rawfile = filename.replace("purpose_", "")
        raw_drug_df = parse_and_read_drugs(json_list, rawfile, "pickle")

        # preprocess and write preprocessed dataframe
        preprocessing.preprocess_and_write_df(raw_drug_df, filename)

    return preprocessing.read_preprocessed_to_pkl(filename)
