import time, os
import pandas as pd, numpy as np
import ijson
import collections, itertools

import gensim
from gensim.test.utils import get_tmpfile
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform

# custom files
import preprocessing, parse_json, main

# parse json into csv (dataframes)
def write_and_read_train_data(file_list, filename):
    if not os.path.isfile("pkl/" + filename + ".pkl"):
        write_train_data(file_list, filename)

    return read_train_data(filename)

# parse dataframe entries into tokenized words only to save
def write_and_read_tokenized_data(full_df, filename):
    if not os.path.isfile("pkl/" + filename + ".pkl"):
        # tokenize
        columns_to_tokenize = ["active_ingredient", "inactive_ingredient", "warnings", "mechanism_of_action",
                               "dosage_and_administration", "indications_and_usage", "contraindications"]
        train_df = preprocessing.tokenize_columns(full_df, columns_to_tokenize)

        # write
        write_tokenized_data(train_df, filename)

    return read_tokenized_data(filename)

# write tokenized version of non-purpose data
def write_tokenized_data(train_df, filename):
    write_t0 = time.time()
    train_df.to_pickle("pkl/" + filename + ".pkl")
    print("Write tokenized data:", str(time.time() - write_t0))

# read tokenized version of non-purpose data
def read_tokenized_data(filename):
    read_t0 = time.time()
    train_df = pd.read_pickle("pkl/" + filename + ".pkl")
    print("Read tokenized data:", str(time.time() - read_t0))

    return train_df

# write raw data from json into dataframe
def write_train_data(file_list, filename):
    parse_t0 = time.time()
    df_cols = ["id", "package_label_principal_display_panel",
               "active_ingredient", "inactive_ingredient", "warnings",
               "brand_name", "product_type", "route",
               "mechanism_of_action", "clinical_pharmacology",
               "dosage_and_administration", "indications_and_usage", "contraindications"]
    train_df = pd.DataFrame(columns=df_cols)

    # extract wanted information from json files
    for parse_file in file_list:
        print("Parsing train: ", parse_file)
        current = []
        nested_current = []
        with open(parse_file) as file:
            objects = ijson.items(file, "results.item")

            # read wanted information into dataframe
            # openfda is in every entry of subset of data
            for o in objects:
                if "purpose" not in o.keys() or o["purpose"][0] == "": # pull specifically data NOT used for display
                    nested_current.append([o["id"] if "id" in o.keys() else None,
                                       o["package_label_principal_display_panel"][0]
                                       if "package_label_principal_display_panel" in o.keys() else None,
                                       o["active_ingredient"][0] if "active_ingredient" in o.keys() else None,
                                       o["inactive_ingredient"][0] if "inactive_ingredient" in o.keys() else None,
                                       o["warnings"][0] if "warnings" in o.keys() else None,
                                       o["openfda"]["brand_name"][0] if ("brand_name" in o["openfda"].keys()) else None,
                                       o["openfda"]["product_type"][0] if "product_type" in o[
                                           "openfda"].keys() else None,
                                       o["openfda"]["route"][0] if "route" in o["openfda"].keys() else None,
                                        o["mechanism_of_action"][0] if "mechanism_of_action" in o.keys() else None,
                                           o["clinical_pharmacology"][0] if "clinical_pharmacology" in o.keys() else None,
                                       o["dosage_and_administration"][0] if "dosage_and_administration" in o.keys() else None,
                                       o["indications_and_usage"][0] if "indications_and_usage" in o.keys() else None,
                                       o["contraindications"][0] if "contraindications" in o.keys() else None])

                current.append(o)

        # compile master dataframe
        current_df = pd.DataFrame(nested_current, columns=df_cols)
        train_df = pd.concat([train_df, current_df])

    print("Parse train:", str(time.time() - parse_t0))

    # pickle it
    train_df.to_pickle("pkl/" + filename + ".pkl")
    print("Wrote:", "pkl/" + filename + ".pkl")

# read training data from pkl
def read_train_data(filename):
    read_t0 = time.time()
    train_df = pd.read_pickle("pkl/" + filename + ".pkl")
    print("Read train:", time.time() - read_t0)
    return train_df

# tokenize data for training into model
def process_training_data(train_df, field):
    # split relevant field column into lists of data
    split_t0 = time.time()
    train_field = train_df.dropna(subset=[field]) # only train on non-null entries
    train_field = train_field[field].tolist()
    print("Length of training data:", str(len(train_field)))

    # convert to tagged documents
    train_corpus = []
    for i, tokens in enumerate(train_field):
        train_corpus.append(gensim.models.doc2vec.TaggedDocument(tokens.split(), [i]))
    print("Number of training token lists:", str(len(train_corpus)))
    print("Split into TaggedDocuments:", str(time.time() - split_t0))

    return train_corpus

# train the model on the split training data
def train_and_save_model(train_corpus, field):
    # train
    build_t0 = time.time()
    model = gensim.models.doc2vec.Doc2Vec(vector_size=100, min_count=1, epochs=40)
    model.build_vocab(train_corpus)
    print("Create and build model:", str(time.time() - build_t0))

    train_t0 = time.time()
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    print("Train model:", str(time.time() - train_t0))

    # save
    save_t0 = time.time()
    model.save(field + ".model")
    # reduce memory
    # model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    print("Save model:", str(time.time() - save_t0))

# load doc2vec model for a field
def load_doc2vec(field):
    load_t0 = time.time()
    model = gensim.models.doc2vec.Doc2Vec.load(field + ".model")
    print("Load model for checkup:", str(time.time() - load_t0))

    return model

# ensure model is working
def check_model(train_corpus, field):
    # load model
    model = load_doc2vec(field)

    print(model.wv.most_similar("sanitizer"))

    # from gensim doc2vecmentation, ensure similarity
    ranks = []
    second_ranks = []
    rank_t0 = time.time()
    for doc_id in range(len(train_corpus)):
        inferred_vector = model.infer_vector(train_corpus[doc_id].words)
        similars = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
        rank = [id for id, similar in similars].index(doc_id)
        ranks.append(rank)

        second_ranks.append(similars[1])

    print("Ranked:", str(time.time() - rank_t0))

    counter_t0 = time.time()
    # count number of ranking most similars
    counter = collections.Counter(ranks)
    print(counter)
    print("Count:", str(time.time() - counter_t0))

    # do another arbitrary word comparison, from gensim documentation
    # doc_id = 2
    # print('Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
    # print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
    # for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(similars) // 2), ('LEAST', len(similars) - 1)]:
    #     print(u'%s %s: «%s»\n' % (label, similars[index], ' '.join(train_corpus[similars[index][0]].words)))

# run model on testing data
def test_model(test_df, purpose, field):
    # only compute pairwise similarity between combinations of documents of a particular purpose/field
    # select only inner purposes with valid field
    purpose_field = test_df.dropna(subset=["id", "brand_name", "route", "product_type", field])
    purpose_field = purpose_field.loc[purpose_field["purpose_cluster"].str.contains(purpose)]
    purpose_field = purpose_field.reset_index()
    print("Number of valid field rows:", str(len(purpose_field)))

    # construct attributes
    num_to_name, attr_dict = main.generate_attr_mappings(purpose_field, purpose_field["id"].tolist())

    # load model
    model = load_doc2vec(field)

    corpus_t0 = time.time()
    # create list of tokens to compare (based on indices)
    test_list = list(map(str.split, purpose_field[field].tolist()))
    # convert test list into tagged_documents
    test_corpus = []
    for i, test_tokens in enumerate(test_list):
        test_corpus.append(gensim.models.doc2vec.TaggedDocument(test_tokens, [i]))
    # remove unheard of vocabulary (also happens because of typos)
    print("Create test corpus:", str(time.time() - corpus_t0))

    print("Create list of tokens:", str(len(test_corpus)))
    print(test_corpus[0].words)

    # create all test vectors at once for comparison
    vectors_t0 = time.time()
    # use list of words in TaggedDocument and reshape as 1 feature to conform to sklearn
    # n_samples x n_features
    test_vectors = np.array([model.infer_vector(para.words) for para in test_corpus])
    print("Create test vectors:", str(time.time() - vectors_t0))
    print(test_vectors.shape)

    # create adjacency matrix
    adj_t0 = time.time()
    adj_mat = cosine_similarity(test_vectors, test_vectors)
    print("Verify adj_mat:")
    print(adj_mat)

    # add weights now
    adj_mat[adj_mat != 0] = (adj_mat[adj_mat != 0] + .1) * 2
    print("Adjusted weights:")
    print(adj_mat)

    # avoid redundant information and send networkx only lower triangle
    adj_mat = np.tril(adj_mat)
    np.fill_diagonal(adj_mat, 0)  # mask 1's (field similarity to itself)
    print("Create adjacency matrix:", str(time.time() - adj_t0))
    print(adj_mat, adj_mat.shape)

    # create sparse matrix
    sparse_mat = main.restrict_adjacency_matrix(adj_mat, num_to_name, 8)

    venn_G = main.generate_purpose_graph(sparse_mat, attr_dict, num_to_name)

    return (venn_G, adj_mat, attr_dict, num_to_name)

if __name__ == "__main__":
    field = "indications_and_usage"
    json_list = ["drug-label-0001-of-0009.json", "drug-label-0002-of-0009.json", "drug-label-0003-of-0009.json",
                 "drug-label-0004-of-0009.json", "drug-label-0005-of-0009.json", "drug-label-0006-of-0009.json",
                 "drug-label-0007-of-0009.json", "drug-label-0008-of-0009.json", "drug-label-0009-of-0009.json"]
    full_df = write_and_read_train_data(json_list, "full_train_df")
    train_df = write_and_read_tokenized_data(full_df, "tokenized_train_df")
    train_corpus = process_training_data(train_df, field)
    # train_and_save_model(train_corpus, field)
    # check_model(train_corpus, field)

    test_df = parse_json.obtain_preprocessed_drugs(json_list, "purpose_full_drug_df")
    purpose = "sunscreen purposes uses protectant skin"
    venn_G, adj_mat, attr_dict, num_to_name = test_model(test_df, purpose, field)

    main.generate_graph_plot(venn_G, purpose, field)
