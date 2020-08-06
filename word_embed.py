import time, os
import pandas as pd, numpy as np
import ijson
import collections, itertools

import gensim
from gensim.test.utils import get_tmpfile
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans

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
                if "purpose" not in o.keys() or o["purpose"][0] == "" and "indications_and_usage" not in o.keys(): # pull specifically data NOT used for clustering
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
    # obtain all fields of similar_products (only relevant products to purpose)
    purpose_df = main.find_df_fitting_purpose(test_df, purpose, field)
    print("Number of valid field rows:", str(len(purpose_df)))

    # load model
    model = load_doc2vec(field)

    test_corpus = create_test_corpus(purpose_df)

    # create all test vectors at once for comparison
    vectors_t0 = time.time()
    # use list of words in TaggedDocument and reshape as 1 feature to conform to sklearn
    # n_samples x n_features
    testX = np.array([model.infer_vector(para.words) for para in test_corpus])
    print("Create test vectors:", str(time.time() - vectors_t0))
    print(testX.shape)

    # run density model
    density_cluster = main.run_density_cluster(testX)

    # build ordered dictionary of cluster:corresponding drug numbers
    density_ref = main.build_ordered_cluster_dict(density_cluster.labels_)

    # calculate adjacency matrix of supernodes by averaging cosine similarity of edges in cluster
    # create adjacency matrix
    full_mat = cosine_similarity(testX, testX)
    print("Verify full_mat:")
    print(full_mat)

    adj_mat = main.calculate_super_adj_mat(density_ref, full_mat)

    # add weights now
    adj_mat[adj_mat != 0] = (adj_mat[adj_mat != 0] + .1) * 2

    # create mapping dictionaries for cluster-based graph
    num_to_cluster, attr_dict = main.generate_super_attr_mappings(purpose_df, density_ref)

    # adj_mat = main.weight_and_process_adj_mat(adj_mat)

    # assign attributes
    # num_to_name, attr_dict = main.generate_attr_mappings(purpose_df)

    # generate graph with top n edges per node weighted by similarity
    # create sparse adjacency matrix, removing edges
    sparse_mat = main.restrict_adjacency_matrix(adj_mat, 8)
    # only compute pairwise similarity between combinations of documents of a particular purpose/field
    # select only inner purposes with valid field

    # # avoid redundant information and send networkx only lower triangle
    # adj_mat = np.tril(adj_mat)
    # np.fill_diagonal(adj_mat, 0)  # mask 1's (field similarity to itself)
    # print("Create adjacency matrix:", str(time.time() - adj_t0))
    print(adj_mat, adj_mat.shape)
    print(sparse_mat)

    return (adj_mat, sparse_mat, attr_dict, num_to_cluster)

# use the topics clustering method with doc2vec information
def test_model_by_topics(test_df, purpose, field):
    # obtain all fields of similar_products (only relevant products to purpose)
    purpose_df = main.find_df_fitting_purpose(test_df, purpose, field)
    print("Number of valid field rows:", str(len(purpose_df)))

    field_list = purpose_df[field].tolist()
    print(field_list[0:10])

    print(purpose_df)

    # preprocess for LDA
    count_data, words = main.fit_count_vectorizer(field_list)

    best_lda, lda_prob = main.fit_LDA(count_data)
    print(lda_prob)

    topics_dict = main.get_topics_word_dict(best_lda, words)

    # create documents (in order) clustered into topics
    doc_topic_list = main.create_doc_topic_cluster(best_lda, lda_prob)
    print(doc_topic_list)

    # build ordered dictionary of cluster:corresponding drug numbers
    cluster_ref = main.build_ordered_cluster_dict(doc_topic_list)

    # some documents are not assigned to all topics, so reset index
    cluster_ref, topics_dict = main.reset_cluster_index(cluster_ref, topics_dict)

    # load model
    model = load_doc2vec(field)

    test_corpus = create_test_corpus(purpose_df, field)

    # create all test vectors at once for comparison
    vectors_t0 = time.time()
    # use list of words in TaggedDocument and reshape as 1 feature to conform to sklearn
    # n_samples x n_features
    testX = np.array([model.infer_vector(para.words) for para in test_corpus])
    print("Create test vectors:", str(time.time() - vectors_t0))
    print(testX.shape)

    # calculate adjacency matrix of supernodes by averaging cosine similarity of edges in cluster
    # create adjacency matrix
    full_mat = cosine_similarity(testX, testX)
    print("Verify full_mat:")
    print(full_mat)

    adj_mat = main.calculate_super_adj_mat(cluster_ref, full_mat)

    # create mapping dictionaries for cluster-based graph
    num_to_cluster, attr_dict = main.generate_super_attr_mappings(purpose_df, cluster_ref, topics_dict)

    # create full comprehensive html descriptions based on topics
    num_to_html = main.generate_cluster_num_to_html(purpose_df, cluster_ref, field)

    # create sparse adjacency matrix by generating graph with top n edges per node weighted by similarity, removing edges
    max_n = 8
    sparse_mat = main.restrict_adjacency_matrix(adj_mat, max_n if adj_mat.shape[0] >= max_n else adj_mat.shape[0])

    print(adj_mat, adj_mat.shape)
    print(sparse_mat)

    return (adj_mat, sparse_mat, attr_dict, num_to_cluster, num_to_html)

# generate test_corpus TaggedDocuments
def create_test_corpus(purpose_df, field):
    corpus_t0 = time.time()
    # create list of tokens to compare (based on indices)
    test_list = list(map(str.split, purpose_df[field].tolist()))
    # convert test list into tagged_documents
    test_corpus = []
    for i, test_tokens in enumerate(test_list):
        test_corpus.append(gensim.models.doc2vec.TaggedDocument(test_tokens, [i]))

    print("Create test corpus:", str(time.time() - corpus_t0))

    print("Create list of tokens:", str(len(test_corpus)))
    print(test_corpus[0].words)

    return test_corpus

# train a model on purpose and cluster it based on doc2vectors
def train_and_cluster_purpose(purpose_df):
    # loaded dataframe has all purpose fields
    if not os.path.isfile("purpose.model"):
        train_corpus = process_training_data(purpose_df, "purpose")
        train_and_save_model(train_corpus, "purpose")

    # actual dataframe to sort by
    trunc_df = purpose_df.copy()
    trunc_df = trunc_df.dropna(
        subset=["id", "brand_name", "route", "product_type"])  # exclude all rows with columns of null

    model = load_doc2vec("purpose")
    # perform testing using essentially the same purpose data but truncated to valid entries
    test_corpus = create_test_corpus(trunc_df, "purpose")

    # create all test vectors at once for comparison
    vectors_t0 = time.time()
    # use list of words in TaggedDocument and reshape as 1 feature to conform to sklearn
    # n_samples x n_features
    testX = np.array([model.infer_vector(para.words) for para in test_corpus])
    print("Create test vectors:", str(time.time() - vectors_t0))
    print(testX.shape)

    # k-means cluster
    km, labels = perform_k_means(testX)
    print_docs_per_cluster(labels, test_corpus)

    # label with clusters on dataframe, then save
    trunc_df["purpose_cluster"] = labels
    trunc_df.sort_values(by=['purpose_cluster'])
    preprocessing.write_preprocessed_to_pkl(trunc_df, "doc2vec_purpose_drug_df")

# perform k-means clustering on doc vectors
def perform_k_means(vectorX):
    km_t0 = time.time()
    km = KMeans(n_clusters=50)
    km.fit(vectorX)
    print("Fit KMeans: ", str(time.time() - km_t0))

    print("Num KMeans labels: ", str(len(km.labels_)))

    return km, km.labels_

# print all purposes per cluster
def print_docs_per_cluster(labels, test_corpus):
    dict_t0 = time.time()
    # each index of docs corresponds to index of labels
    docs = [para.words for para in test_corpus]

    cluster_dict = {}

    for i, label in enumerate(labels):
        if label not in cluster_dict:
            cluster_dict[label] = []
        cluster_dict[label].append(docs[i])

    print("Create cluster dictionary of docs:", str(time.time() - dict_t0))
    f = open("cluster.txt", "w", encoding="utf-8")
    for key, value in cluster_dict.items():
        f.write('%s:%s\n' % (key, value))
    f.close()

if __name__ == "__main__":
    field = "warnings"
    json_list = ["drug-label-0001-of-0009.json", "drug-label-0002-of-0009.json", "drug-label-0003-of-0009.json",
                 "drug-label-0004-of-0009.json", "drug-label-0005-of-0009.json", "drug-label-0006-of-0009.json",
                 "drug-label-0007-of-0009.json", "drug-label-0008-of-0009.json", "drug-label-0009-of-0009.json"]

    # cluster by purpose
    # train_and_cluster_purpose(preprocessing.read_preprocessed_to_pkl("full_drug_df"))

    # full_df = write_and_read_train_data(json_list, "full_train_df")
    # train_df = write_and_read_tokenized_data(full_df, "tokenized_train_df")
    # train_corpus = process_training_data(train_df, field)
    # train_and_save_model(train_corpus, field)
    # check_model(train_corpus, field)

    total_t0 = time.time()
    test_df = parse_json.obtain_preprocessed_drugs(json_list, "purpose_indic_full_drug_df")
    purpose = "indicate usage patient symptom tablet"

    # adj_mat, sparse_mat, attr_dict, num_to_cluster = test_model(test_df, purpose, field)
    adj_mat, sparse_mat, attr_dict, num_to_cluster, num_to_html = test_model_by_topics(test_df, purpose, field)
    venn_G = main.generate_purpose_graph(sparse_mat, attr_dict, num_to_cluster)
    bokeh_script, bokeh_div = main.generate_graph_plot(venn_G, purpose, field, num_to_html, topics=True)

    main.save_html_template(bokeh_script, bokeh_div, purpose, field)

    print("Total test time:", str(time.time() - total_t0))
