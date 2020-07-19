from flask import Flask, render_template, request, redirect
import sys, time

import observe_data, main, preprocessing

import pandas as pd

app = Flask(__name__)
drug_df = preprocessing.read_preprocessed_to_pkl("purpose_full_drug_df")

@app.route('/')
def index():
    purposes = ["Sunscreen", "Analgesic", "Pain Relief"]
    fields = ["Indications and Usage", "Route"]

    return render_template('index.html',
                           purposes=purposes, fields=fields,
                           obs=observe_data.test_observation(), drug_output=str(drug_df.head()))

@app.route('/res', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        purpose = request.form["purpose_res"]
        field = request.form["field_res"]

        graph_t0 = time.time()
        similar_products = main.find_similar_drugs_from_purpose(drug_df, "topical analgesic menthol suppressant cough", "indications_and_usage")

        # 3. Generate a graph node network of top products and their similarity to each other
        venn_G, adj_mat, attr_dict, num_to_name = \
            main.generate_graph_matching_field_of_purpose(drug_df, similar_products, "analgesic", "indications_and_usage")
        script, div = main.plot_purpose_graph(venn_G)

        print("Graph:", str(time.time() - graph_t0))

    return render_template('result.html', purpose=purpose, field=field, bokeh_script=script, bokeh_div=div)

if __name__ == '__main__':
    app.run(port=33507)
