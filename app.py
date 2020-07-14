from flask import Flask, render_template, request, redirect
import sys

import observe_data
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    purposes = ["Sunscreen", "Analgesic", "Pain Relief"]
    fields = ["Indications and Usage", "Route"]

    drug_df = pd.read_pickle("full_drug_df.pkl", compression="zip")
    return render_template('index.html',
                           purposes=purposes, fields=fields,
                           obs=observe_data.test_observation(), drug_output=str(drug_df.head()))

@app.route('/res', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        purpose = request.form["purpose_res"]
        field = request.form["field_res"]

    return render_template('result.html', purpose=purpose, field=field)

if __name__ == '__main__':
    app.run(port=33507)
