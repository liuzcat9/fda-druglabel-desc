from flask import Flask, render_template, request, redirect
import sys, time

import observe_data, main, preprocessing

import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    purposes = ["Sunscreen", "Analgesic", "Antiseptic"]
    fields = ["Indications and Usage", "Warnings"]

    return render_template('index.html',
                           purposes=purposes, fields=fields,
                           obs=observe_data.test_observation())

@app.route('/res', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        purpose = request.form["purpose_res"]
        field = request.form["field_res"]
        output_file = purpose.lower() + "-" + "_".join(field.lower().split()) + ".html"

        print("File retrieved should be:", output_file)

    return render_template('result.html', purpose=purpose, field=field, output_file=output_file,
                           output_file2="sunscreen_purposes_uses_protectant_skin-indications_and_usage.html")

if __name__ == '__main__':
    app.run(port=33507, debug=True)
