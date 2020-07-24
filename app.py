from flask import Flask, render_template, request, redirect
import sys, time

import observe_data, main, preprocessing, parse_json

import pandas as pd

app = Flask(__name__)
purposes = {"Sunscreen": "sunscreen purposes uses protectant skin", "Analgesic": "analgesic", "Antiseptic": "antiseptic"}
fields = {"Indications and Usage": "indications_and_usage", "Warnings": "warnings"}

@app.route('/')
def index():

    return render_template('index.html',
                           purposes=purposes.keys(), fields=fields.keys(),
                           obs=observe_data.test_observation())

@app.route('/res', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        purpose = request.form["purpose_res"]
        field = request.form["field_res"]
        data_purpose = purposes[purpose]
        data_field = fields[field]

        # call process to generate information for this
        script, div = main.create_web_graph(data_purpose, data_field)
        output_file = purpose.lower() + "-" + "_".join(field.lower().split()) + ".html"

        print("File retrieved should be:", output_file)

    return render_template('result.html', purpose=purpose, field=field, output_file=output_file,
                           output_file2="sunscreen_purposes_uses_protectant_skin-indications_and_usage.html",
                           output_file3="sunscreen_purposes_uses_protectant_skin-indications_and_usage.png",
                           bokeh_script=script, bokeh_div=div)

if __name__ == '__main__':
    app.run(port=33507, debug=True)
