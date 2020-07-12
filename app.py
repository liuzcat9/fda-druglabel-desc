from flask import Flask, render_template, request, redirect
import observe_data

app = Flask(__name__)

@app.route('/')
def index():
    purposes = ["Sunscreen", "Analgesic", "Pain Relief"]
    fields = ["Indications and Usage", "Route"]
    return render_template('index.html',
                           purposes=purposes, fields=fields,
                           obs=observe_data.test_observation())

@app.route('/res', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        purpose = request.form["purpose_res"]
        field = request.form["field_res"]

    return render_template('result.html', purpose=purpose, field=field)

if __name__ == '__main__':
    app.run(port=33507)
