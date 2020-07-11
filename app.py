from flask import Flask, render_template, request, redirect
import observe_data

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', obs=observe_data.test_observation())

if __name__ == '__main__':
    app.run(port=33507)
