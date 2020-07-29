from flask import Flask, render_template, request, redirect, url_for
import sys, time

import observe_data, main, preprocessing, parse_json

import pandas as pd

app = Flask(__name__)
purposes = {"Antiseptic Sanitizer": "sanitizer hand antiseptic antimicrobial skin",
            "Sunscreen": "sunscreen purposes uses protectant skin",
            "Octinoxate Octisalate Sunscreen": "sunscreen active octinoxate ingredients octisalate",
            "Anti-itch Cream": "itch anti cream antipruritic protectant",
            "Skin Antiseptic": "antiseptic purposes skin uses preparation",
            "Antiperspirant": "antiperspirant purpose use enriched enteric",
            "Antibiotic Antiseptic": "aid antiseptic antibiotic alertness debriding",
            "Sunscreen Ointment": "protectant skin ointment sunscreen purposes",
            "Misc Handwashing Antiseptic Treatment": "skin uses handwash antiseptic treatment",
            "Cough Suppressant": "anesthetic oral suppressant cough analgesic",
            "Homeopathic": "homeopathic based indications traditional food",
            "Fever and Pain Relief": "reducer fever reliever pain purposes",
            "Antibacterial Soap and Wipes": "antibacterial soap hand agent wipe",
            "Headache and Sore Relief": "use temporary relief headache sore",
            "Eye Lubricant": "lubricant eye laxative glycol redness",
            "Temporary Pain Relief": "relief temporary indications symptoms pain",
            "Antibacterial Handwashing": "decrease bacteria handwashing skin uses",
            "Nasal Decongestant": "decongestant nasal reliever suppressant pain",
            "Antihistamine Decongestant": "antihistamine otc decongestant analgesic purposes",
            "Stomach Relief": "reliever pain upset stomach antidiarrheal",
            "Antimicrobial Topical": "antimicrobial topical skin purpose purposes",
            "Menthol Topical Analgesic": "topical analgesic anesthetic menthol antiseptic",
            "Sunburn Protection": "sunburn protection sun prevent helps",
            "Runny Nose Throat Relief": "throat nose congestion relieves runny",
            "Acne Treatment": "acne treatment medication gel salicylic",
            "Anti-dandruff": "anti dandruff diarrheal gas seborrheic",
            "Laxative": "laxative saline stimulant bulk forming",
            "Irritation and Burns": "minor uses irritation cuts burns",
            "Antacid": "antacid antigas gas purposes anti",
            "Skin Whitening": "wrinkle anti whitening skin protectant",
            "Bacterial": "bacteria use reduce cause disease",
            "Wrinkle Brightening and Improvement": "brightening wrinkle anti skin improvement",
            "HCL and Phenylephrine": "mg active ingredients hcl phenylephrine",
            "Antifungal Cream": "antifungal vaginal purpose cream topical",
            "OTC Section Misc": "section otc facts drug box",
            "External Skin Antibiotic": "external analgesic protectant skin antibiotic",
            "Antihistamine Cough Suppressant": "suppressant cough antihistamine reliever reducer",
            "Backache and Sprains": "sprains minor arthritis strains backache",
            "Diarrhea Relief": "relieve symptoms uses diarrhea headache",
            "Cough Decongestant": "suppressant cough expectorant purposes decongestant",
            "Wart and Callus": "remover wart corn callus purpose",
            "Acid Reducer": "acid reducer antacid purposes otc",
            "Anticavity Toothpaste": "anticavity toothpaste antihypersensitivity rinse antigingivitis",
            "Expectorant": "expectorant antitussive purposes guaifenesin otc",
            "Stool Softener": "stool softener laxative stimulant softner",
            "Deodorant and Disinfection": "sterilization disinfection rinseing deodorization clean",
            "Anti-perspirant and Enteritis": "perspirant anti 關節酸痛 enriching enteritis",
            "Sleep Aid": "nighttime sleep aid reliever pain",
            "Stop Smoking Aid": "smoking stop aid quitting nicotine",
            "Anti-gas": "antigas antacid purposes antiflatulent otc"}

fields = {"Active Ingredient": "active_ingredient", "Inactive Ingredient": "inactive_ingredient",
          "Warnings": "warnings", "Dosage and Administration": "dosage_and_administration",
          "Indications and Usage": "indications_and_usage"}

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

        print("File retrieved should be:", "_".join(purposes[purpose].split()) + "-" + "_".join(fields[field].split()))

    # redirection needed from POST for dynamic url
    return redirect(url_for('.load_result', purpose=purpose, field=field))

@app.route('/res/<purpose>/<field>')
def load_result(purpose, field):
    output_html = "_".join(purposes[purpose].split()) + "-" + "_".join(fields[field].split()) + ".html"
    output_png = "_".join(purposes[purpose].split()) + "-" + "_".join(fields[field].split()) + ".png"

    return render_template('result.html', purpose=purposes[purpose], field=fields[field], output_png=output_png,
                           output_html=output_html)

if __name__ == '__main__':
    app.run(port=33507, debug=True)
