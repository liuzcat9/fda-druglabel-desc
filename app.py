from flask import Flask, render_template, request, redirect, url_for
import sys, time

import observe_data, main, preprocessing, parse_json

import pandas as pd

app = Flask(__name__)
purposes = {"Acid Reducer": "acid reducer antacid otc purpose",
            "Acne and Pores": "acne treatment medication gel pimple",
            "Antiseptics and Antibiotics": "aid antiseptic antibiotic alertness debride",
            "Cough and Pain Relief": "anesthetic oral suppressant cough topical",
            "Antacids": "antacid antigas gas anti antidiarrheal",
            "Anti-itch and Brightening Skin": "anti wrinkle itch brightening whiten",
            "Handwashing": "antibacterial soap hand agent wipe",
            "Anticavity Toothpaste": "anticavity toothpaste antihypersensitivity rinse antigingivitis",
            "Antifungals": "antifungal vaginal purpose cream topical",
            "Allergy and Cough Relief": "antihistamine suppressant cough otc antitussive",
            "Sanitizer and Disinfectant": "antimicrobial topical skin purpose wash",
            "Deodorant and Antiperspirant": "antiperspirant purpose 關節酸痛 enflées enfant",
            "More Sanitizer and Wipes": "antiseptic skin preparation 關節酸痛 endometritis",
            "Additional Antibacterial Sanitizers": "bacteria decrease handwash skin disease",
            "Yet More Hand Sanitizer": "bacteria reduce help disease hand",
            "General Relief": "claim accept evidence fda evaluate",
            "Laxatives": "constipation occasional bowel generally movement",
            "Anti-dandruff Shampoo": "dandruff anti seborrheic dermatitis psoriasis",
            "Decongestants": "decongestant nasal antihistamine expectorant suppressant",
            "Additional Nose and Cough Relief": "expectorant suppressant cough decongestant nasal",
            "Itch and Pain Relief": "external analgesic protectant skin antibiotic",
            "Eye Drops": "eye irritation redness dryness burning",
            "Hair Treatment": "hair treatment lice regrowth man",
            "Headache Relief": "headache temporary relief congestion cough",
            "Homeopathic Remedies": "homeopathic food review statement administration",
            "Enema and Laxatives": "laxative saline bulk form osmotic",
            "Mineral Oil and Additional Eye Drops": "lubricant eye laxative glycol redness",
            "Phenylephrine Cold Relief": "mg active ingredient hcl phenylephrine",
            "Muscle Pain Relief": "minor muscle pain joint ache",
            "Nighttime Sleep Pain Relief": "nighttime sleep aid reliever pain",
            "Another Antiperspirant": "perspirant anti 關節酸痛 endoscopic energy",
            "Skin Repair": "protectant skin sunscreen vasoconstrictor whitening",
            "Antihistamine, Pain Relief, Fever Reducers": "reducer fever reliever pain antihistamine",
            "Nighttime Cold Relief": "reducer fever suppressant reliever cough",
            "More Natural Temporary Pain Relief": "relief temporary indication symptom pain",
            "Aspirin": "reliever pain oral aid diuretic",
            "Wart Removal": "remover wart corn callus purpose",
            "Hand Sanitizer": "sanitizer hand antiseptic antimicrobial skin",
            "Misc Drug Purpose OTC Facts": "section otc fact drug box",
            "Skincare (Antibiotic, Moisturizer)": "skin help protectant lightener minor",
            "Nicotine": "smoking stop aid quitting nicotine",
            "Stool Softener": "softener stool laxative stimulant softner",
            "Alcohol Disinfectant": "sterilization disinfection rinseing hand skin",
            "Bisacodyl and Laxatives": "stimulant laxative respiratory otc sexual",
            "Broad Spectrum Makeup and Sunscreen": "sunburn protection sun prevent help",
            "Facial and Tooth Repair": "sunscreen antigas antiplaque antigingivitis ingredient",
            "Homeopathic Remedies and Allergy Relief": "symptom relieve allergy reliever pain",
            "Topical Analgesic": "topical analgesic menthol suppressant cough",
            "Anti-diarrheal": "upset diarrheal stomach antidiarrheal anti"
            }

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
