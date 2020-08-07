from flask import Flask, render_template, request, redirect, url_for
import sys, time
import json

import observe_data, main, preprocessing, parse_json

import pandas as pd

app = Flask(__name__)

purposes = {"Hand Sanitizer and Cleanser": 'antiseptic bacteria hand skin decrease',
       "Sterile Sanitizer and Wipes": 'hand rub product dry sterilization',
       "Hand Sanitizer": 'sanitizer hand soap potentially water',
       "Miscellaneous (Mainly Prescription) Drugs": 'indicate usage patient symptom tablet',
       "Antibacterial Sanitizer": 'handwash decrease bacteria skin antibacterial',
       "Sunscreen and Moisturizer": 'sun skin sunscreen protection aging',
       "Pain Relief": 'pain relief temporary minor reliever',
       "Anti-itch and Bug Relief": 'poison minor insect bite itch',
       "Sunburn Prevention (Sunscreen)": 'sunscreen sunburn prevent help octinoxate',
       "Acne Treatments": 'acne blemish pimple new blackhead',
       "Allergy Relief": 'nose itchy runny allergy hay',
       "Nasal and Cold Relief": 'nasal congestion fever temporarily cough',
       "First Aid and Sleep Aids": 'aid cut scrape burn infection',
       "Additional Hand Sanitizers and Wipes": 'sanitizer hand bacteria skin help',
       "Additional Foaming and Cleansing Sanitizers": 'repeat recommend bacteria decrease skin',
       "Muscle and Joint Relief": 'muscle sprain analgesic simple backache',
       "Anti-dandruff and Antifungal": 'dandruff itch tinea flake athlete',
       "Skin/Acne Treatment": 'acne medication management rosacea vulgaris',
       "Aspirin and Additional Pain Relief": 'pain ache fever minor temporarily',
       "Cough Relief": 'cough suppressant throat irritation sore',
       "Skin Repair and Rash Relief": 'protect skin protectant help chap',
       "Antiperspirant": 'underarm antiperspirant wetness reduce perspiration',
       "Homeopathic Remedies": 'homeopathic base traditional review statement',
       "Acid Reducer and Heartburn Relief": 'stomach heartburn acid indigestion sour',
       "Anti-diarrheal": 'diarrhea diarrheal traveler anti symptom',
       "Fever and Cold Reducer": 'fever pain flu throat sore',
       "Eye and Discomfort Relief": 'eye lubricant irritation discomfort dryness',
       "Chest Congestion Relief": 'bronchial cough mucus loosen phlegm',
       "Viral and Bacterial/Fungal Prevention": 'herpe infection vaginal alzheimer acyclovir',
       "Hypnotics and Bipolar Disorder": 'disorder bipolar 14 clinical study',
       "Oral Medication": 'oral temporary direction relief headache',
       "Diabetes Control": 'mellitus diabetes glycemic type diet',
       "Blood Pressure Control": 'blood pressure cardiovascular reduction risk',
       "Narcotics, Opioids, Analgesics": 'opioid analgesia tolerate alternative analgesic',
       "Ulcer Treatment": 'ulcer gerd duodenal week erosive',
       "Anxiety Relief, Muscle Relaxer": 'anxiety disorder panic tension term',
       "Toothpaste and Fluoride": 'anticavity cavity aids dental prevention',
       "Bacterial Infection Treatment": 'infection cause susceptible streptococcus susceptibility',
       "Nausea and Vomiting Prevention": 'nausea vomiting antiemetic motion emetogenic',
       "Skin Condition Treatment": 'corticosteroid pruritic dermatosis responsive manifestation',
       "Laxatives": 'constipation laxative movement bowel generally',
       "High Cholesterol Treatment": 'ldl diet patient risk hypercholesterolemia',
       "Anti-depressants": 'depressive major disorder week depression',
       "Joint and Arthritis Relief": 'sign tablet rheumatoid arthritis osteoarthritis',
       "Anti-gas": 'gas bloating refer commonly antigas',
       "Anti-nausea": 'upset stomach diarrhea nausea overindulgence',
       "Anti-seizure": 'seizure partial epilepsy onset adjunctive',
       "Hypothyroidism Treatment": 'hypothyroidism thyroid levothyroxine goiter pituitary',
       "Nicotine": 'smoking quitting nicotine craving withdrawal',
       "Miscellaneous Anti-inflammatory": 'acute allergic dermatitis disease arthritis'}

fields = {"Active Ingredient": "active_ingredient", "Inactive Ingredient": "inactive_ingredient",
          "Warnings": "warnings", "Dosage and Administration": "dosage_and_administration"}

disabled_field_dict = {'acne blemish pimple new blackhead': [],
                       'acne medication management rosacea vulgaris': [],
                       'acute allergic dermatitis disease arthritis': [],
                       'aid cut scrape burn infection': [],
                       'anticavity cavity aids dental prevention': [],
                       'antiseptic bacteria hand skin decrease': [],
                       'anxiety disorder panic tension term': [],
                       'blood pressure cardiovascular reduction risk': ['active_ingredient'],
                       'bronchial cough mucus loosen phlegm': [],
                       'constipation laxative movement bowel generally': [],
                       'corticosteroid pruritic dermatosis responsive manifestation': ['active_ingredient', 'inactive_ingredient'],
                       'cough suppressant throat irritation sore': [],
                       'dandruff itch tinea flake athlete': [],
                       'depressive major disorder week depression': ['active_ingredient', 'inactive_ingredient'],
                       'diarrhea diarrheal traveler anti symptom': [],
                       'disorder bipolar 14 clinical study': ['active_ingredient', 'inactive_ingredient'],
                       'eye lubricant irritation discomfort dryness': [],
                       'fever pain flu throat sore': [],
                       'gas bloating refer commonly antigas': [],
                       'hand rub product dry sterilization': [],
                       'handwash decrease bacteria skin antibacterial': [],
                       'herpe infection vaginal alzheimer acyclovir': [],
                       'homeopathic base traditional review statement': [],
                       'hypothyroidism thyroid levothyroxine goiter pituitary': ['active_ingredient', 'inactive_ingredient'],
                       'indicate usage patient symptom tablet': [],
                       'infection cause susceptible streptococcus susceptibility': [],
                       'ldl diet patient risk hypercholesterolemia': ['active_ingredient', 'inactive_ingredient'],
                       'mellitus diabetes glycemic type diet': ['active_ingredient', 'inactive_ingredient'],
                       'muscle sprain analgesic simple backache': [],
                       'nasal congestion fever temporarily cough': [],
                       'nausea vomiting antiemetic motion emetogenic': [],
                       'nose itchy runny allergy hay': [],
                       'opioid analgesia tolerate alternative analgesic': ['active_ingredient'],
                       'oral temporary direction relief headache': [],
                       'pain ache fever minor temporarily': [],
                       'pain relief temporary minor reliever': [],
                       'poison minor insect bite itch': [],
                       'protect skin protectant help chap': [],
                       'repeat recommend bacteria decrease skin': [],
                       'sanitizer hand bacteria skin help': [],
                       'sanitizer hand soap potentially water': [],
                       'seizure partial epilepsy onset adjunctive': ['active_ingredient'],
                       'sign tablet rheumatoid arthritis osteoarthritis': ['active_ingredient', 'inactive_ingredient'],
                       'smoking quitting nicotine craving withdrawal': [],
                       'stomach heartburn acid indigestion sour': [],
                       'sun skin sunscreen protection aging': [],
                       'sunscreen sunburn prevent help octinoxate': [],
                       'ulcer gerd duodenal week erosive': ['active_ingredient', 'inactive_ingredient'],
                       'underarm antiperspirant wetness reduce perspiration': [],
                       'upset stomach diarrhea nausea overindulgence': []}

@app.route('/')
def index():
    reversed_fields = {val: key for key, val in fields.items()}
    return render_template('index.html',
                           purposes=purposes, fields=fields,
                           reversed_fields=reversed_fields,
                           disabled_field_dict=disabled_field_dict,
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

    return render_template('result.html', purpose=purpose, purpose_key=purposes[purpose], field=field,
                           output_png=output_png, output_html=output_html)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(port=33507, debug=True)
