from flask import Flask, render_template, request, redirect, url_for
import sys, time

import observe_data, main, preprocessing, parse_json

import pandas as pd

app = Flask(__name__)

purposes = {"Anti-plaque Mouthwash": 'antiplaque antigingivitis antibacterial 092 060',
            "Acid Reducer": 'acid reducer antacid otc purpose',
            "Acne and Pores": 'acne treatment medication gel pimple',
            "Antiseptics": 'aid antiseptic alertness debride agent',
            "Cough and Pain Relief": 'anesthetic oral suppressant cough analgesic',
            "Antacids": 'antacid antigas gas anti antidiarrheal',
            "Brightening Skin": 'wrinkle anti brightening skin whiten',
            "Handwashing": 'antibacterial soap hand agent wipe',
            "Anticavity Toothpaste": 'anticavity toothpaste antihypersensitivity rinse antigingivitis',
            "Antifungals (Vaginal)": 'vaginal antifungal miconazole nitrate clotrimazole',
            "Cough Relief": 'suppressant cough antihistamine topical analgesic',
            "Sanitizer and Disinfectant": 'antimicrobial topical skin purpose agent',
            "Deodorant and Antiperspirant": 'antiperspirant purpose 關節酸痛 enflées enfant',
            "More Sanitizer and Wipes": 'antiseptic skin preparation 關節酸痛 endometritis',
            "Additional Antibacterial Sanitizers": 'bacteria decrease skin handwash disease',
            "Yet More Hand Sanitizer": 'reduce bacteria help disease hand',
            "Allergy Symptom Relief": 'antihistamine otc decongestant analgesic suppressant',
            "Cold and Chest Relief": 'cold throat sore congestion nose',
            "Mainly Anti-dandruff Shampoo": 'anti dandruff itch perspirant diarrheal',
            "Nasal and Fever Decongestion": 'decongestant nasal antihistamine expectorant suppressant',
            "Mucus Decongestion": 'expectorant suppressant cough decongestant nasal',
            "Antibiotics, External Analgesics": 'antibiotic aid unit external analgesic',
            "Eyewash": 'eyewash eyesaline emergency flush skin',
            "Hair Nourishment": 'hair regrowth treatment man loss',
            "Homeopathic Headache Remedies": 'temporary relief headache sore pain',
            "Homeopathic Remedies": 'homeopathic drug base indication traditional',
            "Laxatives": 'laxative saline stimulant bulk form',
            "Skin Relief": 'relief skin temporary indication lubricant',
            "Phenylephrine Cold Relief": 'mg active ingredient hcl phenylephrine',
            "Muscle and Joint Pain Relief": 'muscle joint minor pain ache',
            "Nighttime Sleep Aid and Pain Relief": 'nighttime sleep aid reliever pain',
            "Antifungal Topical Cream": 'antifungal purpose cream topical treatment',
            "Skin Repair": 'protectant skin external analgesic ointment',
            "Antihistamine, Pain Relief, Fever Reducers": 'reducer fever reliever pain antihistamine',
            "Nighttime Cold Relief": 'reducer fever reliever suppressant cough',
            "Natural Temporary Pain Relief": 'symptom relieve allergy reliever pain',
            "Aspirin": 'reliever pain oral aid diuretic',
            "Handwash - Healthcare": 'handwash antiseptic antispetic personnel healthcare',
            "Hand Sanitizer": 'sanitizer hand antiseptic antimicrobial skin',
            "Misc Drug Purpose OTC Facts": 'section otc box formulate fact',
            "Sunscreen, Moisturizer": 'sun screen skin measure protection',
            "Nicotine": 'smoking stop aid quitting nicotine',
            "Stool Softener": 'softener stool laxative stimulant softner',
            "Alcohol Disinfectant": 'sterilization disinfection rinseing hand skin',
            "Antigas": 'antigas antacid antiflatulent otc endolorissement',
            "Protective Sunscreen": 'sunburn protection prevent help spf',
            "Octisalate Sunscreen": 'sunscreen avobenzone octisalate octocrylene active',
            "Eye and Nose, Handwash": 'wash hand nasal antiseptic eye',
            "Topical Analgesic": 'analgesic topical external anesthetic menthol',
            "Anti-nausea Anti-diarrheal": 'upset stomach antidiarrheal reliever diarrheal'
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

    return render_template('result.html', purpose=purpose, purpose_key=purposes[purpose], field=field,
                           output_png=output_png, output_html=output_html)

if __name__ == '__main__':
    app.run(port=33507, debug=True)
