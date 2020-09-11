from flask import Flask, render_template, request, redirect, url_for
import sys, time, os, re
import json
from joblib import load

import observe_data, main, preprocessing, parse_json
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy import create_engine

app = Flask(__name__)

purposes = {"Skin Antiseptic": 'skin aid antiseptic antibiotic help',
        "Hand Sanitizer": 'sanitizer hand antiseptic skin wipe',
       "Antiseptic, Sanitizer, and Wipes (2)": 'antiseptic skin deodorant formulate instant',
       "Sanitizer Gel (3)": 'bacteria disease cause decrease skin',
       "Hand Sanitizer and Wipes (4)": 'antibacterial agent wipe hand soap',
       "Hand Sanitizer and Wipes (5)": 'antimicrobial hand sanitizer purpose topical',
        "Hand Sanitizer and Handwash (6)": 'fact drug box otc section',
        "Hand Sanitizer (7)": 'sanitizer skin antiseptic deodorant available',
        "Alcohol Disinfectant": 'sterilization disinfection rinseing disinfe ction',
       "Moisturizer and Sunscreen": 'sunscreen protectant skin lightener lip',
        "Sunscreen and Makeup (2)": 'sun skin protection measure risk',
       "Sunscreen (3)": 'sunscreen octinoxate octisalate avobenzone octocrylene',
        "Skin Renewal": 'wrinkle anti brightening skin whiten',
        "Skin Protection": 'protectant skin ointment sunscreen section',
        "Wart and Callus Removers": 'remover wart corn callus purpose',
       "Anti-Itch": 'itch anti antipruritic cream relief',
       "Acne Control": 'medication acne product antiacne hyland',
        "Acne Cleanser (2)": 'acne gel pimple cream blemish',
       "Topical Analgesic": 'topical analgesic anesthetic menthol camphor',
        "External Analgesic (2)": 'external analgesic protectant skin antibiotic',
        "Antiseptics and First Aid": 'aid antiseptic antimicrobial topical pain',
        "Muscle and Allergy Relief": 'relief temporary pain muscle minor',
        "Allergy Decongestant (2)": 'decongestant nasal antihistamine expectorant suppressant',
       "Allergy Relief (3)": 'antihistamine succinate doxylamine suppressant guaifenesin',
        "Allergies and Cramp Relief (4)": 'symptom relieve allergy reliever pain',
       "Sleep Aid and Pain Relief": 'nighttime sleep aid reliever pain',
        "Headache Relief": 'headache temporary relief migraine dull',
        "Pain Relief": 'reliever pain fever reducer aid',
        "Nighttime Cold Relief": 'fever reducer suppressant reliever decongestant',
       "Homeopathic Remedies": 'homeopathic food statement review administration',
       "Cough Decongestant": 'expectorant suppressant cough decongestant nasal',
        "Acetaminophen Dextromethorphan Cough Suppressant (2)": 'mg hcl acetaminophen phenylephrine dextromethorphan',
        "Cough Drops": 'oral anesthetic suppressant cough analgesic',
        "Eye Relief": 'lubricant eye laxative glycol redness',
        "Anticavity Fluoride": 'anticavity rinse antihypersensitivity antigingivitis antisensitivity',
        "Toothpaste and Mouthwash": 'toothpaste anticavity antigingivitis antisensitivity whiten',
       "Anti-Dandruff": 'dandruff anti dermatitis seborrheic psoriasis',
       "Antiperspirant and Deodorant": 'antiperspirant purpose aluminum chlorohydrate zirconium',
       "Acid Reducers": 'acid reducer antacid salicylic otc',
        "Antacids": 'antacid antigas gas anti antidiarrheal',
       "Anti-Diarrheal": 'diarrheal stomach upset anti antidiarrheal',
       "Anti-Gas": 'antigas antacid antiflatulent hydroxide simethicone',
       "Antiseptic Mouth Rinse": 'antiplaque antigingivitis antibacterial 060 092',
       "Anti-Nausea": 'antiemetic meclizine hydrochloride hcl oxide',
        "Antifungals": 'antifungal vaginal purpose cream section',
       "Laxatives": 'stimulant laxative respiratory reliever aromatic',
        "Natural Laxatives (2)": 'laxative saline phosphate sodium monobasic',
       "Hemorrhoid Treatment": 'vasoconstrictor protectant local anesthetic 25',
       "Stool Softeners": 'stool softener laxative stimulant softner',
        "Nicotine": 'smoking stop aid regain cessation'}

fields = {"Active Ingredient": "active_ingredient", "Inactive Ingredient": "inactive_ingredient",
          "Warnings": "warnings", "Dosage and Administration": "dosage_and_administration",
          "Indications and Usage": "indications_and_usage"}

disabled_field_dict = {'acid reducer antacid salicylic otc': [], 'acne gel pimple cream blemish': [],
                       'aid antiseptic antimicrobial topical pain': [], 'antacid antigas gas anti antidiarrheal': [],
                       'antibacterial agent wipe hand soap': [], 'anticavity rinse antihypersensitivity antigingivitis antisensitivity': [],
                       'antiemetic meclizine hydrochloride hcl oxide': [], 'antifungal vaginal purpose cream section': [],
                       'antigas antacid antiflatulent hydroxide simethicone': [], 'antihistamine succinate doxylamine suppressant guaifenesin': [],
                       'antimicrobial hand sanitizer purpose topical': [], 'antiperspirant purpose aluminum chlorohydrate zirconium': [],
                       'antiplaque antigingivitis antibacterial 060 092': [], 'antiseptic skin deodorant formulate instant': [],
                       'bacteria disease cause decrease skin': [], 'dandruff anti dermatitis seborrheic psoriasis': [],
                       'decongestant nasal antihistamine expectorant suppressant': [], 'diarrheal stomach upset anti antidiarrheal': [],
                       'expectorant suppressant cough decongestant nasal': [], 'external analgesic protectant skin antibiotic': [],
                       'fact drug box otc section': [], 'fever reducer suppressant reliever decongestant': [],
                       'headache temporary relief migraine dull': [], 'homeopathic food statement review administration': [],
                       'itch anti antipruritic cream relief': [], 'laxative saline phosphate sodium monobasic': [],
                       'lubricant eye laxative glycol redness': [], 'medication acne product antiacne hyland': [],
                       'mg hcl acetaminophen phenylephrine dextromethorphan': [], 'nighttime sleep aid reliever pain': [],
                       'oral anesthetic suppressant cough analgesic': [], 'protectant skin ointment sunscreen section': [],
                       'relief temporary pain muscle minor': [], 'reliever pain fever reducer aid': [],
                       'remover wart corn callus purpose': [], 'sanitizer hand antiseptic skin wipe': [],
                       'sanitizer skin antiseptic deodorant available': [], 'skin aid antiseptic antibiotic help': [],
                       'smoking stop aid regain cessation': [], 'sterilization disinfection rinseing disinfe ction': [],
                       'stimulant laxative respiratory reliever aromatic': [], 'stool softener laxative stimulant softner': [],
                       'sun skin protection measure risk': [], 'sunscreen octinoxate octisalate avobenzone octocrylene': [],
                       'sunscreen protectant skin lightener lip': [], 'symptom relieve allergy reliever pain': [],
                       'toothpaste anticavity antigingivitis antisensitivity whiten': [], 'topical analgesic anesthetic menthol camphor': [],
                       'vasoconstrictor protectant local anesthetic 25': [], 'wrinkle anti brightening skin whiten': []}

postgres_engine = create_engine(os.getenv("POSTGRES_URI"))

@app.route('/')
def index():
    print(postgres_engine.execute("""SELECT brand_name, purpose_cluster
                                      FROM purpose_drug_labels
                                      WHERE purpose_cluster = 'aid antiseptic antimicrobial topical pain'
                                      LIMIT 5;""").fetchall())
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

# routes for predicting purpose

@app.route('/predict-purpose')
def predict_purpose():
    return render_template('predict_purpose.html')

@app.route('/predict-purpose/result', methods=['GET', 'POST'])
def predict_purpose_result():
    # load model to predict
    mnb = load("models/purpose_model.joblib")

    if request.method == 'POST':
        active = request.form["active_ingredient"]
        inactive = request.form["inactive_ingredient"]

        total_ingred = active + "," + inactive
        tokenized_text_list = re.split(r'\s?,\s?', total_ingred)
        tokenized_text = " ".join(tokenized_text_list)

        purpose_result = mnb.predict([tokenized_text])[0]

    return render_template('predict_purpose_result.html',
                           tokenized_text_list=tokenized_text_list, purpose_result=purpose_result)

if __name__ == '__main__':
    app.run(port=33507, debug=True)
