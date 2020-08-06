import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import parse_json, preprocessing

def plot_product_type(drug_df):
    # plot OTC/Prescription/NaN for full drug (purpose) dataset
    product_type_counts = drug_df["product_type"].value_counts()
    percentages = [num / len(drug_df.index) * 100.0 for index, num in product_type_counts.items()]
    percentages.append((len(drug_df.index) - product_type_counts.sum()) / len(drug_df.index) * 100.0)
    labels = [index[:-5] for index, num in product_type_counts.items()]
    labels.append("UNLISTED")

    plt.figure()
    plt.pie(percentages, labels=labels, autopct='%1.1f%%')
    plt.title("Distribution of OTC/Prescription/Unlisted Drugs in Subset")
    plt.show()

# plot numbers of drug labels by purpose cluster for tokenized purpose dataset
def plot_cluster_distribution(purpose_df, purpose_key):
    cluster_counts = purpose_df["purpose_cluster"].value_counts()
    labels = [purpose_key[key] for key, val in cluster_counts.items()]

    # all purposes
    print(sum(cluster_counts))
    plt.figure(figsize=(10, 6))
    plt.xticks(rotation=90)
    plt.bar(labels, cluster_counts)
    plt.xlabel("Labels of Clusters")
    plt.ylabel("Number of Drug Labels")
    plt.title("Number of Drug Labels Per Cluster in Preprocessed Data")
    plt.tight_layout()
    plt.show()

    # purposes with brand names, routes
    trunc_df = purpose_df.dropna(
        subset=["id", "brand_name", "route", "product_type"])  # exclude all rows with columns of null

    cluster_counts = trunc_df["purpose_cluster"].value_counts()

    plt.figure(figsize=(10, 6))
    plt.xticks(rotation=90)
    plt.bar(labels, cluster_counts)
    plt.xlabel("Labels of Clusters")
    plt.ylabel("Number of Drug Labels")
    plt.title("Number of Drug Labels Per Cluster in Preprocessed Data \n (Valid Fields Only)")
    plt.tight_layout()
    plt.show()

def test_observation():
    return list(np.arange(9))

def main():
    purposes = {"Anti-plaque Toothpaste": 'antiplaque antigingivitis antibacterial 092 060',
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

    # reverse dictionary
    purpose_key = {val: key for key, val in purposes.items()}

    json_list = ["drug-label-0001-of-0009.json", "drug-label-0002-of-0009.json", "drug-label-0003-of-0009.json",
                 "drug-label-0004-of-0009.json", "drug-label-0005-of-0009.json", "drug-label-0006-of-0009.json",
                 "drug-label-0007-of-0009.json", "drug-label-0008-of-0009.json", "drug-label-0009-of-0009.json"]
    drug_df = parse_json.parse_and_read_drugs(json_list, "full_drug_df")
    purpose_df = preprocessing.read_preprocessed_to_pkl("purpose_full_drug_df")

    plot_product_type(drug_df)
    plot_cluster_distribution(purpose_df, purpose_key)

    # test output
    print(drug_df.isnull().any(axis=0))
    print(drug_df)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(drug_df.loc[drug_df["id"] == "1c0bf028-e240-4667-887b-7a7695196fe1", :])

if __name__ == "__main__":
    main()