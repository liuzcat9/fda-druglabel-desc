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

    # reverse dictionary
    purpose_key = {val: key for key, val in purposes.items()}

    json_list = ["drug-label-0001-of-0009.json", "drug-label-0002-of-0009.json", "drug-label-0003-of-0009.json",
                 "drug-label-0004-of-0009.json", "drug-label-0005-of-0009.json", "drug-label-0006-of-0009.json",
                 "drug-label-0007-of-0009.json", "drug-label-0008-of-0009.json", "drug-label-0009-of-0009.json"]
    drug_df = parse_json.parse_and_read_drugs(json_list, "indic_full_drug_df")
    purpose_df = preprocessing.read_preprocessed_to_pkl("purpose_indic_full_drug_df")

    plot_product_type(drug_df)
    plot_cluster_distribution(purpose_df, purpose_key)

    # test output
    print(drug_df.isnull().any(axis=0))
    print(drug_df)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(drug_df.loc[drug_df["id"] == "1c0bf028-e240-4667-887b-7a7695196fe1", :])

if __name__ == "__main__":
    main()