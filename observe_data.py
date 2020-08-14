import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import parse_json, preprocessing

def plot_product_type(drug_df):
    print("Full dataset:", str(drug_df.shape))
    # plot OTC/Prescription/NaN for full drug (purpose) dataset
    product_type_counts = drug_df["product_type"].value_counts()
    percentages = [num / len(drug_df.index) * 100.0 for index, num in product_type_counts.items()]
    percentages.append((len(drug_df.index) - product_type_counts.sum()) / len(drug_df.index) * 100.0)
    labels = [index[:-5] for index, num in product_type_counts.items()]
    labels.append("UNLISTED")

    plt.figure()
    plt.pie(percentages, labels=labels, autopct='%1.1f%%')
    plt.title("Distribution of OTC/Prescription/Unlisted Drugs in Purpose Dataset")
    plt.show()

# plot numbers of drug labels by purpose cluster for tokenized purpose dataset
def plot_cluster_distribution(purpose_df, purpose_key):
    cluster_counts = purpose_df["purpose_cluster"].value_counts()
    labels = [purpose_key[key] for key, val in cluster_counts.items()]

    # all purposes
    print(sum(cluster_counts))
    plt.figure(figsize=(10, 10))
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

    plt.figure(figsize=(10, 10))
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

    # reverse dictionary
    purpose_key = {val: key for key, val in purposes.items()}

    json_list = ["drug-label-0001-of-0009.json", "drug-label-0002-of-0009.json", "drug-label-0003-of-0009.json",
                 "drug-label-0004-of-0009.json", "drug-label-0005-of-0009.json", "drug-label-0006-of-0009.json",
                 "drug-label-0007-of-0009.json", "drug-label-0008-of-0009.json", "drug-label-0009-of-0009.json"]
    drug_df = parse_json.parse_and_read_drugs(json_list, "drug_df")
    purpose_df = preprocessing.read_preprocessed_to_pkl("purpose_drug_df")

    plot_product_type(drug_df)
    plot_cluster_distribution(purpose_df, purpose_key)

    # test output
    print(drug_df.isnull().any(axis=0))
    print(drug_df)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(drug_df.loc[drug_df["id"] == "1c0bf028-e240-4667-887b-7a7695196fe1", :])

if __name__ == "__main__":
    main()