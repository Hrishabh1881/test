# dorisclinicaltrials

# Clinical Trials API

This project provides two API endpoints to handle clinical trial queries. The first endpoint allows users to search for clinical trials based on a query and location, while the second endpoint retrieves detailed information about clinical trials based on their NCT number.

## Endpoints

### 1. ClinicalTrialsLLMViewHybridZipLocator

This endpoint processes clinical trial queries with a specified ZIP code and radius for location-based filtering.

**URL:** `/chat/clinicaltrialwithzip`

**Method:** `POST`

**Request Body:**
- `query` (string, optional): The search query for clinical trials.
- `zip_code` (string, required): The ZIP code for location-based filtering.
- `radius` (integer, optional, default=120): The radius in miles for location-based filtering.

**Response:**
- `Message` (JSON object): The payload containing the clinical trials information. An empty dictionary is returned if no data is found or if the ZIP code is not provided.

**Example Request:**
```json
{
    "query":"triple negative",
    "zip_code": 60610,
    "radius":150
}
```
**Example Response:**
```json
[{
            "NCT_NUMBER": "NCT02977468",
            "STUDY_TITLE": "Effects of MK-3475 (Pembrolizumab) on the Breast Tumor Microenvironment in Triple Negative Breast Cancer",
            "STUDY_URL": "https://www.clinicaltrials.gov/study/NCT02977468",
            "STUDY_STATUS": "RECRUITING",
            "CONDITIONS": [
                "Triple Negative Breast Cancer"
            ],
            "PHASES": [
                "PHASE1"
            ],
            "LOCATIONS": [
                {
                    "Location Facility": "Loyola University Chicago",
                    "Location City": "Maywood",
                    "Location State": "Illinois",
                    "Location Zip": "60153",
                    "Location Country": "United States",
                    "Distance": 11.0
                }
            ],
            "START_DATE": "2017-10-25",
            "COMPLETION_DATE": "2024-12-31",
            "MINIMUMAGE": "21 Years",
            "MAXIMUMAGE": "80 Years",
            "POINT_OF_CONTACT": {
                "Name": null,
                "Organization": null,
                "Email": null,
                "Phone": null
            },
            "PHASE": null,
            "SEX": "FEMALE",
            "ELIGIBILITY_CRITERIA": "Eligible participants are women aged 21-80 with histologically proven invasive breast carcinoma that is triple negative. They must have a clinically ≤ 3 cm unifocal lesion, be clinically node negative with no evidence of metastatic disease, and have not received any prior anti-cancer therapy within 6 months of study entry. Breast size must be B cup or larger for the IORT procedure, and participants must have an ECOG Performance Status of 0 or 1. Adequate organ function and negative pregnancy tests are required for women of childbearing potential, who must also agree to use contraception. Exclusions include multifocal disease, lesions >3 cm, metastatic disease, active or recent malignancies requiring treatment, contraindications to radiotherapy, immunodeficiency, severe hypersensitivity to pembrolizumab, active autoimmune disease requiring systemic treatment within the past 2 years, non-infectious pneumonitis history, active infection requiring systemic therapy, conditions that might confound study results, psychiatric or substance abuse disorders, pregnancy or breastfeeding, prior therapy with anti-PD-1, anti-PD-L1, or anti-PD-L2 agents, active Hepatitis B or C, recent live vaccines, history of allogenic stem cell or solid organ transplant.",
            "PRIMARY_OUTCOMES": "The primary outcome measure is the number of subjects with a significant mean percent change in tumor-infiltrating lymphocytes (TILs) in tumor tissue. This will assess if MK-3475 therapy increases TILs in newly diagnosed triple negative breast cancer tumors by comparing the mean percent change in TILs from initial biopsy samples to pathology samples from surgery after two cycles of MK-3475, over a timeframe of 3 months.",
            "SECONDARY_OUTCOMES": "No secondary outcome measures are specified.",
            "INTERVENTIONS": "Participants in this study receive Merck 3475 Pembrolizumab (Keytruda) by vein over approximately 30 minutes on Day 1 of one to two cycles. Additionally, participants will receive Intraoperative radiation therapy (IORT) on the day of surgery using the Intrabeam® Photon Radiosurgery System, which delivers low energy X-rays directly to the tumor site after excision, for 20-35 minutes.",
            "STUDY_TYPE": "INTERVENTIONAL",
            "ENROLLMENT": 15,
            "DRUGS": [
                "Mk-3475 (Pembrolizumab)",
                " Keytruda"
            ],
            "BIOMARKERS": [
                "TILs"
            ]
        },
        {
            "NCT_NUMBER": "NCT05029999",
            "STUDY_TITLE": "CD40 Agonist, Flt3 Ligand, and Chemotherapy in Triple Negative Breast Cancer",
            "STUDY_URL": "https://www.clinicaltrials.gov/study/NCT05029999",
            "STUDY_STATUS": "RECRUITING",
            "CONDITIONS": [
                "Metastatic Triple Negative Breast Cancer"
            ],
            "PHASES": [
                "PHASE1"
            ],
            "LOCATIONS": [
                {
                    "Location Facility": "University of Chicago Comprehensive Cancer Center",
                    "Location City": "Chicago",
                    "Location State": "Illinois",
                    "Location Zip": "60637",
                    "Location Country": "United States",
                    "Distance": 9.0
                }
            ],
            "START_DATE": "2022-04-20",
            "COMPLETION_DATE": "2025-04-20",
            "MINIMUMAGE": "18 Years",
            "MAXIMUMAGE": "99 Years",
            "POINT_OF_CONTACT": {
                "Name": null,
                "Organization": null,
                "Email": null,
                "Phone": null
            },
            "PHASE": null,
            "SEX": "ALL",
            "ELIGIBILITY_CRITERIA": "Eligibility includes adults (18+) with unresectable Stage III or IV Triple Negative Breast Cancer, ECOG performance status 0-2, life expectancy ≥ 12 weeks, and documented progressive disease. Initial safety cohort participants must be in second to third line treatment with 1 to 2 prior regimens for metastatic or unresectable disease. Dose expansion participants can be in first to third line treatment with 0 to 2 prior regimens. PD-L1 negative patients not eligible for FDA approved standard care may enroll. Required screening laboratory values are specified. Exclusions include severe hypersensitivity to mAbs, prior anti-CD40 or rhuFlt3L treatment, anthracycline in metastatic setting, prior progression on anthracycline, AML or Flt3 mutation, recent immunotherapy, chemotherapy, kinase inhibitors, major surgery, investigational drugs, immunosuppressives, other malignancies, untreated CNS metastases, pregnancy, active autoimmune disease, significant cardiovascular disease, prior anthracycline therapy above a specified dose, live vaccines, pneumonitis, active infection, and other conditions that increase risk or interfere with results.",
            "PRIMARY_OUTCOMES": "The primary outcome is the safety and tolerability of the drug combination of CDX-1140, CDX-301, and PLD, measured by the number of participants with Dose Limiting Toxicity (DLT) based on severity criteria including serious adverse events and events at Grade 3 or above, with a timeframe from baseline up to 12 months.",
            "SECONDARY_OUTCOMES": "Secondary outcomes include anti-tumor immune response measured by on-treatment CD8 T cell infiltrate, change in CD8 T cell infiltrate from baseline after cycle 1, median Progression Free Survival by RECIST v1.1, Overall Response Rate (ORR) by RECIST v1.1, Duration of Response (DoR) by RECIST v1.1, and Clinical Benefit Rate (CBR) by RECIST v1.1 at 6 months.",
            "INTERVENTIONS": "Interventions involve PLD chemotherapy administered at 40 mg/m2 as an intravenous injection once per cycle until toxicity or progression. CDX-1140 is administered at 1.5mg/kg as an intravenous injection once per cycle until toxicity or progression for up to 24 months. CDX-301 is administered at 75µg/kg as a subcutaneous injection daily for 5 days in cycles 1 and 2 only, with variations in administration across different cohorts.",
            "STUDY_TYPE": "INTERVENTIONAL",
            "ENROLLMENT": 45,
            "DRUGS": [
                "Cd40 Agonist",
                " Flt3 Ligand",
                " Chemotherapy",
                " Cdx-1140",
                " Cdx-301",
                " Pld",
                " Mabs",
                " Anthracycline",
                " Kinase Inhibitors",
                " Immunosuppressives"
            ],
            "BIOMARKERS": [
                "PD-L1",
                " AML",
                " Flt3 mutation"
            ]
        }]
```

### 2. GetClinicalTrialDetailsView

This endpoint retrieves detailed information about clinical trials based on their NCT number. It also extracts the names of drugs and biomarkers from the trial interventions using the GPT-4 model.

**URL:** `/chat/clinicaltrialsdetails/`

**Method:** `GET`

**Query Parameters:**
- `nct_number` (string, required): The NCT number of the clinical trial.

**Response:**
- `filtered_trials` (JSON object): The payload containing the detailed clinical trials information, including drugs and biomarkers. An error message is returned if the trial information is unavailable.


**Example Request:**

- /chat/clinicaltrialsdetails/?nct_number=NCT06229392

**Example Response:**

``` json
{
    "filtered_trials": [
        {
            "NCT_NUMBER": "NCT06229392",
            "STUDY_TITLE": "A Study to Evaluate the Intratumoral Influenza Vaccine Administration in Patients With Breast Cancer",
            "STUDY_URL": "https://www.clinicaltrials.gov/study/NCT06229392",
            "STUDY_STATUS": "RECRUITING",
            "CONDITIONS": [
                "Breast Cancer",
                "Breast Cancer Triple Negative"
            ],
            "PHASES": [
                "PHASE1"
            ],
            "LOCATIONS": [
                {
                    "Location Facility": "Rush University Medical Center",
                    "Location City": "Chicago",
                    "Location State": "Illinois",
                    "Location Zip": "60612",
                    "Location Country": "United States"
                }
            ],
            "START_DATE": "2024-02-28",
            "COMPLETION_DATE": "2025-12-31",
            "MINIMUMAGE": "18 Years",
            "MAXIMUMAGE": null,
            "POINT_OF_CONTACT": "{'Name': None, 'Organization': None, 'Email': None, 'Phone': None}",
            "PHASE": null,
            "SEX": "ALL",
            "ELIGIBILITY_CRITERIA": "Eligibility includes ECOG 0-2, planning to receive standard of care neoadjuvant chemotherapy per NCCN guidelines, with histologically or cytologically confirmed invasive breast cancer that is either triple-negative or HER2+. Exclusions are uncontrolled illnesses, conditions making participation not in the best interest per investigator's opinion, egg allergy, current anticoagulant, corticosteroids, or immunosuppressive therapy, allergic reactions to similar compounds as the flu vaccine, or history of Guillain-Barré syndrome.",
            "PRIMARY_OUTCOMES": "The primary outcome is to assess the safety of intratumoral influenza vaccine administration in breast cancer patients receiving neoadjuvant chemotherapy, specifically by evaluating the percentage of patients experiencing dose-limiting toxicity (DLT) within 3 months post-surgery.",
            "SECONDARY_OUTCOMES": "No secondary outcomes are specified.",
            "INTERVENTIONS": "The study involves administering 2 doses of the seasonal flu vaccine directly into the breast cancer tissue. Participants are divided into cohorts receiving fixed doses of Fluzone Quadrivalent, with the first cohort receiving a standard dose intratumorally on day -6 +/- 1 day and day 0 or 1, and the second cohort receiving a high dose on the same schedule. The vaccine administration is performed by a surgeon/oncologist or via guided ultrasound by a breast radiologist, depending on tumor palpability.",
            "STUDY_TYPE": "INTERVENTIONAL",
            "ENROLLMENT": 18,
            "ZIP_STR": "60612",
            "DRUGS": [
                "Fluzone Quadrivalent"
            ],
            "BIOMARKERS": []
        }
    ]
}
``` 