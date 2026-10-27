# Phishing Website Classification — ML Assignment 2

**Student:** Aravinth
**BITS ID:** 2025NS04081
**Program:** M.Tech (DSE), Semester 1
**Subject:** Machine Learning (DSEZG565)

---

## a. Problem Statement

Phishing websites impersonate legitimate services to steal credentials, financial
details, and personal information from unsuspecting users. Manually identifying
phishing URLs does not scale with the volume of new malicious domains created
daily. This project builds and compares 5 classical machine learning classifiers
that predict whether a website is **phishing** or **legitimate**, using
structural/lexical features extracted from the URL and the webpage itself
(no need to actually visit or render the site). The best-performing model is
deployed as an interactive Streamlit web app so predictions can be explored
without writing any code.

## b. Dataset Description

- **Name:** Web Page Phishing Detection Dataset
- **Source:** [Kaggle — shashwatwork/web-page-phishing-detection-dataset](https://www.kaggle.com/datasets/shashwatwork/web-page-phishing-detection-dataset)
- **Instances:** 11,430 websites
- **Features:** 87 numeric features engineered from the URL structure (e.g.
  `length_url`, `nb_dots`, `nb_hyphens`, `nb_at`), the domain (e.g.
  `domain_age`, `domain_registration_length`, `whois_registered_domain`), and
  the page content (e.g. `nb_hyperlinks`, `ratio_intHyperlinks`, `login_form`,
  `iframe`). This comfortably exceeds the assignment's minimum of 12 features.
- **Target:** `status` — binary (`legitimate` / `phishing`), remapped to
  `Result` (0 = legitimate, 1 = phishing) during preprocessing.
- **Class balance:** perfectly balanced — 5,715 legitimate / 5,715 phishing
  (50% / 50%).
- **Missing values:** none.
- **Identifier column:** `url` (the raw website URL) is retained throughout
  the pipeline as a **display-only** column — it is never used as a model
  feature, only shown in the app's results table so predictions are
  human-readable.
- **Train/test split:** 80% / 20% (stratified on `Result`), `random_state=42`
  → 9,144 training rows, 2,286 test rows (saved as `test_data.csv`).

### Setting up the dataset

If you download the dataset fresh from Kaggle, just drop the CSV into the
`data/` folder **without renaming it** — Kaggle exports it as
`dataset_phishing.csv`, and `model/data_prep.py` expects it at exactly
`data/dataset_phishing.csv`. No other setup needed, the script handles the
rest (cleaning, split, saving).

## c. GitHub Repository Link

**https://github.com/2025NS04081/dsezg565-ml-phishing-classifier**

**Live Streamlit App:** https://dsezg565-ml-phishing-classifier.streamlit.app/

## d. Models Used

All 5 models were trained on the identical train/test split, using
standardized features (`StandardScaler`, fit on training data only) and a
fixed `random_state=42` for reproducibility:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Naive Bayes Classifier (Gaussian)
5. Support Vector Machine (RBF kernel)

### Comparison Table (held-out test set, 2,286 rows)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9361 | 0.9814 | 0.9392 | 0.9326 | 0.9359 | 0.8723 |
| Decision Tree | 0.9353 | 0.9353 | 0.9307 | 0.9405 | 0.9356 | 0.8706 |
| kNN | 0.9396 | 0.9788 | 0.9539 | 0.9239 | 0.9387 | 0.8797 |
| Naive Bayes | 0.6881 | 0.9277 | 0.9424 | 0.4007 | 0.5623 | 0.4597 |
| **Support Vector Machine** | **0.9541** | **0.9883** | **0.9577** | **0.9501** | **0.9539** | **0.9082** |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong, well-calibrated baseline (AUC 0.9814) — the phishing/legitimate classes are largely linearly separable once features are standardized. Fast to train, good precision-recall balance. |
| Decision Tree | Matches Logistic Regression on Accuracy/F1 but has a noticeably lower AUC (0.9353) than the other models, because a single unpruned tree produces coarse, poorly-calibrated probability estimates (few distinct leaf-node probabilities) rather than a smooth score. |
| kNN | Highest Precision (0.9539) among all models — when it predicts phishing, it is rarely wrong — but slightly lower Recall, meaning it misses a few more phishing sites than SVM/Logistic Regression. Performance is highly dependent on feature scaling, which was applied here. |
| Naive Bayes | Clearly the weakest model: high Precision (0.9424) but very low Recall (0.4007), so it misses roughly 60% of actual phishing sites. This happens because Gaussian Naive Bayes assumes features are conditionally independent given the class, but many URL features are strongly correlated (e.g. `length_url`, `nb_dots`, `nb_slash` all increase together for complex/suspicious URLs) — violating this assumption biases the model toward under-predicting the positive class. |
| Support Vector Machine | **Best performer on every single metric** (Accuracy, AUC, Precision, Recall, F1, MCC). The RBF kernel can model non-linear decision boundaries between legitimate and phishing sites that the purely linear Logistic Regression cannot, at the cost of being the most computationally expensive model to train among the five. |
| **Overall, Winner for your dataset?** | **Support Vector Machine** — it dominates across all 6 metrics, most notably MCC (0.9082), which is the most reliable single metric for a balanced binary classification problem since it accounts for all four confusion-matrix quadrants simultaneously. |

---

## Project Structure

```
dsezg565-ml-phishing-classifier/
├── app.py                          # Streamlit web application
├── requirements.txt
├── README.md
├── test_data.csv                   # held-out test split (required deliverable)
├── sample_upload_test.csv          # small 50-row sample for testing the app's upload feature
├── data/
│   ├── dataset_phishing.csv        # raw CSV as downloaded from Kaggle
│   └── raw_phishing_websites.csv   # cleaned full dataset (11,430 rows)
└── model/
    ├── data_prep.py                # loads data, creates train/test split
    ├── train_models.py             # trains all 5 models, computes metrics
    ├── train_data.csv
    ├── metrics_results.csv         # 5 models x 6 metrics (used by README + app)
    └── saved_models/               # trained model + scaler artifacts (.joblib)
```

## How to Run Locally

```bash
git clone https://github.com/2025NS04081/dsezg565-ml-phishing-classifier.git
cd dsezg565-ml-phishing-classifier
pip install -r requirements.txt

# data/dataset_phishing.csv is already included in this repo, so this
# step is only needed if you want to regenerate the split from scratch
python model/data_prep.py      # regenerate train/test split
python model/train_models.py   # retrain all 5 models + metrics

streamlit run app.py           # launch the web app locally
```

## Streamlit App Features

- **Dataset upload (CSV):** upload your own test CSV, or use the bundled
  `test_data.csv` with one click. A smaller `sample_upload_test.csv` (50 rows,
  same columns) is also included in the repo if you just want to quickly try
  the upload feature without using the full test set.
- **Model selection dropdown:** switch between all 5 trained models.
- **Evaluation metrics display:** both the official held-out comparison table
  and live-computed metrics on whatever data you upload.
- **Confusion matrix & classification report:** generated on the fly for the
  selected model and uploaded data.
- **Model Agreement Dashboard:** runs all 5 models simultaneously on the same
  uploaded rows and shows a per-row vote breakdown, consensus prediction, and
  overall agreement rate — useful for spotting genuinely ambiguous URLs.
- **Feature Importance:** visualizes which features (e.g. `nb_dots`,
  `length_url`) most influenced the Decision Tree / Logistic Regression
  predictions.
