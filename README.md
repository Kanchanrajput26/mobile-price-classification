# Mobile Price Range Classification

## a. Problem Statement
The goal is to predict the price range of a mobile phone (low, medium, high or very high cost) from its hardware specifications such as RAM, battery power, screen resolution, memory and connectivity features. This is a **multi-class classification** problem with 4 classes. Five classifiers are trained and compared on the same dataset using six evaluation metrics, and the trained models are served through an interactive Streamlit web app.

## b. Dataset Description  [1 mark]
- **Source:** Kaggle, "Mobile Price Classification" (`train.csv`)
- **Instances:** 2,000
- **Features:** 20 (battery_power, blue, clock_speed, dual_sim, fc, four_g, int_memory, m_dep, mobile_wt, n_cores, pc, px_height, px_width, ram, sc_h, sc_w, talk_time, three_g, touch_screen, wifi)
- **Target:** `price_range` with 4 balanced classes (0 = low, 1 = medium, 2 = high, 3 = very high), 500 samples each
- **Missing values:** none
- **Preprocessing:** median imputation and standard scaling for numeric features (one-hot encoding is built in for categorical features, though this dataset has none), applied through a scikit-learn Pipeline so the saved model includes its preprocessing
- **Train/test split:** 80/20, stratified, `random_state=42` (400 test rows, saved as `test_data.csv`)

## c. GitHub Repository  [1 mark]
https://github.com/Kanchanrajput26/mobile-price-classification

## d. Models Used  [5 marks]

Models implemented on the same dataset: Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes and Support Vector Machine (RBF kernel).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|:--------------|---------:|----:|----------:|-------:|---:|----:|
| Logistic Regression | 0.9650 | 0.9986 | 0.9654 | 0.9650 | 0.9651 | 0.9534 |
| Decision Tree | 0.8275 | 0.8922 | 0.8288 | 0.8275 | 0.8267 | 0.7709 |
| KNN | 0.5375 | 0.7827 | 0.5565 | 0.5375 | 0.5418 | 0.3851 |
| Naive Bayes | 0.8075 | 0.9529 | 0.8064 | 0.8075 | 0.8069 | 0.7434 |
| Support Vector Machine | 0.9100 | 0.9878 | 0.9112 | 0.9100 | 0.9104 | 0.8802 |

*Notes: AUC is computed one-vs-rest (weighted) for the 4 classes. Precision, Recall and F1 use weighted averaging.*

### Observations on model performance  [3 marks]

| ML Model Name | Observation about model performance |
|:--------------|:------------------------------------|
| Logistic Regression | Best performer on every metric (accuracy 96.5%, AUC 0.999, MCC 0.953). Price range appears to be almost linearly separable in the scaled feature space, with RAM being a very strong predictor. |
| Decision Tree | Moderate performance (accuracy 82.8%). A single tree with limited depth splits on one feature at a time, so it cannot capture the smooth linear relationship as well, and it is prone to overfitting. |
| KNN | Weakest model (accuracy 53.8%, MCC 0.385). With 20 features, many of them weakly informative or binary, distances are diluted by irrelevant dimensions, so neighbors are not reliably from the same price class. |
| Naive Bayes | Reasonable accuracy (80.8%) and a high AUC (0.953), so its probability ranking is good even where the hard predictions are weaker. The independence assumption is only partly true for phone specs. |
| Support Vector Machine | Second best (accuracy 91.0%, AUC 0.988). The RBF kernel handles the boundaries well, but it is slightly less accurate than the linear model, which suggests the data is close to linear. |
| **Overall, Winner for your dataset?** | **Logistic Regression**, with the highest Accuracy, AUC, Precision, Recall, F1 and MCC. It is also simple, fast and interpretable. |

*Note: the test set has only 400 rows, so scores vary a few percent with the random split, but the ranking of the models was consistent across runs.*

## Streamlit App
- **Live app:** https://mobile-price-classification-klqkhy2ovafdtgpkgevjtm.streamlit.app/
- **Features:** CSV test-data upload, model selection dropdown, evaluation metrics display, confusion matrix and classification report. The app loads the saved model and does not retrain.

## Repository Structure
```
project-folder/
 -- app.py
 -- train.py
 -- metrics_utils.py
 -- requirements.txt
 -- README.md
 -- test_data.csv
 -- model/   (saved .joblib files for all five classifiers + label encoder)
```

## How to Run
```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
```
