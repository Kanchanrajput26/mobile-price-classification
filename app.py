import glob, os, joblib, pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from metrics_utils import compute_metrics

st.set_page_config(page_title="Classifier Evaluator", layout="wide")
st.title("ML Classification Model Evaluator")

TARGET = joblib.load("model/target_name.joblib")
le = joblib.load("model/label_encoder.joblib")
files = {os.path.basename(f)[:-7].replace("_", " ").title(): f
         for f in glob.glob("model/*.joblib")
         if not f.endswith(("label_encoder.joblib", "target_name.joblib"))}

st.sidebar.header("Controls")
choice = st.sidebar.selectbox("Select model", sorted(files))
up = st.sidebar.file_uploader("Upload test data (CSV)", type="csv")
st.sidebar.caption(f"CSV must include all features and the target column '{TARGET}'.")

if up is None:
    st.info("Upload a test CSV from the sidebar to begin.")
    st.stop()

df = pd.read_csv(up)
if TARGET not in df.columns:
    st.error(f"Target column '{TARGET}' not found in uploaded file.")
    st.stop()

st.subheader("Data preview")
st.dataframe(df.head())

model = joblib.load(files[choice])          # no retraining
X = df.drop(columns=[TARGET])
y = le.transform(df[TARGET])
metrics, pred = compute_metrics(model, X, y)

st.subheader(f"Evaluation metrics: {choice}")
cols = st.columns(len(metrics))
for c, (k, v) in zip(cols, metrics.items()):
    c.metric(k, f"{v:.4f}")

left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y, pred), annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    st.pyplot(fig)
with right:
    st.subheader("Classification report")
    st.text(classification_report(y, pred, target_names=[str(c) for c in le.classes_]))
