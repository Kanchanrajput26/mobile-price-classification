import os, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from metrics_utils import compute_metrics

# ====== EDIT THESE ======
DATA_PATH = "train.csv"      # your downloaded Kaggle/UCI csv
TARGET = "price_range"              # name of the label column
# ========================

df = pd.read_csv(DATA_PATH).dropna(subset=[TARGET])
print("Shape:", df.shape, "| features:", df.shape[1] - 1)

X = df.drop(columns=[TARGET])
le = LabelEncoder()
y = le.fit_transform(df[TARGET])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=40, stratify=y)

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

def make_pre():
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore",
                                               sparse_output=False))]), cat_cols),
    ])

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(probability=True, random_state=42),
}

os.makedirs("model", exist_ok=True)
rows = []
for name, clf in models.items():
    pipe = Pipeline([("pre", make_pre()), ("clf", clf)])
    pipe.fit(X_train, y_train)
    m, _ = compute_metrics(pipe, X_test, y_test)
    rows.append({"ML Model Name": name, **{k: round(v, 4) for k, v in m.items()}})
    joblib.dump(pipe, f"model/{name.replace(' ', '_').lower()}.joblib")

joblib.dump(le, "model/label_encoder.joblib")
# test_data.csv keeps the ORIGINAL target column for upload in the app
test_df = X_test.copy()
test_df[TARGET] = le.inverse_transform(y_test)
test_df.to_csv("test_data.csv", index=False)
joblib.dump(TARGET, "model/target_name.joblib")

res = pd.DataFrame(rows)
res.to_csv("results.csv", index=False)
print(res.to_markdown(index=False))   # paste this table into README
