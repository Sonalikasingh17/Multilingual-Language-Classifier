#  multilingual_classifier_notebook.py
#  Run from project root:  python multilingual_classifier_notebook.py
# ---------------------------------------------------------------
import os, sys, json, pickle, joblib, warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_curve, auc)
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, \
                                          QuadraticDiscriminantAnalysis
warnings.filterwarnings("ignore")

# ---------------------- constants --------------------------------
LANGS: List[str] = [
    'af-ZA','da-DK','de-DE','en-US','es-ES','fr-FR','fi-FI','hu-HU','is-IS','it-IT',
    'jv-ID','lv-LV','ms-MY','nb-NO','nl-NL','pl-PL','pt-PT','ro-RO','ru-RU','sl-SL',
    'sv-SE','sq-AL','sw-KE','tl-PH','tr-TR','vi-VN','cy-GB'
]
continent_lookup = {
    'ZA':'Africa','KE':'Africa','AL':'Europe','GB':'Europe','DK':'Europe','DE':'Europe',
    'ES':'Europe','FR':'Europe','FI':'Europe','HU':'Europe','IS':'Europe','IT':'Europe',
    'ID':'Asia','LV':'Europe','MY':'Asia','NO':'Europe','NL':'Europe','PL':'Europe',
    'PT':'Europe','RO':'Europe','RU':'Europe','SL':'Europe','SE':'Europe','PH':'Asia',
    'TR':'Asia','VN':'Asia','US':'North America'
}
ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)

# ---------------------- helper functions -------------------------
def map_continent(locale:str)->str:
    return continent_lookup[locale.split('-')[1]]

def load_massive_splits(locales:List[str])->Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """Loads train/validation/test once per locale (cached)."""
    buckets = {"train":[], "validation":[], "test":[]}
    for loc in locales:
        ds = load_dataset("qanastek/MASSIVE", loc, trust_remote_code=True)
        for split in buckets:
            tmp = pd.DataFrame(ds[split])[["utt"]].copy()
            tmp["locale"] = loc
            buckets[split].append(tmp)
    dfs = {k: pd.concat(v, ignore_index=True) for k,v in buckets.items()}
    for df in dfs.values():
        df["continent"] = df["locale"].apply(map_continent)
    return dfs["train"], dfs["validation"], dfs["test"]

# ---------------------- load data --------------------------------
print("🗃  Loading MASSIVE splits ... (first run may take ~2-3 min)")
train_df, val_df, test_df = load_massive_splits(LANGS)
print("Shapes:", train_df.shape, val_df.shape, test_df.shape)

# ---------------------- EDA plots --------------------------------
sns.set_theme(style="darkgrid")
fig1 = plt.figure(figsize=(10,4))
sns.countplot(data=train_df, y="locale", order=train_df.locale.value_counts().index)
plt.title("Train samples per locale"); plt.tight_layout()
fig1.savefig(ARTIFACTS/"samples_per_locale.png"); plt.close()

# ---------------------- LANGUAGE MODEL ---------------------------
print("\n🔠 Training language classifier ...")
lang_pipe = make_pipeline(
    TfidfVectorizer(analyzer="char_wb", ngram_range=(1,3), max_features=10000),
    MultinomialNB(alpha=0.1))
lang_pipe.fit(train_df.utt, train_df.locale)

val_pred = lang_pipe.predict(val_df.utt)
test_pred= lang_pipe.predict(test_df.utt)
print("Validation acc:", accuracy_score(val_df.locale,val_pred).round(4))
print("Test acc:", accuracy_score(test_df.locale,test_pred).round(4))
print(classification_report(val_df.locale,val_pred)[:700])

cm = confusion_matrix(val_df.locale, val_pred, labels=LANGS)
fig2 = plt.figure(figsize=(12,10))
sns.heatmap(cm, cmap="mako", cbar=False, xticklabels=False, yticklabels=False)
plt.title("Language confusion matrix (val)"); plt.tight_layout()
fig2.savefig(ARTIFACTS/"lang_confusion.png"); plt.close()

# ---------------------- CONTINENT MODEL --------------------------
print("\n🌍 Training continent classifier ...")
vec = TfidfVectorizer(max_features=15000)
X_train = vec.fit_transform(train_df.utt)
X_val   = vec.transform(val_df.utt)
X_test  = vec.transform(test_df.utt)

svd = TruncatedSVD(n_components=100, random_state=42)
X_train_r = svd.fit_transform(X_train)
X_val_r   = svd.transform(X_val)
X_test_r  = svd.transform(X_test)

lda = LinearDiscriminantAnalysis()
qda = QuadraticDiscriminantAnalysis()

lda.fit(X_train_r, train_df.continent)
qda.fit(X_train_r, train_df.continent)

print("LDA Validation acc:", accuracy_score(val_df.continent, lda.predict(X_val_r)).round(4))
print("LDA Test acc      :", accuracy_score(test_df.continent, lda.predict(X_test_r)).round(4))
print("\n",classification_report(val_df.continent, lda.predict(X_val_r)))

print("QDA Validation acc:", accuracy_score(val_df.continent, qda.predict(X_val_r)).round(4))
print("QDA Test acc      :", accuracy_score(test_df.continent, qda.predict(X_test_r)).round(4))

# ROC curve for LDA (one-vs-rest)
from sklearn.preprocessing import label_binarize
classes = ["Africa","Asia","Europe","North America"]
y_val_bin = label_binarize(val_df.continent, classes)
y_score   = lda.predict_proba(X_val_r)
fig3 = plt.figure(figsize=(6,4))
for i,c in enumerate(classes):
    fpr,tpr,_ = roc_curve(y_val_bin[:,i], y_score[:,i])
    plt.plot(fpr,tpr,label=f"{c} AUC:{auc(fpr,tpr):.2f}")
plt.plot([0,1],[0,1],'k--'); plt.legend(); plt.title("LDA ROC curves"); plt.tight_layout()
fig3.savefig(ARTIFACTS/"lda_roc.png"); plt.close()

# ---------------------- SAVE ARTEFACTS ---------------------------
print("\n💾 Saving artefacts to 'artifacts/' ...")
joblib.dump(lang_pipe,          ARTIFACTS/"language_pipeline.pkl")
joblib.dump(vec,                ARTIFACTS/"continent_vectorizer.pkl")
joblib.dump(svd,                ARTIFACTS/"continent_svd.pkl")
joblib.dump(lda,                ARTIFACTS/"continent_lda_model.pkl")
joblib.dump(qda,                ARTIFACTS/"continent_qda_model.pkl")

# Label encoder for continents
import pickle, json
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder().fit(classes)
joblib.dump(le, ARTIFACTS/"continent_label_encoder.pkl")

perf = {
    "language_val_acc": float(accuracy_score(val_df.locale,val_pred)),
    "language_test_acc": float(accuracy_score(test_df.locale,test_pred)),
    "continent_val_acc": float(accuracy_score(val_df.continent, lda.predict(X_val_r))),
    "continent_test_acc": float(accuracy_score(test_df.continent, lda.predict(X_test_r)))
}
with open(ARTIFACTS/"model_performance.pkl","wb") as f: pickle.dump(perf,f)

# ---------------------- SMOKE TEST -------------------------------
print("\n🚦 Reloading pickles for smoke-test ...")
lang_pipe2 = joblib.load(ARTIFACTS/"language_pipeline.pkl")
lda2       = joblib.load(ARTIFACTS/"continent_lda_model.pkl")
vec2       = joblib.load(ARTIFACTS/"continent_vectorizer.pkl")
svd2       = joblib.load(ARTIFACTS/"continent_svd.pkl")
le2        = joblib.load(ARTIFACTS/"continent_label_encoder.pkl")

print("Reloaded language test acc:",
      accuracy_score(test_df.locale, lang_pipe2.predict(test_df.utt)).round(4))
X_test_r2 = svd2.transform(vec2.transform(test_df.utt))
print("Reloaded LDA continent test acc:",
      accuracy_score(test_df.continent, lda2.predict(X_test_r2)).round(4))

samples = ["Hello how are you?",
           "Bonjour, comment ça va?",
           "Guten Tag, wie geht's?"]
print("\nSample predictions:")
for s in samples:
    lang = lang_pipe2.predict([s])[0]
    cont = le2.inverse_transform([lda2.predict(svd2.transform(vec2.transform([s])))])
    print(f"  '{s[:25]}...' ➜ {lang} / {cont}")

print("Done ✔")
# ---------------------------------------------------------------
