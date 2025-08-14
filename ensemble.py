# ensemble_full_lightgbm.py
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier

os.makedirs("results/ensemble_models", exist_ok=True)

def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)

def ensure_text_column(df):
    if 'text' not in df.columns and 'comment' in df.columns:
        df = df.rename(columns={'comment': 'text'})
    return df

def check_alignment(base, other, name):
    if not base.equals(other):
        raise ValueError(f"Text alignment mismatch with {name} file.")

def build_and_run_lightgbm(X_df, y_series, out_texts, outfile_path):
    # convert everything to string before encoding to avoid mixed-type problems
    # fit label encoder on union of true labels + all feature values
    all_values = pd.concat([y_series.astype(str)] + [X_df[col].astype(str) for col in X_df.columns], ignore_index=True)
    le = LabelEncoder().fit(all_values)
    y_enc = le.transform(y_series.astype(str))
    X_enc = X_df.copy()
    for col in X_enc.columns:
        X_enc[col] = le.transform(X_enc[col].astype(str))

    model = LGBMClassifier(random_state=42)
    model.fit(X_enc, y_enc)
    preds_enc = model.predict(X_enc)
    acc = accuracy_score(y_enc, preds_enc)

    # inverse transform preds to original labels (strings)
    preds = le.inverse_transform(preds_enc)
    out_df = pd.DataFrame({
        'text': out_texts,
        'actual': y_series.astype(str),
        'ensemble_predicted': preds
    })
    out_df.to_csv(outfile_path, index=False)
    return acc

# -----------------------
# SENTIMENT ENSEMBLE
# -----------------------
# file paths (your folder structure)
rf_path_sent   = "results/ml_models/predictions_randomforest_sentiment.csv"
svm_path_sent  = "results/ml_models/predictions_svm_sentiment.csv"
nb_path_sent   = "results/ml_models/predictions_naivebayes_sentiment.csv"
cnn_path_sent  = "results/dl_models/predictions_cnn_sentiment.csv"
lstm_path_sent = "results/dl_models/predictions_lstm_sentiment.csv"
rule_path_sent = "results/rule_based_models/rule_based_predictions.csv"

# load
rf_df = load_csv(rf_path_sent)
svm_df = load_csv(svm_path_sent)
nb_df = load_csv(nb_path_sent)
cnn_df = load_csv(cnn_path_sent)
lstm_df = load_csv(lstm_path_sent)
rule_df = load_csv(rule_path_sent)

# ensure 'text' present
rf_df  = ensure_text_column(rf_df)
svm_df = ensure_text_column(svm_df)
nb_df  = ensure_text_column(nb_df)
cnn_df = ensure_text_column(cnn_df)
lstm_df= ensure_text_column(lstm_df)
rule_df= ensure_text_column(rule_df)

# alignment (use rf_df['text'] as base)
base_text = rf_df['text']
check_alignment(base_text, svm_df['text'], "SVM")
check_alignment(base_text, nb_df['text'], "NaiveBayes")
check_alignment(base_text, cnn_df['text'], "CNN")
check_alignment(base_text, lstm_df['text'], "LSTM")
# rule-based has vader/textblob - uses same file, check its text
check_alignment(base_text, rule_df['text'], "Rule-based")

# assemble features and true labels
X_sent = pd.DataFrame({
    'rf': rf_df['predicted'],
    'svm': svm_df['predicted'],
    'nb': nb_df['predicted'],
    'cnn': cnn_df['predicted'],
    'lstm': lstm_df['predicted'],
    'vader': rule_df['vader_sentiment'],
    'textblob': rule_df['textblob_sentiment']
})

# find true label column in rf_df (try multiple names)
for colname in ('actual', 'true_label', 'sentiment'):
    if colname in rf_df.columns:
        y_sent = rf_df[colname]
        break
else:
    raise ValueError("No actual/true_label/sentiment column found in RandomForest sentiment CSV.")

sent_out_path = "results/ensemble_models/ensemble_lightgbm_sentiment.csv"
acc_sent = build_and_run_lightgbm(X_sent, y_sent, base_text, sent_out_path)
print(f"Sentiment Ensemble (LightGBM) Accuracy: {acc_sent*100:.2f}%")

# -----------------------
# EMOTION ENSEMBLE
# -----------------------
rf_path_em = "results/ml_models/predictions_randomforest_emotion.csv"
svm_path_em = "results/ml_models/predictions_svm_emotion.csv"
nb_path_em = "results/ml_models/predictions_naivebayes_emotion.csv"
cnn_path_em = "results/dl_models/predictions_cnn_emotion.csv"
lstm_path_em = "results/dl_models/predictions_lstm_emotion.csv"

rf_em = load_csv(rf_path_em)
svm_em = load_csv(svm_path_em)
nb_em = load_csv(nb_path_em)
cnn_em = load_csv(cnn_path_em)
lstm_em = load_csv(lstm_path_em)

rf_em  = ensure_text_column(rf_em)
svm_em = ensure_text_column(svm_em)
nb_em  = ensure_text_column(nb_em)
cnn_em = ensure_text_column(cnn_em)
lstm_em= ensure_text_column(lstm_em)

base_text_em = rf_em['text']
check_alignment(base_text_em, svm_em['text'], "SVM (emotion)")
check_alignment(base_text_em, nb_em['text'], "NaiveBayes (emotion)")
check_alignment(base_text_em, cnn_em['text'], "CNN (emotion)")
check_alignment(base_text_em, lstm_em['text'], "LSTM (emotion)")

X_emot = pd.DataFrame({
    'rf': rf_em['predicted'],
    'svm': svm_em['predicted'],
    'nb': nb_em['predicted'],
    'cnn': cnn_em['predicted'],
    'lstm': lstm_em['predicted']
})

# find true label column in rf_em
for colname in ('actual', 'true_label', 'emotion'):
    if colname in rf_em.columns:
        y_em = rf_em[colname]
        break
else:
    raise ValueError("No actual/true_label/emotion column found in RandomForest emotion CSV.")

em_out_path = "results/ensemble_models/ensemble_lightgbm_emotion.csv"
acc_em = build_and_run_lightgbm(X_emot, y_em, base_text_em, em_out_path)
print(f"Emotion Ensemble (LightGBM) Accuracy: {acc_em*100:.2f}%")
