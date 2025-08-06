import pandas as pd
from collections import Counter
import os

def majority_vote(predictions_list):
    counter = Counter(predictions_list)
    most_common = counter.most_common()

    # Tie-breaking logic
    if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
        for pred in predictions_list:
            if pred not in ['neutral', 'unemotional']:
                return pred
        return most_common[0][0]
    return most_common[0][0]

def build_ensemble(output_csv, model_files):
    all_preds = {}
    for model, (filename, colname) in model_files.items():
        full_path = os.path.join("results_ALL", filename)
        try:
            df = pd.read_csv(full_path)
            if colname not in df.columns:
                print(f"⚠️ Column '{colname}' not found in {filename}")
                continue
            all_preds[model] = df[colname]
        except FileNotFoundError:
            print(f"⚠️ File not found: {filename}")
    
    if not all_preds:
        print("❌ No valid predictions found. Exiting.")
        return
    
    result_df = pd.DataFrame(all_preds)
    result_df["ensemble_result"] = result_df.apply(lambda row: majority_vote(row.tolist()), axis=1)
    result_df.to_csv(output_csv, index=False)
    print(f"✅ Ensemble results saved to '{output_csv}'")

# Define model files

sentiment_models = {
    "vader": ("rule_based_predictions.csv", "vader_sentiment"),
    "textblob": ("rule_based_predictions.csv", "textblob_sentiment"),
    "svm": ("predictions_svm_sentiment.csv", "predicted"),
    "randomforest": ("predictions_randomforest_sentiment.csv", "predicted"),
    "naivebayes": ("predictions_naivebayes_sentiment.csv", "predicted"),
    "cnn": ("predictions_cnn_sentiment.csv", "predicted"),
    "lstm": ("predictions_lstm_sentiment.csv", "predicted"),
}

emotion_models = {
    "vader": ("rule_based_predictions.csv", "vader_emotion"),
    "textblob": ("rule_based_predictions.csv", "textblob_emotion"),
    "svm": ("predictions_svm_emotion.csv", "predicted"),
    "randomforest": ("predictions_randomforest_emotion.csv", "predicted"),
    "naivebayes": ("predictions_naivebayes_emotion.csv", "predicted"),
    "cnn": ("predictions_cnn_emotion.csv", "predicted"),
    "lstm": ("predictions_lstm_emotion.csv", "predicted"),
}

# Run ensemble generation
build_ensemble("sentiment_ensemble_output.csv", sentiment_models)
build_ensemble("emotion_ensemble_output.csv", emotion_models)
