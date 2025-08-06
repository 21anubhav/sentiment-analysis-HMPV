import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
import nltk

# Download stopwords if not present
nltk.download('stopwords')

# Load the input CSV
df = pd.read_csv("input_data.csv")

# Show basic info
print("Total Samples:", len(df))
print("\nAvailable Columns:", df.columns.tolist())

# --- Emotion Distribution ---
print("\n📊 Emotion Value Counts:")
emotion_counts = df['emotion'].value_counts()
print(emotion_counts)

# --- Sentiment Distribution ---
print("\n📊 Sentiment Value Counts:")
sentiment_counts = df['sentiment'].value_counts()
print(sentiment_counts)

# --- Most Common Words ---
print("\n🔍 Extracting most common words from 'comment' column...")

def clean_text(text):
    text = re.sub(r"http\S+|www\S+", '', text.lower())
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Clean and tokenize comments
all_words = []
stop_words = set(stopwords.words("english"))
df['comment'] = df['comment'].astype(str)  # Ensure no NaNs

for comment in df['comment']:
    cleaned = clean_text(comment)
    words = cleaned.split()
    filtered_words = [word for word in words if word not in stop_words]
    all_words.extend(filtered_words)

word_freq = Counter(all_words)
most_common_words = word_freq.most_common(20)

print("\n📝 Top 20 Most Common Words:")
for word, count in most_common_words:
    print(f"{word}: {count}")

# --- Optional Visualizations ---

# Bar Plot: Emotion
plt.figure(figsize=(10, 4))
emotion_counts.plot(kind='bar', color='skyblue')
plt.title("Emotion Distribution")
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Bar Plot: Sentiment
plt.figure(figsize=(6, 4))
sentiment_counts.plot(kind='bar', color='lightcoral')
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment (-1 = Neg, 0 = Neutral, 1 = Pos)")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# WordCloud of most common words
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title("Word Cloud of Top Words in Comments")
plt.tight_layout()
plt.show()
