import feedparser
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer


nltk.download('vader_lexicon')


sia = SentimentIntensityAnalyzer()


rss_url = 'https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en'
feed = feedparser.parse(rss_url)


headlines = [entry.title for entry in feed.entries]


data = []
for headline in headlines:
    score = sia.polarity_scores(headline)
    compound = score['compound']
    sentiment = "Positive" if compound >= 0.05 else "Negative" if compound <= -0.05 else "Neutral"
    data.append({
        'Headline': headline,
        'Compound Score': compound,
        'Sentiment': sentiment
    })


df = pd.DataFrame(data)


print(df.head())


plt.figure(figsize=(8, 5))
sns.countplot(x='Sentiment', data=df, palette='coolwarm')
plt.title('Sentiment Distribution of News Headlines')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
sns.histplot(df['Compound Score'], bins=20, kde=True, color='skyblue')
plt.title('Distribution of Compound Sentiment Scores')
plt.xlabel('Compound Score')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
