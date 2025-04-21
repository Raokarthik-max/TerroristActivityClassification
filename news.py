import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk


nltk.download('vader_lexicon')


sia = SentimentIntensityAnalyzer()


headlines = [
    "Stock markets rally as tech shares surge",
    "Global economic outlook worsens due to conflict",
    "New innovation in AI brings hope to healthcare",
    "Unemployment rates continue to rise in urban areas",
    "Company X reports record profits for the second quarter",
    "Severe weather causes damage across the Midwest"
]


data = []
for headline in headlines:
    scores = sia.polarity_scores(headline)
    compound = scores['compound']
    if compound >= 0.05:
        sentiment = "Positive"
    elif compound <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    data.append({'Headline': headline, 'Compound Score': compound, 'Sentiment': sentiment})


df = pd.DataFrame(data)


print(df)
