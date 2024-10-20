import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
try:
    from transformers import pipeline
except ImportError:
    pipeline = None

class SentimentAnalysis: 
    def __init__(self, df=None):
        """初始化时可以选择传入数据集"""
        self.df = df

    def load_data(self, path):
        """从指定路径加载数据集"""
        self.df = pd.read_csv(path)

        return pd.DataFrame(text_columns, columns=['Column Name', 'Average Entry Length', 'Unique Entries'])



    def get_text_columns(self):
        """Identifies text columns and returns a DataFrame with their details."""
        text_columns = []
        for col in self.df.columns:
            if self.df[col].dtype == 'object':  # Checking for text columns
                avg_len = self.df[col].str.len().mean()
                unique_entries = self.df[col].nunique()
                text_columns.append([col, avg_len, unique_entries])
        return pd.DataFrame(text_columns, columns=['Column Name', 'Average Entry Length', 'Unique Entries'])

    def vader_sentiment_analysis(self, data):
        """Performs sentiment analysis using VADER."""
        analyzer = SentimentIntensityAnalyzer()
        scores, sentiments = [], []
        for text in data.fillna(''):
            score = analyzer.polarity_scores(text)['compound']
            scores.append(score)
            sentiments.append('positive' if score > 0 else 'negative' if score < 0 else 'neutral')
        return scores, sentiments

    def textblob_sentiment_analysis(self, data):
        """Performs sentiment analysis using TextBlob."""
        scores, sentiments, subjectivities = [], [], []
        for text in data.fillna(''):
            blob = TextBlob(text)
            score = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            scores.append(score)
            subjectivities.append(subjectivity)
            sentiments.append('positive' if score > 0 else 'negative' if score < 0 else 'neutral')
        return scores, sentiments, subjectivities

    def distilbert_sentiment_analysis(self, data):
        """Performs sentiment analysis using a DistilBERT pipeline."""
        if pipeline is None:
            raise ImportError("Transformers library not installed.")
        sentiment_pipeline = pipeline('sentiment-analysis')
        scores, sentiments = [], []
        for text in data.fillna(''):
            result = sentiment_pipeline(text)[0]
            scores.append(result['score'])
            sentiments.append(result['label'])
        return scores, sentiments
