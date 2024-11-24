from flask import Flask, render_template, send_from_directory
import pandas as pd
import json
import os
import plotly.express as px
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# Load data functions
def load_influencers():
    return pd.read_csv(r"C:\\Users\\abhishek\\data\\influencers.csv")

def load_posts():
    with open(r"C:\Users\abhishek\data\social_media_posts_meaningful.json") as f:
        return pd.json_normalize(json.load(f))

def load_text_data():
    with open(r"C:\Users\abhishek\data\generated_text_data.txt") as f:
        return pd.DataFrame({'text': f.readlines()})

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for influencer data
@app.route('/influencers')
def influencers():
    df = load_influencers()
    return render_template('influencers.html', influencers=df.to_html())

# Route for sentiment analysis
@app.route('/sentiment')
def sentiment():
    text_data = load_text_data()
    text_data['sentiment_score'] = text_data['text'].apply(lambda text: TextBlob(text).sentiment.polarity)
    return render_template('sentiment.html', sentiment=text_data.to_html())

# Route for engagement prediction
@app.route('/predictions')
def predictions():
    df = pd.read_csv(r"C:\Users\abhishek\data\output\predicted_engagement.csv")
    return render_template('prediction.html', predictions=df.to_html())

# Route for hashtag analysis
@app.route('/hashtags')
def hashtags():
    df = pd.read_csv(r"C:\Users\abhishek\data\output\hashtag_counts.csv")
    return render_template('hashtag_analysis.html', hashtags=df.to_html())

# Route for visualizations
@app.route('/visualizations')
def visualizations():
    df = load_posts()
    fig = px.bar(df, x='category', y='total_engagement', title="Engagement by Category")
    graph_html = fig.to_html(full_html=False)
    return render_template('visualizations.html', graph_html=graph_html)

# Static route for downloading outputs
@app.route('/output/<filename>')
def download_file(filename):
    return send_from_directory('output', filename)

if __name__ == '__main__':
    app.run(debug=True)
