import torch
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import os
from transformers import pipeline

# Load the sentiment analysis model from local path
model_path = "../sentimentAnalyzer/Models/models--distilbert--distilbert-base-uncased-finetuned-sst-2-english/snapshots/714eb0fa89d2f80546fda750413ed43d93601a13"

# Define the pipeline
analyzer = pipeline("text-classification", model=model_path)

# Function to analyze a single review
def sentiment_analyzer(review):
    sentiment = analyzer(review)
    return sentiment[0]['label']

# Function to generate a pie chart of sentiment counts
def sentiment_bar_chart(df):
    sentiment_counts = df['Sentiment'].value_counts()

    # Create a pie chart
    fig, ax = plt.subplots()
    sentiment_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%', colors=['green', 'red'])
    ax.set_title('Review Sentiment Distribution')
    ax.set_ylabel('')  # Remove y-label for cleaner look
    return fig

# Main function to handle file input and perform analysis
def read_reviews_and_analyze_sentiment(file_object):
    if file_object is None:
        raise ValueError("No file uploaded.")

    file_path = file_object.name if hasattr(file_object, "name") else file_object

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_excel(file_path)

    # Ensure the Excel file contains the expected 'Reviews' column
    if 'Reviews' not in df.columns:
        raise ValueError("Excel file must contain a column named 'Reviews'.")

    # Perform sentiment analysis
    df['Sentiment'] = df['Reviews'].apply(sentiment_analyzer)

    # Generate the sentiment pie chart
    chart_object = sentiment_bar_chart(df)

    return df, chart_object

# Gradio interface
demo = gr.Interface(
    fn=read_reviews_and_analyze_sentiment,
    inputs=[gr.File(file_types=[".xlsx"], label="Upload your Excel file (must contain a 'Reviews' column)")],
    outputs=[
        gr.Dataframe(label="Sentiment Results"),
        gr.Plot(label="Sentiment Distribution Chart")
    ],
    title="📊 Sentiment Analyzer",
    description="Upload an Excel (.xlsx) file containing customer reviews in a column named 'Reviews'. The model will analyze and visualize sentiment."
)

# Launch the app
demo.launch()
