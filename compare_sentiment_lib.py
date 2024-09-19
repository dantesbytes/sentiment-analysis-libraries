import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from flair.models import TextClassifier
from flair.data import Sentence


nltk.download('vader_lexicon', quiet=True)


nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('spacytextblob')

flair_sentiment = TextClassifier.load('en-sentiment')

def analyze_sentiment(text):
    print("Analyzing sentiment for:", text)
    print("\n1. NLTK (VADER) Analysis:")
    nltk_analysis(text)
    
    print("\n2. TextBlob Analysis:")
    textblob_analysis(text)
    
    print("\n3. spaCy Analysis:")
    spacy_analysis(text)
    
    print("\n4. Flair Analysis:")
    flair_analysis(text)

def nltk_analysis(text):
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(text)
    compound_score = scores['compound']
    
    if compound_score >= 0.05:
        overall_sentiment = "Positive"
    elif compound_score <= -0.05:
        overall_sentiment = "Negative"
    else:
        overall_sentiment = "Neutral"
    
    print(f"Overall Sentiment: {overall_sentiment}")
    print(f"Compound Score: {compound_score:.4f} (on a scale of -1 to 1)")
    print(f"Breakdown: Negative: {scores['neg']:.3f}, Neutral: {scores['neu']:.3f}, Positive: {scores['pos']:.3f}")
    print("Reasoning: VADER picks up on sentiment-laden words and applies rules for things like punctuation and word-order.")

def textblob_analysis(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    if polarity > 0:
        overall_sentiment = "Positive"
    elif polarity < 0:
        overall_sentiment = "Negative"
    else:
        overall_sentiment = "Neutral"
    
    print(f"Overall Sentiment: {overall_sentiment}")
    print(f"Polarity: {polarity:.4f} (on a scale of -1 to 1)")
    print(f"Subjectivity: {subjectivity:.4f} (on a scale of 0 to 1)")
    print("Reasoning: TextBlob uses a simple lexicon-based approach, considering the polarity of individual words.")

def spacy_analysis(text):
    doc = nlp(text)
    polarity = doc._.blob.polarity
    subjectivity = doc._.blob.subjectivity
    
    if polarity > 0:
        overall_sentiment = "Positive"
    elif polarity < 0:
        overall_sentiment = "Negative"
    else:
        overall_sentiment = "Neutral"
    
    print(f"Overall Sentiment: {overall_sentiment}")
    print(f"Polarity: {polarity:.4f} (on a scale of -1 to 1)")
    print(f"Subjectivity: {subjectivity:.4f} (on a scale of 0 to 1)")
    print("Reasoning: spaCy with the TextBlob plugin uses a similar approach to TextBlob, but with spaCy's advanced tokenization.")

def flair_analysis(text):
    sentence = Sentence(text)
    flair_sentiment.predict(sentence)
    label = sentence.labels[0]
    
    print(f"Overall Sentiment: {label.value}")
    print(f"Confidence: {label.score:.4f}")
    print("Reasoning: Flair uses a more advanced model based on contextual string embeddings, allowing it to capture more nuanced sentiments.")

# example comment from instagram

text = "This the guy who won't let a fan greet him you should be a shame of yourself you were probably his hero now he'll remember that for the rest of his life"
analyze_sentiment(text)