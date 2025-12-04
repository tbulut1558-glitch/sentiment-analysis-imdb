import pandas as pd
from sklearn.model_selection import train_test_split
import re
from nltk.corpus import stopwords
import numpy as np
import nltk

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-z\s]', '', text) 
    
    english_stopwords = set(stopwords.words('english'))
    
    negation_words = {"not", "no", "don't", "didn't", "n't", "never", "none", "neither", "nor"}
    final_stopwords = english_stopwords - negation_words
    
    text = ' '.join([word for word in text.split() if word not in final_stopwords])
    
    return text

def charger_et_diviser_donnees(chemin_donnees):
    print("Application du pré-traitement (Nettoyage de texte avec conservation des négations)...")
    
    df = pd.read_csv(chemin_donnees)
    
    df['review'] = df['review'].apply(clean_text)
    
    df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
    
    balance = df['sentiment'].value_counts(normalize=True)
    print(f"Bilan du Dataset (0=Négatif, 1=Positif):\n{balance.round(4)}")
    
    X = df['review']  
    y = df['sentiment'] 
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y 
    ) 
    
    return X_train, X_test, y_train, y_test