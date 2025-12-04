from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.sparse import issparse
from joblib import dump
import os

def creer_caracteristiques(X_train, X_test, chemin_modeles):
    vectoriseur_tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english', min_df=2, max_df=0.95) 
    
    X_train_vec = vectoriseur_tfidf.fit_transform(X_train)
    X_test_vec = vectoriseur_tfidf.transform(X_test)
    
    dump(vectoriseur_tfidf, os.path.join(chemin_modeles, 'vectoriseur_tfidf.joblib'))
    return X_train_vec, X_test_vec

def appliquer_scaling(X_train_vec, X_test_vec, scaler_type):
    if scaler_type == 'Original':
        return X_train_vec, X_test_vec
    
    X_train_dense = X_train_vec.toarray() if issparse(X_train_vec) else X_train_vec
    X_test_dense = X_test_vec.toarray() if issparse(X_test_vec) else X_test_vec
    
    if scaler_type == 'Standard':
        scaler = StandardScaler(with_mean=False) 
    elif scaler_type == 'MinMax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Type de Scaler non supporté.")

    X_train_scaled = scaler.fit_transform(X_train_dense)
    X_test_scaled = scaler.transform(X_test_dense)
    
    return X_train_scaled, X_test_scaled