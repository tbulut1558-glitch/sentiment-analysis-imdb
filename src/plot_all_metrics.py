# src/plot_all_metrics.py
"""
Plot all ROC and Confusion Matrices using:
- the EXACT test set saved by main_run (models/test_set.pkl)
- the TF-IDF vectorizer saved by main_run (models/vectoriseur_tfidf.joblib)
- the scaler objects saved by main_run (models/scaler_Standard.joblib, models/scaler_MinMax.joblib)
This guarantees plots are in the SAME FEATURE SPACE as training.
"""

import os
import pickle
from joblib import load
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from scipy.sparse import issparse
import numpy as np

# chemins
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
CHEMIN_MODELES = os.path.join(BASE_DIR, 'models')
CHEMIN_FIGURES = os.path.join(BASE_DIR, 'reports', 'figures')
os.makedirs(CHEMIN_FIGURES, exist_ok=True)

SCALER_TYPES = ['Original', 'Standard', 'MinMax']
MODEL_TYPES = ['CART', 'KNN', 'RF', 'LR', 'LSVM']


def charger_test_set_et_vectorizer():
    chemin_test = os.path.join(CHEMIN_MODELES, "test_set.pkl")
    if not os.path.exists(chemin_test):
        raise FileNotFoundError("test_set.pkl manquant. Exécutez main_run.py pour créer le jeu de test.")
    X_test_raw, y_test = pickle.load(open(chemin_test, "rb"))

    # charger vectorizer
    vect_path = os.path.join(CHEMIN_MODELES, 'vectoriseur_tfidf.joblib')
    if not os.path.exists(vect_path):
        raise FileNotFoundError("vectoriseur_tfidf.joblib introuvable. Exécutez main_run.py d'abord.")
    vectoriseur = load(vect_path)

    # transformer le texte brut en TF-IDF (sparse)
    X_test_vec = vectoriseur.transform(X_test_raw)
    return X_test_vec, y_test


def apply_scaler_to_test(X_test_vec, scaler_name):
    """
    Applique le même scaling que lors de l'entraînement.
    - Original => retourne la matrice TF-IDF sparse
    - Standard/MinMax => charge scaler_{scaler_name}.joblib et transforme en dense
    """
    if scaler_name == "Original":
        return X_test_vec

    # charger scaler
    scaler_path = os.path.join(CHEMIN_MODELES, f"scaler_{scaler_name}.joblib")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler non trouvé: {scaler_path}. Assurez-vous que main_run.py a sauvegardé le scaler.")
    scaler = load(scaler_path)

    # scaler attend un array dense
    if issparse(X_test_vec):
        X_dense = X_test_vec.toarray()
    else:
        X_dense = X_test_vec

    X_scaled = scaler.transform(X_dense)
    return X_scaled


def plot_roc_and_cm(model, X_test, y_test, name):
    # prédictions
    y_pred = model.predict(X_test)

    # scores pour ROC
    y_score = None
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        try:
            y_score = model.decision_function(X_test)
        except Exception:
            y_score = y_pred

    # ROC
    try:
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, lw=2, color='darkorange', label=f"AUC = {roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], '--', color='navy')
        plt.title(f"ROC - {name}")
        plt.legend()
        plt.savefig(os.path.join(CHEMIN_FIGURES, f"ROC_{name}.png"))
        plt.close()
    except Exception as e:
        print(f"[plot] Impossible de tracer ROC pour {name}: {e}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Négatif','Positif'])
    plt.figure(figsize=(6,5))
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f"Confusion Matrix - {name}")
    plt.savefig(os.path.join(CHEMIN_FIGURES, f"CM_{name}.png"))
    plt.close()


if __name__ == '__main__':
    print("[plot_all_metrics] Chargement du test set et vectorizer...")
    X_test_vec, y_test = charger_test_set_et_vectorizer()

    print("[plot_all_metrics] Création des graphiques (ROC + CM) pour chaque combinaison...")
    for scaler in SCALER_TYPES:
        for model in MODEL_TYPES:
            name = f"{scaler}_{model}"
            model_path = os.path.join(CHEMIN_MODELES, f"{name}_modele.joblib")
            if not os.path.exists(model_path):
                print(f"[plot_all_metrics] Modèle manquant: {name} (ignoré)")
                continue

            # charger modèle
            model_obj = load(model_path)

            # appliquer scaler correspondant sur X_test_vec
            try:
                X_test_for_model = apply_scaler_to_test(X_test_vec, scaler)
            except Exception as e:
                print(f"[plot_all_metrics] Erreur lors de l'application du scaler pour {name}: {e}")
                continue

            # tracer
            try:
                plot_roc_and_cm(model_obj, X_test_for_model, y_test, name)
                print(f"[plot_all_metrics] Graphs enregistrés pour {name}")
            except Exception as e:
                print(f"[plot_all_metrics] Erreur en traçant {name}: {e}")

    print("\n[plot_all_metrics] Terminé. Les images sont dans reports/figures/")
