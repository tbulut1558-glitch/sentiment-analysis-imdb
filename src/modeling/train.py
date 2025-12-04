from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, average_precision_score
from joblib import dump
import os
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np

def evaluer_modele(y_vrai, y_predit, modele, X_test):
    # Obtention des scores pour AUC / PR_AUC
    try:
        y_score = modele.predict_proba(X_test)[:, 1]
    except AttributeError:
        y_score = modele.decision_function(X_test)
    
    # Matrice de Confusion
    cm = confusion_matrix(y_vrai, y_predit)
    tn, fp, fn, tp = cm.ravel()
    
    print("\n  Matrice de Confusion (VN, FP, FN, VP):")
    print(f"  [[{tn} {fp}]")
    print(f"  [{fn} {tp}]]")

    return {
        'Precision': precision_score(y_vrai, y_predit),
        'Recall': recall_score(y_vrai, y_predit),
        'F1-Score': f1_score(y_vrai, y_predit),
        'AUC': roc_auc_score(y_vrai, y_score),
        'PR_AUC': average_precision_score(y_vrai, y_score),
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp
    }

def entrainer_et_evaluer(modele, X_train, y_train, X_test, y_test, nom_modele, chemin_modeles):
    
    print(f"\n--- Entraînement du Modèle: {nom_modele} ---")
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    
    
    modele.fit(X_train, y_train)
    y_pred = modele.predict(X_test)
    
    dump(modele, os.path.join(chemin_modeles, f'{nom_modele}_modele.joblib'))
    
    resultats = evaluer_modele(y_test, y_pred, modele, X_test)
    
    print(f"[{nom_modele}] F1: {resultats['F1-Score']:.4f}, AUC: {resultats['AUC']:.4f}, PR_AUC: {resultats['PR_AUC']:.4f}")
    

    
    return resultats

