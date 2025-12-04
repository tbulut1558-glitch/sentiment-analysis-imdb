import os
import sys
import pickle
from joblib import dump
from joblib import load
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

try:
    from src.modeling.train import entrainer_et_evaluer
    from src.dataset import charger_et_diviser_donnees
    from src.features import creer_caracteristiques
    
except ImportError:
    from modeling.train import entrainer_et_evaluer
    from dataset import charger_et_diviser_donnees
    from features import creer_caracteristiques

# modèles
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# scalers
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.sparse import issparse

# chemins
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
CHEMIN_DONNEES_BRUTES = os.path.join(BASE_DIR, 'data', 'raw', 'IMDB Dataset.csv')
CHEMIN_MODELES = os.path.join(BASE_DIR, 'models')
CHEMIN_REPORTS = os.path.join(BASE_DIR, 'reports')
os.makedirs(CHEMIN_MODELES, exist_ok=True)
os.makedirs(CHEMIN_REPORTS, exist_ok=True)


def dense_if_sparse(X):
    return X.toarray() if issparse(X) else X


if __name__ == '__main__':

    MODELE_REF = os.path.join(CHEMIN_MODELES, 'Original_LR_modele.joblib')
    MODELE_EXISTE = os.path.exists(MODELE_REF)

    if not MODELE_EXISTE:
        print("=== Démarrage entraînement complet ===")

        #Chargement + split + nettoyage (dans charger_et_diviser_donnees)
        X_train_raw, X_test_raw, y_train, y_test = charger_et_diviser_donnees(CHEMIN_DONNEES_BRUTES)

        #Sauvegarder test set brut (texte + labels) pour réutilisation exacte dans plotting
        pickle.dump((X_test_raw, y_test), open(os.path.join(CHEMIN_MODELES, "test_set.pkl"), "wb"))
        print("[main_run] test_set.pkl sauvegardé dans models/")

        #TF-IDF vectorizer (creer_caracteristiques va aussi sauvegarder le vectorizer)
        X_train_vec, X_test_vec = creer_caracteristiques(X_train_raw, X_test_raw, CHEMIN_MODELES)

        #Définition des scalers et modèles
        SCALER_TYPES = ['Original', 'Standard', 'MinMax']
        MODEL_MAP = {
            'CART': DecisionTreeClassifier(random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'RF': RandomForestClassifier(n_estimators=150, max_depth=100, random_state=42, n_jobs=-1),
            'LR': LogisticRegression(solver='liblinear', random_state=42, max_iter=4000, penalty='l2'),
            'LSVM': LinearSVC(random_state=42, max_iter=200, penalty='l2', dual=False)
        }

        tous_les_resultats = []

        for scaler_name in SCALER_TYPES:
            print(f"\n--- SCALER: {scaler_name} ---")

            
            if scaler_name == 'Original':
                X_train_scaled = X_train_vec
                X_test_scaled = X_test_vec
                scaler_obj = None
            else:
                # Convertir en dense car Scaler exige dense arrays
                X_train_dense = dense_if_sparse(X_train_vec)
                X_test_dense = dense_if_sparse(X_test_vec)

                if scaler_name == 'Standard':
                    scaler_obj = StandardScaler(with_mean=False)  # with_mean=False pour données clairsemées
                else: 
                    scaler_obj = MinMaxScaler()

                # Fit scaler sur training et transformer train+test
                X_train_scaled = scaler_obj.fit_transform(X_train_dense)
                X_test_scaled = scaler_obj.transform(X_test_dense)

                # Sauvegarder le scaler pour usage ultérieur (plotting)
                dump(scaler_obj, os.path.join(CHEMIN_MODELES, f"scaler_{scaler_name}.joblib"))
                print(f"[main_run] Scaler sauvegardé: models/scaler_{scaler_name}.joblib")

            #Entraînement des modèles (sur X_train_scaled)
            for model_name, model_obj in MODEL_MAP.items():
                # on clone la classe via get_params
                model = model_obj.__class__(**model_obj.get_params())
                nom_complet = f"{scaler_name}_{model_name}"
                print(f"[main_run] Entraînement: {nom_complet}")

                res = entrainer_et_evaluer(model, X_train_scaled, y_train, X_test_scaled, y_test, nom_complet, CHEMIN_MODELES)
                # Ajouter info contexte
                res_record = {'Scaler': scaler_name, 'Model': model_name, **res}
                tous_les_resultats.append(res_record)

        #Exporter résultats en Excel
        df = pd.DataFrame(tous_les_resultats)
        colonnes_finales = [
        'Scaler', 'Model', 'F1-Score', 'AUC', 'PR_AUC', 
        'Precision', 'Recall', 'TP', 'FP', 'FN', 'TN']
        df = df[colonnes_finales]   
        rapport_path = os.path.join(CHEMIN_REPORTS, 'rapport_scores_globaux_FINAL.xlsx')
        df.to_excel(rapport_path, index=False)
        print(f"\n[main_run] Rapport sauvegardé: {rapport_path}")

    else:
        print("=== Modèles déjà présents. Si vous voulez réentraîner, supprimez les fichiers sous models/ ===")

    # Mode interactif
    print("\n=== Mode prédiction interactive ===")
    # charger vectorizer et meilleur modèle si existant
    try:
        vectoriseur = load(os.path.join(CHEMIN_MODELES, 'vectoriseur_tfidf.joblib'))
    except:
        vectoriseur = None

    while True:
        texte = input("\nEntrez une critique (ou 'quitter'):\n> ")
        if texte.strip().lower() in ['quitter', 'exit', 'q']:
            print("Fin.")
            break
        if vectoriseur is None:
            print("Vectoriseur introuvable. Exécutez l'entraînement d'abord.")
            continue
        # on choisit LR Original s'il existe, sinon RF Original
        best_path = os.path.join(CHEMIN_MODELES, 'Original_LR_modele.joblib')
        if not os.path.exists(best_path):
            best_path = os.path.join(CHEMIN_MODELES, 'Original_RF_modele.joblib')
        if not os.path.exists(best_path):
            print("Aucun modèle optimal trouvé.")
            continue
        modele = load(best_path)
        texte_nettoye = texte  # on suppose clean_text déjà appliqué dans pipeline; sinon on peut importer clean_text
        X_vect = vectoriseur.transform([texte_nettoye])
        try:
            pred = modele.predict(X_vect)[0]
        except Exception:
            try:
                pred = modele.predict(X_vect.toarray())[0]
            except Exception as e:
                print("Erreur prédiction:", e)
                continue
        label = "POSITIF" if pred == 1 else "NÉGATIF"
        print(f"Prediction: {label}")
