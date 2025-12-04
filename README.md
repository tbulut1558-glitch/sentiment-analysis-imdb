Sentiment Analysis of IMDB Movie Reviews 


Author
Tugba BULUT
Master 1 IBD - Techniques d’Apprentissage Artificiel


Project Overview


This project is centered on applying supervised machine learning techniques to the IMDB Movie Reviews dataset.

The primary objective is twofold:

Model Identification: To systematically evaluate various classification algorithms (such as Logistic Regression, CART, KNN, Random Forest, and SVM) in order to identify the most suitable model configuration for the high-dimensional text data.

Inference: To utilize the best-performing model to automatically classify new, incoming movie reviews into distinct POSITIVE or NEGATIVE sentiment categories.

This work ensures the final output is a reliable and functional data-driven system capable of predicting public opinion from raw text in an effective manner.





Project Structure


This project adheres to the standard cookiecutter-data-science layout.

``
sentiment-analysis-imdb/
│
├── data/
│ └── raw/          -> Contains the original IMDB Dataset.csv.
│
├── models/         -> Stores all trained .joblib models and the TF-IDF vectorizer.
│
├── reports/
│ └── figures/      -> Generated ROC curves and Confusion Matrices (30 total visuals).
│ └── rapport_scores_globaux_FINAL.xlsx
└── src/
    ├── dataset.py      -> Data loading, Stratified splitting, and Negation handling.
    ├── features.py     -> TF-IDF (N-gram 1,2) and Scaling transformation logic.
    ├── plot_all_metrics.py -> Final reporting 
    ├── modeling/
    │   └── train.py    -> ML Training logic, Cross-Validation, and Evaluation.
    └── main_run.py     -> Main pipeline execution, persistence check, and live inference.``
    
    

Dataset


The Dataset: IMDB Dataset of 50K Movie Reviews

https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

The dataset used is the IMDB Movie Reviews Dataset, who includes 50k review and sentiments.

Characteristics:
Data Type: Unstructured Text (Customer Reviews).

Volume: 50,000 reviews in total.

Target Task: Binary Classification (Positive/Negative sentiment).



Methodology and Model Implementation



Text Cleaning: Implemented a custom cleaning function that preserves negation terms (not, no, etc.) to correctly capture polarity inversion.

TF-IDF Vectorizer: This tool converts text into weighted numerical features, giving higher scores to unique, distinctive words (like "disaster") over common ones to ensure the model focuses on meaning.

N-grams: Allows the model to recognize multi-word phrases, such as "not good" or "utterly predictable," as single contextual features, which significantly improves understanding of the sentiment.

Negation Handling: This step prevents the system from removing words like "not" or "never," ensuring that the model does not accidentally misread a negative sentence as positive.

Cross-Validation (CV): Repeatedly testing the model on different parts of the training data to confirm that our final performance score is dependable and not just a lucky guess from a single test.


Scaling: Three versions of the data are tested: 

        -Original (with no normalisation)
        -Standard Scaler
        -Min Max Scaler

Models Used:

    -CART
    -KNN
    -Random Forest
    -Logistic Regression (LR)
    -Linear Support Vector Machine (LSVM)


Evaluation Metrics

    -F1 Score
    -AUC ROC
    -PR AUC
    -Precision
    -Recall
    -Confusion Matrix


Project Execution

    1.Install Dependencies: 

    pip install pandas scikit-learn numpy matplotlib joblib nltk openpyxl

    2. Launch Full Pipeline (Training & Report): This command trains all 15 models, saves them to models/, and generates the final Excel report with all metrics.

    python src/main_run.py

    3. Generate Visualizations:

    python src/plot_all_metrics.py
    


Reports


The report/ folder contains:
    -ROC graphics
    -Confusion matrices
    -Numerical results table


    

Key Results Summary



Logistic Regression (LR): Optimal classifier that we are using. Fastest and most reliable model for separating text classes, yielding the highest score.

Linear SVM (LSVM):	A strong linear alternative for the project. Confirmed the power of linear separation on high-dimensional text data.

Random Forest (RF):	Really stable; used in the project to verify that performance was completely unaffected by data scaling.

Decision Tree (CART): Used as the single-tree reference point to demonstrate the superior stability of the RF approach.

KNN: It proved that standard scaling can destroy the structure of sparse text features, causing catastrophic model failure.




Possible Improvements



-Adding more advanced text preprocessing 

-Testing richer vectorization methods

-Evaluating additional models

-Add more robust cross-validation




