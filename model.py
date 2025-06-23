# -*- coding: utf-8 -*-
"""
Script de création et de sauvegarde d'un modèle d'IA factice.
"""
import joblib
from sklearn.linear_model import LogisticRegression

def train_and_save_model():
    """
    Entraîne un modèle simple et le sauvegarde sur disque.
    """
    # Création d'un modèle factice
    X_dummy = [[1], [2], [3], [4]]
    y_dummy = [0, 0, 1, 1]
    model = LogisticRegression()
    model.fit(X_dummy, y_dummy)

    # Sauvegarde du modèle
    model_filename = 'simple_text_classifier.joblib'
    joblib.dump(model, model_filename)
    print(f"Modèle sauvegardé dans '{model_filename}'")
    return model_filename

def predict(input_data):
    """
    Charge le modèle et effectue une prédiction.
    """
    model = joblib.load('simple_text_classifier.joblib')
    return model.predict(input_data)

if __name__ == '__main__':
    train_and_save_model()
    # Exemple de prédiction
    print(f"Prédiction pour [[5]]: {predict([[5]])}")
