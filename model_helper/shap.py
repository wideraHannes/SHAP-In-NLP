import os.path
import pickle

def calculate_and_save(shap_values_file, explainer, data):
    # Überprüfen, ob die Datei existiert
    if not os.path.exists(shap_values_file):
        # Berechnen Sie die SHAP-Werte, wenn die Datei nicht existiert
        shap_values = explainer(data)

        # Speichern Sie die SHAP-Werte in einer Datei
        with open(shap_values_file, "wb") as f:
            pickle.dump(shap_values, f)
    else:
        # Laden Sie die SHAP-Werte aus der Datei, wenn sie existiert
        with open(shap_values_file, "rb") as f:
            shap_values = pickle.load(f)

    return shap_values
