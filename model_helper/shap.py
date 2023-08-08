import os.path
import pickle

def calculate_and_save(shap_values_file, explainer, data):
    """The calculate_and_save function either calculates and saves SHAP values using the provided explainer for the given data or loads
    the SHAP values from the file if it already exists.
    """
    if not os.path.exists(shap_values_file):
        shap_values = explainer(data)
        with open(shap_values_file, "wb") as f:
            pickle.dump(shap_values, f)
    else:
        with open(shap_values_file, "rb") as f:
            shap_values = pickle.load(f)

    return shap_values
