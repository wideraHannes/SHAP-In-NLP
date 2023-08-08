import pandas as pd

def predict_and_save(model, name, corpus_test, output_file):
    """ Predict probabilities for a binary classification model and save the results to a CSV file."""
    predictions = model.predict_proba(corpus_test)

    predictions_class_0 = predictions[:, 0]
    predictions_class_1 = predictions[:, 1]


    results = pd.DataFrame({
        f'{name}_0': predictions_class_0,
        f'{name}_1': predictions_class_1
    })

    results.to_csv(output_file)