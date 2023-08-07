import pandas as pd

def predict_and_save(model, name, corpus_test, output_file):
    # Generate predictions
    predictions = model.predict_proba(corpus_test)

    # Destructure probabilities for class_1 and class_2
    predictions_class_0 = predictions[:, 0]
    predictions_class_1 = predictions[:, 1]

    # Create a DataFrame containing the test samples and BERT predictions
    results = pd.DataFrame({
        f'{name}_0': predictions_class_0,
        f'{name}_1': predictions_class_1
    })

    # Save the DataFrame to a CSV file
    results.to_csv(output_file)