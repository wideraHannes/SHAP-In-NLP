import pandas as pd
import numpy as np

def get_disagreements(y_test, LOGREG_PREDICTION_FILE, DTC_PREDICTION_FILE, BERT_PREDICTION_FILE):
  df_logreg = pd.read_csv(LOGREG_PREDICTION_FILE)
  df_dtc = pd.read_csv(DTC_PREDICTION_FILE)
  df_bert = pd.read_csv(BERT_PREDICTION_FILE)

  df_labels = pd.DataFrame(y_test, columns=['label'])
  df = pd.concat([df_logreg, df_dtc, df_bert, df_labels], axis=1)

  # renove columns Unnamed: 0
  df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

  # Berechne die Differenz der Wahrscheinlichkeiten zwischen den Modellen
  df['logreg_dtc_diff'] = abs(df['logreg_1'] - df['dtc_1'])
  df['logreg_bert_diff'] = abs(df['logreg_1'] - df['bert_1'])
  df['dtc_bert_diff'] = abs(df['dtc_1'] - df['bert_1'])

  # Sortiere den DataFrame nach den größten Unterschieden
  dtc_bert_sorted_by_diff = df.sort_values(by=['dtc_bert_diff'], ascending=False)
  logreg_bert_sorted_by_diff = df.sort_values(by=['logreg_bert_diff'], ascending=False)
  logreg_dtc_sorted_by_diff = df.sort_values(by=['logreg_dtc_diff'], ascending=False)
  return dtc_bert_sorted_by_diff, logreg_bert_sorted_by_diff, logreg_dtc_sorted_by_diff


def get_misclassifications(y_test, PREDICTION_FILE):
    df = pd.read_csv(PREDICTION_FILE)
    df_labels = pd.DataFrame(y_test, columns=['label'])
    df = pd.concat([df, df_labels], axis=1)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Finde den Spaltennamen für die vorhergesagten Wahrscheinlichkeiten der Klasse 1
    proba_col_name = [col for col in df.columns if col.endswith('_1')][0]

    # Berechne die Differenz zwischen tatsächlichen Klassen und vorhergesagten Wahrscheinlichkeiten
    df['diff'] = np.abs(df['label'] - df[proba_col_name])

    # Assuming logreg_misclassifications is a DataFrame with columns 'label', 'logreg_0', 'logreg_1'

  # Create a new column 'misclassified' and set its value based on the misclassification condition
    df['misclassified'] = df.apply(lambda row: (row['label'] == 1 and row[1] <= 0.5) or (row['label'] == 0 and row[1] > 0.5),axis=1)
    # remove all not misclassified rows
    df = df[df['misclassified'] == True]

    # Sortiere den DataFrame nach der 'diff' Spalte in absteigender Reihenfolge
    df_misclassified = df.sort_values(by='diff', ascending=False)

    return df_misclassified