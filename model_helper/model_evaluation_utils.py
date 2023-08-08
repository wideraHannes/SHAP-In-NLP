import pandas as pd
import numpy as np

def get_disagreements(y_test, LOGREG_PREDICTION_FILE, DTC_PREDICTION_FILE, BERT_PREDICTION_FILE):
  """
  The function takes the paths of three prediction files (LOGREG_PREDICTION_FILE, DTC_PREDICTION_FILE, 
  and BERT_PREDICTION_FILE) and the true labels (y_test) as input. It reads the prediction files and the true labels into
  pandas DataFrames, and then it calculates the absolute differences in probabilities between the models.
  The function returns three DataFrames, each sorted by the largest differences in probabilities between the corresponding 
  pairs of models (dtc and bert, logreg and bert, logreg and dtc).
  """
  df_logreg = pd.read_csv(LOGREG_PREDICTION_FILE)
  df_dtc = pd.read_csv(DTC_PREDICTION_FILE)
  df_bert = pd.read_csv(BERT_PREDICTION_FILE)

  df_labels = pd.DataFrame(y_test, columns=['label'])
  df = pd.concat([df_logreg, df_dtc, df_bert, df_labels], axis=1)

  df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

  df['logreg_dtc_diff'] = abs(df['logreg_1'] - df['dtc_1'])
  df['logreg_bert_diff'] = abs(df['logreg_1'] - df['bert_1'])
  df['dtc_bert_diff'] = abs(df['dtc_1'] - df['bert_1'])

  dtc_bert_sorted_by_diff = df.sort_values(by=['dtc_bert_diff'], ascending=False)
  logreg_bert_sorted_by_diff = df.sort_values(by=['logreg_bert_diff'], ascending=False)
  logreg_dtc_sorted_by_diff = df.sort_values(by=['logreg_dtc_diff'], ascending=False)
  return dtc_bert_sorted_by_diff, logreg_bert_sorted_by_diff, logreg_dtc_sorted_by_diff


def get_misclassifications(y_test, PREDICTION_FILE):
  """
  The function takes the true labels y_test and the path of a prediction file (PREDICTION_FILE) as input. 
  It reads the prediction file and the true labels into a pandas DataFrame, then calculates the absolute difference between the actual 
  classes and predicted probabilities.  It returns a DataFrame containing the misclassified samples, sorted by the absolute
  difference in probabilities.
  """
  df = pd.read_csv(PREDICTION_FILE)
  df_labels = pd.DataFrame(y_test, columns=['label'])
  df = pd.concat([df, df_labels], axis=1)
  df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

  # Find the column name for the predicted probabilities of class 1.
  proba_col_name = [col for col in df.columns if col.endswith('_1')][0]

  # Calculate the difference between actual classes and predicted probabilities.
  df['diff'] = np.abs(df['label'] - df[proba_col_name])

  # Create a new column 'misclassified' and set its value based on the misclassification condition
  df['misclassified'] = df.apply(lambda row: (row['label'] == 1 and row[1] <= 0.5) or (row['label'] == 0 and row[1] > 0.5),axis=1)
  df = df[df['misclassified'] == True]
  df_misclassified = df.sort_values(by='diff', ascending=False)

  return df_misclassified