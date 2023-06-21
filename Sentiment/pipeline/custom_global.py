import pandas as pd
import matplotlib.pyplot as plt


def calculate_positive_mean(feature):
    # check if feature is bigger than 0
    positive_impact =  feature[feature > 0]
    if positive_impact.size > 10:
        return positive_impact.mean()
    else:
        return 0
    
def calculate_negative_mean(feature):
    # check if feature is bigger than 0
    positive_impact =  feature[feature < 0]
    if positive_impact.size > 10:
        return positive_impact.mean()
    else:
        return 0

def custom_global_explanation(shap_values):
  values = shap_values.values
  feature_names = shap_values.feature_names  

  data = []

  for i,(row_values,row_features) in enumerate(zip(values, feature_names)):
      row_dict = {}
      for j,value in enumerate(row_values):
          word = row_features[j]
          row_dict[word] = value
      data.append(row_dict)

  len(data)
  df = pd.DataFrame(data)
  df = df.fillna(0)
  positive_impact = df.apply(calculate_positive_mean, axis=0).sort_values(ascending=False)[:10]
  negative_impact = df.apply(calculate_negative_mean, axis=0).sort_values(ascending=True)[:10]
  # plot the mean absolute value of the features
  plt.figure(figsize=(5, 5))
  plt.title("Feature importance based on Positive SHAP values")
  plt.barh(positive_impact.index, positive_impact.values)
  plt.figure(figsize=(5, 5))
  plt.title("Feature importance based on Negative SHAP values")
  plt.barh(negative_impact.index, negative_impact.values)
