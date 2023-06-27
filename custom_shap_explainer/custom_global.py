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

def custom_global_explanation(shap_values, num_words=10):
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
  positive_impact = df.apply(calculate_positive_mean, axis=0).sort_values(ascending=False)[:num_words]
  negative_impact = df.apply(calculate_negative_mean, axis=0).sort_values(ascending=True)[:num_words]
  # plot the mean absolute value of the features
  plt.figure(figsize=(5, 5))
  plt.title("Feature importance based on Positive SHAP values")
  plt.barh(positive_impact.index, positive_impact.values)
  plt.figure(figsize=(5, 5))
  plt.title("Feature importance based on Negative SHAP values")
  plt.barh(negative_impact.index, negative_impact.values)


def custom_global_shap_distribution(shap_values, threshold=0.01):
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
    positive_impact = df.apply(calculate_positive_mean, axis=0).sort_values(ascending=False)
    negative_impact = df.apply(calculate_negative_mean, axis=0).sort_values(ascending=True)

    # filter by threshold
    positive_impact = positive_impact[positive_impact > threshold]
    negative_impact = negative_impact[negative_impact < -threshold]

    # plot the mean absolute value of the features
    plt.figure(figsize=(20, 5))
    plt.title("Feature importance based on Positive SHAP values")
    plt.bar(positive_impact.index, positive_impact.values)

    # Calculate and plot the mean of positive values
    positive_mean = positive_impact.mean()
    plt.axhline(y=positive_mean, color='r', linestyle='--', label=f'Mean: {positive_mean:.2f}')
    plt.legend()

    # Remove x-axis tick labels
    ax = plt.gca()
    ax.set_xticks([])

    plt.figure(figsize=(20, 5))
    plt.title("Feature importance based on Negative SHAP values")
    plt.bar(negative_impact.index, negative_impact.values)

    # Calculate and plot the mean of negative values
    negative_mean = negative_impact.mean()
    plt.axhline(y=negative_mean, color='r', linestyle='--', label=f'Mean: {negative_mean:.2f}')
    plt.legend()

    # Remove x-axis tick labels
    ax = plt.gca()
    ax.set_xticks([])


