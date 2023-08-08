import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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

def custom_global_explanation(shapley_values, num_words=10):
    """
    This function explains the impact of words globally using SHAP values.

    Parameters:
    shapley_values (object): SHAP values object with feature names and values.
    num_words (int): The number of top impacting words to plot.

    Returns:
    None
    """
    values = shapley_values.values
    feature_names = shapley_values.feature_names  

    word_dict = _populate_word_dict(values, feature_names)
    word_dict = {word: values for word, values in word_dict.items() if len(values) >= 3}

    absolute_mean_impact = {word: np.mean(values) for word, values in word_dict.items()}

    positive_words, positive_impact = _get_sorted_impacts(absolute_mean_impact, num_words, reverse=True)
    negative_words, negative_impact = _get_sorted_impacts(absolute_mean_impact, num_words, reverse=False)

    _plot_words_impact(positive_words, positive_impact, 'Positive Words Impact', 'Positive Words')
    _plot_words_impact(negative_words, negative_impact, 'Negative Words Impact', 'Negative Words')


def _populate_word_dict(values, feature_names):
    word_dict = {}
    for row_values, row_features in zip(values, feature_names):
        for value, word in zip(row_values, row_features):
            if word in word_dict:
                word_dict[word].append(value)
            else:
                word_dict[word] = [value]
    return word_dict


def _get_sorted_impacts(absolute_mean_impact, num_words, reverse):
    sorted_words = sorted(absolute_mean_impact.items(), key=lambda x: x[1], reverse=reverse)[:num_words]
    words, impact = zip(*sorted_words)
    return words, impact


def _plot_words_impact(words, impact, title, ylabel):
    plt.figure(figsize=(7, 4))
    plt.barh(words, impact)
    plt.xlabel('MEAN Shap Value')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def custom_global_boxplot(shap_values, num_words=10, name=""):
    values = shap_values.values
    feature_names = shap_values.feature_names

    word_dict = {}
    for row_values, row_features in zip(values, feature_names):
        for value, word in zip(row_values, row_features):
            if word in word_dict:
                word_dict[word].append(value)
            else:
                word_dict[word] = [value]

    word_dict = {word: values for word, values in word_dict.items() if len(values) >= 3}
    absolute_mean_impact = {}
    for word, values in word_dict.items():
        absolute_mean_impact[word] = np.abs(values).mean()

    sorted_impact = sorted(absolute_mean_impact.items(), key=lambda x: x[1], reverse=True)
    top_words = dict(sorted_impact[:num_words])

    boxplot_data = [word_dict[word] for word in top_words.keys()]
    labels = list(top_words.keys())

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(boxplot_data, labels=labels, vert=False, showfliers=False)
    ax.set_xlabel('SHAP Values')
    plt.axvline(x=0, color='r', linestyle='dotted')
    ax.set_ylabel('Top Words')
    # print name in title
    plt.title(f"Boxplot of Top Words Impact on {name}")
    plt.show()

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



def custom_global_boxplot_word(shap_values, explain_word, name=""):
    values = shap_values.values
    feature_names = shap_values.feature_names

    shap_values = [value for row_values, row_features in zip(values, feature_names)
                   for value, word in zip(row_values, row_features) if word == explain_word]
    
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.boxplot(shap_values, labels=[explain_word], vert=False, showfliers=False)
    ax.set_xlabel('SHAP Values')
    plt.axvline(x=0, color='r', linestyle='dotted')
    # print name in title
    plt.show()


