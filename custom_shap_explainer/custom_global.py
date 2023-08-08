import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def custom_global_explanation(shapley_values, num_words=10):
    """
    This function explains the impact of words globally using SHAP values.
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


def custom_global_boxplot(shap_values, num_words=10, name=""):
    """The function creates a custom global boxplot for visualizing the impact of top words based on their SHAP values.
    It takes the SHAP values as input and calculates the mean impact for each word. It then identifies the top words with the highest absolute mean
    impact and creates a boxplot to display their distribution.
    """
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



def custom_global_boxplot_word(shap_values, explain_word, name=""):
    """ The custom_global_boxplot_word function creates a custom global boxplot for visualizing the impact of SHAP values for a specific explain_word.
    It takes the SHAP values as input and filters the SHAP values for the specified word.
    Then, it plots a boxplot to display the distribution of SHAP values for that word. 
    """
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





def _plot_words_impact(words, impact, title, ylabel):
    plt.figure(figsize=(7, 4))
    plt.barh(words, impact)
    plt.xlabel('MEAN Shap Value')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()


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