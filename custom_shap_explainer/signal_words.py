from IPython.core.display import HTML,display
import numpy as np


def highlight_signal_words(shap_values,round_shap_values=3, top_words=5):
    """
    The function takes SHAP values, rounds them to a specified precision, and highlights the top signal
    words based on their importance in the provided SHAP values. The function determines the high and low thresholds for identifying the
    most significant and least significant words and then generates HTML code to display the text with highlighted signal words in different 
    colors (#57c2ff for low importance and #ff6284 for high importance).
    """
    words = shap_values.feature_names
    values = shap_values.values

    values = [round(value, round_shap_values) for value in values]
    low_percentile = top_words
    high_percentile = 100 - top_words
    
    low_threshold, high_threshold = _calculate_thresholds(values, low_percentile, high_percentile)
    lowest_word_index_value, highest_word_index_value = _find_lowest_highest_words_indices(words, values, low_threshold, high_threshold)

    lowest_indices = [word_index for word, word_index, value in lowest_word_index_value]
    highest_indices = [word_index for word, word_index, value in highest_word_index_value]

    highlighted_text = []
    for index, word in enumerate(words):
        if index in lowest_indices:
            word, _, value = lowest_word_index_value[lowest_indices.index(index)]
            highlighted_text.append(_highlight_word(word, value, '#57c2ff'))
        elif index in highest_indices:
            word, _, value = highest_word_index_value[highest_indices.index(index)]
            highlighted_text.append(_highlight_word(word, value, '#ff6284'))
        else:
            highlighted_text.append(word)
    
    text = f"<div style='font-size: 1.3em; line-height: 2em; background-color:transparent'>{' '.join(highlighted_text)}</div>"
    display(HTML(text))

def _calculate_thresholds(values, low_percentile, high_percentile):
    low_threshold = np.percentile(values, low_percentile)
    high_threshold = np.percentile(values, high_percentile)
    return low_threshold, high_threshold

def _find_lowest_highest_words_indices(words, values, low_threshold, high_threshold):
    word_index_value_list = [(word, index, value) for index, (word, value) in enumerate(zip(words, values))]
    lowest_word_index_value = [(word, index, value) for word, index, value in word_index_value_list if value <= low_threshold]
    highest_word_index_value = [(word, index, value) for word, index, value in word_index_value_list if value >= high_threshold]
    return lowest_word_index_value, highest_word_index_value

def _highlight_word(word, value, background_color):
    return f"""<div style='position:relative; padding: 8px 10px; background-color: {background_color}; display: inline-block'>
                <span>{word}</span>
                <span style='font-size: 0.7em; position: absolute; top: -7px; left: 2px;'>{value}</span>
              </div>"""