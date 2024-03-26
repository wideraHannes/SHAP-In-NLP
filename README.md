# Overview Of SHapley Additive exPlanations (SHAP) In Natural Language Processing


## Introduction

This project is a part of my Thesis "Overview Of SHapley Additive exPlanations (SHAP) In Natural Language Processing", where I primarily worked with Python Notebooks. Throughout the thesis, I dealt with two datasets: the Large Movie Review Dataset, which I used for sentiment analysis, and the IBM Debater: Evidence Sentences Dataset, on which I performed argument mining.
Additionally i added my final Thesis as PDF.

``German grade 1.0 (very good)``

## Sentiment Analysis

The Sentiment analysis code can be found in the `Sentiment` folder. The main notebook for Sentiment Analysis is located inside the `pipeline` folder and named "Sentiment_Analysis_SHAP". In this notebook, I conducted Sentiment analysis using three different models: Decision Tree, Logistic regression, and BERT. After the training, I locally interpreted certain misclassifications and performed global analysis on the important words.

## Argument Mining

Similarly, the code for Argument mining is located inside the `Argument` folder. The dedicated notebook for Argument Mining can be found inside the `pipeline` folder, named "Claim_Detection_SHAP". In this notebook, I trained and analyzed the exact same models as in Sentiment Analysis but on a different task, which is Argument mining.

## Custom Shap Explainer

For further investigation of certain classifications and to gain a better global overview of the important words, I created a custom Shap explainer. This custom explainer is located inside the `custom_shap_explainer` folder. It includes custom plots, such as a local word highlighting plot that highlights only the words with the biggest impact, and a global plot that displays a boxplot to provide more information about the distribution of the most important words.

## Getting Started

To get started with the code in this repository, follow these steps:

1. Clone this repository to your local machine.
2. Make sure you have the required dependencies installed.
3. Open the respective notebooks for Sentiment Analysis and Argument Mining located in the `pipeline` folder.
4. Run the code cells in the notebooks to perform the analyses.
