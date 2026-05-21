# Overview of SHapley Additive exPlanations (SHAP) in Natural Language Processing

## Introduction

This repository contains the project work from my thesis, **“Overview of SHapley Additive exPlanations (SHAP) in Natural Language Processing”**.

The thesis investigates how SHAP can be applied to NLP models and focuses on interpreting text-based machine learning models through Python notebooks. The experiments cover two NLP tasks:

- **Sentiment analysis** using the Large Movie Review Dataset
- **Argument mining** using the IBM Debater: Evidence Sentences Dataset

In **Chapter 3**, the thesis explains the theoretical foundation of SHAP, including Shapley values, additive feature attribution methods, and the mathematical properties behind SHAP. It also describes the main components required to apply SHAP to NLP models, such as the Partition Explainer, text maskers, explanations, and visualization methods.

A major contribution of this project is the implementation of two custom SHAP-based visualization methods for NLP:

- a **local highlight plot** that emphasizes the most relevant words in longer text samples
- a **global boxplot visualization** that shows the distribution of SHAP values for the most influential words

For a quick read of the main findings, see [Advantages and Challenges of SHAP](06-advantages-and-challenges-of-shap.md).

The repository also includes the final thesis as a PDF.

**Final grade:** German grade 1.0 (*very good*)

## What is SHAP?

SHAP is a framework for explaining machine learning predictions. It is based on **Shapley values**, a concept from game theory that was introduced to solve the problem of fairly distributing a reward among several players.

A simple way to understand this is the pirate example: imagine a group of pirates lifting a heavy stone to reach a hidden treasure. Each pirate contributes differently to lifting the stone. To divide the treasure fairly, we need to know how much each pirate actually contributed. The Shapley value measures this by checking how much the result changes when a specific pirate joins different possible groups of pirates.

In machine learning, the idea is transferred as follows: the pirates become the input features, and the treasure becomes the model output. For NLP models, these features are usually words, tokens, or subwords. SHAP assigns each feature an importance value that describes how much it contributed to a specific prediction.

SHAP belongs to the class of **additive feature attribution methods**. These methods explain a complex model by approximating its output with a simpler explanation model:

$$
g(x') = \phi_0 + \sum_{i=1}^{M} \phi_i x_i'
$$

Here, `g(x')` is the explanation model, `phi_0` is the base value, and each `phi_i` is the contribution of feature `i`. The simplified input `x'` indicates whether a feature is present or missing. This means that the final prediction can be explained as the base value plus the contributions of the individual features.

The important point is that SHAP is not just a random explanation method. Shapley values satisfy four fairness properties: **efficiency**, **symmetry**, **dummy**, and **additivity**. In the pirate example, efficiency means that the whole treasure is distributed and nothing is lost. Symmetry means that two pirates who contribute equally receive the same share. Dummy means that a pirate who does not help lift the stone receives nothing. Additivity means that if the pirates find multiple treasures, it does not matter whether the shares are calculated separately or together; each pirate's total share remains consistent.

Lundberg and Lee showed that, for additive feature attribution methods, these properties lead to a unique solution based on Shapley values. This is why SHAP can be understood as a unified framework for feature attribution: it combines several existing explanation approaches under one theoretically grounded solution.

In practice, SHAP values describe how the model moves from a base value, usually the average prediction, to the final prediction for a specific input. Positive SHAP values push the prediction upward, while negative SHAP values push it downward. These values can then be visualized with plots, making it possible to inspect which words or tokens influenced a model prediction the most.

For a more detailed explanation of the theory behind SHAP, including the mathematical background and the connection to additive feature attribution methods, see **Chapter 3** of the thesis.

## Sentiment Analysis

The sentiment analysis code is located in the [`Sentiment`](Sentiment) folder. The main notebook can be found in the [`Sentiment/pipeline`](Sentiment/pipeline) folder under the name [`Sentiment_Analysis_SHAP.ipynb`](Sentiment/pipeline/Sentiment_Analysis_SHAP.ipynb).

In this notebook, I trained and analyzed three different models:

- Decision Tree
- Logistic Regression
- BERT

After training, I used SHAP to locally interpret selected misclassifications and to perform a global analysis of important words. This helped reveal which words had the strongest influence on the model predictions and where the models failed to understand the actual sentiment of a text.

<img width="912" height="657" alt="Bildschirmfoto 2026-01-13 um 14 13 48" src="https://github.com/user-attachments/assets/87d11528-35b1-4ef6-ba0d-d531cb06ed3c" />

## Argument Mining

The argument mining code is located in the [`Argument`](Argument) folder. The corresponding notebook can be found in the [`Argument/pipeline`](Argument/pipeline) folder under the name [`Claim_Detection_SHAP.ipynb`](Argument/pipeline/Claim_Detection_SHAP.ipynb).

In this notebook, I trained and analyzed the same three model types used for sentiment analysis: Decision Tree, Logistic Regression, and BERT. However, instead of classifying movie reviews by sentiment, the models were applied to an argument mining task using the IBM Debater: Evidence Sentences Dataset.

The goal was to investigate whether SHAP can also provide useful explanations for more complex NLP tasks, such as detecting evidence or argumentative content in text.

## Custom SHAP Visualizations

To further investigate individual classifications and gain a better global overview of important words, I implemented custom SHAP-based visualization methods. These can be found in the [`custom_shap_explainer`](custom_shap_explainer) folder.

The custom visualizations include:

- a **local word highlighting plot** that highlights only the words with the strongest impact on a prediction
- a **global boxplot visualization** that provides more information about the distribution of SHAP values for the most important words

These plots were created because the default SHAP visualizations are useful, but can become difficult to read for longer texts or larger sets of local explanations.

<img width="606" height="204" alt="Bildschirmfoto 2026-01-13 um 14 10 53" src="https://github.com/user-attachments/assets/423667c1-48a7-4676-a659-b1a5aa0f7f12" />

<img width="624" height="376" alt="Bildschirmfoto 2026-01-13 um 14 11 56" src="https://github.com/user-attachments/assets/fd75c05f-8be4-4d49-8794-5e2c3c800a11" />

## Getting Started

To get started with the code in this repository:

1. Clone the repository to your local machine.
2. Install the required dependencies.
3. Open the relevant notebook in the corresponding `pipeline` folder:
   - [`Sentiment/pipeline/Sentiment_Analysis_SHAP.ipynb`](Sentiment/pipeline/Sentiment_Analysis_SHAP.ipynb)
   - [`Argument/pipeline/Claim_Detection_SHAP.ipynb`](Argument/pipeline/Claim_Detection_SHAP.ipynb)
4. Run the notebook cells to train the models and generate the SHAP explanations.
