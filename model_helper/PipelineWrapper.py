import pandas as pd
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

class PipelineWrapper():
    """The class is a Python wrapper designed to streamline text classification 
    tasks using scikit-learn pipelines. It encapsulates a model and vectorizer within a pipeline, 
    offering simple methods for fitting,
    predicting, generating classification reports, and retrieving feature names.
    """
    def __init__(self, model, vectorizer, corpus_test, corpus_train, y_test, y_train):
        self.corpus_test = corpus_test
        self.corpus_train = corpus_train
        self.y_test = y_test
        self.y_train = y_train
        self.pipeline = Pipeline([('vectorizer', vectorizer), ('model', model)])


    def fit(self):
        self.pipeline.fit(self.corpus_train, self.y_train)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)
    
    def report(self):
        report = classification_report(self.y_test, self.predict(self.corpus_test),  output_dict=True)
        return pd.DataFrame(report)
    
    # named steps
    def get_feature_names(self):
        return self.pipeline.named_steps['vectorizer'].get_feature_names_out()