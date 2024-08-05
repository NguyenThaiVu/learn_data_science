import os 
import random
import numpy as np
import pandas as pd 
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


import mlflow
import mlflow.sklearn

def load_dataset():
    data = load_wine()
    X = data.data
    y = data.target
    return (X, y)


# Hyper parameter
list_criterion = ['gini', 'entropy', 'log_loss']
list_max_depth = [3, 5, 7]
list_min_samples_leaf = [5, 10]
max_trials = 5


if __name__ == "__main__":

    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Create MLFlow experiment to tracking model
    my_exp = mlflow.set_experiment("auto_experiment")

    for trial in range(max_trials):

        # Grid search random hyper-parameters
        criterion = random.choice(list_criterion)
        max_depth = random.choice(list_max_depth)
        min_samples_leaf = random.choice(list_min_samples_leaf)

        with mlflow.start_run(experiment_id=my_exp.experiment_id):
            
            # Train model
            model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
            model.fit(X_train, y_train)

            # Evaluation
            y_test_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_test_pred)
            f1 = f1_score(y_test, y_test_pred, average='macro')

            # Log params
            # mlflow.log_params({'criterion':criterion,\
            #                 'max_depth':max_depth,\
            #                     'min_samples_leaf':min_samples_leaf})
            
            # mlflow.log_metrics({'accuracy': accuracy, 'f1': f1})

            # mlflow.sklearn.log_model(model, 'model')
            mlflow.autolog()


