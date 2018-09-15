import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from naive_bayes import NaiveBayesClassifier

# Upload Dataset
spams = pd.read_csv("spam.csv", engine="python")

# Clean the DataFrame
spams = spams.dropna(axis=1)
spams.columns = ["spam", "body"]
spams = spams[["body", "spam"]]

# Encode the label
spams["spam"] = LabelEncoder().fit_transform(spams["spam"])

emails = spams["body"]
labels = spams["spam"]

X_train, X_test, y_train, y_test = train_test_split(emails, labels, test_size=0.3)

train_data = pd.concat([X_train, y_train], axis=1)

# Train and classify
nc = NaiveBayesClassifier()
nc.train(train_data)
print(nc)
print(nc.classify("sign up today and win a prize"))
print(nc.classify("At what time would you like to meet"))

# Notes: this type of models work better on small datasets