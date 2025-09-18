import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
credit_card_df = pd.read_csv("creditcard.csv")

# Separate legit and fraud
legit = credit_card_df[credit_card_df.Class == 0]
fraud = credit_card_df[credit_card_df.Class == 1]

# Balance dataset
legit_sample = legit.sample(n=102, random_state=2)
credit_card_df = pd.concat([legit_sample, fraud], axis=0)

# Features & labels
x = credit_card_df.drop("Class", axis=1)
y = credit_card_df["Class"]

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=2
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# Predictions & accuracy
ypred = model.predict(x_test)
print("Accuracy:", accuracy_score(ypred, y_test))
