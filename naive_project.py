#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 20:00:36 2026

@author: Subham
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# What is the business objective?
'''
1.1 Business Objective

The objective is to automatically identify whether a tweet about a disaster is real or fake using a Naïve Bayes classification model.

This helps organizations like:

Disaster response teams

News agencies

Emergency services


1.2 Business Constraints
How to make a model that should identify the real disaster.

The system should process tweets quickly.

'''
#================================== Data Pre-processing =====================================


'''2.1 Data Cleaning'''
  


# 1 Loading the data set
Disaster_data = pd.read_csv("/Users/Subham/Desktop/data/360_back_up/360 assignment / Naive Bayes/Datasets_Naive Bayes/Disaster_tweets_NB.csv", encoding = "ISO-8859-1")

Disaster_data.info()

Disaster_data.head(10)

# Step 3 Remove unnecessary column
#in case of dropping multiple columns
####Disaster_data.drop(['id','location','keyword'])

Disaster_data= Disaster_data.drop("id",axis=1)

#Step 4 Handle missing values


Disaster_data.isnull().sum()

Disaster_data["keyword"] = Disaster_data["keyword"].fillna("unknown") #here we are not dropping the rows because its a big set of data so making it 
Disaster_data["location"] = Disaster_data["location"].fillna("unknown")# with unknown

#Step 5 Text Cleaning
Disaster_data['text']

import re

def clean_text(text):

    text = text.lower()

    text = re.sub("[^a-zA-Z]", " ", text)
    text = re.sub(r'\b\w{10,}\b', '', text)
    text = re.sub("\s+"," ",text)
    return text.strip()



Disaster_data["text"] = Disaster_data["text"].apply(clean_text)

#================================== 4.EDA=====================================



Disaster_data.describe()
Disaster_data.head(10)



#Univariate Analysis
Disaster_data["target"].value_counts()

# real 1
# fake 0

sns.countplot(x="target",data=Disaster_data)

Disaster_data["keyword"].value_counts().head(10)

#As per  the keywords which one is coming when
pd.crosstab(Disaster_data.keyword, Disaster_data.target)

# top 5 location for the real disaster 

real_disaster=Disaster_data[Disaster_data['target']==1] #filtered separate dataframe


top_location = real_disaster["location"].value_counts().head(5)
# in visual form

plt.figure(figsize=(8,5))

sns.barplot(
    x=top_location.values,
    y=top_location.index
)

plt.title("Top 5 Real Disaster Tweets by Location")
plt.xlabel("Number of Tweets")
plt.ylabel("Location")


print(f"{top_location}The analysis shows that the majority of real disaster tweets originate from locations such as USA, London, and Canada. These locations appear to have higher reporting activity related to disasters on Twitter.")

# top 5 keywords for the real disaster 


top_keyword = real_disaster["keyword"].value_counts().head(5)

plt.figure(figsize=(10,7))

sns.barplot(x=top_keyword,y=top_keyword.index)

"""Keywords such as fire, earthquake,
      and flood appear most frequently in 
      real disaster tweets. These keywords are strong 
      indicators of real disaster-related events."""


#=====================Model Building========================

from sklearn.model_selection import train_test_split
# Train Test Split

X = Disaster_data["text"]
y = Disaster_data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# Convert Text to Numbers (data preparation for the model) so here we are using countvectorizer

vectorizer = CountVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)

X_test_vec = vectorizer.transform(X_test)


#Building Naive Bayes Model

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()

model.fit(X_train_vec, y_train)

# Model Validation

y_pred = model.predict(X_test_vec)


#confusion matrix

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))

sns.heatmap(cm, 
            annot=True, 
            fmt="d", 
            cmap="Blues")

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.show()


# accuracy score

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

accuracy_value = accuracy * 100

plt.figure(figsize=(5,4))

plt.bar(["Accuracy"], [accuracy_value])

plt.ylim(0,100)

plt.title("Model Accuracy")

plt.ylabel("Percentage")

plt.show()



# Precision & Recall

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))



# ===========================model tuning again i.e experimenting more to get better accuracy  ===========================

# so before we tried everything with countvectorizer now we are trying it with
#TfidfVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
#tfidf = TfidfVectorizer(stop_words="english") as we see these are reducing the feature for MultinomialNB model .
#tfidf = TfidfVectorizer(stop_words="english")

X_train_tfidf = tfidf.fit_transform(X_train)

X_test_tfidf = tfidf.transform(X_test)


# Building Naive Bayes Model

model = MultinomialNB(alpha=0.5)

model.fit(X_train_tfidf, y_train)



# Model Validation

y_pred = model.predict(X_test_vec)



from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))

sns.heatmap(cm, 
            annot=True, 
            fmt="d", 
            cmap="Blues")

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.show()



# accuracy score

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)


#auc roc curve


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

y_prob = model.predict_proba(X_test_tfidf)[:,1] #This gives probability of class 1 (real disaster).

#Compute ROC values
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

#Calculate AUC
roc_auc = auc(fpr, tpr)

#display
plt.figure(figsize=(6,5))

plt.plot(fpr, tpr, label="ROC curve (AUC = %0.2f)" % roc_auc)
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Disaster Tweet Model")
plt.legend()
plt.show()

#==============================Deployment==================================


import pickle

# save model
pickle.dump(model, open("model.pkl", "wb"))

# save vectorizer
pickle.dump(tfidf, open("vectorizer.pkl", "wb"))


