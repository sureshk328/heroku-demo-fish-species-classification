import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping

df=pd.read_csv('Fish.csv')

def convert_to_int(word):
    word_dict = {'Bream':0, 'Roach':1, 'Whitefish':2, 'Parkki':3, 'Perch':4, 'Pike':5, 'Smelt':6}
    return word_dict[word]

df['Species'] = df['Species'].apply(lambda x : convert_to_int(x))

feature_cols = ['Weight', 'Length1', 'Length2', 'Length3','Height','Width']
X = df[feature_cols] # Features
y = df.Species # Target variable


logisticRegr = LogisticRegression(max_iter=100000)
logisticRegr.fit(X,y)

# Saving model to disk
pickle.dump(logisticRegr, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[19.9, 13.8,15.4, 16,2.52,2.02]]))
