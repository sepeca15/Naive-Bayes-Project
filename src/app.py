import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

data = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/naive-bayes-project-tutorial/main/playstore_reviews_dataset.csv')

data = data.drop(columns = 'package_name')
data['review'] = data['review'].str.strip().str.lower()

X = data['review']
y = data['polarity']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.25, random_state = 42)

vec = CountVectorizer(stop_words = 'english')
X_train = vec.fit_transform(X_train).toarray()
X_test = vec.transform(X_test).toarray()
model = MultinomialNB()
model.fit(X_train, y_train)

filename = '../models/playStoreReview_Model.pickle'
pickle.dump(model, open(filename,'wb'))