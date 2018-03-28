# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# разделение по табам, так как в сообщениях никогда не встречаются табы
# и так можно разделить текст сообщения от его эмоционального оттенка
dataset = pd.read_csv('Restaurant_Reviews.tsv',
                      delimiter = '\t', # tab separated text
                      quoting = 3) # ignore "" simboles in messages

# Cleaning the texts // dataset['Review'][0] get first review
import re
import nltk
nltk.download('stopwords') # скачиваем словарь лишних слов
from nltk.corpus import stopwords

corpus = [] # collection of texts

for i in range(0, 1000) :
  # delet all chars and set ' ' in dataset
  review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
  # to lower letters
  review = review.lower()
  review = review.split() # make list of words ['wow', 'loved', 'this', ...]

  # word проходимся по review и оставляем там только те слова, которые есть
  # в stopwords
  # python работает быстрее с set() чем с list!!!
  # удаляем 'this'

  # stemming words => loved = love for review (ps.stem(word))
  from nltk.stem.porter import PorterStemmer
  ps = PorterStemmer()

  review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

  # join all words in one line and split by ' '
  review = ' '.join(review)
  corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
# помимо max_features, размерность можно уменьшить 9 dimensionality reduction
cv = CountVectorizer(max_features = 1500) # max columns of features 1500
X = cv.fit_transform(corpus).toarray() # fit_transform ???
y = dataset.iloc[:, 1].values

# для классификации в NLP обычно используют NaiveBayes, Descicion Tree C, RFC

# Naive Bayes
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling не нужно, так как мы в большей степени имеем 0
# совсем немного 1 и еще меньше 2, 3 и 4, поэтому не имеет смысл применять
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)

# Fitting the classifire to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test) # предсказываем данные из X_test

# Making the Confusion Matrix # узнаем насколько правильная модель
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) # in console cm


















