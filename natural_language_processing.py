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
# также размер можно уменьшить из 9 раздела dimensionality reduction
cv = CountVectorizer(max_features = 1500) # max columns of features 1500
X = cv.fit_transform(corpus).toarray() # fit_transform ???
y = dataset.iloc[:, 1].values




















