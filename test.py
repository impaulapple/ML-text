# 教學網站
# https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
# %%

from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
import Utils

categories = ["alt.atheism", "soc.religion.christian", "comp.graphics", "sci.med"]
twenty_train = fetch_20newsgroups(
    subset="train", categories=categories, shuffle=True, random_state=42
)
twenty_train.target_names
len(twenty_train.data)
len(twenty_train.filenames)
print("\n".join(twenty_train.data[0].split("\n")[:3]))
print(twenty_train.target_names[twenty_train.target[0]])
# df = pd.DataFrame({
#     '類別': twenty_train.target,
#     '文字': twenty_train.data
# })
# Utils.save_to_excel(df, "text_data")
#%% 
# Print the part of dataset
print(twenty_train.target[:10])
for t in twenty_train.target[:10]:
    print(twenty_train.target_names[t])
    
    
#%% Tokenizing text with scikit-learn

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape
count_vect.vocabulary_.get(u'algorithm')

#%% From occurrences to frequencies (TF-IDF 跟上面選一個就好)
#
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape
# %% Training a classifier
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

docs_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))
# %% Building a pipeline
from sklearn.pipeline import Pipeline
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])
text_clf.fit(twenty_train.data, twenty_train.target)
# %% Evaluation of the performance on the test set
import numpy as np
twenty_test = fetch_20newsgroups(subset='test',
    categories=categories, shuffle=True, random_state=42)
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
np.mean(predicted == twenty_test.target)
# %%  support vector machine (SVM)
from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, random_state=42,
                          max_iter=5, tol=None)),
])
text_clf.fit(twenty_train.data, twenty_train.target)
predicted = text_clf.predict(docs_test)
np.mean(predicted == twenty_test.target)
# %%
