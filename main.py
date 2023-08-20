#%%
# 加载库
import nltk
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# 加载文本数据样本
documents = [["Text data example one"], 
             ["Second example of text data"],
             ["Third text data example"]] 

labels = [0, 1, 0] # 文档对应标签

# 分词
tokenizer = nltk.RegexpTokenizer(r"\w+")
# documents = [tokenizer.tokenize(doc) for doc in documents]

# 创建tfidf向量  
tfidf = TfidfTransformer()
tfidf_vectors = tfidf.fit_transform(documents)

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(tfidf_vectors, labels, test_size=0.2, random_state=42)

# 训练和预测
clf = MultinomialNB().fit(X_train, y_train)
predicted = clf.predict(X_test)

# 评估
print("Accuracy:", accuracy_score(y_test, predicted))
# %%





