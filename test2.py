#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import Utils

# 假設 pdf_texts 是一個包含 100 個 PDF 文本的列表
# 假設 labels 是一個包含 100 個標籤的列表

# 使用 read_excel 方法讀取文件
df = pd.read_excel('text_data.xlsx')

# 步驟 2: 特徵工程
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(df['文字'].values)

# Convert to DataFrame
tfidf_df = pd.DataFrame(X.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Print the DataFrame
print(tfidf_df)
# Utils.save_to_excel(tfidf_df, 'text_vector')

# 步驟 3: 分割數據集
X_train, X_test, y_train, y_test = train_test_split(X, df['類別'], test_size=0.2)

# 步驟 4: 選擇和訓練模型
model = SVC()
model.fit(X_train, y_train)

# 步驟 5: 評估模型
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")

# %% Plot the Confusion Matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title(f'{model.__class__.__name__} Confusion Matrix')
plt.show()

#%%
# Get feature importance
importance = model.feature_importances_

# Create a DataFrame to view importance
importance_df = pd.DataFrame({
    'Feature': tfidf_vectorizer.get_feature_names_out(),
    'Importance': importance
})

# Sort by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)
importance_df
Utils.save_to_excel(importance_df, 'text_importance')

# %%
