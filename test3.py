# %%
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Sample texts
texts = ['cat dog',
         'dog bird', 
         'bird cat']

texts_2 = [
    'Her interest in cat dog.',
    'With a she distinct passion for dog bird.',
    'In her little workshop, bird cat.'
]

text_3 = [
    "她對機器人的興趣特別濃厚，她那舒適的工作坊內充滿了各式各樣的機械創造，大小不一，從微小到巨大無比。",
    "憑藉對機器人工程學的獨特熱情，她在自己緊湊的工作坊中放置了各種各樣的機械構造，每一個都在形狀和大小上獨一無二。",
    "在她那小小的工作坊裡，每一個可以想像的形狀和尺寸的機械奇蹟都蘊藏著生命，反映出她對機器人領域的非凡迷戀。"
]
# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Apply fit_transform
tfidf_matrix = vectorizer.fit_transform(text_3)

# Convert to DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# Print the DataFrame
print(tfidf_df)

# %%
