# %%
from gensim.models import Word2Vec
import jieba

# Sample sentences
sentences = [
    "她對機器人的興趣特別濃厚，她那舒適的工作坊內充滿了各式各樣的機械創造，大小不一，從微小到巨大無比。",
    "憑藉對機器人工程學的獨特熱情，她在自己緊湊的工作坊中放置了各種各樣的機械構造，每一個都在形狀和大小上獨一無二。",
    "在她那小小的工作坊裡，每一個可以想像的形狀和尺寸的機械奇蹟都蘊藏著生命，反映出她對機器人領域的非凡迷戀。"
]

sentences_new = []
for sentence in sentences:
    words = jieba.cut(sentence)
    sentences_new.append(list(words))
    

# Train the Word2Vec model
model = Word2Vec(sentences_new, vector_size=100, window=5, min_count=1, workers=4)

# Save the trained model
model.save("word2vec.model")
aaa = model.wv
aaa.index_to_key
#%%
# Load the model
model = Word2Vec.load("word2vec.model")


# Find vector for a word
vector = model.wv['興趣']
print("Vector for '興趣':", vector)

# Find similar words
similar_words = model.wv.most_similar('她', topn=3)
print("Words similar to '她':", similar_words)

# %%
