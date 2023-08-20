#%%
import jieba

sentence = "我愛機器學習和自然語言處理"

# Use jieba to split the sentence into words
words = jieba.cut(sentence)

# Convert the result to a list (optional)
words = list(words)

# Print the list of words
print(words)

# %%
