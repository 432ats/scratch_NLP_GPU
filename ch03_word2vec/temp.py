import sys
sys.path.append('/home/ats432/projects/Matsuzaki_Lab/scratch_NLP')
import numpy as np 
from common.util import preprocess, create_contexts_target, convert_one_hot

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)


#コンテキストとターゲットの生成
contexts, target = create_contexts_target(corpus, window_size=1)


#one-hot表現への変換
vocab_size = len(word_to_id)
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)
print(contexts)
print(target)