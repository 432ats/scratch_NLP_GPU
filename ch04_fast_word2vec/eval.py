import sys
sys.path.append('/home/ats432/projects/Matsuzaki_Lab/scratch_NLP')
from common.util import most_similar
import pickle

pkl_file = 'cbow_params.pkl'

with open(pkl_file, 'rb') as f:
    params = pickle.load(f)
    word_vecs = params['word_vecs']
    word_to_id = params['word_to_id']
    id_to_word = params['id_to_word']

querys = ['you', 'car', 'toyota', 'black', 'japanese']

for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)
