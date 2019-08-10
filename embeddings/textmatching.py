from six.moves import cPickle
from embeddings.text_feature import text_feature

#!!!for the following: will download huge fasttext model to cache if it was not in the cache
#comment it out if don't want to load it


'''
#get the word embeddings for classes and save
class_emb = embedder.preLoadVec("./coco_classes.txt")
with open("class_vector.pkl", "wb") as f:
    cPickle.dump(class_emb, f)
print("saved class emb")
'''
#the following is to load vectors only for words in coco classes
'''
dict = None
with open("class_vector.pkl", "rb") as f:
    dict = cPickle.load(f)
'''

#print(dict['hair drier']) #some two-word phrases have word vectors


#get word embeddings for queries

def get_embedding(string1, embedder):
	query_emb = embedder.embQuery(string1)
	return query_emb
