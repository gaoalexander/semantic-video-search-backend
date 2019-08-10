import torch
import spacy
import re
import torchtext
from torchtext.vocab import FastText


#drop stop word
class text_feature():
    def __init__(self, mode="single"):
        self.mode = mode
        self.spacy_nlp = spacy.load('en_core_web_sm')
        self.vectors = FastText()

    def cleanText(self, input):
        '''
        input: string, search query
        output: list of words/single word without stop words in lowercase
        '''
        lowerinput = re.sub(r'[^\w\s]','', input).lower()
        if self.mode == "single":
            lowerinput = lowerinput.replace(" ", "")
            print(lowerinput)
        s = self.spacy_nlp(lowerinput)
        tokens = []
        if self.mode != "single":
            tokens = [token.text for token in s if not token.is_stop]
        else:
            tokens = [s.text]
        #print(tokens)
        return tokens

    def embText(self, tokens):
        '''
        tokens: a (list of) words
        output: a (list of) tensors of size [300]
        '''
        return [self.vectors[tk].numpy() for tk in tokens]

    def preLoadVec(self, path):
        '''
        extract and store word vectors for all the object classes
        '''
        dict = {}
        file = open(path, "r")
        f = file.read().splitlines()
        for line in f:
            tokens = self.cleanText(line)
            dict[line] = self.embText(tokens)

        file.close()
        return dict

    def embQuery(self, query):
        tokens = self.cleanText(query)
        return self.embText(tokens)








#embedding with fasttext
