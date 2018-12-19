#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jinghaoyan
"""

from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding


import os
import pandas as pd
from sklearn.model_selection import train_test_split


os.chdir('/Users/jinghaoyan/Documents/GT/Fall2018/ANLY580/Project/data')


fields = ['Title', 'Tags']
train = pd.read_csv("Train.csv", sep=',', quotechar = '"',low_memory=True, skip_blank_lines=True, dtype=str, encoding='utf-8', usecols=fields).dropna(how='any')

train['Title'] = train['Title'].replace('\<p\>', ' ', regex=True).replace('\</p\>', ' ', regex=True)

train1, test1 = train_test_split(train, test_size=0.2)

#train1.to_csv('train1.csv', index=False)
test1.to_csv('test1.csv', index=False, quotechar='"')



# new entity label
LABEL = 'STACKEX'


#c = 0
TRAIN_DATA = []
for row in range(500000):
    mst = {}
    #c += 1
    title = train1.iloc[row]['Title'].lower()
    tags = train1.iloc[row]['Tags'].lower()
    #print(str(c) + ': ' + str(title), str(tags))
    for tag in tags.split(' '):
        if tag in title:
            mst[str(tag)] = 1
            dct = {}
            dct2 = {}
            #print(str(tag))
            #print(title[title.find(str(tag)):title.find(str(tag))+len(str(tag))])
            if str(tag) not in dct:
                dct[str(tag)] = 1
                dct2['entities'] = [(title.find(str(tag)), title.find(str(tag))+len(str(tag)), 'STACKEX')]
                TRAIN_DATA.append((str(title), dct2))
    if mst == {}:
        dct3 = {}
        dct3['entities'] = []
        TRAIN_DATA.append((str(title), dct3))




@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    new_model_name=("New model name for model meta.", "option", "nm", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int))
def main(model='en', new_model_name='test_model2', output_dir='/Users/jinghaoyan/Documents/GT/Fall2018/ANLY580/Project/testmodel/', n_iter=10):
    """Set up the pipeline and entity recognizer, and train the new entity."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")
        
        
    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe('ner')

    ner.add_label(LABEL)   # add new entity label to entity recognizer
    
    if model is None:
        optimizer = nlp.begin_training()
    else:
        # Note that 'begin_training' initializes the models, so it'll zero out
        # existing entity types.
        optimizer = nlp.entity.create_optimizer()

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4., 32., 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.35,
                           losses=losses)
            print('Losses', losses)

    # test the trained model
    test_text = 'What about Linux R and Python java sql?'
    doc = nlp(test_text)
    print("Entities in '%s'" % test_text)
    for ent in doc.ents:
        print(ent.label_, ent.text)

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta['name'] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

#        # test the saved model
#        print("Loading from", output_dir)
#        nlp2 = spacy.load(output_dir)
#        p = 0
#        for row in range(50):
#            title = test1.iloc[row]['Title'].lower()
#            actual = test1.iloc[row]['Tags'].lower()
#            doc2 = nlp2(str(title))
#            for ent in doc2.ents:
#                if ent.text in str(actual):
#                    #print(str(actual), ent.text)
#                    p += 1
#        print(p/50)
                    

#        doc2 = nlp2(test_text)
#        for ent in doc2.ents:
#            print(ent.label_, ent.text)


if __name__ == '__main__':
    plac.call(main)