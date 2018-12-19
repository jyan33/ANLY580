#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jinghaoyan
"""

import spacy

import os
import pandas as pd


os.chdir('/Users/jinghaoyan/Documents/GT/Fall2018/ANLY580/Project/data')


output_dir = '/Users/jinghaoyan/Documents/GT/Fall2018/ANLY580/Project/testmodel/'


fields = ['Title']


test = pd.read_csv("Test.csv", sep=',', quotechar = '"',low_memory=True, skip_blank_lines=True, dtype=str, encoding='utf-8', usecols=fields).dropna(how='any')
test['Title'] = test['Title'].replace('\<p\>', ' ', regex=True).replace('\</p\>', ' ', regex=True)


test2 = test.sample(n=100000)

test2.to_csv('test2.csv', index=False, quotechar='"')

with open('predict_new_model.txt', 'w') as fout:
    fout.write('')

print("Loading from", output_dir)
nlp2 = spacy.load(output_dir)
with open('predict_new_model.txt', 'a') as fout:
    for row in range(len(test2)):
        lst = []
        title = test2.iloc[row]['Title'].lower()
        doc2 = nlp2(str(title))
        for ent in doc2.ents:
            lst.append(ent.text)
        fout.write(str(title) + '|' + str(lst) + '\n')

