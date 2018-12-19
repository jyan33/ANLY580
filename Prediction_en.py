#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jinghaoyan
"""

import spacy

import os
import pandas as pd


os.chdir('/Users/jinghaoyan/Documents/GT/Fall2018/ANLY580/Project/data')


fields = ['Title']


test2 = pd.read_csv("test2.csv", sep=',', quotechar = '"',low_memory=True, skip_blank_lines=True, dtype=str, encoding='utf-8', usecols=fields).dropna(how='any')
test2['Title'] = test2['Title'].replace('\<p\>', ' ', regex=True).replace('\</p\>', ' ', regex=True)

with open('predict_en.txt', 'w') as fout:
    fout.write('')

nlp2 = spacy.load('en')
with open('predict_en.txt', 'a') as fout:
    for row in range(len(test2)):
        lst = []
        title = test2.iloc[row]['Title'].lower()
        doc2 = nlp2(str(title))
        for ent in doc2.ents:
            lst.append(ent.text)
        fout.write(str(title) + '|' + str(lst) + '\n')

