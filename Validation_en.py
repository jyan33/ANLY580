#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jinghaoyan
"""

import spacy

import os
import pandas as pd
import numpy as np

os.chdir('/Users/jinghaoyan/Documents/GT/Fall2018/ANLY580/Project/data')


fields = ['Title', 'Tags']


test1 = pd.read_csv("test1.csv", sep=',', quotechar = '"',low_memory=True, skip_blank_lines=True, dtype=str, encoding='utf-8', usecols=fields).dropna(how='any')
test1['Title'] = test1['Title'].replace('\<p\>', ' ', regex=True).replace('\</p\>', ' ', regex=True)


prob = []

for num in range(5):
    
    test11 = test1.sample(n=100000)

    nlp2 = spacy.load('en')
    p = 0
    for row in range(len(test11)):
        title = test11.iloc[row]['Title'].lower()
        actual = test11.iloc[row]['Tags'].lower()
        doc2 = nlp2(str(title))
        for ent in doc2.ents:
            if ent.text in str(actual):
                #print(str(actual), ent.text)
                p += 1
    print(p/len(test11))
    prob.append(p/len(test11))

print(np.mean(prob))
