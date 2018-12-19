#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jinghaoyan
"""


from rake_nltk import Rake
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
    
    r = Rake()
    
    p = 0
    for row in range(len(test11)):
        title = test11.iloc[row]['Title'].lower()
        tags = test11.iloc[row]['Tags'].lower()
        r.extract_keywords_from_text(title)
        r.get_ranked_phrases()
        if r.get_ranked_phrases() != []:
            for item in r.get_ranked_phrases():
                if item.lower() in tags:
                    p += 1
    print(p/len(test11))
    prob.append(p/len(test11))

print(np.mean(prob))

