#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jinghaoyan
"""

from rake_nltk import Rake
import os
import pandas as pd


os.chdir('/Users/jinghaoyan/Documents/GT/Fall2018/ANLY580/Project/data')

fields = ['Title']


test2 = pd.read_csv("test2.csv", sep=',', quotechar = '"',low_memory=True, skip_blank_lines=True, dtype=str, encoding='utf-8', usecols=fields).dropna(how='any')
test2['Title'] = test2['Title'].replace('\<p\>', ' ', regex=True).replace('\</p\>', ' ', regex=True)


r = Rake()

with open('predict_rake.txt', 'w') as fout:
    fout.write('')

with open('predict_rake.txt', 'a') as fout:
    for row in range(len(test2)):
        title = test2.iloc[row]['Title'].lower()
        r.extract_keywords_from_text(title)
        r.get_ranked_phrases()
        if r.get_ranked_phrases() != []:
            fout.write(str(title) + '|' + str(r.get_ranked_phrases()) + '\n')
