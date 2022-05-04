#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 17:25:13 2022

@author: anes
"""


#%%
import pandas as pd
import numpy as np
import os,cv2

#%%

dataframe = pd.read_csv('jaypee_metadata.csv')
dataset = np.array(dataframe)

#%%

for dirname, _, filenames in os.walk('./images'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


#%%
