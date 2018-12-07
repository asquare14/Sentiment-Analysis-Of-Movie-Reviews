import numpy as np 
import pandas as pd 
pd.set_option('display.max_colwidth', -1)
from time import time
import re
import string
import os
from pprint import pprint
import collections

def generate(df_test,test_df_flags,var,name):
    df_test_logistic = df_test.copy()[["PhraseId"]]
    df_test_logistic['Sentiment'] = var.predict(test_df_flags)
    df_test_logistic.to_csv(name+'.csv', index = False)