# -*- coding: utf-8 -*-
"""
Spyderエディタ

これは一時的なスクリプトファイルです
"""

import numpy as np
import pandas as pd
import os

def xlsx_to_text(directory, startswith, sheet_name):
    for filename in os.listdir(directory):
        if filename.startswith(startswith) and filename.endswith(".xlsx"):
            print("now reading ",os.path.join(directory, filename), "...")
            #extract sheet
            df=pd.read_excel(directory + "/" + filename, sheet_name = sheet_name)
            df_dat=df.iloc[:,range(8,df.shape[1],9)]
            df_dat.to_csv(filename.replace(".xlsx",".csv"))
            continue
        else:
            continue

xlsx_to_text(directory = "sandbox", 
             startswith = "ダイナミック" ,
             sheet_name = "RawVolts_IIRgram")