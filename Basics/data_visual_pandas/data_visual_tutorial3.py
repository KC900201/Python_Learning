# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 15:43:27 2020

@author: KwongCheongNg
@filename: data_visual_tutorial3.py
@coding: utf-8
========================
Date          Comment
========================
02232020      First revision
"""

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def data_visual(min_wage_corr):
    labels = []
    
    
    plt.matshow(min_wage_corr)
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv("../datasets/Minimum Wage Data.csv", encoding="latin")
    
    #print(df.head())
    
    df.to_csv("../datasets/minwage.csv", encoding="utf-8")
    df = pd.read_csv("../datasets/minwage.csv")
    #print(df.head())
    
    # Groupby function
    gb = df.groupby("State")
    #print(gb.get_group("Alabama").set_index("Year").head())
    
    act_min_wage = pd.DataFrame()
    
    for name, group in df.groupby("State"):
        if act_min_wage.empty:
            act_min_wage = group.set_index("Year")[["Low.2018"]].rename(columns={"Low.2018":name})
        else:
            act_min_wage = act_min_wage.join(group.set_index("Year")[["Low.2018"]].rename(columns={"Low.2018":name}))
    
    min_wage_corr = act_min_wage.replace(0, np.NaN).dropna(axis=1).corr()
    min_wage_corr.head()

#    print("Act_min_wage.head(): ")
#    print(act_min_wage.head())
    
#    print(act_min_wage.describe())
    
    #print(act_min_wage.corr().head())
    '''
    issue_df = df[df['Low.2018']==0]

    print(issue_df.head())
    print(issue_df['State'].unique())
    print(act_min_wage.replace(0, np.NaN).dropna(axis=1).corr().head())
    
    for problem in issue_df['State'].unique():
        if problem in min_wage_corr.columns:
            print("Missing something here....")
            
    grouped_issues = issue_df.groupby("State")
    
    print(grouped_issues.get_group("Alabama").head(3))
    print(grouped_issues.get_group("Alabama")['Low.2018'].sum())
    '''