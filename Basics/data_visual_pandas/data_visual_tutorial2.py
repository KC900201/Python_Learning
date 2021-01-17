# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 2020

@author: Kwong Cheong Ng
@filename: data_visual_tutorial2.py
@coding: utf-8
========================
Date          Comment
========================
02192020      First revision
"""

import matplotlib
import numpy as np
import pandas as pd


def to_dataframe(df):
    graph_df = pd.DataFrame()
    
#    for region in df['region'].unique()[:16]:
    for region in df['region'].unique():
        print(region)
        region_df = df.copy()[df['region']==region]
        region_df.set_index('Date', inplace=True)
        region_df.sort_index(inplace=True)
        region_df[f"{region}_price25ma"] = region_df["AveragePrice"].rolling(25).mean()
        
        if graph_df.empty:
            graph_df = region_df[[f"{region}_price25ma"]] # note the double square brackets!
        else:
            graph_df = graph_df.join(region_df[f"{region}_price25ma"])
    
    return graph_df

if __name__ == '__main__':

    # Read csv data
    df = pd.read_csv("datasets/avocado.csv")

    # Convert string to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.copy()[df['type']=='organic']
    df.sort_values(by="Date", ascending=True, inplace=True)
    
    graph_df = to_dataframe(df)
    
#    albany_df = df[df['region'] == 'Albany']
    # Create a copy of df from existing to avoid pandas Warning
#    albany_df = df.copy()[df['region'] == "Albany"]
#    albany_df.set_index("Date", inplace=True)
#    albany_df.sort_index(inplace=True)    
    
#    print(albany_df["AveragePrice"].plot())
    
    # Apply smoothing to graph, then plot
    #albany_df["AveragePrice"].rolling(25).mean().plot()

    # Create new column - average price
#    albany_df["price25ma"] = albany_df["AveragePrice"].rolling(25).mean()
#    print(albany_df.head())
#    print(albany_df.tail())

    # Convert value to array, array to list
#    print(set(df['region'].values.tolist()))
    # Show unique values of column
#    print(df['region'].unique())
    print(graph_df.tail())
    print(graph_df.plot(figsize=(8,5), legend=False))
    


