# -*- coding: utf-8 -*-
"""
Created on Thur Jan 23  2020

@author: Kwong Cheong Ng
@filename: data_visual_tutorial.py
@coding: utf-8
========================
Date          Comment
========================
01232020      First revision
"""

# Import libraries
import pandas as pd
import matplotlib

if __name__ == '__main__':
    # Read csv
#    df = pd.read_csv("datasets/risk_prediction_graph.csv")
    df = pd.read_csv("../datasets/avocado.csv")
    df['Date'] = pd.to_datetime(df['Date'])
#    print(df)

#    print(df.head(10))
#    print(df.tail(10))
    
    # reference specific columns
#    print(df['risk of interval [0,9]'].head())
#    print(df['risk of interval [10,19]'].head())
#    print(df['risk of interval [20,29]'].head())
    
#    print(df.index)
    
    #avocado.csv
    albany_df = df[df['region']=="Albany"]
    albany_df.set_index("Date", inplace=True)
    albany_df.sort_index(inplace=True)
 #   albany_df["AveragePrice"].rolling(25).mean().plot()
    
    # set new column
    albany_df["price25ma"] = albany_df["AveragePrice"].rolling(25).mean()
    print(albany_df.head())
    print(albany_df['AveragePrice'].plot())
    #risk_prediction_graph.csv
    '''
    #Plot graph
    anan_df = df[df['dataset'] == 'A3D']
    anan_df.set_index('risk_class', inplace = True)
    anan_df['risk of interval [0,9]'].rolling(25).mean().plot()
#    anan_df.plot()
    
    dashcam_df = df[df['dataset'] == 'Dashcam_Accident']
    dashcam_df.set_index('risk_class', inplace = True)
    dashcam_df['risk of interval [0,9]'].rolling(25).mean().plot()
#    dashcam_df.plot()
    '''
