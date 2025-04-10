import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

nba_data = pd.read_csv("data/NBA_Regular_Season.csv", sep = ";", encoding = 'latin-1')

new_data = (nba_data[['Rk', 'Player', 'Pos', 'PTS', 'AST' ,'TRB']].groupby('Rk', as_index=False).agg({
    'Player': 'first',  
    'Pos': 'first',
    'PTS': 'mean',
    'AST': 'mean',
    'TRB': 'mean'
}))

new_data = new_data[new_data['Pos'].isin(['PG', 'SG', 'SF', 'PF', 'C'])].reset_index(drop = True)

all_star_dict = {"Player": ["Tyrese Haliburton", "Damian Lillard", "Giannis Antetokounmpo", "Jayson Tatum", "Joel Embiid",
                 "Jalen Brunson", "Tyrese Maxey", "Donovan Mitchell", "Trae Young", "Paolo Banchero", "Scottie Barnes", "Jaylen Brown",
                 "Julius Randle", "Bam Adebayo", "Luka Don?i?", "Shai Gilgeous-Alexander", "Kevin Durant", "LeBron James", "Nikola Joki?",
                 "Devin Booker", "Stephen Curry", "Anthony Edwards", "Paul George", "Kawhi Leonard", "Karl-Anthony Towns", "Anthony Davis"],
                 "All-Star": [int(1)]*26}

all_star_data = pd.DataFrame(all_star_dict)

final_dataset = (pd.merge(new_data, all_star_data, how = "outer", on = "Player").fillna(0))
final_dataset['Rk'] = final_dataset['Rk'].astype(int)
final_dataset['All-Star'] = final_dataset['All-Star'].astype(int)

st.title("Exploring Machine Learning Classification Models")

path = st.radio("Choose your path!", ["Upload my own dataset", "Become an NBA All-Star"])

if path == "Become an NBA All-Star":
    