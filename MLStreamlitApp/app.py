import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

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
positions = {
    'PG':1,
    'SG':2,
    'SF':3,
    'PF':4,
    'C':5
}
final_dataset['Pos'] = final_dataset['Pos'].map(positions)

st.title("Exploring Machine Learning Classification Models")

path = st.radio("Choose your path!", ["Upload my own dataset", "Become an NBA All-Star"])

if path == "Become an NBA All-Star":
    st.subheader("Predicting NBA All-Star Status with Machine Learning")
    st.write("Excellent choice! In this portion of the app, you get to pretend that you're a basketball player in the NBA during the 2023-24 season!")
    st.write("You'll first input your position, as well as your season statistics for Points per Game, Assists per Game, and Rebounds per Game. Then, you'll choose what kind of classification model you'd like to use to predict your All-Star status -- either logistic regression, decision tree, or k-nearest neighbors. Finally, you'll tune the hyperparameters of the corresponding model and hit 'Go'!")
    st.write("The app will spit out your probability of being an All-Star, as well as display some of the model metrics to give you a sense of how accurate the prediction is. On your mark, get set, go!")

    position = st.selectbox("Select your position:", ['PG', 'SG', 'SF', 'PF', 'C'])   

    points = st.slider("Enter your average points per game:", min_value = 0.0, max_value = final_dataset['PTS'].max(), value = 10.0, step = 0.1)

    assists = st.slider("Enter your average assists per game:", min_value = 0.0, max_value = final_dataset['AST'].max(), value = 3.0, step = 0.1)

    rebounds = st.slider("Enter your average rebounds per game:", min_value = 0.0, max_value = final_dataset['TRB'].max(), value = 4.0, step = 0.1)

    model_choice = st.radio("Choose a classification model:", ['Logistic Regression', 'Decision Tree', 'K-Nearest Neighbors'])

    if model_choice == "Logistic Regression":
        scale_question = st.radio("Would you like to scale the data? (Scaling adjusts for unit bias)", ['Yes', 'No'])
    
        if st.button("Run!"):
            features = ['Pos', 'PTS', 'AST', 'TRB']
            X = final_dataset[features]
            y = final_dataset['All-Star']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                                random_state = 99)
            user_data = [[positions[position], points, assists, rebounds]]
            if scale_question == "Yes":
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                user_data = scaler.transform(user_data)
            model = LogisticRegression()
            model.fit(X_train, y_train)
            prob = model.predict_proba(user_data)
            st.write(f"Your probability of being an all-star is {prob[0][1]:.2%}!")
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"This model has an accuracy of {accuracy:.4f}. The give you a more complete picture of the model's accuracy, check out the confusion matrix and classification report below.")
            col1, col2 = st.columns(2)
            with col1:
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot = True, cmap = "Blues")
                st.subheader("Confusion Matrix for Logistic Regression")
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                st.pyplot(plt)
                plt.clf()
            with col2:
                st.subheader("Classification Report for Logistic Regression")
                st.text(classification_report(y_test, y_pred))


    

