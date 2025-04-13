import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

def data_prep(dataset, features, target):
    X = dataset[features]
    y = dataset[target]
    return X,y

def model_prob(model, user_data):
    prob = model.predict_proba(user_data)
    st.write(f"Your probability of being an all-star is **{prob[0][1]:.2%}**!")

def model_prob2(model, user_data):
    prob = model.predict_proba(user_data)
    st.write(f"The probability that '{target}' is '{input_data[target].unique()[1]}' is **{prob[0][1]:.2%}**!")

def model_metrics(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    st.markdown(f"""
                Main model metrics:
                - Accuracy is the percentage of correct predictions made by the model. This model has an accuracy of **{accuracy:.2%}**.
                - Precision answers the question: 'Out of all datapoints classified as positive, what percentage were actually positive?' This model's precision is **{precision:.2%}**.
                - Recall answers the question: 'Out of all datapoints that were actually positive, what percentage did we catch?' This model has a recall of **{recall:.2%}**.
                """)
    st.write("For a more complete picture of the model's predictive power, check out the confusion matrix, classification report, and ROC Curve/AUC below.")

def plot_confusion(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot = True, cmap = "Blues", fmt = 'g')
    st.write("### Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(plt)
    plt.clf()

def show_classification(y_test, y_pred):
    st.write("### Classification Report")
    st.text(classification_report(y_test, y_pred))
    st.write("Note: F1-score gives the harmonic mean of precision and recall.")

def plot_roc(fpr, tpr, roc_auc):
    plt.figure(figsize = (4,4))
    plt.plot(fpr, tpr, label = f'ROC Curve, AUC = {roc_auc:.4f}')
    plt.plot([0,1], [0,1], linestyle = '--', label = "Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    st.write("### ROC Curve and AUC (Area Under Curve)")
    plt.legend(loc = "lower right")
    st.pyplot(plt)
    st.write("Note: A model with an AUC of 0.80 or above generally has good predictive power.")

def display_visuals(y_test, y_pred, X_test):
    col1, col2 = st.columns(2)
    with col1:
        plot_confusion(y_test, y_pred)
    with col2:
        show_classification(y_test, y_pred)
    y_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, threshold = roc_curve(y_test, y_probs)
    roc_auc = roc_auc_score(y_test, y_probs)
    col3, col4, col5 = st.columns([0.25,.5,.25])
    with col4:
        plot_roc(fpr, tpr, roc_auc)

def scale_data(X_train, X_test, user_data):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    user_data = scaler.transform(user_data)
    return X_train, X_test, user_data


# Loading in and re-formatting the sample dataset.
nba_data = pd.read_csv("data/NBA_Regular_Season.csv", sep = ";", encoding = 'latin-1')
new_data = (nba_data[['Rk', 'Player', 'Pos', 'PTS', 'AST' ,'TRB']].groupby('Rk', as_index=False).agg({
    'Player': 'first',  
    'Pos': 'first',
    'PTS': 'mean',
    'AST': 'mean',
    'TRB': 'mean'
}))
new_data = new_data[new_data['Pos'].isin(['PG', 'SG', 'SF', 'PF', 'C'])].reset_index(drop = True)

# Merging the NBA dataset with another handmade dataset, which specifies the players who were all-stars during the 2023-24 season.
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

# Main text of the app
st.title("Exploring Machine Learning Classification Models")
path = st.radio("Choose a path!", ["Upload my own dataset!", "Become an NBA All-Star!"])

# If the person decides to play around with the NBA dataset...
if path == "Become an NBA All-Star!":
    st.subheader("Predicting NBA All-Star Status with Machine Learning 🏀")
    st.write("Excellent choice! In this portion of the app, you get to pretend that you're a basketball player in the NBA during the 2023-24 season!")
    st.write("You'll first input your position, as well as your season statistics for Points per Game, Assists per Game, and Rebounds per Game. Then, you'll choose what kind of classification model you'd like to use to predict your All-Star status -- either logistic regression, decision tree, or k-nearest neighbors. Finally, you'll tune the hyperparameters of the corresponding model and hit 'Run'!")
    st.write("The app will spit out your probability of being an All-Star, as well as display some of the model metrics to give you a sense of how accurate the prediction is. On your mark, get set, go!")

    position = st.selectbox("Select your position:", ['PG', 'SG', 'SF', 'PF', 'C'])  
    points = st.slider("Enter your average points per game:", min_value = 0.0, max_value = final_dataset['PTS'].max(), value = 10.0, step = 0.1)
    assists = st.slider("Enter your average assists per game:", min_value = 0.0, max_value = final_dataset['AST'].max(), value = 3.0, step = 0.1)
    rebounds = st.slider("Enter your average rebounds per game:", min_value = 0.0, max_value = final_dataset['TRB'].max(), value = 4.0, step = 0.1)
    model_choice = st.radio("Choose a classification model:", ['Logistic Regression', 'Decision Tree', 'K-Nearest Neighbors'])

    features = ['Pos', 'PTS', 'AST', 'TRB']
    target = 'All-Star'
    user_data = [[positions[position], points, assists, rebounds]]

    if model_choice == "Logistic Regression":
        scale_question = st.radio("Would you like to scale the data? (Scaling adjusts for unit bias.)", ['Yes', 'No'])
        if st.button("Run!"):
            X,y = data_prep(final_dataset, features, target)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                                random_state = 99)
            if scale_question == "Yes":
                X_train, X_test, user_data = scale_data(X_train, X_test, user_data)
            model = LogisticRegression()
            model.fit(X_train, y_train)
            model_prob(model, user_data)
            y_pred = model.predict(X_test)
            model_metrics(y_test, y_pred)
            display_visuals(y_test, y_pred, X_test)
    
    if model_choice == 'Decision Tree':
        hyper_choice = st.radio("Would you like to choose your own model hyperparameters, or have the model optimize them for you?",
                                ["I'll tune them myself.", "Tune them for me!"])
        if hyper_choice == "I'll tune them myself.":
            criterion = st.radio("Select a criterion algorithm for optimizing each split (gini is simpler but slightly faster):", ['gini', 'entropy'])
            max_depth = st.slider("Select a maximum tree depth (higher = more precise, but risk overfitting):", min_value = 1, max_value = 10, step = 1)
            min_samples_split = st.slider("Select the minimum number of samples required to split an internal node (lower = more precise, but risk overfitting):",
                                          min_value = 2, max_value = 10, step = 1)
            min_samples_leaf = st.slider("Select the minimum number of samples required to be in a leaf node (lower = more precise, but risk overfitting):",
                                         min_value = 1, max_value = 10, step = 1)
        else:
            param_grid = {
                        'criterion': ['gini', 'entropy'],
                        'max_depth': [None,2,4,6,8],
                        'min_samples_split': list(range(2,11,2)),
                        'min_samples_leaf' : list(range(1,11,2))
                        }           
            st.write("You're in good hands. Hit 'Run!' whenever you're ready! This may take a few seconds.")
        if st.button('Run!'):
            X,y = data_prep(final_dataset, features, target)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                                random_state = 99)
            if hyper_choice == "I'll tune them myself.":
                model = DecisionTreeClassifier(random_state = 99, criterion = criterion, max_depth = max_depth,
                                               min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf)
                model.fit(X_train, y_train)
            else:
                dtree = DecisionTreeClassifier(random_state = 99)
                grid_search = GridSearchCV(estimator = dtree, param_grid = param_grid, cv = 3, scoring = 'accuracy')
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
            model_prob(model, user_data)
            y_pred = model.predict(X_test)
            model_metrics(y_test, y_pred)
            display_visuals(y_test, y_pred, X_test)

    if model_choice == "K-Nearest Neighbors":
        scale_question = st.radio("Would you like to scale the data? (Scaling adjusts for unit bias.)", ['Yes', 'No'])
        hyper_choice = st.radio("Would you like to choose your own model hyperparameters, or have the model optimize them for you?",
                                ["I'll tune them myself.", "Tune them for me!"])
        if hyper_choice == "I'll tune them myself.":
            n_neighbors = st.slider("Please choose the number of neighbors to use (fewer neighbors = more precise, but risk overfitting):", 
                                    min_value = 1, max_value = 19, step = 2)
            metric = st.radio("Please choose the metric to use for distance computation (use 'euclidean' for continuous data, 'manhattan' for discrete data, and 'minkowski' for flexibility):",
                              ["minkowski", "euclidean", "manhattan"])
        else:
            param_grid = {
                        'n_neighbors': list(range(1, 20, 2)),
                        'metric': ['minkowski', 'euclidean', 'manhattan'],
            } 
            st.write("You're in good hands. Hit 'Run!' whenever you're ready! This may take a few seconds.")
        if st.button('Run!'):
            X,y = data_prep(final_dataset, features, target)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                                random_state = 99)
            if scale_question == "Yes":
                X_train, X_test, user_data = scale_data(X_train, X_test, user_data)
            if hyper_choice == "I'll tune them myself.":
                model = KNeighborsClassifier(n_neighbors = n_neighbors, metric = metric)
                model.fit(X_train, y_train)
            else:
                knn = KNeighborsClassifier()
                grid_search = GridSearchCV(estimator = knn, param_grid = param_grid, cv = 5, scoring = 'accuracy')
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
            model_prob(model, user_data)
            y_pred = model.predict(X_test)
            model_metrics(y_test, y_pred)
            display_visuals(y_test, y_pred, X_test)


if path == "Upload my own dataset!":
    st.subheader("Making Predictions with User-Provided Data 📈📊")
    st.markdown("Love the curiosity! You'll be making predictions and gathering insights in no time! Here's your job:")
    st.markdown("""
                1. Upload your own tidy dataset containing one or more predictor variables (either numerical, categorical, or binary) and one **binary** target variable.
                2. Select different values of your predictor variables, as well as which classification model you'd like to use for prediction -- either logistic regression, decision tree, or k-nearest neighbors.
                3. Tune the hyperparameters of your model (or let the computer tune it for you), and hit 'Run!'.
                4. Observe the probability given by the model, and evaluate the model's predictive power using the given metrics and visualizations.
                """)
    st.write("On your mark, get set, go!")

    uploaded_file = st.file_uploader("Upload a .csv file containing the tidy dataset of interest with at least one binary predictor (and no dates!):", type = "csv")
    if uploaded_file:
        input_data = pd.read_csv(uploaded_file)
        st.write("Here are the first few rows of your dataset (missing values will be dropped):")
        st.dataframe(input_data.head())
        input_data = input_data.dropna()
        pos_binary = []
        for var in input_data.columns:
            if input_data[var].nunique() == 2:
                pos_binary.append(var)
        if len(pos_binary) == 1:
            target = pos_binary[0]
            st.write(f"The only binary variable in your dataset is {target}. This will be used as your target variable.")
        elif len(pos_binary) > 1:
            target = st.selectbox("Please select which binary variable you would like to use as your target:", pos_binary)
        else:
            st.write("This dataset contains no binary variables. Please select a new dataset.")
            st.stop()
        input_data_numeric = input_data.copy()
        if set(input_data[target].unique()) != {0,1}:
            target_map = {input_data[target].unique()[0]:0,
                   input_data[target].unique()[1]:1}
            input_data_numeric[target] = input_data[target].map(target_map)
        vars = list(input_data.columns)
        vars.remove(target)
        features = st.multiselect("Select variables from the dataset to build your model:", vars)
        if features:
            cats = []
            for var in features:
                if input_data[var].dtype == "bool":
                    map = {"False":0,
                    "True":1}
                    input_data_numeric[var] = input_data[var].map(map)
                elif input_data[var].dtype == "object":
                    cats.append(var)
            cat_dict = {}
            for cat in cats:
                map = {}
                for i in range(0, len(input_data[cat].unique())):
                    map[input_data[cat].unique()[i]] = i
                input_data_numeric[cat] = input_data[cat].map(map)
                cat_dict[cat] = map
            user_data = []
            for var in features:
                if input_data[var].dtype == "bool" or input_data[var].dtype == "object":
                    val = st.selectbox(f"Please choose a value for {var}:", list(input_data[var].unique()))
                    user_data.append(cat_dict[var][val])
                elif (input_data[var] % 1 == 0).all():
                    if len(input_data[var].unique()) <= 10:
                        sorted_list = sorted([int(val) for val in input_data[var].unique()])
                        options_list = list(range(min(sorted_list), max(sorted_list) + 1, 1))
                        val = st.selectbox(f"Please choose a value for {var}:", options_list)
                        user_data.append(val)
                    else:
                        val = st.slider(f"Please choose a value for {var}:", min_value = int(input_data[var].min()),
                                        max_value = int(input_data[var].max()), step = 1)
                        user_data.append(val)
                elif input_data[var].dtype == "float64":
                    val = st.number_input(f"Please enter a value for {var}:")
                    user_data.append(val)
            user_data = [user_data]
            model_choice = st.radio("Choose a classification model:", ['Logistic Regression', 'Decision Tree', 'K-Nearest Neighbors'])

            if model_choice == "Logistic Regression":
                scale_question = st.radio("Would you like to scale the data? (Scaling adjusts for unit bias.)", ['Yes', 'No'])
    
                if st.button("Run!"):
                    X,y = data_prep(input_data_numeric, features, target)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                                random_state = 99)
                    if scale_question == "Yes":
                        X_train, X_test, user_data = scale_data(X_train, X_test, user_data)
                    model = LogisticRegression()
                    model.fit(X_train, y_train)
                    model_prob2(model, user_data)
                    y_pred = model.predict(X_test)
                    model_metrics(y_test, y_pred)
                    if set(input_data[target].unique()) != {0,1}:
                        st.write(f"(Here, 0 represents {input_data[target].unique()[0]} and 1 represents {input_data[target].unique()[1]}.)")
                    display_visuals(y_test, y_pred, X_test)

            if model_choice == 'Decision Tree':
                hyper_choice = st.radio("Would you like to choose your own model hyperparameters, or have the model optimize them for you?",
                                ["I'll tune them myself.", "Tune them for me!"])
                if hyper_choice == "I'll tune them myself.":
                    criterion = st.radio("Select a criterion algorithm for optimizing each split (gini is simpler but slightly faster):", ['gini', 'entropy'])
                    max_depth = st.slider("Select a maximum tree depth (higher = more precise, but risk overfitting):", min_value = 1, max_value = 10, step = 1)
                    min_samples_split = st.slider("Select the minimum number of samples required to split an internal node (lower = more precise, but risk overfitting):",
                                          min_value = 2, max_value = 10, step = 1)
                    min_samples_leaf = st.slider("Select the minimum number of samples required to be in a leaf node (lower = more precise, but risk overfitting):",
                                         min_value = 1, max_value = 10, step = 1)
                else:
                    param_grid = {
                        'criterion': ['gini', 'entropy'],
                        'max_depth': [None,2,4,6,8],
                        'min_samples_split': list(range(2,11,2)),
                        'min_samples_leaf' : list(range(1,11,2))
                        }
                    st.write("You're in good hands. Hit 'Run!' whenever you're ready! This may take a few seconds.")
                if st.button('Run!'):
                    X,y = data_prep(input_data_numeric, features, target)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                                random_state = 99)
                    if hyper_choice == "I'll tune them myself.":
                        model = DecisionTreeClassifier(random_state = 99, criterion = criterion, max_depth = max_depth,
                                                       min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf)
                        model.fit(X_train, y_train)
                    else:
                        dtree = DecisionTreeClassifier(random_state = 99)
                        grid_search = GridSearchCV(estimator = dtree, param_grid = param_grid, cv = 3, scoring = 'accuracy')
                        grid_search.fit(X_train, y_train)
                        model = grid_search.best_estimator_
                    model_prob2(model, user_data)
                    y_pred = model.predict(X_test)
                    model_metrics(y_test, y_pred)
                    if set(input_data[target].unique()) != {0,1}:
                        st.write(f"(Here, 0 represents {input_data[target].unique()[0]} and 1 represents {input_data[target].unique()[1]}.)")
                    display_visuals(y_test, y_pred, X_test)

            if model_choice == "K-Nearest Neighbors":
                scale_question = st.radio("Would you like to scale the data? (Scaling adjusts for unit bias.)", ['Yes', 'No'])
                hyper_choice = st.radio("Would you like to choose your own model hyperparameters, or have the model optimize them for you?",
                                ["I'll tune them myself.", "Tune them for me!"])
                if hyper_choice == "I'll tune them myself.":
                    n_neighbors = st.slider("Please choose the number of neighbors to use (fewer neighbors = more precise, but risk overfitting):", 
                                    min_value = 1, max_value = 19, step = 2)
                    metric = st.radio("Please choose the metric to use for distance computation (use 'euclidean' for continuous data, 'manhattan' for discrete data, and 'minkowski' for flexibility):",
                              ["minkowski", "euclidean", "manhattan"])
                else:
                    param_grid = {
                        'n_neighbors': list(range(1, 20, 2)),
                        'metric': ['minkowski', 'euclidean', 'manhattan'],
                    } 
                    st.write("You're in good hands. Hit 'Run!' whenever you're ready! This may take a few seconds.")
                if st.button('Run!'):
                    X,y = data_prep(input_data_numeric, features, target)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                                random_state = 99)
                    if scale_question == "Yes":
                        X_train, X_test, user_data = scale_data(X_train, X_test, user_data)
                    if hyper_choice == "I'll tune them myself.":
                        model = KNeighborsClassifier(n_neighbors, metric = metric)
                        model.fit(X_train, y_train)
                    else:
                        knn = KNeighborsClassifier()
                        grid_search = GridSearchCV(estimator = knn, param_grid = param_grid, cv = 5, scoring = 'accuracy')
                        grid_search.fit(X_train, y_train)
                        model = grid_search.best_estimator_
                    model_prob2(model, user_data)
                    y_pred = model.predict(X_test)
                    model_metrics(y_test, y_pred)
                    if set(input_data[target].unique()) != {0,1}:
                        st.write(f"(Here, 0 represents {input_data[target].unique()[0]} and 1 represents {input_data[target].unique()[1]}.)")
                    display_visuals(y_test, y_pred, X_test)




        
            

    