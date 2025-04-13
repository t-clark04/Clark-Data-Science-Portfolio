# -----------------------------------------------
# Loading in dependencies
# -----------------------------------------------
import streamlit as st
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

# All of these required for executing machine learning algorithms.
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from pathlib import Path # Needed for relative path when deploying app online

# -----------------------------------------------
# Helper Functions
# -----------------------------------------------
def data_prep(dataset, features, target): # Gives the X and y subsets of the data.
    X = dataset[features]
    y = dataset[target]
    return X,y

def model_prob(model, user_data):
    # Calculate the probability of being an All-Star given the inputted season statistics.
    prob = model.predict_proba(user_data)
    # Return that probability.
    st.write(f"Your probability of being an all-star is **{prob[0][1]:.2%}**!")

def model_prob2(model, user_data):
    # Calculate the general probability of a '1' given the user's data.
    prob = model.predict_proba(user_data)
    # Return that probability.
    st.write(f"The probability that '{target}' is '{input_data[target].unique()[1]}' is **{prob[0][1]:.2%}**!")

def model_metrics(y_test, y_pred):
    # Calculate accuracy, precision, and recall.
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    # Define these three metrics and return them for this model.
    st.markdown(f"""
                Main model metrics:
                - Accuracy is the percentage of correct predictions made by the model. This model has an accuracy of **{accuracy:.2%}**.
                - Precision answers the question: 'Out of all datapoints classified as positive, what percentage were actually positive?' This model's precision is **{precision:.2%}**.
                - Recall answers the question: 'Out of all datapoints that were actually positive, what percentage did we catch?' This model has a recall of **{recall:.2%}**.
                """)
    st.write("For a more complete picture of the model's predictive power, check out the confusion matrix, classification report, and ROC Curve/AUC below.")

def plot_confusion(y_test, y_pred):
    # Calculate the confusion matrix for this model.
    cm = confusion_matrix(y_test, y_pred)
    # Build a heatmap with the confusion matrix data.
    sns.heatmap(cm, annot = True, cmap = "Blues", fmt = 'g') #'fmt' argument ensures that full values are displayed.
    # Title and labels
    st.write("### Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(plt)
    plt.clf() # Ensures we start fresh before plotting again.

def show_classification(y_test, y_pred):
    st.write("### Classification Report")
    # Build and display the classification report.
    st.text(classification_report(y_test, y_pred))
    st.write("Note: F1-score gives the harmonic mean of precision and recall.")

def plot_roc(fpr, tpr, roc_auc):
    # Set the canvas.
    plt.figure(figsize = (4,4))
    # Plot the ROC curve and random guess lines on the same plot.
    plt.plot(fpr, tpr, label = f'ROC Curve, AUC = {roc_auc:.4f}')
    plt.plot([0,1], [0,1], linestyle = '--', label = "Random Guess")
    # Title, labels, and legend.
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    st.write("### ROC Curve and AUC (Area Under Curve)")
    plt.legend(loc = "lower right")
    st.pyplot(plt)
    st.write("Note: Generally, a model with an AUC of 0.80 or above is considered to have good predictive power.")

def display_visuals(y_test, y_pred, X_test): # Calls on the above three helper functions to format the visualizations.
    # Plot the confusion matrix and classification report in two different columns.
    col1, col2 = st.columns(2)
    with col1:
        plot_confusion(y_test, y_pred)
    with col2:
        show_classification(y_test, y_pred)
    # Calculate all of the required inputs to the plot_roc helper function.
    y_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, threshold = roc_curve(y_test, y_probs)
    roc_auc = roc_auc_score(y_test, y_probs)
    # Using columns to center the ROC curve.
    col3, col4, col5 = st.columns([0.25,.5,.25])
    with col4:
        plot_roc(fpr, tpr, roc_auc)

def scale_data(X_train, X_test, user_data):
    # Set the scalar based on the X_train data, then
    # use it to transform X_train, X_test, and the inputted data.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    user_data = scaler.transform(user_data)
    return X_train, X_test, user_data

def data_info():
    # Create an expander for more information on the original NBA dataset.
    with st.expander("Want to learn more about the data?"):
        st.markdown("### 2023-24 NBA Statistics Dataset")
        # Provide a description and link to the full dataset.
        st.markdown("""
                 The data for this portion of the app was adapted from a dataset on Kaggle linked [here](https://www.kaggle.com/datasets/vivovinco/2023-2024-nba-player-stats?resource=download&select=2023-2024+NBA+Player+Stats+-+Regular.csv).
                 The original dataset contains 30 columns of data on the NBA players who took part in the 2023-24 season, though only six were used in this app.
                 To explore variables like player name, team, games played, field goals, and more, clink the link to the full dataset above, or check out the first few rows of the dataset below:
                """)
        # Display the first ten rows of the full dataset.
        st.dataframe(nba_data.head(10))

# -----------------------------------------------
# Loading and Re-formatting the NBA Dataset
# -----------------------------------------------
# Set the proper path for deploying the app online.
DATA_PATH = Path(__file__).parent / "data" / "NBA_Regular_Season.csv"
# Read in the data from the 'data' folder.
nba_data = pd.read_csv("data/NBA_Regular_Season.csv", sep = ";", encoding = 'latin-1')
# Take a subset of the variables and combine rows with the same player.
# Keep the first instance of their name and position, and average over their points, assists, and rebounds.
new_data = (nba_data[['Rk', 'Player', 'Pos', 'PTS', 'AST' ,'TRB']].groupby('Rk', as_index=False).agg({
    'Player': 'first',  
    'Pos': 'first',
    'PTS': 'mean',
    'AST': 'mean',
    'TRB': 'mean'
}))
# Exclude rows with multiple positions listed.
new_data = new_data[new_data['Pos'].isin(['PG', 'SG', 'SF', 'PF', 'C'])].reset_index(drop = True)

# Merge the NBA dataset with another handmade dataset, which specifies the players who were all-stars during the 2023-24 season.
all_star_dict = {"Player": ["Tyrese Haliburton", "Damian Lillard", "Giannis Antetokounmpo", "Jayson Tatum", "Joel Embiid",
                 "Jalen Brunson", "Tyrese Maxey", "Donovan Mitchell", "Trae Young", "Paolo Banchero", "Scottie Barnes", "Jaylen Brown",
                 "Julius Randle", "Bam Adebayo", "Luka Don?i?", "Shai Gilgeous-Alexander", "Kevin Durant", "LeBron James", "Nikola Joki?",
                 "Devin Booker", "Stephen Curry", "Anthony Edwards", "Paul George", "Kawhi Leonard", "Karl-Anthony Towns", "Anthony Davis"],
                 "All-Star": [int(1)]*26}
all_star_data = pd.DataFrame(all_star_dict)
final_dataset = (pd.merge(new_data, all_star_data, how = "outer", on = "Player").fillna(0))

# Convert the 'Rk' and 'All-Star' variables to integers
final_dataset['Rk'] = final_dataset['Rk'].astype(int)
final_dataset['All-Star'] = final_dataset['All-Star'].astype(int)

# Convert the categorical position values to integers as well.
positions = {
    'PG':1,
    'SG':2,
    'SF':3,
    'PF':4,
    'C':5
}
final_dataset['Pos'] = final_dataset['Pos'].map(positions)

# -----------------------------------------------
# App Title and Path Options
# -----------------------------------------------
st.title("Exploring Machine Learning Classification Models")
path = st.radio("Choose a path!", ["Upload my own dataset!", "Become an NBA All-Star!"])

# -----------------------------------------------
# NBA All-Star Path Information
# -----------------------------------------------
if path == "Become an NBA All-Star!":
    st.subheader("Predicting NBA All-Star Status with Machine Learning 🏀")
    # Instructions
    st.write("Excellent choice! In this portion of the app, you get to pretend that you're a basketball player in the NBA during the 2023-24 season! Here's how it works:")
    st.markdown("""
                1. Input your position, as well as your season statistics for Points per Game, Assists per Game, and Rebounds per Game.
                2. Choose what kind of classification model you'd like to use to predict your All-Star status -- either logistic regression, decision tree, or k-nearest neighbors.
                3. Tune the hyperparameters of the corresponding model and hit 'Run!'.
                4. Observe your probability of being an All-Star, and evaluate the model's predictive power using the given metrics and visualizations.
                """)
    st.write("On your mark, get set, go!")
    
    # Obtaining and storing the user-inputted NBA data.
    position = st.selectbox("Select your position:", ['PG', 'SG', 'SF', 'PF', 'C'])  
    points = st.slider("Enter your average points per game:", min_value = 0.0, max_value = final_dataset['PTS'].max(), value = 10.0, step = 0.1)
    assists = st.slider("Enter your average assists per game:", min_value = 0.0, max_value = final_dataset['AST'].max(), value = 3.0, step = 0.1)
    rebounds = st.slider("Enter your average rebounds per game:", min_value = 0.0, max_value = final_dataset['TRB'].max(), value = 4.0, step = 0.1)
    model_choice = st.radio("Choose a classification model:", ['Logistic Regression', 'Decision Tree', 'K-Nearest Neighbors'])

    # Defining our feature and target variables, and storing the inputs as 'user_data'.
    features = ['Pos', 'PTS', 'AST', 'TRB']
    target = 'All-Star'
    user_data = [[positions[position], points, assists, rebounds]]

    # Logistic Regression path
    if model_choice == "Logistic Regression":
        # Only "hyperparameter" for logistic regression in this model is scaled/unscaled data.
        scale_question = st.radio("Would you like to scale the data? (Scaling adjusts for unit bias.)", ['Yes', 'No'])
        # Where model execution starts.
        if st.button("Run!"):
            X,y = data_prep(final_dataset, features, target) # Subset the data into X,y.
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 99) # Split into training and testing sets.
            if scale_question == "Yes": # Re-scale the data if desired by user.
                X_train, X_test, user_data = scale_data(X_train, X_test, user_data)
            model = LogisticRegression()
            model.fit(X_train, y_train) # Fit the logistic regression model.
            model_prob(model, user_data) # Use it to make and display predictions on the user's data.
            y_pred = model.predict(X_test)
            model_metrics(y_test, y_pred) # Calculate and display the model's evaluation metrics.
            st.write(f"(Here, a '1' represents an All-Star.)")
            display_visuals(y_test, y_pred, X_test) # Display confusion matrix, classification report, and ROC curve.
            data_info() # Provide an expander with more information on the data.

    # Decision Tree path
    if model_choice == 'Decision Tree':
        # Offers user the option to tune the hyperparameters or let the algorithm do it.
        hyper_choice = st.radio("Would you like to choose your own model hyperparameters, or have the model optimize them for you?",
                                ["I'll tune them myself.", "Tune them for me!"])
        if hyper_choice == "I'll tune them myself.":
            # Gathering and storing the user's hyperparameter choices, since they have chosen to tune them.
            criterion = st.radio("Select a criterion algorithm for optimizing each split (gini is simpler but slightly faster):", ['gini', 'entropy'])
            max_depth = st.slider("Select a maximum tree depth (higher = more precise, but risk overfitting):", min_value = 1, max_value = 10, step = 1)
            min_samples_split = st.slider("Select the minimum number of samples required to split an internal node (lower = more precise, but risk overfitting):",
                                          min_value = 2, max_value = 10, step = 1)
            min_samples_leaf = st.slider("Select the minimum number of samples required to be in a leaf node (lower = more precise, but risk overfitting):",
                                         min_value = 1, max_value = 10, step = 1)
        else: # User wants the computer to optimize the hyperparameter choices.
            # Defining the parameter grid for the GridSearchCV algorithm to cycle through.
            param_grid = {
                        'criterion': ['gini', 'entropy'],
                        'max_depth': [None,2,4,6,8],
                        'min_samples_split': list(range(2,11,2)),
                        'min_samples_leaf' : list(range(1,11,2))
                        }           
            st.write("You're in good hands. Hit 'Run!' whenever you're ready! This may take a few seconds.")
        # Where model execution starts.
        if st.button('Run!'):
            X,y = data_prep(final_dataset, features, target) # Subset the data into X,y.
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 99) # Split into training and testing sets.
            if hyper_choice == "I'll tune them myself.": # If the user supplied hyperparameters...
                # Create and fit the decision tree model with the user's hyperparameter choices.
                model = DecisionTreeClassifier(random_state = 99, criterion = criterion, max_depth = max_depth,
                                               min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf)
                model.fit(X_train, y_train)
            else: # Otherwise, carry out a grid search and select the model with the best hyperparameter choices.
                dtree = DecisionTreeClassifier(random_state = 99)
                grid_search = GridSearchCV(estimator = dtree, param_grid = param_grid, cv = 3, scoring = 'accuracy')
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
            model_prob(model, user_data) # Use the model to make and display predictions on the user's data.
            y_pred = model.predict(X_test)
            model_metrics(y_test, y_pred) # Calculate and display the model metrics.
            st.write(f"(Here, a '1' represents an All-Star.)")
            display_visuals(y_test, y_pred, X_test) # Display the confusion matrix, classification report, and ROC curve.
            data_info() # Provide more information on the NBA dataset.
    
    # K-Nearest Neighbors path
    if model_choice == "K-Nearest Neighbors":
        # Allow the user to scale the data if they desire.
        scale_question = st.radio("Would you like to scale the data? (Scaling adjusts for unit bias.)", ['Yes', 'No'])
        # Offer user the option to tune the hyperparameters or let the algorithm do it.
        hyper_choice = st.radio("Would you like to choose your own model hyperparameters, or have the model optimize them for you?",
                                ["I'll tune them myself.", "Tune them for me!"])
        if hyper_choice == "I'll tune them myself.":
            # Gathering and storing the user's hyperparameter choices, since they have chosen to tune them.
            n_neighbors = st.slider("Please choose the number of neighbors to use (fewer neighbors = more precise, but risk overfitting):", 
                                    min_value = 1, max_value = 19, step = 2)
            metric = st.radio("Please choose the metric to use for distance computation (use 'euclidean' for continuous data, 'manhattan' for discrete data, and 'minkowski' for flexibility):",
                              ["minkowski", "euclidean", "manhattan"])
        else: # User wants the computer to optimize the hyperparameter choices.
            # Defining the parameter grid for the GridSearchCV algorithm to cycle through.
            param_grid = {
                        'n_neighbors': list(range(1, 20, 2)),
                        'metric': ['minkowski', 'euclidean', 'manhattan'],
            } 
            st.write("You're in good hands. Hit 'Run!' whenever you're ready! This may take a few seconds.")
        # Where model execution starts.
        if st.button('Run!'):
            X,y = data_prep(final_dataset, features, target) # Subset the data into X,y.
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 99) # Split into training and testing sets.
            if scale_question == "Yes": # Re-scale the data if the user desires.
                X_train, X_test, user_data = scale_data(X_train, X_test, user_data)
            if hyper_choice == "I'll tune them myself.": # If the user supplied hyperparameters...
                # Create and fit the K-nearest neighbors model with the user's hyperparameter choices.
                model = KNeighborsClassifier(n_neighbors = n_neighbors, metric = metric)
                model.fit(X_train, y_train)
            else: # Otherwise, carry out a grid search and select the model with the best hyperparameter choices.
                knn = KNeighborsClassifier()
                grid_search = GridSearchCV(estimator = knn, param_grid = param_grid, cv = 5, scoring = 'accuracy')
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
            model_prob(model, user_data) # Use the model to make and display predictions on the user's data.
            y_pred = model.predict(X_test)
            model_metrics(y_test, y_pred) # Calculate and display the model metrics.
            st.write(f"(Here, a '1' represents an All-Star.)")
            display_visuals(y_test, y_pred, X_test) # Display the confusion matrix, classification report, and ROC curve.
            data_info() # Provide more information on the NBA dataset.

# -----------------------------------------------
# User-Provided Data Path Information
# -----------------------------------------------
if path == "Upload my own dataset!":
    st.subheader("Making Predictions with User-Provided Data 📈📊")
    # Instructions
    st.markdown("Love the curiosity! You'll be making predictions and gathering insights in no time! Here's your job:")
    st.markdown("""
                1. Upload your own tidy dataset containing one or more predictor variables (either numerical, categorical, or binary) and one **binary** target variable.
                2. Select different values of your predictor variables, as well as which classification model you'd like to use for prediction -- either logistic regression, decision tree, or k-nearest neighbors.
                3. Tune the hyperparameters of your model (or let the computer tune it for you), and hit 'Run!'.
                4. Observe the probability given by the model, and evaluate the model's predictive power using the given metrics and visualizations.
                """)
    st.write("On your mark, get set, go!")

    # Prompt the user to upload a file.
    uploaded_file = st.file_uploader("Upload a .csv file containing the tidy dataset of interest with at least one binary predictor (and no dates!):", type = "csv")
    if uploaded_file: # If they do upload a csv file...
        input_data = pd.read_csv(uploaded_file) # Read it in.
        st.write("Here are the first few rows of your dataset (missing values will be dropped):")
        st.dataframe(input_data.head()) # Display the first 5 rows.
        input_data = input_data.dropna() # Drop all missing values (two of the ML models can't deal with them).
        # Create a list with all possible binary variables in the inputted dataset.
        pos_binary = []
        for var in input_data.columns:
            if input_data[var].nunique() == 2:
                pos_binary.append(var)
        if len(pos_binary) == 1: # If there's only one, that's the target variable.
            target = pos_binary[0]
            st.write(f"The only binary variable in your dataset is {target}. This will be used as your target variable.")
        elif len(pos_binary) > 1: # If there are multiple, prompt the user to select one as target.
            target = st.selectbox("Please select which binary variable you would like to use as your target:", pos_binary)
        else: # If there are no binary variables in the dataset, ask the user to upload a different one.
            st.write("This dataset contains no binary variables. Please select a new dataset.")
            st.stop()
        input_data_numeric = input_data.copy() # Create a copy of the dataset before converting all categorical variables to numeric.
        # If the values of the target variables aren't already 0 and 1, changes them to 0s and 1s,
        # then store the map so we can recover what the 0s and 1s mean.
        if set(input_data[target].unique()) != {0,1}:
            target_map = {input_data[target].unique()[0]:0,
                   input_data[target].unique()[1]:1}
            input_data_numeric[target] = input_data[target].map(target_map)
        # Have the user select their feature variables from those given in the inputted dataset.
        vars = list(input_data.columns)
        vars.remove(target)
        features = st.multiselect("Select variables from the dataset to build your model:", vars)
        if features: # Once they select at least one variable...
            # Convert all of the "bool" columns to 0s and 1s, then store all possible categorical
            # variables in a list called cats.
            cats = []
            for var in features:
                if input_data[var].dtype == "bool":
                    map = {"False":0,
                    "True":1}
                    input_data_numeric[var] = input_data[var].map(map)
                elif input_data[var].dtype == "object":
                    cats.append(var)
            # Go through each of the variables in the cats list and convert them to numeric values in the 
            # data frame. Store the value map for each variable in a large dictionary called cat_dict, so
            # that we can recover what each numeric value represents later on.
            cat_dict = {}
            for cat in cats:
                map = {}
                for i in range(0, len(input_data[cat].unique())):
                    map[input_data[cat].unique()[i]] = i
                input_data_numeric[cat] = input_data[cat].map(map)
                cat_dict[cat] = map
            # Begin the process of gathering user data for their inputted dataset.
            user_data = []
            for var in features: # Go through each of the variables selected as features.
                # If it's a boolean or a categorical variable, display the original values of the variable
                # as options in a selectbox, but store the numeric counterpart of the value in our user_data
                # list, according to the map defined above.
                if input_data[var].dtype == "bool" or input_data[var].dtype == "object":
                    val = st.selectbox(f"Please choose a value for {var}:", list(input_data[var].unique()))
                    user_data.append(cat_dict[var][val])
                # If it's an integer variable with 10 or fewer values, use a selectbox widget to gather
                # the user input and store it accordingly. If it's an integer variable with more than 10
                # unique values, use a slider instead.
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
                # If it's a decimal variable, use the general number_input widget to gather the user's data
                # and store it in the user_data list.
                elif input_data[var].dtype == "float64":
                    val = st.number_input(f"Please enter a value for {var}:")
                    user_data.append(val)
            user_data = [user_data] # Reformatting the user_data list according to the requirements of later functions.
            model_choice = st.radio("Choose a classification model:", ['Logistic Regression', 'Decision Tree', 'K-Nearest Neighbors'])

            # Logistic Regression path
            if model_choice == "Logistic Regression":
                # Only "hyperparameter" for logistic regression in this model is scaled/unscaled data.
                scale_question = st.radio("Would you like to scale the data? (Scaling adjusts for unit bias.)", ['Yes', 'No'])
                # Where model execution starts.
                if st.button("Run!"):
                    X,y = data_prep(input_data_numeric, features, target) # Subset data into X,y.
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 99) # Split into training and testing sets.
                    if scale_question == "Yes": # Re-scale the data if the user desires.
                        X_train, X_test, user_data = scale_data(X_train, X_test, user_data)
                    model = LogisticRegression()
                    model.fit(X_train, y_train) # Define and fit the logistic regression model.
                    model_prob2(model, user_data) # Use the model to make and display predictions on the user's data.
                    y_pred = model.predict(X_test)
                    model_metrics(y_test, y_pred) # Calcualte and display the model metrics.
                    if set(input_data[target].unique()) != {0,1}: # Specify what 0 and 1 represent in the confusion matrix and classification report.
                        st.write(f"(Here, 0 represents {input_data[target].unique()[0]} and 1 represents {input_data[target].unique()[1]}.)")
                    display_visuals(y_test, y_pred, X_test) # Display the confusion matrix, classification report, and ROC curve.

            # Decision Tree path
            if model_choice == 'Decision Tree':
                # Offers user the option to tune the hyperparameters or let the algorithm do it.
                hyper_choice = st.radio("Would you like to choose your own model hyperparameters, or have the model optimize them for you?",
                                ["I'll tune them myself.", "Tune them for me!"])
                if hyper_choice == "I'll tune them myself.":
                    # Gathering and storing the user's hyperparameter choices, since they have chosen to tune them.
                    criterion = st.radio("Select a criterion algorithm for optimizing each split (gini is simpler but slightly faster):", ['gini', 'entropy'])
                    max_depth = st.slider("Select a maximum tree depth (higher = more precise, but risk overfitting):", min_value = 1, max_value = 10, step = 1)
                    min_samples_split = st.slider("Select the minimum number of samples required to split an internal node (lower = more precise, but risk overfitting):",
                                          min_value = 2, max_value = 10, step = 1)
                    min_samples_leaf = st.slider("Select the minimum number of samples required to be in a leaf node (lower = more precise, but risk overfitting):",
                                         min_value = 1, max_value = 10, step = 1)
                else: # User wants the computer to optimize the hyperparameter choices.
                    # Defining the parameter grid for the GridSearchCV algorithm to cycle through.
                    param_grid = {
                        'criterion': ['gini', 'entropy'],
                        'max_depth': [None,2,4,6,8],
                        'min_samples_split': list(range(2,11,2)),
                        'min_samples_leaf' : list(range(1,11,2))
                        }
                    st.write("You're in good hands. Hit 'Run!' whenever you're ready! This may take a few seconds.")
                # Where model execution starts.
                if st.button('Run!'):
                    X,y = data_prep(input_data_numeric, features, target) # Subset the data into X,y.
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 99) # Split into training and testing sets.
                    if hyper_choice == "I'll tune them myself.": # If the user supplied hyperparameters...
                        # Create and fit the decision tree with the user's hyperparameter choices.
                        model = DecisionTreeClassifier(random_state = 99, criterion = criterion, max_depth = max_depth,
                                                       min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf)
                        model.fit(X_train, y_train)
                    else: # Otherwise, carry out a grid search and select the model with the best hyperparameter choices.
                        dtree = DecisionTreeClassifier(random_state = 99)
                        grid_search = GridSearchCV(estimator = dtree, param_grid = param_grid, cv = 3, scoring = 'accuracy')
                        grid_search.fit(X_train, y_train)
                        model = grid_search.best_estimator_
                    model_prob2(model, user_data) # Use the model to make and display predictions on the user's data.
                    y_pred = model.predict(X_test)
                    model_metrics(y_test, y_pred) # Calculate and display the model metrics.
                    if set(input_data[target].unique()) != {0,1}: # Specify what 0 and 1 represent in the confusion matrix and classification report.
                        st.write(f"(Here, 0 represents {input_data[target].unique()[0]} and 1 represents {input_data[target].unique()[1]}.)")
                    display_visuals(y_test, y_pred, X_test) # Display confusion matrix, classification report, and ROC curve.
            
            # K-Nearest Neighbors Path
            if model_choice == "K-Nearest Neighbors":
                # Allow the user to scale the data if they desire.
                scale_question = st.radio("Would you like to scale the data? (Scaling adjusts for unit bias.)", ['Yes', 'No'])
                # Offer user the option to tune the hyperparameters or let the algorithm do it.
                hyper_choice = st.radio("Would you like to choose your own model hyperparameters, or have the model optimize them for you?",
                                ["I'll tune them myself.", "Tune them for me!"])
                if hyper_choice == "I'll tune them myself.":
                    # Gathering and storing the user's hyperparameter choices, since they have chosen to tune them.
                    n_neighbors = st.slider("Please choose the number of neighbors to use (fewer neighbors = more precise, but risk overfitting):", 
                                    min_value = 1, max_value = 19, step = 2)
                    metric = st.radio("Please choose the metric to use for distance computation (use 'euclidean' for continuous data, 'manhattan' for discrete data, and 'minkowski' for flexibility):",
                              ["minkowski", "euclidean", "manhattan"])
                else: # User wants the computer to optimize the hyperparameter choices.
                    # Defining the parameter grid for the GridSearchCV algorithm to cycle through.
                    param_grid = {
                        'n_neighbors': list(range(1, 20, 2)),
                        'metric': ['minkowski', 'euclidean', 'manhattan'],
                    } 
                    st.write("You're in good hands. Hit 'Run!' whenever you're ready! This may take a few seconds.")
                # Where execution starts.
                if st.button('Run!'):
                    X,y = data_prep(input_data_numeric, features, target) # Subset the data into X,y.
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 99) # Split into training and testing sets.
                    if scale_question == "Yes": # Re-scale the data if the user desires.
                        X_train, X_test, user_data = scale_data(X_train, X_test, user_data)
                    if hyper_choice == "I'll tune them myself.": # If the user supplied hyperparameters...
                        # Create and fit the K-nearest neighbors model with the user's hyperparameter choices.
                        model = KNeighborsClassifier(n_neighbors, metric = metric)
                        model.fit(X_train, y_train)
                    else: # Otherwise, carry out a grid search and select the model with the best hyperparameter choices.
                        knn = KNeighborsClassifier()
                        grid_search = GridSearchCV(estimator = knn, param_grid = param_grid, cv = 5, scoring = 'accuracy')
                        grid_search.fit(X_train, y_train)
                        model = grid_search.best_estimator_
                    model_prob2(model, user_data) # Use the model to make and display predictions on the user's data.
                    y_pred = model.predict(X_test)
                    model_metrics(y_test, y_pred) # Calculate and display the model metrics.
                    if set(input_data[target].unique()) != {0,1}: # Specify what 0 and 1 represent in the confusion matrix and classification report.
                        st.write(f"(Here, 0 represents {input_data[target].unique()[0]} and 1 represents {input_data[target].unique()[1]}.)")
                    display_visuals(y_test, y_pred, X_test) # Display confusion matrix, classification report, and ROC curve.




        
            

    