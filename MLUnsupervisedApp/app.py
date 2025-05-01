# -----------------------------------------------
# Loading in dependencies
# -----------------------------------------------
import streamlit as st
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

# All of these required for executing machine learning algorithms.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pathlib import Path # Needed for relative path when deploying app online
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram

# -----------------------------------------------
# Loading and Re-formatting the MLB Pitchers Dataset
# -----------------------------------------------
# Set the proper path for deploying the app online.
DATA_PATH = Path(__file__).parent / "data" / "MLB_Pitchers.csv"
# Read in the data from the 'data' folder.
MLB_data = pd.read_csv(DATA_PATH, encoding = 'latin-1')
# Drop the unnecessary columns, restrict our scope only to pitchers with more 
# than 20 innings pitched, and drop any missing values.
MLB_data = (MLB_data.drop(columns = ["Lg", "Awards", "Player-additional"]).query('IP > 20').dropna())
duplicates = MLB_data[MLB_data.duplicated('Rk', keep = False)]
team_dict = dict()
for index, row in duplicates.iterrows():
    if row['Rk'] not in list(team_dict.keys()):
        team_dict[row['Rk']] = [row['Team']]
    else:
        team_dict[row['Rk']].append(row['Team'])
for elem in team_dict:
    del team_dict[elem][0]
team_df = pd.DataFrame(list(team_dict.items()), columns = ['Rk', 'Team'])
MLB_data = MLB_data.groupby('Rk').agg("first").reset_index()
MLB_data['Team'] = MLB_data['Team'].apply(lambda x: [x])
MLB_data.set_index('Rk', inplace = True)
team_df.set_index('Rk', inplace = True)
MLB_data.update(team_df)
MLB_data = MLB_data.reset_index().drop(columns = 'Rk')
Identifiers = MLB_data[['Player', 'Team']]
MLB_reduced = MLB_data.drop(columns = ['Player', 'Team'])

# We don't need to numerically encode any categorical variables for this dataset.
# -----------------------------------------------
# App Title and Path Options
# -----------------------------------------------
st.title("Exploring the Uses of Unsupervised Machine Learning")
st.write("Welcome! In this app, you'll explore the two main functions of unsupervised machine learning: clustering and dimensionality reduction!")
path = st.radio("Select a path to get started!", ["Upload my own dataset!", "Become an MLB analyst!"])

# -----------------------------------------------
# MLB Analyst Path Information
# -----------------------------------------------
if path == "Become an MLB analyst!":
    st.subheader("Sabermetrics with MLB Pitcher Data")
    # Instructions
    st.write("In this portion of the app, you'll get to pretend that you're an analyst in the front office of your favorite Major League Baseball team, and your team desperately needs a pitcher. However, all you've been given is a dataset with the statistics for each pitcher in the MLB during the 2024 season.")
    st.write("With no target variable to predict in the dataset, you'll use unsupervised machine learning to group together similar players into clusters and then visualize those clusters in the xy-plane. So, if there's a desirable pitcher who's out of your pay range, you can look within that pitcher's cluster to find some more affordable talent for your team.")
    st.write("Here's your job:")
    st.markdown("""
                1. Take a look at the dataset you've been given, as well as the definitions of all of the pitching statistics, and choose a group of variables to use for clustering similar pitchers together.
                2. Pick the model you'd like to use to construct the clusters -- either KMeans or Hierarchical Clustering. 
                3. Tune the hyperparameters of the corresponding model and hit 'Run!'.
                4. Observe the outputted model feedback and re-tune the hyperparameters, if necessary.
                5. Pick the dimensionality reduction model you'd like to use to visualize your clusters -- either Principal Components Analysis (PCA) or t-SNE. 
                6. Pick which variables you'd like to display when you hover over a datapoint, as well as if you'd like a particular team to be highlighted (to help you locate a certain player). Then, hit "Visualize!".
                7. Explore your pitcher options, and track down the next Cy Young winner!
                """)
    st.write("Let's go!")
    st.markdown("""
        #### Clustering the Data
    """)
    st.write("To start, check out the first few rows of data below, as well as the provided glossary:")
    st.dataframe(MLB_data.head(5))
    with st.expander("**Glossary**"):
        st.markdown(
            """
            **Explanations of variables as provided by Baseball Reference**:
            - **Age** - As of June 30 of the 2024 season.
            - **WAR (Wins Above Replacement)** - A single number that presents the number of wins the player added to the team above what a replacement player would add. This value includes defensive support and includes additional value for high leverage situations. Scale: 8+ MVP Quality, 5+ All-Star Quality, 2+ Starter, 0-2 Reserve, < 0 Replacement Level. Developed by Sean Smith of BaseballProjection.com
            - **W** - Wins
            - **L** - Losses
            - **W-L% (Win-Loss Percentage)** - Wins / (Wins + Losses)
            - **ERA** - 9 * Earned Runs Allowed / Innings Pitched.
            - **G** - Games Pitched
            - **GS** - Games Started
            - **GF** - Games Finished
            - **CG** - Complete Games
            - **SHO** - Shutouts (No runs allowed and a complete game)
            - **SV** - Saves
            - **IP** - Innings Pitched
            - **H** - Hits Allowed
            - **R** - Runs Allowed
            - **ER** - Earned Runs Allowed
            - **HR** - Home Runs Allowed
            - **BB** - Bases on Balls (or Walks)
            - **IBB** - Intentional Bases on Balls
            - **SO** - Strikeouts
            - **HBP** - Times Hit by a Pitch
            - **BK** - Balks
            - **WP** - Wild Pitches
            - **BF** - Batters Faced
            - **ERA+** - 100*[League Earned Run Average/Earned Run Average] (Adjusted to the player’s ballpark(s))
            - **FIP (Fielding Independent Pitching)** - Measures a pitcher's effectiveness at preventing HR, BB, HBP and causing SO. Calculated as (13*HR + 3*(BB+HBP) - 2*SO)/IP + Constantlg. The constant is set so that each season major-league average FIP is the same as the major-league average ERA.
            - **WHIP** - (Walks + Hits)/Innings Pitched
            - **H9** - 9 x Hits / Innings Pitched
            - **HR9** - 9 x Home Runs / Innings Pitched
            - **BB9** - 9 x Walks / Innings Pitched
            - **SO9** - 9 x Strikeouts / Innings Pitched
            - **SO/BB** - Strikeouts/Walks
        """)
    
    # Obtaining and storing selected variables.
    features = st.multiselect("Select the variables you'd like to use for clustering (at least 2, preferably more):", MLB_reduced.columns)  
    # Asking user which model they'd like to use for clustering.
    cluster_model = st.radio("Choose a clustering model:", ['KMeans Clustering', 'Hierarchical Clustering'])
    
    # KMeans path
    if cluster_model == "KMeans Clustering":
        # Gathering and storing the user's desired number of clusters.
        n_clusters = st.slider("Select your desired number of clusters, k:", min_value = 2, max_value = 20, step = 1)
        # Where model execution starts.
        if st.button('Run!'):
            # Creating the dataset we will use to build the model
            X = MLB_data[features]
            # Scaling the data
            scaler = StandardScaler()
            X_std = scaler.fit_transform(X)
            # Building the model
            kmeans = KMeans(n_clusters = n_clusters, random_state = 99)
            clusters = kmeans.fit_predict(X_std)
            # Calculating the WCSS and Silhouette Scores for each possible k
            ks = range(2,21)
            wcss = []
            silhouette_scores = []
            for k in ks:
                km = KMeans(k, random_state = 99)
                km.fit(X_std)
                wcss.append(km.inertia_) # Grabs the WCSS
                labels = km.labels_
                silhouette_scores.append(silhouette_score(X_std, labels))
            # Plotting the elbow plot and silhouette score plot
            col1, col2 = st.columns(2)
            with col1:
                plt.figure()
                plt.plot(ks, wcss, marker='o')
                plt.xlabel('Number of clusters (k)')
                plt.xticks(range(1,21))
                plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
                plt.title('Elbow Method for Optimal k')
                plt.grid(True)
                st.pyplot(plt)
                st.write("Note: The elbow plot displays the within-cluster variance for different numbers of clusters. Typically, the optimal k value is found at the 'elbow' of the plot, or where the slope changes sharply.")
            with col2:
                plt.figure()
                plt.plot(ks, silhouette_scores, marker='o', color='green')
                plt.xlabel('Number of clusters (k)')
                plt.xticks(range(1,21))
                plt.ylabel('Silhouette Score')
                plt.title('Silhouette Score for Optimal k')
                plt.grid(True)
                st.pyplot(plt)
                st.write("Note: The silhouette score measures the average similarity of data points within a cluster. Thus, the highest silhouette score is most desirable.")
            st.write("**After viewing these plots, you are encouraged to adjust the number of clusters in your model to the optimal number, then re-run the model before continuing.**")
            st.markdown("""
                    #### Visualizing and Analyzing the Clusters
                    """)
            dim_red = st.radio("Pick a dimensionality reduction model to visualize your clusters in 2D-space:", 
                                ["PCA", "t-SNE"])
            
            
    
    # Hierarchical Clustering path
    if cluster_model == "Hierarchical Clustering":
        # Gathering and storing the user's desired hyperparameter choices.
        n_clusters = st.slider("Select your desired number of clusters, k:", min_value = 2, max_value = 20, step = 1)
        link = st.radio("Please choose the rule to use for linking new clusters (default = ward):",
                              ["ward", "complete", "average", "single"])
        # Where model execution starts.
        if st.button('Run!'):
            # Creating the dataset we will use to build the model
            X = MLB_data[features]
            # Scaling the data
            scaler = StandardScaler()
            X_std = scaler.fit_transform(X)
            # Building the model
            hierarch = AgglomerativeClustering(n_clusters = n_clusters, linkage = link)
            clusters = hierarch.fit_predict(X_std)
            # Constructing the dendrogram
            Z = linkage(X_std, method = link)
            fig, ax = plt.subplots(figsize = (10,5))
            dgram = dendrogram(Z, ax = ax, truncate_mode = "lastp")
            plt.ylabel("Distance")
            plt.title("Dendrogram of Hierarchical Clustering for MLB Pitchers")
            st.pyplot(fig)
            st.write("Note: The dendrogram displays the bottom-up clustering process carried out by the hierarchical algorithm. Inspecting the plot helps you determine where to cut the tree and decide on your optimal number of clusters.")
            # Calculating the Silhouette Scores for each possible k
            ks = range(2,21)
            silhouette_scores = []
            for k in ks:
            # Fit hierarchical clustering
                labels = AgglomerativeClustering(n_clusters=k, linkage= link).fit_predict(X_std)
                score = silhouette_score(X_std, labels)
                silhouette_scores.append(score)
            col3, col4, col5 = st.columns([0.25,.5,.25])
            with col4:
                plt.figure()
                plt.plot(ks, silhouette_scores, marker='o', color='green')
                plt.xlabel('Number of clusters (k)')
                plt.xticks(range(1,21))
                plt.ylabel('Silhouette Score')
                plt.title('Silhouette Score for Optimal k')
                plt.grid(True)
                st.pyplot(plt)
                st.write("Note: The silhouette score measures the average similarity of data points within a cluster. Thus, the highest silhouette score is most desirable.")
            st.write("**After viewing these plots, you are encouraged to adjust the number of clusters in your model to the optimal number, then re-run the model before continuing.**")

    

