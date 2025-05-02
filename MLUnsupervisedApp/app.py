# -----------------------------------------------
# Loading in dependencies
# -----------------------------------------------
# Needed to suppress a memory leak warning from KMeans.
import os
os.environ["OMP_NUM_THREADS"] = "3"

import streamlit as st # To run the app.
import pandas as pd # To work with dataframes.
import seaborn as sns # To build dendrograms.
import matplotlib.pyplot as plt # To make scatterplots and line charts.

# All of these are required for executing the machine learning algorithms.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from pathlib import Path # Needed for relative path when deploying app online.

# -----------------------------------------------
# Helper Functions
# -----------------------------------------------
def scale_X(X): # Re-scale the data and return it.
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    return X_std

def calculate_metrics(X_std):
    ks = range(2,21) # Take k from 2 to 20.
    wcss = []
    silhouette_scores = []
    # For each k, fit a KMeans model (I only use this for KMeans),
    # and append the WCSS and Silhouette scores to the running lists,
    # then return them both.
    for k in ks: # For each k,
        km = KMeans(k, random_state = 99)
        km.fit(X_std)
        wcss.append(km.inertia_)
        labels = km.labels_
        silhouette_scores.append(silhouette_score(X_std, labels))
    return(wcss, silhouette_scores)

def plot_elbow_silhouette(ks, wcss, silhouette_scores):
    # Split the canvas up into two columns.
    col1, col2 = st.columns(2)
    # In the first column, plot each k value against its corresponding
    # WCSS, return the scatterplot, and describe to the user what
    # the graph is displaying.
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
    # Do the same thing in the second column, except now with
    # the silhouette scores. Write another note to the user about
    # how to read the graph.
    with col2:
        plt.figure()
        plt.plot(ks, silhouette_scores, marker='o', color='green')
        plt.xlabel('Number of clusters (k)')
        plt.xticks(range(1,21))
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score for Optimal k')
        plt.grid(True)
        st.pyplot(plt)
        st.write("Note: The silhouette score measures the average similarity of data points within a cluster. Thus, a higher silhouette score generally indicates a better model.")

def plot_dendrogram(X_std, link):
    # Start off by performing the hierarchical clustering, according
    # to the linkage rule specified by the user.
    Z = linkage(X_std, method = link)
    # Create the plotting canvas.
    fig, ax = plt.subplots(figsize = (10,5))
    # Build the dendrogram and label it accordingly.
    dendrogram(Z, ax = ax, truncate_mode = "lastp")
    plt.ylabel("Distance")
    plt.title("Dendrogram of Hierarchical Clustering for MLB Pitchers")
    # Display it for the user.
    st.pyplot(fig)
    # Return a description for the dendrogram.
    st.write("Note: The dendrogram displays the bottom-up clustering process carried out by the hierarchical algorithm. Inspecting the tree and cutting it at a specific height helps you determine your optimal number of clusters.")

def hover_labs(dim_red, feature_labs):
    # Create a dictionary that tells plotly to display the 
    # user's desired hover labels and not to display the 
    # PCA or t-SNE coordinates.
    hover_dict = dict()
    for var in feature_labs:
        hover_dict[var] = True
    hover_dict[f"{dim_red}1"] = False
    hover_dict[f"{dim_red}2"] = False
    return hover_dict

def build_MLB_traces(final_df, dim_red):
    # Create a separate dataframe with only the team that 
    # the user wants to highlight in the graph and store the
    # opacity variable as 1 for each observation. All of this needs to
    # be done because plotly doesn't allow for variable opacity (you can
    # only have one opacity value per graph), so we work around it.
    df_highlight = final_df[final_df['opacity'] == 1.0]
    # Put the rest of the datapoints in a separate dataframe,
    # with an opacity value of 0.4 (more transparent).
    df_faded = final_df[final_df['opacity'] == 0.4]
    # Create a dictionary of fake datapoints so that we can use
    # its legend and ensure that all possible clusters are included
    # in the legend (even if the user's desired team has no players
    # in one of the clusters).
    dict_fake = {f"{dim_red}1": [float('nan')]*n_clusters, f"{dim_red}2": [float('nan')]*n_clusters, 'Cluster': final_df['Cluster'].unique()}
    df_fake = pd.DataFrame(dict_fake)
    # Create a plotly scatterplot for the datapoints that are meant to 
    # be highlighted. Assign each cluster to a different color, pass
    # in our hover_dictionary that we built previously, and set the
    # opacity of the dots to 1. Then, make the dots slightly bigger
    # as well.
    fig_highlight = px.scatter(df_highlight, x=f"{dim_red}1", y=f"{dim_red}2", color="Cluster",
                            hover_data=hover_dict, 
                            color_discrete_sequence=px.colors.qualitative.Set1,
                            opacity=1.0,
                            category_orders={'Cluster': sorted(final_df['Cluster'].unique())})
    fig_highlight.update_traces(marker=dict(size=6.5))
    # Do the same thing again, but with all of the datapoints not included in the highlighted
    # data (i.e. not on the user's desired team). Set the opacity of these dots to 0.4 instead
    # of 1 (more transparent).
    fig_faded = px.scatter(df_faded, x=f"{dim_red}1", y=f"{dim_red}2", color="Cluster",
                        hover_data=hover_dict,
                        color_discrete_sequence=px.colors.qualitative.Set1,
                        opacity=0.4,
                        category_orders={'Cluster': sorted(final_df['Cluster'].unique())})
    fig_faded.update_traces(marker=dict(size=6.5))
    # Finally, plot the "fake" data, and make sure the cluster labels line up as well.
    fig_fake = px.scatter(df_fake, x=f"{dim_red}1", y=f"{dim_red}2", color="Cluster",
                            color_discrete_sequence=px.colors.qualitative.Set1,
                            opacity=1,
                            category_orders={'Cluster': sorted(final_df['Cluster'].unique())})
    # Suppress the legend for the higlighted and for the more transparent data, and only use
    # the legend for the "fake" data.
    for trace in fig_faded.data:
        trace.showlegend = False
    for trace in fig_highlight.data:
        trace.showlegend = False
    # Combine these three graphs into one, and return it.
    fig = go.Figure(data=fig_faded.data + fig_highlight.data + fig_fake.data)
    return fig

def update_fig_format(fig, dim_red):
    # Adjust the graph aesthetics to our liking (much more
    # complicated because we are using plotly for a more interactive
    # interface).
    fig.update_layout(
        margin = dict(t = 40), # Lower the title.
        hoverlabel = dict(font_color = "black", font_size = 12), # Format the hover label text.
        # Add a title.
        title=dict(
            text=f"Pitcher Clusters in 2-D Plane via {dim_red}",
            x=0.5,
            xanchor='center',
            font=dict(color="black", size=17)),
        # Make all text size 15 and black.
        font=dict(
            size=15,
            color="black"),
        # Adjust the size of the graph.
        width=900,
        height=500,
        # Label the x-axis, add a grid, and adjust the font to our liking.
        xaxis=dict(
            title="Component 1",
            showgrid=True,
            gridcolor='lightgray',
            title_font=dict(color="black"),
            tickfont=dict(color="black")),
        # Do the same for the y-axis.
        yaxis=dict(
            title="Component 2",
            showgrid=True,
            gridcolor='lightgray',
            title_font=dict(color="black"),
            tickfont=dict(color="black")),
        # Adjust the font for the legend
        legend_title=dict(
            text="Cluster",
            font=dict(size=15, color="black")),
        # Make the background white.
        plot_bgcolor="white",
        paper_bgcolor="white",
        # Make sure the cluster labels are displayed in
        # ascending order.
        legend=dict(
            traceorder='normal',
            tracegroupgap=2,
            font=dict(color="black", size = 15))
        )
    # Add a note to the bottom of the graph to explain why
    # some pitchers have an asterisk after their names while
    # others don't.
    fig.add_annotation(
        text="* indicates a left-handed pitcher.",
        xref="paper", yref="paper",
        x=0, y=-0.2,  # Positioning below the plot
        showarrow=False,
        font=dict(size=14),
        )
    
def data_info():
    # Create an expander for more information on the original MLB dataset.
    with st.expander("Want to learn more about the data?"):
        # Provide a description and link to the full dataset.
        st.markdown("""
                - The data and glossary used in this portion of the app both come from Baseball-Reference.com.
                - To check out the full dataset, head over to their website linked [here](https://www.baseball-reference.com/leagues/majors/2024-standard-pitching.shtml).
                - Thank you to Baseball Reference for making this project possible!
                """)

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
# Locate the pitchers who played for multiple teams, then iterate through them.
# Create a dictionary for each pitcher, where the corresponding value is a list
# of teams that they played for in 2024. Then delete the initial entry, since it
# is not a team at all. It is just a placeholder like '2TM' that indicates they
# played for two teams.
duplicates = MLB_data[MLB_data.duplicated('Rk', keep = False)]
team_dict = dict()
for index, row in duplicates.iterrows():
    if row['Rk'] not in list(team_dict.keys()):
        team_dict[row['Rk']] = [row['Team']]
    else:
        team_dict[row['Rk']].append(row['Team'])
for elem in team_dict:
    del team_dict[elem][0]
# Turn that dictionary into a dataframe containing the ranks of pitchers
# who played for multiple teams and a column of lists containing the teams.
team_df = pd.DataFrame(list(team_dict.items()), columns = ['Rk', 'Team'])
# Collapse down each of the rows containing pitchers who played for different
# teams and keep the first row's data, since already it contains sums and averages 
# where appropriate, but get rid of the specified pitchers since they played in 
# multiple leagues during the 2024 season, so they are just too complicated to
# work with. There aren't many of them anyway, so it's not that big of a loss. 
MLB_data = MLB_data.groupby('Rk').agg("first").reset_index()
MLB_data = MLB_data[~MLB_data['Rk'].isin([162,288,375,400,417,431,467,476,514,529])]
# Turn each entry for Team in the data frame into a list, and for the pitchers with
# multiple teams, replace their current entry for Team with the list we just created
# in the team_df.
MLB_data['Team'] = MLB_data['Team'].apply(lambda x: [x])
MLB_data.set_index('Rk', inplace = True)
team_df.set_index('Rk', inplace = True)
MLB_data.update(team_df)
# Reset the indices in the dataframe (drop rank), store the Player and Team variables as a 
# separate dataset called Identifiers to be used later, then make a reduced data frame
# without those variables. The reduced dataset will be used to offer the user their variable
# choices for clustering the pitchers.
MLB_data = MLB_data.reset_index().drop(columns = 'Rk')
Identifiers = MLB_data[['Player', 'Team']]
MLB_reduced = MLB_data.drop(columns = ['Player', 'Team'])

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
    st.subheader("Sabermetrics with MLB Pitcher Data ⚾📈")
    # Instructions
    st.write("In this portion of the app, you'll get to pretend that you're an analyst in the front office of your favorite Major League Baseball team, and your team desperately needs a pitcher. However, all you've been given is a dataset with the statistics for each MLB pitcher during the 2024 season.")
    st.write("With no target variable to predict in the dataset, you'll use unsupervised machine learning to group together similar players into clusters and then visualize those clusters in the xy-plane. That way, if you know there's a desirable pitcher who's out of your pay range, you can look within that pitcher's cluster to find some more affordable talent for your team.")
    st.write("Here's your job:")
    st.markdown("""
                1. Take a look at the dataset you've been given, as well as the glossary of pitching metrics, and choose a group of variables to cluster similar pitchers together.
                2. Pick the model you'd like to use to construct the clusters -- either KMeans or Hierarchical Clustering -- and tune the model hyperparameters to start (you can re-adjust them later).
                3. Choose the dimensionality reduction model you'd like to use to visualize your clusters -- either Principal Components Analysis (PCA) or t-SNE. 
                4. Pick which variables you'd like to display when you hover over a datapoint in your plot, as well is if you'd like a particular team to be highlighted (to help you locate a certain player). Then, hit "Go!".
                5. Observe the outputted model feedback and re-tune the hyperparameters, if necessary.
                6. Explore your pitching options, and track down the next Cy Young winner!
                """)
    st.write("Let's get to work!")
    st.divider() # Breaks up the app a little bit.
    st.write("To start, check out the first few rows of data below, as well as the provided glossary:")
    st.dataframe(MLB_data.head(5)) # First five rows of the big data frame.
    # Glossary of pitching metrics from Baseball Reference to help the user 
    # pick which variables they want to use.
    with st.expander("**Glossary**"):
        st.markdown(
            """
            **Explanations of variables as provided by Baseball-Reference.com**:
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
    cluster_model = st.radio("Choose a clustering model (KMeans = top-down, Hierarchical = bottom-up):", ['KMeans Clustering', 'Hierarchical Clustering'])
    
    # KMeans path
    if cluster_model == "KMeans Clustering":
        # Gathering and storing the user's desired number of clusters.
        n_clusters = st.slider("Select your desired number of clusters, k, to start out (can be adjusted later):", min_value = 2, max_value = 20, step = 1)
        # Storing their choice for the dimensionarlity reduction model.
        dim_red = st.radio("Pick a dimensionality reduction model to visualize your clusters in 2D-space (t-SNE = better, but slower):", 
                            ["PCA", "t-SNE"])
        # Asking them to which variables they want to be displayed upon hovering.
        feature_labs = st.multiselect("Select which variables you would like displayed when you hover:", options = ["Player", "Team"] + features, default = ["Player", "Team"])
        # Asking the user which team to highlight in the plot.
        high_team = st.selectbox("Select a team you would like highlighted in the plot, if any:", 
                                    options = ["None"] + sorted(list(set(val for sublist in Identifiers['Team'] for val in sublist))))
        # Where execution starts.
        if st.button("Go!"):
            if dim_red == "PCA":
                # Creating the dataset we will use to build the model, 
                # and scaling it.
                X = MLB_data[features]
                X_std = scale_X(X)
                # Building the KMeans model and storing the clusters as a dataframe.
                kmeans = KMeans(n_clusters = n_clusters, random_state = 99)
                clusters = pd.DataFrame(kmeans.fit_predict(X_std), columns = ["Cluster"])
                # Calculating the WCSS and Silhouette Scores for each possible k.
                ks = range(2,21)
                wcss, silhouette_scores = calculate_metrics(X_std)
                # Plotting the elbow plot and silhouette score plot.
                plot_elbow_silhouette(ks, wcss, silhouette_scores)
                st.write("**After viewing these plots, you are encouraged to adjust the number of clusters in your model to the optimal number, then re-run the model.**")
                # Doing the dimensionality reduction and storing principle components as a
                # data frame.
                pca = PCA(n_components=2)
                X_pca = pd.DataFrame(pca.fit_transform(X_std), columns = ["PCA1", "PCA2"])
                # Concatenating all of the necessary dataframes into one for plotting.
                final_df = pd.concat([X_pca, clusters, X, Identifiers], axis = 1)
                # Converting the cluster column to a categorical variable so that the 
                # color scale on the graph is discrete rather than continuous.
                final_df['Cluster'] = final_df['Cluster'].astype('category')
                # Create an opacity column in the final_dataset to dictate which team
                # should be highlighted and which teams should be more transparent.
                if high_team == "None":
                    final_df['opacity'] = 1
                else:            
                    final_df['opacity'] = final_df['Team'].apply(lambda x: 1 if high_team in x else 0.4)
                # Creating our dictionary to tell plotly which variables to include in the hover labels.
                hover_dict = hover_labs(dim_red, feature_labs)
                # Building our graph.
                fig = build_MLB_traces(final_df, dim_red)
                # Updating it for the aesthetics.
                update_fig_format(fig, dim_red)
                # Displaying it to the user.
                st.plotly_chart(fig, use_container_width=True)
                # Providing information on the data source.
                data_info()
            if dim_red == "t-SNE":
                # Creating the dataset we will use to build the model, 
                # and scaling it.
                X = MLB_data[features]
                X_std = scale_X(X)
                # Building the KMeans model and storing the clusters as a dataframe.
                kmeans = KMeans(n_clusters = n_clusters, random_state = 99)
                clusters = pd.DataFrame(kmeans.fit_predict(X_std), columns = ["Cluster"])
                # Calculating the WCSS and Silhouette Scores for each possible k.
                ks = range(2,21)
                wcss, silhouette_scores = calculate_metrics(X_std)
                # Plotting the elbow plot and silhouette score plot.
                plot_elbow_silhouette(ks, wcss, silhouette_scores)
                st.write("**After viewing these plots, you are encouraged to adjust the number of clusters in your model to the optimal number, then re-run the model.**")
                # Doing the dimensionality reduction and storing t-SNE components as a
                # data frame.
                tsne = TSNE(n_components=2)
                X_tsne = pd.DataFrame(tsne.fit_transform(X_std), columns = ["t-SNE1", "t-SNE2"])
                # Concatenating all of the necessary dataframes into one for plotting.
                final_df = pd.concat([X_tsne, clusters, X, Identifiers], axis = 1)
                # Converting the cluster column to a categorical variable so that the 
                # color scale on the graph is discrete rather than continuous.
                final_df['Cluster'] = final_df['Cluster'].astype('category')
                # Create an opacity column in the final_dataset to dictate which team
                # should be highlighted and which teams should be more transparent.
                if high_team == "None":
                    final_df['opacity'] = 1
                else:            
                    final_df['opacity'] = final_df['Team'].apply(lambda x: 1 if high_team in x else 0.4)
                # Creating our dictionary to tell plotly which variables to include in the hover labels.
                hover_dict = hover_labs(dim_red, feature_labs)
                # Building our graph.
                fig = build_MLB_traces(final_df, dim_red)
                # Updating the aesthetics.
                update_fig_format(fig, dim_red)
                # Displaying it to the user.
                st.plotly_chart(fig, use_container_width=True)
                # Providing information on the data source.
                data_info()

    
    # Hierarchical Clustering path
    if cluster_model == "Hierarchical Clustering":
        # Gathering and storing the user's desired hyperparameter choices.
        n_clusters = st.slider("Select your desired number of clusters, k, to start out (can be adjusted later):", min_value = 2, max_value = 20, step = 1)
        # Asking the user to decide on a linkage rule.
        link = st.radio("Please choose your desired rule for linking new clusters (default = ward, see README file for full descriptions):",
                              ["ward", "complete", "average", "single"])
        # Storing their choice for the dimensionarlity reduction model.
        dim_red = st.radio("Pick a dimensionality reduction model to visualize your clusters in 2D-space (t-SNE = better, but slower):", ["PCA", "t-SNE"])
        # Asking them to which variables they want to be displayed upon hovering.
        feature_labs = st.multiselect("Select which variables you would like displayed when you hover:", options = ["Player", "Team"] + features, default = ["Player", "Team"])
        # Asking the user which team to highlight in the plot.
        high_team = st.selectbox("Select a team you would like highlighted in the plot, if any:", 
                                    options = ["None"] + sorted(list(set(val for sublist in Identifiers['Team'] for val in sublist))))
        # Where model execution starts.
        if st.button('Go!'):
            if dim_red == "PCA":
                # Creating the dataset we will use to build the model, 
                # and scaling it.
                X = MLB_data[features]
                X_std = scale_X(X)
                # Building the hierarchical clustering model and storing the 
                # generated clusters as a data frame.
                hierarch = AgglomerativeClustering(n_clusters = n_clusters, linkage = link)
                clusters = pd.DataFrame(hierarch.fit_predict(X_std), columns = ["Cluster"])
                # Constructing and displaying the dendrogram.
                plot_dendrogram(X_std, link)
                st.write("**After viewing the dendrogram, you are encouraged to adjust the number of clusters in your model to the optimal number, then re-run the model before continuing.**")
                # Doing the dimensionality reduction and storing principle components as a
                # data frame.
                pca = PCA(n_components=2)
                X_pca = pd.DataFrame(pca.fit_transform(X_std), columns = ["PCA1", "PCA2"])
                # Concatenating all of the necessary dataframes into one for plotting.
                final_df = pd.concat([X_pca, clusters, X, Identifiers], axis = 1)
                # Converting the cluster column to a categorical variable so that the 
                # color scale on the graph is discrete rather than continuous.
                final_df['Cluster'] = final_df['Cluster'].astype('category')
                # Create an opacity column in the final_dataset to dictate which team
                # should be highlighted and which teams should be more transparent.
                if high_team == "None":
                    final_df['opacity'] = 1
                else:            
                    final_df['opacity'] = final_df['Team'].apply(lambda x: 1 if high_team in x else 0.4)
                # Creating our dictionary to tell plotly which variables to include in the hover labels.
                hover_dict = hover_labs(dim_red, feature_labs)
                # Building our graph.
                fig = build_MLB_traces(final_df, dim_red)
                # Updating the aesthetics.
                update_fig_format(fig, dim_red)
                # Displaying it to the user.
                st.plotly_chart(fig, use_container_width=True)
                # Providing information on the data source.
                data_info()
            if dim_red == "t-SNE":
                # Creating the dataset we will use to build the model, 
                # and scaling it.
                X = MLB_data[features]
                X_std = scale_X(X)
                # Building the hierarchical clustering model and storing the 
                # generated clusters as a data frame.
                hierarch = AgglomerativeClustering(n_clusters = n_clusters, linkage = link)
                clusters = pd.DataFrame(hierarch.fit_predict(X_std), columns = ["Cluster"])
                # Constructing and displaying the dendrogram.
                plot_dendrogram(X_std, link)
                st.write("**After viewing the dendrogram, you are encouraged to adjust the number of clusters in your model to the optimal number, then re-run the model before continuing.**")
                # Doing the dimensionality reduction and storing t-SNE components as a
                # data frame
                tsne = TSNE(n_components=2)
                X_tsne = pd.DataFrame(tsne.fit_transform(X_std), columns = ["t-SNE1", "t-SNE2"])
                # Concatenating all of the necessary dataframes into one for plotting.
                final_df = pd.concat([X_tsne, clusters, X, Identifiers], axis = 1)
                # Converting the cluster column to a categorical variable so that the 
                # color scale on the graph is discrete rather than continuous.
                final_df['Cluster'] = final_df['Cluster'].astype('category')
                # Create an opacity column in the final_dataset to dictate which team
                # should be highlighted and which teams should be more transparent.
                if high_team == "None":
                    final_df['opacity'] = 1
                else:            
                    final_df['opacity'] = final_df['Team'].apply(lambda x: 1 if high_team in x else 0.4)
                # Creating our dictionary to tell plotly which variables to include in the hover labels.
                hover_dict = hover_labs(dim_red, feature_labs)
                # Building our graph.
                fig = build_MLB_traces(final_df, dim_red)
                # Updating the aesthetics.
                update_fig_format(fig, dim_red)
                # Displaying it to the user.
                st.plotly_chart(fig, use_container_width=True)
                # Providing information on the data source.
                data_info()
    
# -----------------------------------------------
# User-Provided Data Path Information
# -----------------------------------------------
if path == "Upload my own dataset!":
    st.subheader("Building Clusters with User-Provided Data 🧠🧩")
    # Instructions
    st.markdown("Awesome choice! You'll be generating your own insights on unlabeled data in no time! Here's what you'll need to do:")
    st.markdown("""
                1. Upload your own tidy dataset containing two or more variables (either numeric, categorical, or binary).
                2. Take a look at the data you've uploaded, and choose a collection of variables to use for clustering similar observations together.
                3. Pick the model you'd like to use to construct the clusters -- either KMeans or Hierarchical Clustering -- and tune the model hyperparameters to start (you can re-adjust them later).
                4. Choose the dimensionality reduction model you'd like to use to visualize your clusters -- either Principal Components Analysis (PCA) or t-SNE. 
                5. Pick which variables you'd like to display when you hover over a datapoint in your plot. Then, hit "Go!".
                6. Observe the outputted model feedback and re-tune the hyperparameters, if necessary.
                7. Explore your clustered data!
                """)
    st.write("Let's get exploring!")
    st.divider() # Breaks up the app a little bit.
    # Prompt the user to upload a file.
    uploaded_file = st.file_uploader("Upload a .csv file containing your tidy dataset of interest (no dates!):", type = "csv")
    if uploaded_file: # If they do upload a csv file...
        input_data = pd.read_csv(uploaded_file) # Read it in.
        st.write("Here are the first few rows of your dataset. Missing values will be dropped, and any categorical variables will be converted to numeric:")
        st.dataframe(input_data.head()) # Display the first 5 rows.
        input_data = input_data.dropna() # Drop all missing values (ML models can't deal with them).
        input_data_numeric = input_data.copy() # Create a copy of the dataset before converting all categorical variables to numeric.
        # Have the user select their desired variables from those given in the inputted dataset.
        vars = list(input_data.columns)
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
             # Asking user which model they'd like to use for clustering.
            cluster_model = st.radio("Choose a clustering model (KMeans = top-down, Hierarchical = bottom-up):", ['KMeans Clustering', 'Hierarchical Clustering'])
            
