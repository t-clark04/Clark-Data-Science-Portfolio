# Welcome to my streamlit app! This app allows users to filter and explore
# different variables within the Palmer's Penguins dataset, a set of measurement
# data collected on penguins from the Palmer Archipelago in Antarctica between
# 2007 and 2009.

# To get started with the app, head over to the terminal. Use the ls and cd commands
# to navigate your current directory to the Clark-Data-Science-Portfolio folder on 
# your computer. For example, to enter an existing folder in your current working 
# directory, enter "cd folder_name". To exit the folder, enter "cd ..". Once you're in
# the data science portfolio, type "cd basic-streamlit-app". Then, enter "streamlit run 
# main.py". This will open up the streamlit app in a new browser window. Good luck,  
# and happy exploring!

# Importing the packages I will need to create the app and load in the data.
import streamlit as st
import pandas as pd

# Adding a title, loading in the penguin data, and converting the year column to
# string type to avoid added commas.
st.title("Exploring the Palmer's Penguins Dataframe")
penguins_df = pd.read_csv("data/penguins.csv")
penguins_df["year"] = penguins_df["year"].astype(str)

# Basic description and instructions for navigating the data.
st.write("The Palmer's Penguins dataset contains measurement data collected between 2007 and 2009 on three different species of penguins living on the Palmer Archipelago. The data frame contains 344 observations of penguin data across 9 different variables. The purpose of this streamlit application is to allow you to explore the data a bit better!")
st.write("Start out by choosing a metric you would like to sort by (or choose '(none)' to see the complete dataset), then interact with the resulting widget to narrow down the data. The filtered dataset will be displayed for you!")

# Providing variable descriptions (even though they should be mostly self explanatory).
st.markdown(
"""
For your reference:
- "id" is the penguin identification number, which runs from 0 to 343.
- "species" is the penguin species, with choices Adelie, Gentoo, or Chinstrap.
- "island" is the particular island they came from, either Torgersen, Biscoe, or Dream.
- "bill_length_mm" is the bill length in millimeters, running from 32.1 to 59.6.
- "bill_depth_mm" is likewise the bill depth in millimeters, ranging from 13.1 to 21.5.
- "flipper_length_mm" is the flipper length in millimeters, which runs from 172 to 231.
- "body_mass_g" is the body mass of the penguin in grams, with range 2,700 to 6,300.
- "sex" is the sex of the penguin -- either male, female, or not given.
- "year" is the year in which they were studied (2007, 2008, or 2009).
""")

# Adding a selectbox for the user to choose which variable they would like to filter 
# by, setting '(none)' as the default.
var = st.selectbox("Choose a variable to filter by: ", ["(none)"] + penguins_df.columns.tolist())

# Implementing a series of if/elif statements so that the user is prompted with an
# appropriate new widget, depending on variable what they selected in the initial
# selectbox. The program then displays the penguins dataframe, filtered by whatever 
# conditions they chose. The particular widgets used in this app include selectboxes, 
# sliders, and radio buttons.
if var == "(none)":
    st.dataframe(penguins_df)

elif var == "id":
    id_slider_values = st.slider("Select a range of penguin IDs to explore: ", min_value = 0, max_value = 343, value = (100, 200))
    st.dataframe(penguins_df[(penguins_df["id"] >= id_slider_values[0]) & (penguins_df["id"] <= id_slider_values[1])])

elif var == "species":
    species_select = st.selectbox("Select a penguin species to explore: ", penguins_df["species"].unique())
    st.dataframe(penguins_df[penguins_df["species"] == species_select])

elif var == "island":
    island_select = st.selectbox("Which island would you like to look at?", penguins_df["island"].unique())
    st.dataframe(penguins_df[penguins_df["island"] == island_select])

elif var == "bill_length_mm":
    length_slider_values = st.slider("Select a range of bill lengths to explore: ", min_value = penguins_df["bill_length_mm"].min(), max_value = penguins_df["bill_length_mm"].max(), value = (38.5, 48.5))
    st.dataframe(penguins_df[(penguins_df["bill_length_mm"] >= length_slider_values[0]) & (penguins_df["bill_length_mm"] <= length_slider_values[1])])

elif var == "bill_depth_mm":
    depth_slider_values = st.slider("Select a range of bill depths to explore: ", min_value = penguins_df["bill_depth_mm"].min(), max_value = penguins_df["bill_depth_mm"].max(), value = (17.5, 20.5))
    st.dataframe(penguins_df[(penguins_df["bill_depth_mm"] >= depth_slider_values[0]) & (penguins_df["bill_depth_mm"] <= depth_slider_values[1])])

elif var == "flipper_length_mm":
    flipper_slider_values = st.slider("Select a range of flipper lengths to explore: ", min_value = int(penguins_df["flipper_length_mm"].min()), max_value = int(penguins_df["flipper_length_mm"].max()), value = (190, 205), step = 1)
    st.dataframe(penguins_df[(penguins_df["flipper_length_mm"] >= flipper_slider_values[0]) & (penguins_df["flipper_length_mm"] <= flipper_slider_values[1])])

elif var == "body_mass_g":
    mass_slider_values = st.slider("Select a range of penguin body mass values: ", min_value = int(penguins_df["body_mass_g"].min()), max_value = int(penguins_df["body_mass_g"].max()), value = (4100, 5100), step = 5)
    st.dataframe(penguins_df[(penguins_df["body_mass_g"] >= mass_slider_values[0]) & (penguins_df["body_mass_g"] <= mass_slider_values[1])])

elif var == "sex":
    sex = st.radio("Select which sex you would like to filter by: ", options = ["male", "female", "not given"])
    if (sex == "male") | (sex == "female"):
        st.dataframe(penguins_df[penguins_df["sex"] == sex])
    else:
        st.dataframe(penguins_df[~penguins_df["sex"].isin(["male", "female"])])

elif var == "year":
    year = st.radio("Choose a year of penguin data to explore: ", options = penguins_df["year"].unique())
    st.dataframe(penguins_df[penguins_df["year"] == year])

# Providing a link to more information about the Palmer's Penguins dataset.
st.write("For more information on the dataset, see: https://allisonhorst.github.io/palmerpenguins/articles/intro.html")

# Thanking the user.
st.write("Thanks for checking out my app!")