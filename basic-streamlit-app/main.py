import streamlit as st
import pandas as pd

st.title("Filtering the Palmer's Penguins Dataset")
penguins_df = pd.read_csv("data/penguins.csv")

st.write("The Palmer's Penguins dataset contains 344 observations of penguin data across 9 different variables. The purpose of this streamlit application is to allow you to explore the data a bit better!")
st.write("Start out by choosing a metric you would like to sort by, then interact with the resulting widget to narrow down the data. The filtered dataset will then be displayed for you!")

var = st.selectbox("Choose a variable to filter by: ", penguins_df.columns.tolist())

if var == "id":
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