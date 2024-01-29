# Importing necessary modules
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go


title = "Projet : Reconnaissance de plantes"
sidebar_name = "Exploration"

# Function to load dataframes
@st.cache_data
def load_dataframe():
    df_kaggle_data = pd.read_csv("./streamlit_app/dataframes/df_KaggleData.csv")
    return df_kaggle_data

def choose_df():
    st.subheader("Comparaison des datasets PlantVillage, Plant Disease et New Plant Diseases")
    dataset_selection = st.selectbox("Sélectionnez le dataset à afficher", ["PlantVillage", "Plant Disease", "New Plant Diseases", "All Datasets"], index=0)
    return dataset_selection

# Function to plot interactive graphs in "Exploration" tab
@st.cache_data
def plot_interactive_graphs(df_kaggle_data, dataset_selection):

    if dataset_selection == "PlantVillage":
        dataset_df = df_kaggle_data[df_kaggle_data["Dataset"] == "PlantVillage"]
        custom_palette = sns.color_palette("Paired", 2).as_hex()
    elif dataset_selection == "Plant Disease":
        dataset_df = df_kaggle_data[df_kaggle_data["Dataset"] == "PlantDisease"]
        custom_palette = sns.color_palette("Set1", 14).as_hex()
    elif dataset_selection == "New Plant Disease":
        dataset_df = df_kaggle_data[df_kaggle_data["Dataset"] == "NewPlantDisease"]
        custom_palette = sns.color_palette("husl", 34).as_hex()
    else:
        dataset_df = df_kaggle_data  # For All Datasets, use the entire dataframe
        custom_palette = px.colors.qualitative.Plotly

    st.write(f"Nombre total d'images dans le dataset sélectionné: {len(dataset_df)}")
    st.write(dataset_df)

    if dataset_selection == "All Datasets":
        # Interactive Graph 1 - Count Plot for 'Saine' column
        st.subheader("Distribution de la classe 'Saine'")
        dataset_df['Saine'].replace({1: 'Saine', 0: 'Malade'}, inplace=True)
        fig1 = px.bar(dataset_df, x='Saine', color='Dataset', color_discrete_sequence=custom_palette, barmode='group')
        fig1.update_layout(height=300, width=500)
        st.plotly_chart(fig1)

        # Interactive Graph 2 - Bar Plot for 'Plante' column
        st.subheader("Distribution de la classe 'Plante'")
        fig2 = px.bar(dataset_df, x='Plante', color='Dataset', color_discrete_sequence=custom_palette, barmode='group')
        fig2.update_layout(height=500, width=800)
        st.plotly_chart(fig2)

        # Interactive Graph 3 - Bar Plot for 'Categorie' column
        st.subheader("Distribution de la classe 'Categorie'")
        fig3 = px.bar(dataset_df, x='Categorie', color='Dataset', color_discrete_sequence=custom_palette, barmode='group')
        fig3.update_layout(height=700, width=1200)
        st.plotly_chart(fig3)

    else:
        # Interactive Graph 1 - Bar Plot for 'Saine' column
        st.subheader("Distribution de la classe 'Saine'")
        dataset_df['Saine'].replace({1: 'Saine', 0: 'Malade'}, inplace=True)
        saine_counts = dataset_df['Saine'].value_counts()
        fig1 = px.bar(saine_counts, x=saine_counts.index, y=saine_counts.values, labels={'x': 'Classe de Saine', 'y': 'Nombre'}, color=saine_counts.index, color_discrete_sequence=custom_palette)
        fig1.update_layout(height=300, width=500)
        st.plotly_chart(fig1)

        # Interactive Graph 2 - Bar Plot for 'Plante' column
        st.subheader("Distribution de la classe 'Plante'")
        plante_counts = dataset_df['Plante'].value_counts()
        custom_palette_2 = sns.color_palette("Set1", len(plante_counts)).as_hex()
        fig2 = px.bar(plante_counts, x=plante_counts.index, y=plante_counts.values, labels={'x': 'Classe de Plante', 'y': 'Nombre'}, color=plante_counts.index, color_discrete_sequence=custom_palette_2)
        st.plotly_chart(fig2)

        # Interactive Graph 3 - Count Plot for 'Categorie' column
        st.subheader("Distribution de la classe 'Categorie'")
        categorie_counts = dataset_df['Categorie'].value_counts()
        custom_palette_3 = sns.color_palette("husl", len(categorie_counts)).as_hex()
        fig3 = px.bar(categorie_counts, x=categorie_counts.index, y=categorie_counts.values, labels={'x': 'Classe de Maladie', 'y': 'Nombre'}, color=categorie_counts.index, color_discrete_sequence=custom_palette_3)
        fig3.update_layout(height=700, width=1100)
        st.plotly_chart(fig3)


def run():

    st.title(title)
    st.header(sidebar_name)
    st.markdown("---")

    df_kaggle_data = load_dataframe()
    dataset_selection = choose_df()
    plot_interactive_graphs(df_kaggle_data, dataset_selection)
