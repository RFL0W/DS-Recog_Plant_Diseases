# Importing necessary modules
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go


title = "Projet : Reconnaissance de plantes"
sidebar_name = "Focus sur NPD"

# Function to load dataframes
@st.cache_data
def load_dataframe():
    df_npd = pd.read_csv("./streamlit_app/dataframes/df_NPD.csv")
    df_npd['Rotation'] = df_npd['Rotation'].astype(str)
    df_npd['Luminosite'] = df_npd['Luminosite'].astype(str)
    return df_npd

# Function for "New Plant Disease" tab
@st.cache_data
def plot_new_plant_disease_graphs(df_npd):
    rotations_seq = ['0', '30', '90', '180', '200', '270']

    st.subheader("Images avec modification de Rotation (par catégorie de feuilles)")
    fig1 = px.bar(df_npd, x='Categorie', color='Rotation', category_orders={"Rotation": rotations_seq})
    fig1.update_layout(height=600, width=900)
    fig1.update_yaxes(range=[0, 2700])
    st.plotly_chart(fig1)

    st.subheader("Images avec modification de Symétrie (par catégorie de feuilles)")
    fig2 = px.bar(df_npd, x='Categorie', color='Symetrie')
    fig2.update_layout(height=600, width=900)
    fig2.update_yaxes(range=[0, 2700])
    st.plotly_chart(fig2)

    st.subheader("Images avec modification de Luminosité (par catégorie de feuilles)")
    fig3 = px.bar(df_npd, x='Categorie', color='Luminosite')
    fig3.update_layout(height=600, width=900)
    fig3.update_yaxes(range=[0, 2700])
    st.plotly_chart(fig3)

    st.subheader("Nombre d'images avec une modification (par catégorie de feuilles)")
    fig4 = px.bar(df_npd[((df_npd['Symetrie']!='0') | (df_npd['Rotation']!='0') | (df_npd['Luminosite']!='0'))], x='Categorie')
    fig4.update_layout(height=600, width=900)
    fig4.update_yaxes(range=[0, 2700])
    st.plotly_chart(fig4)

    st.subheader("Nombre d'images sans mofication (par catégorie de feuilles)")
    fig5 = px.bar(df_npd[((df_npd['Symetrie']=='0') | (df_npd['Rotation']=='0') | (df_npd['Luminosite']=='0'))], x='Categorie')
    fig5.update_layout(height=600, width=900)
    fig5.update_yaxes(range=[0, 2700])
    st.plotly_chart(fig5)

    st.subheader("Répartition des fichiers par taille (en octets)")
    fig6 = go.Figure(data=[go.Histogram(y=df_npd['Taille du fichier'], marker_color='cyan')])
    fig6.update_layout(height=600, width=900)
    st.plotly_chart(fig6)

def run():

    st.title(title)
    st.header("Focus sur le dataset 'New Plant Disease'")
    st.markdown("---")

    df_npd = load_dataframe()
    plot_new_plant_disease_graphs(df_npd)
