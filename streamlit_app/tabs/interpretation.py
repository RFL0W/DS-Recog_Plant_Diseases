import streamlit as st
import pandas as pd
import numpy as np


title = "Projet : Reconnaissance de plantes"
sidebar_name = "Interpretabilité"


def run():

    st.title(title)
    st.header("Interprétabilité des modèles avec Grad-Cam")
    st.markdown("---")

    st.markdown(
        """
        ## Définition
        - **Grad**ient-weighted **C**lass **A**ctivation **M**apping : technique utilisée en deep pour **comprendre les points d’attention d’un modèle** d’analyse d’image
        - Calcul le **gradient de la classe de sortie par rapport** aux cartes de caractéristiques de la **dernière couche convolutive** pour créer une carte d’activation pondérée

        ## Remarques
        - **Tentative infructueuse** d’application d’un Grad-CAM sur un **modèle avec transfert learning**, le calcul du gradient posant un problème d’accès entre une sous-couche de la couche importée et la couche de sortie
        - **Incertitude** sur la possibilité d’appliquer un Grad-CAM ou équivalent **sur le modèle ViT** n’utilisant pas de CNN

        ## Résultats
        - **Feuille saine** : le contour et les nervures de la feuille semble déterminants
        """
        )

    st.image("./streamlit_app/assets/img-interpret-01.png", width=500)

    st.markdown(
        """
        - **Présence d’une partie dégradée** : cette partie sera le point de focalisation du modèle 
        """
        )

    st.image("./streamlit_app/assets/img-interpret-02.png", width=500)

    st.markdown(
        """
        - **Présence de tâches** : ces taches seront le point de focalisation du modèle
        """
        )

    st.image("./streamlit_app/assets/img-interpret-03.png", width=500)

    st.markdown(
        """
        - **Particularité de maladie** : focalisation sur le **pourtour** pour la maladie des « feuilles en cuillère »
        """
        )

    st.image("./streamlit_app/assets/img-interpret-04.png", width=500)

    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
