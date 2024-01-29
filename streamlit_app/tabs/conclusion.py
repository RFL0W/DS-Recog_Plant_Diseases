import streamlit as st
import pandas as pd
import numpy as np


title = "Projet : Reconnaissance de plantes"
sidebar_name = "Conclusion"


def run():

    st.title(title)
    st.header(sidebar_name)
    st.markdown("---")

    st.image("./streamlit_app/assets/plants_wind.gif", width=700)

    st.markdown(
        """
        ## Des performances étonnantes
        - Performances en machine learning bien meilleures qu’escompté lors de l’exploration des données
        - Performance en deep learning impressionnantes et laissant présager de nombreuses révolutions
        """
        )

    st.image("./streamlit_app/assets/img-end-01.png", width=300)

    st.markdown(
        """
        ## Des pistes d’amélioration à explorer
        - Grad-CAM à appliquer aux modèles avec transfert learning et explorer les possibilités sur le ViT
        - Méta-modèle combinant les outputs des différents modèles pour améliorer les résultats
        - Développement de modèles spécifiques pour traiter les plus grandes confusions
        """
        )

    st.image("./streamlit_app/assets/img-end-02.png", width=700)
        
    st.markdown(
        """
        ## Des ajouts nécessaires pour une utilisation en conditions réelles
        - Détourage des photos de feuilles prises en conditions réelles
        - Identification des cas d’utilisation et de leurs contraintes, par exemple appareils mobiles impliquant une réduction des tailles des modèles
        """
        )

    st.image("./streamlit_app/assets/img-end-03.png", width=700)

    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")

    st.markdown("---")
    st.image("./streamlit_app/assets/plants_plant.gif", width=700)
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    