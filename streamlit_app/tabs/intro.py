import streamlit as st


title = "Projet : Reconnaissance de plantes"
sidebar_name = "Introduction"


def run():

    st.title(title)
    st.header(sidebar_name)
    st.markdown("---")

    # TODO: choose between one of these GIFs
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/1.gif")
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/2.gif")
    #st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/3.gif")
    st.image("./streamlit_app/assets/plants_rain.gif", width=700)

    st.title("Exemples de classification attendue")
    st.image("./streamlit_app/assets/img-intro.png", width=300)

    st.title("Application possible")
    st.markdown("Permettre aux jardiniers amateurs et/ou professionnels d’identifier facilement les plantes de leur jardin et de diagnostiquer leurs éventuelles maladies")
    st.markdown("")

    st.title("Attentes méthodologiques")
    st.markdown("Voici les points attendus pour la méthodologie du projet :")
    st.markdown("- Études des bases de données à disposition")
    st.markdown("- Développement de différents modèles de machine et deep learning couvrant différents types de classifications (malades/saines, espèces, espèces-maladie)")
    st.markdown("- Recherche des meilleures performances (accuracy) par améliorations itératives des modèles")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")

    st.markdown("---")
    st.image("./streamlit_app/assets/plants_grow.gif", width=700)
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")

if __name__ == "__main__":
    run()
