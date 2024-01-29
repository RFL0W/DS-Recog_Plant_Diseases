import streamlit as st
import pandas as pd
import numpy as np


title = "Projet : Reconnaissance de plantes"
sidebar_name = "Modélisation"


def run():

    st.title(title)
    st.header(sidebar_name)
    st.markdown("---")

    # Liste des options pour la sélectbox
    options = [
        "Modélisation Machine-Learning",
        "Modélisation Deep-Learning 2 et 14 Classes",
        "Modélisation Deep-Learning 38 Classes",
        "Récapitulatif des Modèles Choisis"
    ]

    # Sélectbox pour choisir une option
    choix = st.selectbox("Choisir une section :", options)

    if choix == "Modélisation Machine-Learning":
        st.markdown(
        """
        ## Préparation des données
        - Utilisation des **diagrammes d’intensité** tronqués
        - Tests réalisés sur les **images segmentées** ou **complètes**
        - Utilisation des **images non transformées en luminosité**
        - **Undersampling** et **sélection** de certaines augmentations de données pour obtenir une **équirépartition** des classes

        ## Modèles et entrainement
        - **3 types de classifications** : binaire, 14 classes, 38 classes
        - **4 types de modèles** : Random Forest, SVM, XGBoost et Régression logisitique

        ## Résultats
        - **2 classes** : meilleur modèle **Random Forest**, **accuracy** de **0,92**
        """
        )

        st.image("./streamlit_app/assets/ML-Result-2C.png", width=400)

        st.markdown(
        """
        - **14 classes** : meilleur modèle **SVM**, **accuracy** de **0,87**
        """
        )

        st.image("./streamlit_app/assets/ML-Result-14C.png", width=400)

        st.markdown(
        """
        - **38 classes** : meilleur modèle **SVM**, **accuracy** de **0,90**
        """
        )

        st.image("./streamlit_app/assets/ML-Result-38C.png")

        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")

    elif choix == "Modélisation Deep-Learning 2 et 14 Classes":
        st.markdown(
        """
        ## Préparation des données
        - **Oversampling** pour la classification binaire
        - **Re-sampling** pour la classification 14
        - **Génération de bruit** supplémentaire
        - Utilisation des **preprocess dédiés** à chaque modèle importé

        ## Modèles et entrainement
        #### Structure des modèles retenus
        - **2 classes** : **EfficientNetV2B3** + **6 couches** de sortie personnalisées
        - **14 classes** : **EfficientNetV2S** + **6 couches** de sortie personnalisées
        #### Entraînement
        - **Variation des tests** : avec/sans re-sampling, avec/sans modifications d’image
        - **Variation des hyperparamètres** via **GridSearchCV**

        ## Résultats
        - **2 classes** : **accuracy** de **0,997** avec seulement **53** erreurs / 17 571 images
        """
        )

        st.image("./streamlit_app/assets/DL-Result-2C.png", width=300)

        st.markdown(
        """
        - **14 classes** : **accuracy** de **0,996** et **f1-score** supérieur à **0,99** sur toutes les classes, **70 erreurs**  / 17 571 images
        """
        )

        st.image("./streamlit_app/assets/DL-Result-14C.png", width=500)

        st.markdown(
        """
        **A noter** : Les feuilles de **tomates** sont très **confondues** avec celles de **pomme de terre**. Plus de **40% des erreurs** viennent de cette confusion (29 occurrences).
        """
        )

        st.image("./streamlit_app/assets/DL-Result-14C-Bonus.png", width=400)

        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")

    elif choix == "Modélisation Deep-Learning 38 Classes":
        st.markdown(
        """
        ## Préparation des données
        - **Pas d’augmentation** par transformation car New Plant Disease déjà significativement transformée
        - **Normalisation** 1/255
        - **Redimensionnement** en 224, 224
        - Utilisation d’un ImageDataGenerator pour **charger les images à la volée**

        ## Modèles et entrainement
        #### Structure des modèles retenus
        - Features extraction d’un modèle ViT pré-entrainé 
        - Ajout de 6 couches de sortie personnalisées
        #### Entraînement
        - Entrainement en **4** étapes (15 puis 3*10 epochs) avec **dégel progressif** des couches importées
        - Conservation de la **meilleure epoch** (val_accuracy) **à chaque étape** (callback)

        ## Résultats
        - Une **très bonne diagonale** de **heatmap** (36 mauvaises classifications sur 17 571 images de l’ensemble de validation)
        """
        )

        st.image("./streamlit_app/assets/DL-Result-38C-1.png", width=650)

        st.markdown(
        """
        - Un **classification report** donnant un **F1-score** minimal de **0,986** et une **accuracy** de **0,998**
        """
        )

        st.image("./streamlit_app/assets/DL-Result-38C-2.png", width=650)

        st.markdown(
        """
        **Bonus** : Une analyse des **mauvaises classifications** faisant ressortir des **ambiguïtés** possibles
        """
        )

        st.image("./streamlit_app/assets/DL-Result-38C-Bonus.png")

        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")

    elif choix == "Récapitulatif des Modèles Choisis":
        st.header("Comparaison des Modèles")
        st.image("./streamlit_app/assets/img-model-recap.png", width=900)

        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")
        st.markdown("")

if __name__ == "__main__":
    run()
