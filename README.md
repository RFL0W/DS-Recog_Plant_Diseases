# ProjectTemplate

## Presentation and Installation

This repository contains the code for our project "Plant and Diseases recognition", developed during our [Data Scientist training](https://datascientest.com/en/data-scientist-course) at [DataScientest](https://datascientest.com/).

The goal of this project is to recognise a specie and a potential disease based on a leaf picture. Typical application can be recognition and identification of diseases in a personal or professional agriculutral set-up.

This project was developed by the following team :

- Julie Le Vu ([GitHub](https://github.com/JulieLeVu) / [LinkedIn](https://www.linkedin.com/in/julie-le-vu-201b2a261/))
- Florent Maurice ([GitHub](https://github.com/RFL0W/) / [LinkedIn](http://linkedin.com/))
- Hadrien Gremillet([GitHub](https://github.com/HadrienGremillet) / [LinkedIn](https://www.linkedin.com/in/hadrieng/))

You can browse and run the [notebooks](./notebooks). 
- "0 Chargement de données" is needed to load data from files and fill dataframes used during the project
- "1 Exploration des données" countains all data generated during the exploratory phase. Naming of each Notebook is in line with our report
- "2 Modélisation ML" countains all notebooks related to Machine Learning models
- "3 Modélisation deep" countains 3 subfolders corresponding to 2 classes, 14 classes and 38 classes modelisations
    In each sub-folders, there are 2nd level subfolders "sans transfert" meaning models built from scratch and "avec transfert" means model based on transfert learning.

***IMPORTANT NOTE***
Due to Github size limitation, both the data and trained models cannot be uploaded to GitHub so you won't be able to execute directly the Notebooks without adapting the code.
If you want to reuse some of this work, you are encourage to contact one of the team member to assist you.

You will need to install the dependencies (in a dedicated environment) :

numpy==1.25.0
pandas==1.5.3
streamlit==1.25
Pillow==9.4.0
seaborn==0.12.2
matplotlib==3.7.1
plotly==5.9.0
tensorflow==2.10.1
keras==2.10.0
seaborn==0.12.2


## Streamlit App

***IMPORTANT NOTE***
Same caveat than before. Due to size limitations, both data and trained model are not saved on GitHub so you won't be able to execute the Streamlit app without team assistance.

To run the app (be careful with the paths of the files in the app):

```shell
conda create --name plant_recognition python=3.9
conda activate plant_recognition
pip install -r requirements.txt
streamlit run app.py
```

The app should then be available at [localhost:8501](http://localhost:8501).
