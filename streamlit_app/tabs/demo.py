import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as eNetV2_preprocess_input
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_VGG16
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_Inception
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from vit_keras import vit
from tf_keras_vis.gradcam import Gradcam

#####################################################################
#Définition des variables
pathTest = '.\\streamlit_app\\ImagesTest\\'
title = "Projet : Reconnaissance de plantes"
sidebar_name = "Démonstration"

class_dict = {0: 'Apple__Apple_scab',
 1: 'Apple__Black_rot',
 2: 'Apple__Cedar_apple_rust',
 3: 'Apple__healthy',
 4: 'Blueberry__healthy',
 5: 'Cherry_(including_sour)__Powdery_mildew',
 6: 'Cherry_(including_sour)__healthy',
 7: 'Corn_(maize)__Cercospora_leaf_spot Gray_leaf_spot',
 8: 'Corn_(maize)__Common_rust_',
 9: 'Corn_(maize)__Northern_Leaf_Blight',
 10: 'Corn_(maize)__healthy',
 11: 'Grape__Black_rot',
 12: 'Grape__Esca_(Black_Measles)',
 13: 'Grape__Leaf_blight_(Isariopsis_Leaf_Spot)',
 14: 'Grape__healthy',
 15: 'Orange__Haunglongbing_(Citrus_greening)',
 16: 'Peach__Bacterial_spot',
 17: 'Peach__healthy',
 18: 'Pepper,_bell__Bacterial_spot',
 19: 'Pepper,_bell__healthy',
 20: 'Potato__Early_blight',
 21: 'Potato__Late_blight',
 22: 'Potato__healthy',
 23: 'Raspberry__healthy',
 24: 'Soybean__healthy',
 25: 'Squash__Powdery_mildew',
 26: 'Strawberry__Leaf_scorch',
 27: 'Strawberry__healthy',
 28: 'Tomato__Bacterial_spot',
 29: 'Tomato__Early_blight',
 30: 'Tomato__Late_blight',
 31: 'Tomato__Leaf_Mold',
 32: 'Tomato__Septoria_leaf_spot',
 33: 'Tomato__Spider_mites Two-spotted_spider_mite',
 34: 'Tomato__Target_Spot',
 35: 'Tomato__Tomato_Yellow_Leaf_Curl_Virus',
 36: 'Tomato__Tomato_mosaic_virus',
 37: 'Tomato__healthy'}
#####################################################################

#####################################################################
#Définition des fonctions

####
#Fonction permettant à partir du nom du fichier de l'image de rechercher les informations saine/malade, plante, catégorie d'une feuille du jeu de test
def InfoNomImageTest(nom):
    if ("Apple" in nom) and ("Scab" in nom):
      Categorie = "Apple__Apple_scab"
      Plante = "Pommier"
      Saine = "Malade"
      Maladie = "Tavelure du pommier"
    elif ("Apple" in nom) and ("Rot" in nom):
      Categorie = "Apple__Black_rot"
      Plante = "Pommier"
      Saine = "Malade"
      Maladie = "Pourriture noire"
    elif ("Apple" in nom) and ("Rust" in nom):
      Categorie = "Apple__Cedar_apple_rust"
      Plante = "Pommier"
      Saine = "Malade"
      Maladie = "Rouille de Virginie du pommier"
    elif ("Apple" in nom) and ("Healthy" in nom):
      Categorie = "Apple__healthy"
      Plante = "Pommier"
      Saine = "Saine"
      Maladie = "Aucune"
    elif ("Blueberry" in nom) and ("Healthy" in nom):
      Categorie = "Blueberry__healthy"
      Plante = "Myrtille"
      Saine = "Saine"
      Maladie = "Aucune"
    elif ("Cherry" in nom) and ("Mildew" in nom):
      Categorie = "Cherry_(including_sour)__Powdery_mildew"
      Plante = "Cerisier"
      Saine = "Malade"
      Maladie = "Oïdium"
    elif ("Cherry" in nom) and ("Healthy" in nom):
      Categorie = "Cherry_(including_sour)__healthy"
      Plante = "Cerisier"
      Saine = "Saine"
      Maladie = "Aucune"
    elif ("Corn" in nom) and ("Leaf" in nom) and ("Spot" in nom):
      Categorie = "Corn_(maize)__Cercospora_leaf_spot Gray_leaf_spot"
      Plante = "Maïs"
      Saine = "Malade"
      Maladie = "Cercosporiose du maïs (maladie des taches grises)"
    elif ("Corn" in nom) and ("Rust" in nom):
      Categorie = "Corn_(maize)__Common_rust_"
      Plante = "Maïs"
      Saine = "Malade"
      Maladie = "Rouille commune"
    elif ("Corn" in nom) and ("Leaf" in nom) and ("Blight" in nom):
      Categorie = "Corn_(maize)__Northern_Leaf_Blight"
      Plante = "Maïs"
      Saine = "Malade"
      Maladie = "Helminthosporiose du nord"
    elif ("Corn" in nom) and ("Healthy" in nom):
      Categorie = "Corn_(maize)__healthy"
      Plante = "Maïs"
      Saine = "Saine"
      Maladie = "Aucune"
    elif ("Grape" in nom) and ("Rot" in nom):
      Categorie = "Grape__Black_rot"
      Plante = "Vigne"
      Saine = "Malade"
      Maladie = "Pourriture noire"
    elif ("Grape" in nom) and ("Esca" in nom):
      Categorie = "Grape__Esca_(Black_Measles)"
      Plante = "Vigne"
      Saine = "Malade"
      Maladie = "Esca (apoplexie parasite)"
    elif ("Grape" in nom) and ("Leaf" in nom) and ("Spot" in nom):
      Categorie = "Grape__Leaf_blight_(Isariopsis_Leaf_Spot)"
      Plante = "Vigne"
      Saine = "Malade"
      Maladie = "Brûlure de la feuille"
    elif ("Grape" in nom) and ("Healthy" in nom):
      Categorie = "Grape__healthy"
      Plante = "Vigne"
      Saine = "Saine"
      Maladie = "Aucune"
    elif ("Orange" in nom) and ("Haunglongbing" in nom):
      Categorie = "Orange__Haunglongbing_(Citrus_greening)"
      Plante = "Orange"
      Saine = "Malade"
      Maladie = "Maladie du dragon jaune"
    elif ("Peach" in nom) and ("Bacterial" in nom):
      Categorie = "Peach__Bacterial_spot"
      Plante = "Pêcher"
      Saine = "Malade"
      Maladie = "Bactériose des fruits à noyau"
    elif ("Peach" in nom) and ("Healthy" in nom):
      Categorie = "Peach__healthy"
      Plante = "Pêcher"
      Saine = "Saine"
      Maladie = "Aucune"
    elif ("Pepper" in nom) and ("Bacterial" in nom):
      Categorie = "Pepper,_bell__Bacterial_spot"
      Plante = "Poivron"
      Saine = "Malade"
      Maladie = "Tâche bactérienne"
    elif ("Pepper" in nom) and ("Healthy" in nom):
      Categorie = "Pepper,_bell__healthy"
      Plante = "Poivron"
      Saine = "Saine"
      Maladie = "Aucune"
    elif ("Potato" in nom) and ("Early" in nom) and ("Blight" in nom):
      Categorie = "Potato__Early_blight"
      Plante = "Pomme de terre"
      Saine = "Malade"
      Maladie = "Alternariose"
    elif ("Potato" in nom) and ("Late" in nom) and ("Blight" in nom):
      Categorie = "Potato__Late_blight"
      Plante = "Pomme de terre"
      Saine = "Malade"
      Maladie = "Mildiou de la pomme de terre"
    elif ("Potato" in nom) and ("Healthy" in nom):
      Categorie = "Potato__healthy"
      Plante = "Pomme de terre"
      Saine = "Saine"
      Maladie = "Aucune"
    elif ("Raspberry" in nom) and ("Healthy" in nom):
      Categorie = "Raspberry__healthy"
      Plante = "Framboisier"
      Saine = "Saine"
      Maladie = "Aucune"
    elif ("Soybean" in nom) and ("Healthy" in nom):
      Categorie = "Soybean__healthy"
      Plante = "Soja"
      Saine = "Saine"
      Maladie = "Aucune"
    elif ("Squash" in nom) and ("Mildew" in nom):
      Categorie = "Squash__Powdery_mildew"
      Plante = "Courge"
      Saine = "Malade"
      Maladie = "Oïdium"
    elif ("Strawberry" in nom) and ("Scorch" in nom):
      Categorie = "Strawberry__Leaf_scorch"
      Plante = "Fraisier"
      Saine = "Malade"
      Maladie = "Brûlure foliaire"
    elif ("Strawberry" in nom) and ("Healthy" in nom):
      Categorie = "Strawberry__healthy"
      Plante = "Fraisier"
      Saine = "Saine"
      Maladie = "Aucune"
    elif ("Tomato" in nom) and ("Bacterial" in nom):
      Categorie = "Tomato__Bacterial_spot"
      Plante = "Tomate"
      Saine = "Malade"
      Maladie = "Gale bactérienne"
    elif ("Tomato" in nom) and ("Early" in nom) and ("Blight" in nom):
      Categorie = "Tomato__Early_blight"
      Plante = "Tomate"
      Saine = "Malade"
      Maladie = "Alternariose"
    elif ("Tomato" in nom) and ("Late" in nom) and ("Blight" in nom):
      Categorie = "Tomato__Late_blight"
      Plante = "Tomate"
      Saine = "Malade"
      Maladie = "Mildiou de la tomate"
    elif ("Tomato" in nom) and ("Mold" in nom):
      Categorie = "Tomato__Leaf_Mold"
      Plante = "Tomate"
      Saine = "Malade"
      Maladie = "Pourriture foliaire de la tomate"
    elif ("Tomato" in nom) and ("Leaf" in nom) and ("Spot" in nom):
      Categorie = "Tomato__Septoria_leaf_spot"
      Plante = "Tomate"
      Saine = "Malade"
      Maladie = "Tache septorienne"
    elif ("Tomato" in nom) and ("Spider" in nom):
      Categorie = "Tomato__Spider_mites Two-spotted_spider_mite"
      Plante = "Tomate"
      Saine = "Malade"
      Maladie = "Tétranyques"
    elif ("Tomato" in nom) and ("Target" in nom) and ("Spot" in nom):
      Categorie = "Tomato__Target_Spot"
      Plante = "Tomate"
      Saine = "Malade"
      Maladie = "Tache concentrique"
    elif ("Tomato" in nom) and ("Curl" in nom) and ("Virus" in nom):
      Categorie = "Tomato__Tomato_Yellow_Leaf_Curl_Virus"
      Plante = "Tomate"
      Saine = "Malade"
      Maladie = "Virus des feuilles jaunes en cuillère de la tomate"
    elif ("Tomato" in nom) and ("Mosaic" in nom):
      Categorie = "Tomato__Tomato_mosaic_virus"
      Plante = "Tomate"
      Saine = "Malade"
      Maladie = "Mosaïque de la tomate"
    elif ("Tomato" in nom) and ("Healthy" in nom):
      Categorie = "Tomato__healthy"
      Plante = "Tomate"
      Saine = "Saine"
      Maladie = "Aucune"
    else:
      Categorie = "Inconnue"
      Plante = "Inconnue"
      Saine = "Inconnu"
      Maladie = "Inconnue"
    return Saine, Plante, Maladie

#####
#Fonction permettant de définir le nom de la plante en français en fonction du nom anglais
def ValeursPred14(pred):
    if (pred == "Apple"):
        pred = "Pommier"
    elif (pred == "Blueberry"):
        pred = "Myrtille"
    elif (pred == "Cherry_(including_sour)"):
        pred = "Cerisier"
    elif (pred == "Corn_(maize)"):
        pred = "Maïs"
    elif (pred == "Grape"):
        pred = "Vigne"
    elif (pred == "Peach"):
        pred = "Pêcher"
    elif (pred == "Orange"):
        pred = "Orange"
    elif (pred == "Pepper,_bell"):
        pred = "Poivron"
    elif (pred == "Potato"):
        pred = "Pomme de terre"
    elif (pred == "Raspberry"):
        pred = "Framboisier"
    elif (pred == "Soybean"):
        pred = "Soja"
    elif (pred == "Squash"):
        pred = "Courge"
    elif (pred == "Strawberry"):
        pred = "Fraisier"
    elif (pred == "Tomato"):
        pred = "Tomate"
    else:
        pred = "Inconnue"
    return pred
     
#####
#Fonction permettant de définir les 3 valeurs (Etat, Plante, Maladie) en fonction de la catégorie prédite
def ValeursPred38(pred):
    if (pred == "Apple__Apple_scab"):
      Plante = "Pommier"
      Etat = "Malade"
      Maladie = "Tavelure du pommier"
    elif (pred == "Apple__Black_rot"):
      Plante = "Pommier"
      Etat = "Malade"
      Maladie = "Pourriture noire"
    elif (pred == "Apple__Cedar_apple_rust"):
      Plante = "Pommier"
      Etat = "Malade"
      Maladie = "Rouille de Virginie du pommier"
    elif (pred == "Apple__healthy"):
      Plante = "Pommier"
      Etat = "Saine"
      Maladie = "Aucune"
    elif (pred == "Blueberry__healthy"):
      Plante = "Myrtille"
      Etat = "Saine"
      Maladie = "Aucune"
    elif (pred == "Cherry_(including_sour)__Powdery_mildew"):
      Plante = "Cerisier"
      Etat = "Malade"
      Maladie = "Oïdium"
    elif (pred == "Cherry_(including_sour)__healthy"):
      Plante = "Cerisier"
      Etat = "Saine"
      Maladie = "Aucune"
    elif (pred == "Corn_(maize)__Cercospora_leaf_spot Gray_leaf_spot"):
      Plante = "Maïs"
      Etat = "Malade"
      Maladie = "Cercosporiose du maïs (maladie des taches grises)"
    elif (pred == "Corn_(maize)__Common_rust_"):
      Plante = "Maïs"
      Etat = "Malade"
      Maladie = "Rouille commune"
    elif (pred == "Corn_(maize)__Northern_Leaf_Blight"):
      Plante = "Maïs"
      Etat = "Malade"
      Maladie = "helminthosporiose du nord"
    elif (pred == "Corn_(maize)__healthy"):
      Plante = "Maïs"
      Etat = "Saine"
      Maladie = "Aucune"
    elif (pred == "Grape__Black_rot"):
      Plante = "Vigne"
      Etat = "Malade"
      Maladie = "Pourriture noire"
    elif (pred == "Grape__Esca_(Black_Measles)"):
      Plante = "Vigne"
      Etat = "Malade"
      Maladie = "Esca (apoplexie parasite)"
    elif (pred == "Grape__Leaf_blight_(Isariopsis_Leaf_Spot)"):
      Plante = "Vigne"
      Etat = "Malade"
      Maladie = "Brûlure de la feuille"
    elif (pred == "Grape__healthy"):
      Plante = "Vigne"
      Etat = "Saine"
      Maladie = "Aucune"
    elif (pred == "Orange__Haunglongbing_(Citrus_greening)"):
      Plante = "Orange"
      Etat = "Malade"
      Maladie = "Maladie du dragon jaune"
    elif (pred == "Peach__Bacterial_spot"):
      Plante = "Pêcher"
      Etat = "Malade"
      Maladie = "Bactériose des fruits à noyau"
    elif (pred == "Peach__healthy"):
      Plante = "Pêcher"
      Etat = "Saine"
      Maladie = "Aucune"
    elif (pred == "Pepper,_bell__Bacterial_spot"):
      Plante = "Poivron"
      Etat = "Malade"
      Maladie = "Tâche bactérienne"
    elif (pred == "Pepper,_bell__healthy"):
      Plante = "Poivron"
      Etat = "Saine"
      Maladie = "Aucune"
    elif (pred == "Potato__Early_blight"):
      Plante = "Pomme de terre"
      Etat = "Malade"
      Maladie = "Alternariose"
    elif (pred == "Potato__Late_blight"):
      Plante = "Pomme de terre"
      Etat = "Malade"
      Maladie = "Mildiou de la pomme de terre"
    elif (pred == "Potato__healthy"):
      Plante = "Pomme de terre"
      Etat = "Saine"
      Maladie = "Aucune"
    elif (pred == "Raspberry__healthy"):
      Plante = "Framboisier"
      Etat = "Saine"
      Maladie = "Aucune"
    elif (pred == "Soybean__healthy"):
      Plante = "Soja"
      Etat = "Saine"
      Maladie = "Aucune"
    elif (pred == "Squash__Powdery_mildew"):
      Plante = "Courge"
      Etat = "Malade"
      Maladie = "Oïdium"
    elif (pred == "Strawberry__Leaf_scorch"):
      Plante = "Fraisier"
      Etat = "Malade"
      Maladie = "Brûlure foliaire"
    elif (pred == "Strawberry__healthy"):
      Plante = "Fraisier"
      Etat = "Saine"
      Maladie = "Aucune"
    elif (pred == "Tomato__Bacterial_spot"):
      Plante = "Tomate"
      Etat = "Malade"
      Maladie = "Gale bactérienne de la tomate"
    elif (pred == "Tomato__Early_blight"):
      Plante = "Tomate"
      Etat = "Malade"
      Maladie = "Alternariose"
    elif (pred == "Tomato__Late_blight"):
      Plante = "Tomate"
      Etat = "Malade"
      Maladie = "Mildiou de la tomate"
    elif (pred == "Tomato__Leaf_Mold"):
      Plante = "Tomate"
      Etat = "Malade"
      Maladie = "Pourriture foliaire de la tomate"
    elif (pred == "Tomato__Septoria_leaf_spot"):
      Plante = "Tomate"
      Etat = "Malade"
      Maladie = "Tache septorienne"
    elif (pred == "Tomato__Spider_mites Two-spotted_spider_mite"):
      Plante = "Tomate"
      Etat = "Malade"
      Maladie = "Tétranyques"
    elif (pred == "Tomato__Target_Spot"):
      Plante = "Tomate"
      Etat = "Malade"
      Maladie = "Tache concentrique"
    elif (pred == "Tomato__Tomato_Yellow_Leaf_Curl_Virus"):
      Plante = "Tomate"
      Etat = "Malade"
      Maladie = "Virus des feuilles jaunes en cuillère de la tomate"
    elif (pred == "Tomato__Tomato_mosaic_virus"):
      Plante = "Tomate"
      Etat = "Malade"
      Maladie = "Mosaïque de la tomate"
    elif (pred == "Tomato__healthy"):
      Plante = "Tomate"
      Etat = "Saine"
      Maladie = "Aucune"
    else:
      Plante = "Inconnue"
      Etat = "Inconnu"
      Maladie = "Inconnue"
    return Etat, Plante, Maladie

####
#Fonction du calcul de l'histogramme
def CalculHistImage(nomPath):
    #lecture de l'image
    img = cv2.imread(nomPath)
    
    # Calcul des histogrammes pour les 3 couleurs
    histB = cv2.calcHist([img], [0], None, [256], [0, 256])
    histG = cv2.calcHist([img], [1], None, [256], [0, 256])
    histR = cv2.calcHist([img], [2], None, [256], [0, 256])
    # Suppression des valeurs de début et de fin
    histB = histB[10:-10]
    histG = histG[10:-10]
    histR = histR[10:-10]
    # Concaténation des histogrammes et conversion en DataFrame
    df_tabHistInt = pd.DataFrame(np.concatenate([histB, histG, histR], axis=None))
    #on retourne le DataFrame transposé (1 ligne et 708 colonnes)
    return df_tabHistInt.T

#####
#Fonction "predict" pour le modèle de classification binaire EfficientNetV2B3, nommé "BEST-2CLASS-MODEL.h5"
def DL_2c_eNetV2B3_Predict(image_path, model_path):
    classifier = models.load_model(model_path)
    #Chargement et prétraitement de l'image
    test_image = image.load_img(image_path, target_size=(224, 224))
    test_image = image.img_to_array(test_image) / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    processed_test_img = eNetV2_preprocess_input(test_image)

    #Prédiction de la classe sans affichage
    confidence_percentage = round(100 * classifier.predict(processed_test_img, verbose=0)[0][1], 3)
    if confidence_percentage < 50:
        prediction = 0
        confidence_percentage = round(100 - confidence_percentage, 2)
    else:
        prediction = 1
    return(prediction, confidence_percentage)


#####
#Fonction "predict" pour le modèle de classification 14 classes EfficientNetV2S, nommé "BEST-14CLASS-MODEL.h5"
def DL_14c_eNetV2S_Predict(image_path, model_path):
    classifier = models.load_model(model_path)
    plant_class_mapping = {'Apple': 0,
                       'Blueberry': 1,
                       'Cherry_(including_sour)': 2,
                       'Corn_(maize)': 3,
                       'Grape': 4,
                       'Orange': 5,
                       'Peach': 6,
                       'Pepper,_bell': 7,
                       'Potato': 8,
                       'Raspberry': 9,
                       'Soybean': 10,
                       'Squash': 11,
                       'Strawberry': 12,
                       'Tomato': 13}

    #Inversion de la correspondance pour obtenir les noms de classe
    plant_class_names = {v: k for k, v in plant_class_mapping.items()}

    #Chargement et prétraitement de l'image
    test_image = image.load_img(image_path, target_size=(224, 224))
    test_image = image.img_to_array(test_image) / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    processed_test_img = eNetV2_preprocess_input(test_image)

    #Prédiction de la classe
    prediction_plant = classifier.predict(processed_test_img, verbose=0)
    predicted_plant_class_index = np.argmax(prediction_plant)
    predicted_plant_class = plant_class_names[predicted_plant_class_index]
    confidence_percentage = round(100 * prediction_plant[0][predicted_plant_class_index], 3)

    prediction = predicted_plant_class

    return(prediction, confidence_percentage)

#####
#Fonction de prédiction générique pour DL 38 classes
def model_predict(image_path, model38, preprocess_function, image_size, class_dict):
    # Charger l'image
    img = image.load_img(image_path, target_size=image_size)
    img_array = image.img_to_array(img)
    # Appliquer la fonction de prétraitement
    img_array = preprocess_function(img_array)
    # Ajouter une dimension supplémentaire pour indiquer le numéro du lot (batch)
    img_array = np.expand_dims(img_array, axis=0)
    # Faire des prédictions
    predictions = model38.predict(img_array)
    # Obtenir la classe prédite et la probabilité maximale
    predicted_class_index = np.argmax(predictions[0])
    predicted_proba = np.max(predictions[0])
    # Obtenir le nom de la classe prédite
    predicted_class = class_dict[predicted_class_index]
    return predicted_class, predicted_proba*100

#####
# Fonction pour préparer les images pour modèles DL 38 classes
def preprocess_normalisation(img):
    return img / 255.

#####
# Fonction de prédiction ViT_predict
def ViT_predict(image_path,model_path):
    model_ViT = models.load_model(model_path)
    return(model_predict(image_path, model_ViT, preprocess_normalisation, (224,224), class_dict))

#####
# Fonction de prédiction VGG16
def VGG16_predict(image_path,model_path):
    model_VGG16 = models.load_model(model_path)
    return(model_predict(image_path, model_VGG16, preprocess_VGG16, (224,224), class_dict))

#####
# Fonction de prédiction Inception
def Inception_predict(image_path,model_path):
    model_Inception = models.load_model(model_path)
    return(model_predict(image_path, model_Inception, preprocess_Inception, (299,299), class_dict))

#####
# Fonction de prédiction Dropout
def Dropout_predict(image_path,model_path):
    model_Dropout = models.load_model(model_path)
    return(model_predict(image_path, model_Dropout, preprocess_normalisation, (256,256), class_dict))

#####
# Fonction de prédiction 4 couches
def Quatre_couches_predict(image_path,model_path):
    model_Quatre_couches = models.load_model(model_path)
    return(model_predict(image_path, model_Quatre_couches, preprocess_normalisation, (256,256), class_dict))

#####
# Fonction de prédiction LeNet
def LeNet_predict(image_path,model_path):
    model_LeNet = models.load_model(model_path)
    return(model_predict(image_path, model_LeNet, preprocess_normalisation, (256,256), class_dict))

#####
# Fonction permettant d'afficher pour une image l'image de l'histogramme et le GradCam
def affiche_hist_gradcam(image_path, model):
    # Chargement de l'image
    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    
    # Créer une nouvelle figure
    fig, axs = plt.subplots(1,3,figsize=(9,3), dpi=300)
    fig.patch.set_facecolor('black')
    
    #_____________________________________________________
    # Affichage de l'image au centre
    
    axs[1].imshow(image_np)
    axs[1].axis('off')
    
    
    #_____________________________________________________
    # Affichage du diagramme de densité à gauche
    
    # Calculer les histogrammes pour chaque couleur
    color = ('r','g','b')
    histograms = []
    for i,col in enumerate(color):
        histogram = cv2.calcHist([image_np], [i], None, [256], [0,256])
        histograms.append(histogram)
    
    # Affichage des histogrammes
    for i, histogram in enumerate(histograms):
        axs[0].plot(histogram, color = color[i], alpha=0.6, linewidth=2.5)  # Augmenter la largeur de la ligne
    
    # Configurer l'histogramme
    axs[0].set_xlim([0,256])
    axs[0].set_facecolor('black')  # rendre le fond du subplot ax2 noir

    axs[0].set_xlabel("Intensité", weight='bold', color = 'white')
    axs[0].set_ylabel("Nombre de pixels", weight='bold', color = 'white')
    axs[0].set_title("Machine Learning \n diagramme d'intensité", weight='bold', color = 'white')
    axs[0].tick_params(colors='white')
    
    #_____________________________________________________
    # Affichage du Grad-CAM
    
    # Model modifier function to change the last layer activation to linear
    def model_modifier(m):
        m.layers[-1].activation = tf.keras.activations.linear
        return m

    # Charger l'image
    image_PIL = load_img(image_path, target_size=(256, 256))
    image_PIL = img_to_array(image_PIL)
    image_PIL /= 255.
    image_PIL = np.expand_dims(image_PIL, axis=0)
    
    # Obtenir les prédictions du modèle
    predictions = model.predict(image_PIL)
    predicted_class = np.argmax(predictions, axis=1)

    # Loss function for the gradcam. It returns the output of the predicted class.
    def loss(output):
        return (output[0][predicted_class[0]])

    # Utiliser l'objet Gradcam pour obtenir la heatmap
    gradcam = Gradcam(model,
                      model_modifier=model_modifier)
    heatmap = gradcam(loss, image_PIL)

    # Convert the grayscale heatmap to jet color heatmap
    heatmap = cm.jet(heatmap[0])[..., :3]*255
    # Convert float32 to uint8
    heatmap = np.uint8(heatmap)
    
    axs[2].imshow(heatmap)
    axs[2].set_title("Deep Learning \n Grad-CAM \n (modèle Dropout)", weight='bold', color = 'white')
    axs[2].axis('off')
    
    plt.subplots_adjust(top=0.7, bottom=0.2)
    
    # Sauvegarder la figure sous forme de fichier .png
    fig.savefig("temp.png", dpi=300)

    # Fermer l'image pour libérer de la mémoire
    plt.close(fig)
    
    # Charger l'image sauvegardée
    image = cv2.imread("temp.png")

    # Convertir l'image de BGR à RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Supprimer le fichier temp.png
    os.remove("temp.png")

    # Return the image
    return image

#####################################################################

def run():

    st.title(title)
    st.header(sidebar_name)
    st.markdown("---")

    #Init de la liste qui contiendra les images de test
    new_rows = []
    #Parcours des images du répertoire de test
    for uneImage in os.listdir(pathTest):
        #Recherche des infos des images à partir du nom
        Saine, Plante, Maladie  = InfoNomImageTest(uneImage)
        #Création du nom complet de l'image
        nomPathImg = pathTest + uneImage
        #Création de la nouvelle liste
        new_row = {'nomPathImg' : nomPathImg, 'Image' : uneImage, 'Saine' : Saine, 'Plante' : Plante, 'Maladie' : Maladie}
        new_rows.append(new_row)
    #Création du DataFrame de test à partir des images chargées
    df_Test = pd.DataFrame(new_rows)
    lstImg = df_Test.Image.tolist()
    lstImg.insert(0,"Choisir...")

    #Affichage de la listebox et de l'image choisie en 2 colonnes
    #On coupe en 2 colonnes
    selectbox_disabled = False
    colI1, colI2 = st.columns([0.5,0.5])
    with colI1:
        st.subheader("Choix de l'image")
        img_sel = st.selectbox(label = "", options = lstImg, key='selectbox', disabled= selectbox_disabled)
        img_choisie = img_sel
        img_tel = st.file_uploader("", type=["jpg", "jpeg"])
        if img_tel is not None:
            img_sel = lstImg[0]
            img_choisie = img_tel
            selectbox_disabled = True
        else:
            selectbox_disabled = False
            
    with colI2:
        #Recherche de l'index dans df_Test de l'image chosiee
        if (img_sel != "Choisir..."):
            index_df = df_Test.loc[df_Test.Image == img_choisie].index[0]
            st.image(df_Test.loc[index_df].nomPathImg)
        else:
            if img_tel is not None:
                st.image(img_tel)
                
    #Chargement du tableau des modèles
    st.subheader("")
    st.subheader("Choix du modèle")
        
    df_Modele = pd.read_csv(".\\streamlit_app\\dataframes\\Modeles.csv",sep=';')
    df_Modele_Choix = st.data_editor(
        df_Modele,
        column_config={
            "Choix": st.column_config.CheckboxColumn(
                "",
                help="Choisissez un modèle",
                default=False,
            ),
            "Modele" : "Modèle",
            "NbClasses" : "Nombre de classes",
            "Type" : "Type de modèle",
            "TransLearn" : "Transfert Learning",
            "Path": None,
            "Nom": None
        },
        disabled=["Modele","NbClasses","Type","TransLearn","Accuracy"],
        hide_index=True,
    )
    df_Choix = df_Modele_Choix[df_Modele_Choix.Choix]

    if st.button("Lancer la prédiction "):
        #Vérification qu'une image est chargée
        if (img_sel == "Choisir...") and (img_tel is None):
            st.error('Il faut choisir une image', icon="🚨")
        elif (len(df_Choix) == 0):
            st.error('Il faut choisir au moins un modèle', icon="🚨")
        else:
            #Initialisation du tableau des prédictions
            df_Pred = pd.DataFrame(columns=['Nom','Taux de confiance','Etat','Plante','Maladie'])
            
            if img_tel is not None:
                #image téléchargée, on ne connait pas les valeurs réelles
                new_row = pd.DataFrame.from_dict({'Nom' : ["Valeurs réelles"],'Taux de confiance': "",'Etat':"Inconnu",'Plante':"Inconnue",'Maladie':"Inconnue"})
                #on va enregistrer l'image pour pouvoir ensuite avoir le path
                img = Image.open(img_tel)
                # Chemin du dossier temporaire
                rep_temp = "temp_images"
                os.makedirs(rep_temp, exist_ok=True)
                # Chemin complet pour le fichier temporaire
                path_temp = os.path.join(rep_temp, "image_telechargee.jpg")
                # Enregistrer l'image dans le dossier temporaire
                img.save(path_temp)
                path_img = path_temp
            else:
                #Recherche des valeurs réelles de l'image sélectionnée
                path_img = df_Test.loc[index_df].nomPathImg
                new_row = pd.DataFrame.from_dict({'Nom' : ["Valeurs réelles"],'Taux de confiance': "",'Etat':[df_Test.loc[index_df].Saine],'Plante':[df_Test.loc[index_df].Plante],'Maladie':[df_Test.loc[index_df].Maladie]})
            #Ajout des valeurs réelles dans le tableau final
            df_Pred = pd.concat([df_Pred, new_row], ignore_index=True)
        
            #Recherche des prédictions
            #Boucle sur le tableau des choix des modèles
            for i in range(len(df_Choix)):
                #Boucle sur les modèles sélectionnés
                
                #initialisation des variables
                modele_choisi = df_Choix.iloc[i].Nom
                #Recherche du type de modèle choisi
                if(df_Choix.iloc[i].Type == "Machine Learning"):
                    model = joblib.load(df_Choix.iloc[i].Path)
                    data = CalculHistImage(path_img)
                    pred = model.predict(data)[0]
                    taux = ""
                elif(modele_choisi == "DL eNetV2B3 2 classes"):
                    pred, taux = DL_2c_eNetV2B3_Predict(path_img, df_Choix.iloc[i].Path)
                elif(modele_choisi == "DL eNetV2S 14 classes"):
                    pred, taux = DL_14c_eNetV2S_Predict(path_img, df_Choix.iloc[i].Path)
                elif(modele_choisi == "DL VIT 38 classes"):
                    pred, taux = ViT_predict(path_img, df_Choix.iloc[i].Path)
                elif(modele_choisi == "DL VIT 38 classes"):
                    pred, taux = ViT_predict(path_img, df_Choix.iloc[i].Path)
                elif(modele_choisi == "DL LeNet 38 classes"):
                    pred, taux = LeNet_predict(path_img, df_Choix.iloc[i].Path)
                elif(modele_choisi == "DL DropOut 38 classes"):
                    pred, taux = Dropout_predict(path_img, df_Choix.iloc[i].Path)
                elif(modele_choisi == "DL 4Couches 38 classes"):
                    pred, taux = Quatre_couches_predict(path_img, df_Choix.iloc[i].Path)
                elif(modele_choisi == "DL VGG16 38 classes"):
                    pred, taux = VGG16_predict(path_img, df_Choix.iloc[i].Path)
                elif(modele_choisi == "DL Inception 38 classes"):
                    pred, taux = Inception_predict(path_img, df_Choix.iloc[i].Path)
                
                if (taux == ""):
                    strTaux=""
                else:
                    #Mise en forme du taux
                    strTaux = "{:.3f}".format(taux)
                
                #Mise en forme du résultat pour préparer l'affichage
                if(df_Choix.iloc[i].NbClasses == "2 classes"):
                    #Conversion de la valeur prédite en saine ou malade
                    if (pred == 0):
                      pred_Etat = 'Malade'
                    elif (pred == 1):
                      pred_Etat = 'Saine'
                    else:
                      pred_Etat = 'Inconnu'
                    #si prédiction 2 classes alors on n'a pas les prédictions du type de plante ou de la maladie
                    pred_Plante = ""
                    pred_Maladie = ""
                if(df_Choix.iloc[i].NbClasses == "14 classes"):
                    #si prédiction 14 classes alors on n'a pas les prédictions de l'état ou de la maladie
                    pred_Plante = ValeursPred14(pred)
                    pred_Etat = ""
                    pred_Maladie = ""
                if(df_Choix.iloc[i].NbClasses == "38 classes"):
                    #Appel de la fonction permettant la mise en place des valeurs
                    pred_Etat, pred_Plante, pred_Maladie = ValeursPred38(pred)
                
                # construction de la nouvelle ligne du dataFrame des prédictions
                new_row = pd.DataFrame.from_dict({'Nom' : [modele_choisi],'Taux de confiance':[strTaux], 'Etat':[pred_Etat],'Plante':[pred_Plante],'Maladie':[pred_Maladie]})
                
                #Ajout des valeurs pédites dans le tableau final
                df_Pred = pd.concat([df_Pred, new_row], ignore_index=True)
                
            #Affichage
            #Affichage du tableau des prédictions
            st.dataframe(df_Pred)
            #Affichage des images
            model_dropout2 = models.load_model(".\\streamlit_app\\Modeles\\model_avec_dropout_V2.h5")
            st.image(affiche_hist_gradcam(path_img, model_dropout2))
