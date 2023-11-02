import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns


st.title('Projet AB testing')
st.subheader ( 'Par Blandine JATSA NGUETSE')



# Menu de navigation
st.sidebar.title("Menu de Navigation")
selected_option = st.sidebar.radio("Plan de travail", ["Presentation du jeu de données", "Vue d'ensemble des données" , "Analyses des variables continues ( analyse bivariée)", "Reste de l'analyse + Regression logistique"])

if selected_option == "Presentation du jeu de données":
    # Afficher le contenu de l'analyse de données
    #Bouton pour afficher le paragraphe
    afficher_paragraphe = st.checkbox("Objectifs")

    # Vérifier si le bouton a été cliqué
    if afficher_paragraphe:
        # Titre du paragraphe
        #st.expander("Comprehension de la problématique")
        # Contenu du paragraphe
        st.write(" Lorsqu'une banque prete de l'argent à un client, elle prend le risque que cet emprunteur ne rembourse pas cet argent dans le délai convenu. Ce risque est appelé Risque de Crédit. Il est nécessaire pour les institutions  financieres de se protéger contre ce risque, en anticipant la dessus et cela en utilisant des outils de modélisation qui prennent en compte plusieurs paramamètres  notamment,  l'historique de l'emprunteur et son profil financier .")
