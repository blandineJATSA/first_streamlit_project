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
selected_option = st.sidebar.radio("creer un A/B test", ["Concevoir notre A/B testing", "Vue d'ensemble des données" , "Analyses des variables continues ( analyse bivariée)", "Reste de l'analyse + Regression logistique"])

if selected_option == "Concevoir notre A/B testing":
    st.write("Ce projet explique comment concevoir, exécuter et interpréter un A/B test en vue d'améliorer les taux "
             "de conversion d'une page de site Web. Cette méthode est applicable à tout type d'activités marketing.")

    st.write("- Mise en situation et objectifs")
    st.write(" En tant que data scientist, nous sommes solliciter par l'équipe produit. Le concepteur a travaillé très"
             " dur sur une nouvelle version de la page produit, dans l'espoir qu'elle entraînera un taux de conversion "
             "plus élevé. Le taux de conversion actuel est d'environ 13 % en moyenne sur l'ensemble de l'année,"
             " l'équipe aimerait une augmentation de 2 % , ce qui signifie que le nouveau design sera considéré comme "
             "un succès s'il augmente le taux de conversion à 15%."
             "Notre objectif est de tester sur un petit nombre d'utilisateurs pour voir ses performances avant de"
             " déployer le changement."
             " Nous suggérons donc d'exécuter un test A/B.")
