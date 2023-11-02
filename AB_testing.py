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

    st.markdown("- **Mise en situation et objectifs**")
    st.write(" En tant que data scientist, nous sommes solliciter par l'équipe produit. Le concepteur de cette équipe a "
             "travaillé très dur sur une nouvelle version de la page produit, dans l'espoir qu'elle entraînera un taux "
             "de conversion "
             "plus élevé. Le taux de conversion actuel est d'environ 13 % en moyenne sur l'ensemble de l'année,"
             " l'équipe aimerait une augmentation de 2 % , ce qui signifie que le nouveau design sera considéré comme "
             "un succès s'il augmente le taux de conversion à 15%."
             "Notre objectif est de tester sur un petit nombre d'utilisateurs pour voir ses performances avant de"
             " déployer le changement."
             " Nous suggérons donc d'exécuter un test A/B.")

    st.markdown("- **Objectif du test et indicateur à mesurer**")
    st.write("L'exemple d'A/B testing développé dans ce projet porte sur **le design de la page produit**. L'objectif"
             " dans notre cas d'étude est d'observer comment le nouveau design de la page produit influence "
             "le taux de conversion d'un visiteur à lead. Ainsi L'indicateur  mesuré est **le taux de conversion**. "
             "la seule variable testée est **le design**, le texte et les couleurs restant identiques. "
             "En effet, ce test vise à mesurer l'effet du design sur le taux de conversion"
             )

    st.markdown("- **Hypothèses du test**")
    st.write("Etant donné que nous ne savons pas si notre nouveau design influence mieux ou moins le taux de conversion"
             "par rapport à l'ancien, nous choisissons un test bilatéral:")

    st.markdown("$$H_0 : P=P_0 $$")
    st.markdown("$$H_a : P \\neq P_0$$")
    st.write(" où $P$ et $P_0$ représentent respectivement le taux de conversion du nouveau et de l'ancien design. "
             "Nous fixerons également un niveau de confiance de 95 %  soit $\\alpha = 0,05$.")

    st.markdown("- **Choisir une taille d'échantillon**")
    st.write(" Pour notre test nous aurons besoin de deux groupes: Un groupe de referent dont on leur montrera l'ancien "
             "design et un groupe test dont on leur montrera le nouveau design. $<br>$ L'échantillon que nous décidons de"
             " capturer dans chaque groupe aura un effet sur la précision de nos taux de conversion estimés :"
             " plus la taille de l'échantillon est grande, plus nos estimations sont précises (c'est-à-dire plus nos "
             "intervalles de confiance sont petits), plus les chances de détecter une différence sont élevéesdans les"
             " deux groupes, s'il est présent. <br> D’un autre côté, plus notre échantillon est grand, plus notre "
             "étude devient coûteuse (et peu pratique). <br> La taille de l'échantillon dont nous avons besoin est "
             "estimée par quelque chose appelé analyse de puissance , et elle dépend de quelques facteurs :" )

