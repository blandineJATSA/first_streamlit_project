import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px
import streamlit.components.v1 as components
from scipy.stats import kruskal
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector




import datetime

credit = pd.read_csv('credit_scoring.csv')
credit_copy = credit.copy()
varquali =["Comptes", "Historique_credit", "Objet_credit", "Epargne", "Anciennete_emploi", 
              "Situation_familiale", "Garanties", "Biens", "Autres_credit", "Statut_domicile", 
                "Type_emploi", "Nb_pers_charge" , "Telephone" ]

varquanti =["Duree_credit", "Montant_credit", "Taux_effort", "Age","Nb_credits", "Anciennete_domicile"]
vars = varquali + varquanti

credit_copy['Cible'] = credit_copy['Cible'].replace({1:0, 2:1})
credit_copy['Cible']= credit_copy['Cible'].astype('object')
#def main():
st.title('Projet de machine learning : Risque de credit')
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

    afficher_paragraphe2 = st.checkbox("Jeu de données")
    if afficher_paragraphe2:
        st.markdown("""
    Le jeu de données que nous allons utiliser est connu sous le nom "German Credit data". Il contient 1000 dossiers de crédit à la consommation, dont 300 avec des impayés. 
Il se compose de la variable cible ("Cible") qui est bouleen avec la modalité 1 pour la l'abscence d'impayés et 2 pour la presence d'impayés; puis : 

- Trois variables quatitatives continues qui portent sur le credit démandé à savoir: la durée du credit démandés en mois, le montant du credit en euros et l'age du demandeur en années.

- Six variables quantitatives discretes qui portent sur le profil financier du demandeur de credit : Le solde moyen  sur compte courant, l'encours d'épargne, le nombre de credit deja detenu dans la banque, , le taux d'effort ou d'endettement, l'ancienneté au domicile, le nombre de personnes en charge

- Dix variables qualitaties qui portent sur le profil fiancier et personnel du demandeur:  l'objet du credit, l'historique de remboursemment du demandeur, les autres credits detenu ( hors de la banque), les biens de valeur detenus par le demandeur, ses garanties, sa situation familiale, son ancienneté dans l'emploi, son statut au domicile, son type d'emploi occupé, et la fourniture d'un numero de telephone.

""")

    data_head = st.checkbox("afficher les  5 premières lignes de la base de données ")
    if data_head :
        st.dataframe(credit.head(5))
        # Affichez la dimension de la base de données
        num_rows, num_columns = credit.shape
        st.write(f"La base de données a {num_rows} lignes et {num_columns} colonnes.")
    
    
    
    preparation_données = st.checkbox(" Préparation des données ")
    if preparation_données : 
        st.write("Variables Qualitatives :", varquali)
        st.write("Variables Quantitatives :", varquanti)
        # Utilisez la fonction 'value_counts' pour calculer les fréquences
        counts = credit_copy["Cible"].value_counts()

        # Affichez les fréquences dans Streamlit
        st.write("Table des fréquences de la variable cible :")
        st.write(counts)
    
     
elif selected_option == "Vue d'ensemble des données": 
    st.write("""Dans cette section, nous explorons nos données avec des statistiques et des graphiques avant l'étapes de la modélisation afin 

- de se familiariser avec ces données
- de s'assurer de l'absence d'anomalie
- de mesurer la liason des variables explicatives entre elles et avec la varible cible

En bref, d'examiner tous les points suceptibles de devoir etre prise en compte avant après et pendant la modelisation.""")
        
    st.write(credit_copy.describe(include ="all"))
    st.write("Valeurs manquantes :")
    st.write(credit_copy.isnull().sum())
    # Sélection de la variable à afficher avec une liste déroulante
    selected_variable = st.selectbox("Sélectionnez la variable à afficher :", varquanti + varquali)
    st.title(f'Graphique interactif de {selected_variable}')
    if selected_variable in varquanti:
        fig = px.histogram(credit_copy, x=selected_variable, title=f'Histogramme de {selected_variable}')
    else:
        fig = px.histogram(credit_copy, x=selected_variable, title=f'Histogramme de {selected_variable}', color='Cible')
    st.plotly_chart(fig)

    
elif selected_option == "Analyses des variables continues ( analyse bivariée)":
    
    st.write("""Au travers des graphiques nous avons analyser les variables continues afin  de detecter d'enventuelle anomalie et aussi la liaison avec la variable cible.

Nous avons trois variables continues : le Montant_credit,  Age, Duree_credit""")
    
    # Créez une figure seaborn
    sns.set_style("whitegrid")
    g = sns.displot(credit_copy, x="Duree_credit", col="Cible", kde=True)

    # Utilisez Streamlit pour afficher la figure
    st.title("Distribution de Duree_credit par Cible")
    #st.write("Distribution de la variable Duree_credit en fonction de la variable Cible.")
    st.pyplot(g) 
    st.write("On constate assez nettement la plus forte proportion de credits plus longs parmi ceux qui ont des impayés, particulièrement les crédits de 48 mois et plus.")
    
    g1 = sns.displot(credit_copy, x="Montant_credit", col="Cible", kde=True)

    # Utilisez Streamlit pour afficher la figure
    st.title("Distribution de Montant_credit par Cible")
    #st.write("Distribution de la variable Duree_credit en fonction de la variable Cible.")
    st.pyplot(g1) 
    st.write("""Les credits de très petit montant sont rares ( le min est de 250 euros et 95% dépassent 700 euros). Ensuite, on atteint une fraquence maximale autour de 1200 euros, qui ne fait ensuite que decroite. On constate la plus forte proportion de montant plus élevés parmi les crédits qui ont des impayés. 
La distribution ne presente pas d'anomalie apparente.""")
    
    g2 = sns.displot(credit_copy, x="Age", col="Cible", kde=True)

    # Utilisez Streamlit pour afficher la figure
    st.title("Distribution de l'Age par Cible")
    #st.write("Distribution de la variable Duree_credit en fonction de la variable Cible.")
    st.pyplot(g2) 
    st.write(""" L'age des contractants est compris entre 19 et 75 ans. Il a une distribution plus homogène pour les crédits sans impayé que pour les autres. Les impayés concernent majoritairement les contractants de moins de 35 ans lors de l'octroi de credit. Passé cet age, les histogrammes des bons et mauvais dossiers se ressemblent plus.""")
    
    st.title(" test de kruskal-Wallis et Chi-deux normalisé de kruskal-Wallis ")
    st.write(""" Nous voyons que les trois variables continues presentent chacune une liaison avec la variable cible, que nous pouvons mesurer  l'aide d'un test.
Nous choisisons le test de kruskal-walis car c'est un test non-paramétrique; ce type de test permet de s'affranchir des hypothèses de normalités et d'homoscédaticité habituelles dans les tests paramétriques.

test kruskal-wallis:

- avantages : lecture simple, grande généralité d'application

- inconvegnients : sa dependance à l'effectif : plus la population est importante plus le Chi-deux de kruska-wallis est élevé.

- Une solution : L'utilisation du chi-deux normalisé à l'instar du V de Cramer""")
    
    # Créez une liste déroulante pour sélectionner la variable
    variable_choice = st.selectbox("Sélectionnez la variable à analyser :", ['Montant_credit', 'Duree_credit', 'Age'])

    # Filtrer les données en fonction de la variable sélectionnée
    groupes = [credit_copy[credit_copy['Cible'] == 1][variable_choice], credit_copy[credit_copy['Cible'] == 0][variable_choice]]

    # Effectuez le test de Kruskal-Wallis
    statistique, p_valeur = kruskal(*groupes)

    # Affichez les résultats
    #st.title(" Test de Kruskal-Wallis")
    st.write("Variable analysée :", variable_choice)
    st.write("Statistique de test de Kruskal-Wallis :", statistique)
    st.write("P-valeur :", p_valeur)

    # Interprétation des résultats
    alpha = 0.05  # Niveau de signification
    if p_valeur < alpha:
        st.write("Il y a une différence significative entre les groupes.")
    else:
        st.write("Il n'y a pas de différence significative entre les groupes.")

    somme_non_na = np.sum(~np.isnan(credit_copy[variable_choice]))

    # Calculez la racine carrée de la statistique de Kruskal-Wallis divisée par la somme des valeurs non manquantes
    resultat = np.sqrt(statistique / somme_non_na)

    st.write("Résultat :", resultat)
    
    st.title(" Discretisation des variables continues")
    
elif selected_option == "Reste de l'analyse + Regression logistique":
    
    # Age
    credit_copy['Age_Deciles'] = pd.qcut(credit_copy['Age'], q=10)
    contingency_table = pd.crosstab(credit_copy['Age_Deciles'], credit_copy['Cible'])
    # Calculez les proportions par ligne
    row_proportions = contingency_table.div(contingency_table.sum(axis=1), axis=0)
    interval1 = ["(18.999, 23.0]",   "(23.0, 26.0]",   "(26.0, 28.0]",
                    "(28.0, 30.0]",   "(30.0, 33.0]",   "(33.0, 36.0]",
                    "(36.0, 39.0]",   "(39.0, 45.0]",   "(45.0, 52.0]",
                    "(52.0, 75.0]"]
    
    taux_impayes = row_proportions.iloc[:, 1]
    plt.bar(interval1, taux_impayes)
    plt.ylim(0, 0.5)
    plt.title("Age")
    plt.ylabel("Taux impayés")
    plt.xticks(rotation='vertical')
    plt.axhline(y=0.3, color='red', linestyle='--')
    plt.show()
    # st.write(""" Sur la figure se distinguent très nettement les deux premiers déciles, dont le taux d'impayés est sensiblement superieur à la moy. En revanche aucune autre tendance ne se dessine ausi fortement dans les autres déciles. Nous decoupons donc l'age en deux tranches : les moins de 25 ans ( plutot que 26 ans ) et les plus de 25ans."""
    
    credit_copy['qAge']  = pd.cut(credit_copy['Age'], bins=[0, 25, float('inf')])

    contingency_table = pd.crosstab(credit_copy['qAge'], credit_copy['Cible'])

    # Calculez les proportions par ligne
    row_proportions = contingency_table.div(contingency_table.sum(axis=1), axis=0)
    credit_copy['qAge']  = pd.cut(credit_copy['Age'], bins=[0, 25, float('inf')])
    
    q = credit_copy['Duree_credit'].quantile([i/20 for i in range(21)])

    Duree_credit = pd.cut(credit_copy['Duree_credit'], bins=q.unique())

    contingency_table = pd.crosstab(Duree_credit, credit_copy['Cible'])

    # Calculez les proportions par ligne
    row_proportions = contingency_table.div(contingency_table.sum(axis=1), axis=0)
    
    interval = [  "(4.0, 6.0]",   "(6.0, 9.0]", " (9.0, 10.0]", "(10.0, 12.0]",
                  "(12.0, 15.0]"," (15.0, 18.0]", "(18.0, 20.0]", "(20.0, 24.0]",
                  "(24.0, 30.0]", "(30.0, 36.0]"," (36.0, 48.0]", "(48.0, 72.0]"]

    taux_impayes = row_proportions.iloc[:, 1]
    plt.bar(interval, taux_impayes)
    plt.ylim(0, 0.6)
    plt.title("Duree_credit")
    plt.ylabel("Taux impayés")
    plt.xticks(rotation='vertical')
    plt.axhline(y=0.3, color='red', linestyle='--')
    plt.show()
    
    credit_copy['qDuree_credit']  = pd.cut(credit_copy['Duree_credit'], bins=[0, 15, 36, float('inf')])

    contingency_table = pd.crosstab(credit_copy['qDuree_credit'], credit_copy['Cible'])

    # Calculez les proportions par ligne
    row_proportions = contingency_table.div(contingency_table.sum(axis=1), axis=0)
    
    
    q= credit_copy['Montant_credit'].quantile([i/20 for i in range(21)])

    Montant_credit = pd.cut(credit_copy['Montant_credit'], bins=q.unique())

    contingency_table = pd.crosstab(Montant_credit, credit_copy['Cible'])

    # Calculez les proportions par ligne
    row_proportions = contingency_table.div(contingency_table.sum(axis=1), axis=0)
    
    
    interval = [ "(250.0, 708.95]",   "(708.95, 932.0]",  "(932.0, 1157.55]",
                  "(1157.55, 1262.0]",  "(1262.0, 1365.5]",  "(1365.5, 1479.4]",
                 " (1479.4, 1602.65]", "(1602.65, 1906.8]", "(1906.8, 2100.55]",
                  "(2100.55, 2319.5]",  "(2319.5, 2578.0]",  "(2578.0, 2852.4]",
                   "(2852.4, 3187.4]",  "(3187.4, 3590.0]", "(3590.0, 3972.25]",
                 " (3972.25, 4720.0]", "(4720.0, 5969.95]", "(5969.95, 7179.4]",
                   "(7179.4, 9162.7]", "(9162.7, 18424.0]"]

    taux_impayes = row_proportions.iloc[:, 1]
    plt.bar(interval, taux_impayes)
    plt.ylim(0, 0.6)
    plt.title("Montant_credit")
    plt.ylabel("Taux impayés")
    plt.xticks(rotation='vertical')
    plt.axhline(y=0.3, color='red', linestyle='--')
    plt.show()
    
    credit_copy['qMontant_credit']  = pd.cut(credit_copy['Montant_credit'], bins=[0, 4000, float('inf')])

    contingency_table = pd.crosstab(credit_copy['qMontant_credit'], credit_copy['Cible'])

    # Calculez les proportions par ligne
    row_proportions = contingency_table.div(contingency_table.sum(axis=1), axis=0)
    
    colonnes_a_exclure = ['Duree_credit', 'Montant_credit', 'Age', 'Cible', 'Cle', 'Age_Deciles']

    credit2 = credit_copy.drop(colonnes_a_exclure, axis=1)
    
    
    
    # Créez un dictionnaire de remplacement des modalités
    replacements = {
    'A14': 'Pas de compte',
    'A11': 'CC < 0 euros',
    'A12': 'CC [0-200 euros[',
    'A13': 'CC > 200 euros'
}

    credit2['Comptes'] = credit2['Comptes'].replace(replacements)
    
    credit2['Historique_credit'] = np.where(credit2['Historique_credit'] == 'A30', 'Impayés passés',
    np.where(credit2['Historique_credit'] == 'A31', 'Impayé en cours dans autre banque',
    np.where(credit2['Historique_credit'].isin(['A32', 'A33']), 'Pas de crédits ou en cours sans retard', 'Crédits passés sans retard')
    ))
    
    credit2['Objet_credit'] = np.where(credit2['Objet_credit'] == 'A40', 'Voiture neuve',
    np.where(credit2['Objet_credit'] == 'A41', 'Voiture occasion',
    np.where(credit2['Objet_credit'].isin(['A42', 'A44', 'A45']), 'Interieur', np.where(credit2['Objet_credit'] == 'A43', 'Video-HIFI', np.where(credit2['Objet_credit'].isin(['A46', 'A48']) , 'Etudes',  np.where(credit2['Objet_credit'] == 'A47', 'Vacances', np.where(credit2['Objet_credit'] == 'A49', 'Business','Autres')
    ))))))
    
    credit2['Epargne'] = np.where(credit2['Epargne'] == 'A65', 'Sans épargne',
    np.where(credit2['Epargne'].isin(['A61', 'A62']), '< 500 euros', '> 500 euros'))
    
    credit2['Anciennete_emploi'] = np.where(credit2['Anciennete_emploi'] == 'A73', 'entre 1 et 4 ans',
    np.where(credit2['Anciennete_emploi'].isin(['A71', 'A72']), 'Sans emploi ou < 1 an', 'depuis au moins  4 ans'))
    
    credit2['Situation_familiale'] = np.where(credit2['Situation_familiale'] == 'A91', 'Homme divorcé/séparé',
    np.where(credit2['Situation_familiale'] == 'A92', 'Femme divorcée/séparée/mariée',
    np.where(credit2['Situation_familiale'].isin(['A93', 'A94']), 'Homme céliataire/marié/veuf', 'Femme célibataire')))
    
    credit2['Garanties'] = np.where(credit2['Garanties'] == 'A103', 'Avec garant', 'sans garant')
    
    credit2['Biens'] = np.where(credit2['Biens'] == 'A121', 'Immobilier', np.where(credit2['Biens'] == 'A124', 'Aucun bien', 'Non immobilier'))
    
    credit2['Autres_credit'] = np.where(credit2['Autres_credit'] == 'A143', 'Aucun credit exterieur', 'Crédits exterieurs')

    credit2['Statut_domicile'] = np.where(credit2['Statut_domicile'] == 'A152' , 'Propriétaire', 'Non propriétaire')
    
    
    st.title("La regression logistique")
    
    st.write(""" 
Avant de passer aux arbres de decision, aux méthodes d'aggregation et aux autres méthodes de modélisation, nous commencons par chercher un modèle logistique qui nous permettra d'obtenir une première grille de score et d'avoir un point de référence en termes de va discriminantes et d'aire sous la courbe ROC, pour les autres modèles de score.""")
    
    # Renommer les colonnes age, montant_credit et Duree_credit
    credit2.rename(columns={
    'qAge' : 'Age',
    'qDuree_credit' : 'Duree_credit' ,
    'qMontant_credit': 'Montant_credit'
}, inplace=True)
    
    credit2['Age']  = credit2['Age'].astype(object)
    credit2['Duree_credit'] = credit2['Duree_credit'].astype(object)
    credit2['Montant_credit'] = credit2['Montant_credit'].astype(object)
    credit2['Cible'] = credit_copy['Cible'] 
    
    # Creations des échantillons d'apprentissage et de test

    # Sélectionner les variables explicatives et la variable d'intérêt
    X = credit2[['Comptes', 'Historique_credit', 'Objet_credit', 'Epargne',
       'Anciennete_emploi', 'Taux_effort', 'Situation_familiale', 'Garanties',
       'Anciennete_domicile', 'Biens', 'Autres_credit', 'Statut_domicile',
       'Nb_credits', 'Type_emploi', 'Nb_pers_charge', 'Telephone', 'Age',
       'Duree_credit', 'Montant_credit']]

    y = credit2['Cible']
    # Convertir les variables catégorielles en variables indicatrices (dummies)
    X = pd.get_dummies(X, columns=X.columns, drop_first=True)

    import statsmodels.api as sm
    # Ajouter une constante pour l'interception
    X = sm.add_constant(X)
    
    # Diviser les données en ensembles d'apprentissage et de test

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.fit_transform(y_test)

    # Créez un objet de modèle de régression logistique
    lr = LogisticRegression()

    # Créez un objet de sélection de caractéristiques séquentielles
    sfs = SequentialFeatureSelector(lr, n_features_to_select="auto", direction="backward", scoring="roc_auc")
    # Ajustez le modèle pour sélectionner les caractéristiques
    sfs = sfs.fit(X_train, y_train_encoded)
    
    # Créer le modèle de régression logistique
    logit_model = sm.Logit(y_train_encoded, X_train)
    # Ajuster le modèle aux données d'apprentissage
    result = logit_model.fit()
    # Afficher le résumé des résultats de la régression
    st.text(result.summary())

    # Obtenez les prédictions du modèle sur l'ensemble d'entraînement et de test
    y_train_pred = result.predict(X_train)
    y_test_pred = result.predict(X_test)

    # Calculez les courbes ROC et les aires sous la courbe (AUC)
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train_encoded, y_train_pred)
    roc_auc_train = auc(fpr_train, tpr_train)

    fpr_test, tpr_test, thresholds_test = roc_curve(y_test_encoded, y_test_pred)
    roc_auc_test = auc(fpr_test, tpr_test)

    # Créez un graphique avec les courbes ROC
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(fpr_train, tpr_train, color='darkorange', lw=2, label='ROC curve (train) (AUC = %0.2f)' % roc_auc_train)
    ax.plot(fpr_test, tpr_test, color='green', lw=2, label='ROC curve (test) (AUC = %0.2f)' % roc_auc_test)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Taux de faux positifs')
    ax.set_ylabel('Taux de vrais positifs')
    ax.set_title('Courbes ROC')
    ax.legend(loc="lower right")

    # Affichez le graphique dans Streamlit
    st.pyplot(fig)

    
    

    
    
    


    # Afficher le contenu de l'analyse bivariée
    #st.write("Contenu de l'Analyse Bivariée")
else:
    # Autres sections si nécessaire
    pass

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# Obtenir la date actuelle
aujourd_hui = datetime.date.today()
    
# Afficher la date dans votre application Streamlit
#st.write("Date du jour : ", aujourd_hui)
    
# Définir le style CSS pour positionner la date en bas à droite
style = """
<style>
    .bottom-right {
        position: absolute;
        bottom: 10px;
        right: 10px;
        color: #888888;
    }
</style>
"""

# Afficher la date dans votre application Streamlit avec le style personnalisé
st.write(style, unsafe_allow_html=True)
st.markdown(f'<div class="bottom-right">Date du jour : {aujourd_hui}</div>', unsafe_allow_html=True)

   
    
    
    
#if __name__=='__main__':
 #       main()