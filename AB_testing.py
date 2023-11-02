import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.stats.api as sms
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil


st.title('Projet AB testing')
st.subheader ( 'Par Blandine JATSA NGUETSE')


# Some plot styling preferences
#plt.style.use('seaborn-whitegrid')
font = {'family' : 'Helvetica', 'weight' : 'bold','size'   : 14}

mpl.rc('font', **font)
effect_size = sms.proportion_effectsize(0.13, 0.15)    # Calculating effect size based on our expected rates

required_n = sms.NormalIndPower().solve_power(
    effect_size,
    power=0.8,
    alpha=0.05,
    ratio=1
    )               # Calculating sample size needed

required_n = ceil(required_n)    # Rounding up to next whole number
AB_test = pd.read_csv('ab_data.csv')

# Menu de navigation
st.sidebar.title("Menu de Navigation")
selected_option = st.sidebar.radio("creer un A/B test", ["Concevoir notre A/B testing", "Collecte et préparation des données" , "Analyses des variables continues ( analyse bivariée)", "Reste de l'analyse + Regression logistique"])

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
    st.write(" Pour notre test nous aurons besoin de deux groupes: Un groupe de referent dont on leur montrera l'ancien"
             "design et un groupe test dont on leur montrera le nouveau design. L'échantillon que nous décidons de"
             " capturer dans chaque groupe aura un effet sur la précision de nos taux de conversion estimés :"
             " plus la taille de l'échantillon est grande, plus nos estimations sont précises (c'est-à-dire plus nos "
             "intervalles de confiance sont petits), plus les chances de détecter une différence sont élevéesdans les"
             " deux groupes, s'il est présent.")
    st.write("D’un autre côté, plus notre échantillon est grand, plus notre "
             "étude devient coûteuse (et peu pratique). La taille de l'échantillon dont nous avons besoin est "
             "estimée par l'analyse de puissance , et elle dépend de quelques facteurs :")
    st.markdown(" - **Puissance du test :** La probabilité de trouver une différence statistique entre les groupes de notre"
                "test lorsqu'une différence est réellement présente. Ceci est généralement fixé à 0,8 par convention ")
    st.markdown("- **Valeur alpha ($\\alpha$):** La valeur critique que nous avons fixée précédemment à 0,05")
    st.markdown("- **Taille de l'effet:** Quelle différence nous prévoyons qu'il y ait entre les taux de conversion : "
                "Puisque notre équipe se contenterait d’une différence de 2 %, nous pouvons utiliser 13 % et 15 % pour "
                "calculer la taille de l’effet attendue.")

    st.write(" Nous aurions besoin d'au moins ", required_n, "observations pour chaque groupe:")

elif selected_option == "Collecte et préparation des données":
    data_head = st.checkbox("afficher les  5 premières lignes de la base de données ")
    if data_head:
        st.dataframe(AB_test.head(5))
        # Affichez la dimension de la base de données
        num_rows, num_columns = AB_test.shape
        st.write(f"La base de données a {num_rows} lignes et {num_columns} colonnes.")
        #st. write(AB_test.infos())
        st.write("Nous n'utiliserons en fait que les colonnes **group** et **converted** pour l'analyse.")
        st.write("Tableau croisé entre 'group' et 'landing_page' :")
        st.write(pd.crosstab(AB_test['group'], AB_test['landing_page']))

    preparation_données = st.checkbox(" Préparation des données ")
    if preparation_données:
        session_counts = AB_test['user_id'].value_counts(ascending=False)
        multi_users = session_counts[session_counts > 1].count()

        st.write(f"Il y a {multi_users} utilisateurs qui apparaissent plusieurs fois dans l'ensemble de données")
        users_to_drop = session_counts[session_counts > 1].index

        df = AB_test[~AB_test['user_id'].isin(users_to_drop)]
        st.write("L'ensemble de données mis à jour contient désormais",  df.shape[0], "entrées")

        st.write("  **Échantillonnage**")
        st.write("Nous effectuons un échantillonnage aléatoire simple ")
        control_sample = df[df['group'] == 'control'].sample(n=required_n, random_state=22)
        traitement_sample = df[df['group'] == 'treatment'].sample(n=required_n, random_state=22)

        ab_test = pd.concat([control_sample, traitement_sample], axis=0)
        ab_test.reset_index(drop=True, inplace=True)
        #st.write(ab_test.head())

        st.write("  **Visualiser les résultats**")
        st.write("La première chose que nous pouvons faire est de calculer quelques statistiques de base pour avoir"
                 " une idée de ce à quoi ressemblent nos échantillons.")

        conversion_rates = ab_test.groupby('group')['converted']

        std_p = lambda x: np.std(x, ddof=0)  # Std. écart de la proportion
        se_p = lambda x: stats.sem(x, ddof=0)  # Std. erreur de proportion (std / sqrt(n))

        conversion_rates = conversion_rates.agg([np.mean, std_p, se_p])
        conversion_rates.columns = ['conversion_rate', 'std_deviation', 'std_error']

        st.write(conversion_rates.style.format ('{:.3f}'))

        st.write("À en juger par les statistiques ci-dessus, il semble que nos deux conceptions aient fonctionné de "
                 "manière très similaire , notre nouvelle conception étant légèrement meilleure, env. Taux de "
                 "conversion de 12,3 % contre 12,6 % .")

        st.write("Le tracé des données rendra ces résultats plus faciles à comprendre :")

        # Créer un graphique à l'aide de Seaborn
        plt.figure(figsize=(8, 6))
        sns.barplot(x=ab_test['group'], y=ab_test['converted'], ci=False)
        plt.ylim(0, 0.17)
        plt.title('Taux de conversion par groupe', pad=20)
        plt.xlabel('Groupe', labelpad=15)
        plt.ylabel('Converti (proportion)', labelpad=15)

        # Afficher le graphique avec Streamlit
        st.pyplot(plt)

        st.write("Les taux de conversion pour nos groupes sont en effet très proches. Notez également que le taux de"
                 " conversion du controlgroupe est inférieur à ce à quoi nous nous attendions compte tenu de ce que "
                 "nous connaissions de notre moyenne. taux de conversion (12,3% contre 13%). Cela montre qu’il existe"
                 " une certaine variation dans les résultats lors de l’échantillonnage d’une population.")
        st.write("Donc la treatment valeur du groupe est plus élevée. Cette différence est-elle statistiquement"
                 " significative ?")

        st.write("- Tester l'hypothèse")
        st.write("La dernière étape de notre analyse consiste à tester notre hypothèse. Puisque nous disposons d'un "
                 "très grand échantillon, nous pouvons utiliser l' approximation normale pour calculer notre valeur "
                 "p (c'est-à-dire le test z).")

        from statsmodels.stats.proportion import proportions_ztest, proportion_confint

        control_results = ab_test[ab_test['group'] == 'control']['converted']
        treatment_results = ab_test[ab_test['group'] == 'treatment']['converted']
        n_con = control_results.count()
        n_treat = treatment_results.count()
        successes = [control_results.sum(), treatment_results.sum()]
        nobs = [n_con, n_treat]

        z_stat, pval = proportions_ztest(successes, nobs=nobs)
        (lower_con, lower_treat), (upper_con, upper_treat) = proportion_confint(successes, nobs=nobs, alpha=0.05)

        st.write(f"z statistic: {z_stat:.2f}")
        st.write(f'p-value: {pval:.3f}')
        st.write(f'ci 95% for control group: [{lower_con:.3f}, {upper_con:.3f}]')
        st.write(f'ci 95% for treatment group: [{lower_treat:.3f}, {upper_treat:.3f}]')








