#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 14:34:03 2023

@author: raph
"""

import streamlit as st
import geopandas as gpd
import pandas as pd
import xlrd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split



# Fonction pour la régression gamma
def gamma_regression(X, y):
    X = sm.add_constant(X)  # Ajoutez une colonne constante
    model = sm.GLM(y, X, family=sm.families.Gaussian())
    results = model.fit()
    return results


# Fonction pour la visualisation des données
def visualize_data():
    
    # Ajouter un grand titre
    st.title("Étude démographique et économique du monde")

    # Description de l'application
    st.write(
        "Bienvenue dans notre application d'étude démographique et économique du monde. "
        "Cette application vous permet d'explorer les données démographiques et économiques de différents pays à l'aide de cartes interactives. "
        "Vous pouvez choisir parmi plusieurs variables telles que la population totale, le PIB, l'utilisation d'Internet, "
        "l'espérance de vie et les exportations de biens et services. "
        "De plus, vous avez la possibilité d'afficher ces données en échelle logarithmique pour une meilleure visualisation. "
        "Sélectionnez un onglet dans la barre latérale pour explorer les données ou le modèle et la prédiction."
    )

    # Saut de ligne
    st.write("")

    # Saut de ligne
    st.write("")
    
    # Ajoutez votre code de visualisation des données ici
    # Options pour le menu déroulant
    options = [
        "Population, total",
        "GDP, PPP (current international $)",
        "Internet users (per 100 people)",
        "2014 Life expectancy at birth, total (years)",
        "Exports of goods and services (% of GDP)"
    ]

    # Barre latérale à gauche pour la sélection de la variable
    st.sidebar.subheader("Sélectionnez une variable :")
    selected_variable = st.sidebar.selectbox("Variable", options)

    # Filtrer les données pour exclure les lignes avec des valeurs NULL
    filtered_data = data_df.dropna(subset=[selected_variable])

    # Ajouter un subheader pour la carte du monde
    st.subheader(f"Représentation géographique de la variable {selected_variable}")
    
    st.write("Explorez les subtilités de la dynamique mondiale grâce à notre "
             "représentation cartographique, qui illustre la variable sélectionnée "
             "dans le menu à gauche dans tous les pays du monde. Les nuances de bleu "
             "sur la carte varient en intensité en fonction de l'ampleur de la "
             "mesure sélectionnée, ce qui permet de visualiser les disparités et les modèles"
             " régionaux. Qu'il s'agisse du PIB, de la densité de population ou d'une autre"
             " statistique importante, les données de chaque pays s'animent, offrant un aperçu"
             " en un coup d'œil. L'option *Echelle logarithmique* permet d'ajuster la "
             "représentation visuelle afin d'obtenir une compréhension plus nuancée des"
             " zones présentant des valeurs très variées, ce qui permet de faire ressortir"
             " les pays dont les chiffres sont les plus faibles.")
    # Case à cocher pour l'échelle logarithmique
    log_view = st.checkbox("Afficher l'échelle logarithmique")

    # Appliquer une transformation logarithmique à la colonne de données si la case est cochée
    if log_view:
        filtered_data['log_Variable'] = filtered_data[selected_variable].apply(lambda x: np.log10(x) if x > 0 else np.nan)

    # Créez un GeoDataFrame avec la géométrie et les données, y compris la colonne logarithmique si activée
    world_gdf = world.merge(filtered_data, left_on='iso_a3', right_on='Country Code', how='left')

    # Calculez les quantiles en fonction de la variable
    variable_to_use = 'log_Variable' if log_view else selected_variable
    quantiles = np.percentile(world_gdf[variable_to_use], [0, 20, 40, 60, 80, 100])

    # Créez une carte avec une échelle logarithmique pour la variable si activée
    color_range = [np.log10(quantiles[0]), np.log10(quantiles[-1])] if log_view else [quantiles[0], quantiles[-1]]

    # Mettez à jour la légende avec le titre "Légende"
    fig = px.choropleth(world_gdf,
                        locations="iso_a3",
                        color=variable_to_use,
                        hover_name="name",
                        color_continuous_scale=px.colors.sequential.Blues,
                        color_continuous_midpoint=np.nanmean(quantiles),
                        range_color=color_range)

    # Mettez à jour la légende avec le titre "Légende"
    fig.update_layout(coloraxis_colorbar=dict(
        title="Légende",
        tickvals=quantiles,
        tickmode="array"
    ))

    # Afficher la carte avec Plotly Express
    st.plotly_chart(fig)

    # Saut de ligne
    st.write("")

    st.subheader("PIB en fonction de l'espérance de vie")
    st.write("Comprendre le lien entre la richesse d'un pays et la santé de ses citoyens "
             "est essentiel pour les responsables politiques et les chercheurs. Le diagramme "
             "de dispersion ci-dessous illustre cette relation, en représentant le produit "
             "intérieur brut (PIB) par rapport à l'espérance de vie moyenne à la naissance "
             "pour les pays du monde entier. Chaque point représente un pays unique, codé "
             "par couleur pour faciliter l'identification, et positionné en fonction de son"
             " espérance de vie et de sa production économique. Grâce à cette visualisation,"
             "nous pouvons étudier comment la prospérité économique s'aligne sur l'allongement "
             "de la durée de vie et identifier les valeurs aberrantes qui vont à l'encontre "
             "de ces tendances.")
        
    fig = px.scatter(data_df.dropna(subset=['2014 Life expectancy at birth, total (years)', 'GDP, PPP (current international $)']),
                     x='2014 Life expectancy at birth, total (years)', 
                     y='GDP, PPP (current international $)',
                     hover_name='Country Name', 
                     color='Region Code',  # Updated to the correct column name
                     title="GDP vs. Life Expectancy")
    fig.update_xaxes(title_text='Life Expectancy at birth, total (years)')
    fig.update_yaxes(title_text='GDP, PPP (current international $)')
    fig.update_layout(
    hovermode='closest',
    clickmode='event+select'
    )
    fig.update_traces(
        marker=dict(size=10),
        hoverinfo="text",
        text=[f"Country: {country} <br>GDP: {gdp} <br>Life Expectancy: {life_exp}" for country, gdp, life_exp in zip(data_df['Country Name'], data_df['GDP, PPP (current international $)'], data_df['2014 Life expectancy at birth, total (years)'])]
    )
    fig.update_yaxes(type="log")
 #   fig.update_layout(showlegend=False)
    

    st.plotly_chart(fig)

    st.write("")    
    # Graphique à barres des dix pays les plus peuplés pour la variable sélectionnée

    st.subheader(f"Les dix pays les plus importants pour {selected_variable}")
    top_10_countries = filtered_data.nlargest(10, selected_variable)
    fig2 = px.bar(top_10_countries, x=selected_variable, y="Country Name", orientation="h", text=selected_variable,
                  labels={selected_variable: selected_variable})
    fig2.update_traces(marker_color='blue', opacity=0.6, texttemplate='%{text:.2s}', textposition='outside')
    st.plotly_chart(fig2)

    # Ajouter un sous-titre pour la répartition des valeurs
    st.subheader(f"Répartition des valeurs de {selected_variable}")

    # Case à cocher pour supprimer les valeurs extrêmes
    remove_outliers = st.checkbox("Supprimer les valeurs extrêmes")

    # Filtrer les valeurs extrêmes si la case est cochée
    if remove_outliers:
        filtered_data = filtered_data[
            (filtered_data[selected_variable] >= filtered_data[selected_variable].quantile(0.1)) &
            (filtered_data[selected_variable] <= filtered_data[selected_variable].quantile(0.90))
        ]

    # Recalculer le min et le max après suppression des valeurs extrêmes
    min_value = filtered_data[selected_variable].min()
    max_value = filtered_data[selected_variable].max()

    # Recalculer les bins
#    bins = np.linspace(min_value, max_value, 100)

    hist_fig = px.histogram(
        filtered_data, x=selected_variable, nbins=100, range_x=[min_value, max_value],
        labels={selected_variable: f"{selected_variable}"}
    )
    hist_fig.update_layout(showlegend=False)

    st.plotly_chart(hist_fig)

    # Ajouter un sous-titre pour la répartition des valeurs
    st.subheader(f"Box-plot de {selected_variable}")

    # Case à cocher pour supprimer les valeurs extrêmes
    remove_outliers2 = st.checkbox("Supprimer les valeurs extrêmes", key="remove_outliers2")

    # Filtrer les valeurs extrêmes si la case est cochée
    if remove_outliers2:
        filtered_data = filtered_data[
            (filtered_data[selected_variable] >= filtered_data[selected_variable].quantile(0.1)) &
            (filtered_data[selected_variable] <= filtered_data[selected_variable].quantile(0.90))
        ]
        
    # Créer le box plot
    fig3 = px.box(filtered_data, x=selected_variable)


    # Afficher le box plot
    st.plotly_chart(fig3)

# Fonction pour le modèle et la prédiction
def model_and_prediction():
    # Ajoutez votre code pour le modèle et la prédiction ici
    st.title("Modèle et prédiction économique")
    
    st.write("Bienvenue dans l'onglet Modèle et Prédiction. Dans cette section, "
             "nous allons explorer comment notre modèle de régression linéaire nous "
             "permet de comprendre et de prédire les relations entre diverses variables "
             "économiques d'un pays. Notre modèle a été construit à partir de données "
             "mondiales, notamment le PIB (GDP, PPP), la population totale, "
             "l'utilisation d'Internet, l'espérance de vie à la naissance, et les exportations "
             "de biens et de services.")

    
    st.subheader("Le Modèle de Régression Linéaire")
    
    st.write("Notre modèle de régression linéaire est un outil puissant qui nous permet de comprendre comment ces facteurs économiques interagissent les uns avec les autres. Il repose sur l'hypothèse que le PIB peut être modélisé en fonction des autres variables mentionnées.")
    st.write("Le modèle calcule des coefficients pour chaque variable, nous indiquant l'importance de chaque facteur dans la détermination du PIB.")
    st.write(" - **Intercept** : C'est la valeur attendue du PIB lorsque toutes les autres variables sont nulles.")
    st.write(" - **Coefficients** : Chaque coefficient représente la variation attendue du PIB pour une unité de variation dans la variable respective, toutes les autres variables étant maintenues constantes. Par exemple, si le coefficient de la population totale est positif, cela signifie qu'une augmentation de la population est associée à une augmentation du PIB.")

    # Filtrer les données pour inclure les colonnes pertinentes
    filtered_data = data_df[['GDP, PPP (current international $)',
                           'Population, total',
                           '2014 Life expectancy at birth, total (years)']]
    
    # Filtrer les données pour exclure les lignes avec des valeurs NA
    filtered_data = filtered_data.dropna(subset=['Population, total',
                                                  '2014 Life expectancy at birth, total (years)',
                                                  'GDP, PPP (current international $)'])
    
    # Divisez les données en variables indépendantes (X) et la variable dépendante (y)
    X = filtered_data[['Population, total',
                       '2014 Life expectancy at birth, total (years)']]
    X['Population, total'] = X['Population, total'] / 1e6
    y = filtered_data['GDP, PPP (current international $)'] / 1e9
    
    # Split the data into training and testing sets (e.g., 80% training and 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Créez un modèle de régression linéaire
    model = LinearRegression()
    # Fit the model on the training data
    model.fit(X_train, y_train)
    
    # Remplacez le code de régression linéaire par une régression gamma
    model2 = gamma_regression(X_train, y_train)
    
    # Récupérez les coefficients de régression
    intercept = round(model.intercept_, 1)
    coefficients = [round(coef, 1) for coef in model.coef_]

    # Affichez les coefficients de régression
    st.subheader("Modélisation du GDP en fonction des variables indépendantes :")
    st.write(f"**Intercept** : {intercept}")
    st.write(f"**Coefficients :**")
    st.write(f" - Population, total : {coefficients[0]}")
    st.write(f" - 2014 Life expectancy at birth, total (years) : {coefficients[1]}")

    # Créer des données pour la droite y = x
    x_values = np.linspace(min(y), max(y), 100)
    y_values = x_values
    
    # Calculate R-squared (R2) on the test data
    r2 = model.score(X_test, y_test)
    
    # Calculate Mean Squared Error (MSE) on the test data
    predicted_y = model.predict(X_test)
    mse = mean_squared_error(y_test, predicted_y)
    
    # Display R2 and MSE using st.write
    st.write(f"R-squared (R2): {r2:.2f}")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    
    # Créez un graphique interactif avec Plotly
    scatter = go.Scatter(
        x=y,
        y=predicted_y,
        mode='markers',
        text=filtered_data.index,
        marker=dict(size=8, opacity=0.6),
        name='Régression linéaire'
    )

    # Créez la droite y = x
    line = go.Scatter(
        x=x_values,
        y=y_values,
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='y = x'
    )

    layout = go.Layout(
        title="Régression linéaire et y = x",
        xaxis=dict(title="Valeurs observées de GDP, PPP (current international 1e9$)"),
        yaxis=dict(title="Valeurs prédites de GDP, PPP (current international 1e9$)")
    )

    fig = go.Figure(data=[scatter, line], layout=layout)

    # Affichez le graphique Plotly interactif
    st.plotly_chart(fig)
    
    # Ajouter un titre pour la prédiction
    st.subheader("Prédiction du GDP (régression linéaire)")
    
   # Formulaire pour saisir les paramètres
    st.write("Utilisez le formulaire ci-dessous pour saisir les paramètres et obtenir une prédiction du GDP.")
    population = st.number_input("Population totale en millions d'habitants", min_value=0, step=1)
    life_expectancy = st.number_input("Espérance de vie à la naissance en 2014 (années)", min_value=0.0, step=1.0, format="%.3f")

    # Bouton pour effectuer la prédiction
    if st.button("Effectuer la prédiction"):
        # Créez un tableau de données avec les paramètres saisis
        user_input = pd.DataFrame({'Population, total': [population],
                                   '2014 Life expectancy at birth, total (years)': [life_expectancy]})
                                   
        # Effectuez la prédiction en utilisant le modèle
        predicted_gdp = model.predict(user_input)
    
        # Affichez le résultat de la prédiction
        st.write(f"Selon le modèle de régression linéaire, le GDP prévu est d'environ : {predicted_gdp[0]:.2f} milliards de dollars (USD)")
    
    # Ajouter un titre pour la régression GLM log-linéaire
    st.subheader("Modèle GLM log-linéaire")
    
    st.write("Notre modèle GLM log-linéaire est un outil puissant pour comprendre la relation entre les variables économiques. "
             "Il est basé sur une régression généralisée avec une distribution gamma et une fonction de lien log. "
             "Cela signifie que le modèle est adapté aux données économiques qui ont une relation log-linéaire, "
             "ce qui est souvent le cas pour le PIB.")
    st.write("Le modèle calcule des coefficients pour chaque variable, nous indiquant l'effet des variables explicatives sur le PIB.")
    st.write(" - **Intercept** : C'est la valeur attendue du PIB lorsque toutes les autres variables sont nulles.")
    st.write(" - **Coefficients** : Chaque coefficient représente la variation attendue du PIB pour un changement d'une unité dans la variable respective, toutes les autres variables étant maintenues constantes.")
    
    # Afficher les résultats de la régression GLM log-linéaire
    st.write("Régression GLM log-linéaire :")
    st.write(model2.summary())
    
    # Calculate R-squared (R2) on the test data
    r2 = model.score(X_test, y_test)
    
    # Calculate Mean Squared Error (MSE) on the test data
    predicted_y = model.predict(X_test)
    mse = mean_squared_error(y_test, predicted_y)
    
    st.write("")
    
    # Display R2 and MSE using st.write
    st.write(f"R-squared (R2): {r2:.2f}")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    
    st.write("")
    st.write("")
    # Ajouter un titre pour la prédiction GLM log-linéaire
    st.subheader("Prédiction du GDP (régression GLM log-linéaire)")
    
    st.write("Utilisez le formulaire ci-dessous pour saisir les paramètres et obtenir une prédiction du GDP par habitant "
             "en utilisant le modèle GLM log-linéaire.")
    
    # Formulaire pour saisir les paramètres de prédiction
    population_glm = st.number_input("Population totale en millions d'habitants (GLM)", min_value=0, step=1, key="population_glm")
    life_expectancy_glm = st.number_input("Espérance de vie à la naissance en 2014 (années) (GLM)", min_value=0.0, step=1.0, format="%.3f", key="life_expectancy_glm")

    # Bouton pour effectuer la prédiction GLM log-linéaire
    if st.button("Effectuer la prédiction (régression GLM log-linéaire)"):
        # Créez un tableau de données avec les paramètres saisis
        user_input_glm = pd.DataFrame({'const': [1],  # Constant for intercept
                                    'Population, total': [population_glm / 1e6],
                                    '2014 Life expectancy at birth, total (years)': [life_expectancy_glm]})
    
        # Effectuez la prédiction en utilisant le modèle GLM log-linéaire
        predicted_gdp_glm = model2.predict(user_input_glm)
    
        # Affichez le résultat de la prédiction GLM log-linéaire
        st.write(f"Selon le modèle GLM log-linéaire, le GDP prévu est d'environ : {predicted_gdp_glm[0]:.2f} milliards de dollars (USD)")
        
    
    st.subheader("Conclusion")
    
    st.write("En conclusion, nos deux modèles ne sont pas très performants quant à notre jeu de données. Cela s'explique par le faible nombre de données que l'on a, "
             "mais également par le fait que les variables explicatives ne sont pas forcément liées à la variable d'intérêt qui est le GDP.")


def random_forest_model_and_prediction():
    st.title("Machine learning (Random Forest)")
    
    st.write("Bienvenue dans l'onglet Modèle et Prédiction (Random Forest). Dans cette section, "
             "nous allons explorer comment notre modèle de forêt aléatoire nous permet de comprendre et de prédire les relations entre diverses variables "
             "économiques d'un pays. Notre modèle a été construit à partir de données "
             "mondiales, notamment le PIB (GDP, PPP), la population totale, "
             "l'utilisation d'Internet, l'espérance de vie à la naissance, et les exportations "
             "de biens et de services.")
    
    st.subheader("Le Modèle de Forêt Aléatoire")
    
    st.write("Notre modèle de forêt aléatoire est un outil puissant qui nous permet de modéliser des relations complexes entre les variables économiques. Il s'appuie sur un ensemble d'arbres de décision pour effectuer des prédictions.")
    
    # Filtrer les données pour inclure les colonnes pertinentes
    filtered_data = data_df[['GDP, PPP (current international $)',
                           'Population, total',
                           '2014 Life expectancy at birth, total (years)']]
    
    # Filtrer les données pour exclure les lignes avec des valeurs NA
    filtered_data = filtered_data.dropna(subset=['Population, total',
                                                  '2014 Life expectancy at birth, total (years)',
                                                  'GDP, PPP (current international $)'])
    
    # Divisez les données en variables indépendantes (X) et la variable dépendante (y)
    X = filtered_data[['Population, total',
                       '2014 Life expectancy at birth, total (years)']]
    X['Population, total'] = X['Population, total'] / 1e6
    y = filtered_data['GDP, PPP (current international $)'] / 1e9
    
    # Split the data into training and testing sets (e.g., 80% training and 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Fit the model on the training data
    model.fit(X_train, y_train)
    
    # Afficher les caractéristiques importantes
    feature_importances = model.feature_importances_
    st.subheader("Importance des caractéristiques")
    st.write("L'importance des caractéristiques dans le modèle de forêt aléatoire :")
    st.write(f" - Population, total : {feature_importances[0]:.2f}")
    st.write(f" - 2014 Life expectancy at birth, total (years) : {feature_importances[1]:.2f}")
    
    st.write("")
    
    # Prédiction
    st.subheader("Prédiction du GDP (Random Forest)")
    
    st.write("Utilisez le formulaire ci-dessous pour saisir les paramètres et obtenir une prédiction du GDP.")
    population = st.number_input("Population totale en millions d'habitants", min_value=0, step=1)
    life_expectancy = st.number_input("Espérance de vie à la naissance en 2014 (années)", min_value=0.0, step=1.0, format="%.3f")
    
    
    # Bouton pour effectuer la prédiction
    if st.button("Effectuer la prédiction (Random Forest)"):
        # Créez un tableau de données avec les paramètres saisis
        user_input = pd.DataFrame({'Population, total': [population],
                                   '2014 Life expectancy at birth, total (years)': [life_expectancy]})
                                   
        # Effectuez la prédiction en utilisant le modèle de forêt aléatoire
        predicted_gdp = model.predict(user_input)

        # Affichez le résultat de la prédiction
        st.write(f"Selon le modèle de forêt aléatoire, le GDP prévu est d'environ : {predicted_gdp[0]:.2f} milliards de dollars (USD)")
            
    st.subheader("Conclusion")
    
    st.write("Le modèle de forêt aléatoire s'est avéré être une excellente approche pour prédire le GDP en fonction des variables économiques sélectionnées. Il présente plusieurs avantages :")

    st.write("1. **Flexibilité** : La forêt aléatoire peut capturer des relations complexes entre les variables, ce qui est essentiel pour modéliser la complexité des données économiques du monde réel.")
    
    st.write("2. **Robustesse** : Il est résistant au surajustement grâce à l'agrégation d'un grand nombre d'arbres de décision, ce qui en fait un choix adapté pour des ensembles de données de taille limitée comme le nôtre.")
    
    st.write("3. **Importance des caractéristiques** : Le modèle nous permet de déterminer l'importance de chaque caractéristique, ce qui est utile pour comprendre quelles variables ont le plus d'influence sur les prédictions du GDP.")
    
# Ouvrir le fichier Excel
excel_file = "/Users/raph/Downloads/2015 World Bank data by nation and region.xls"
workbook = xlrd.open_workbook(excel_file)

# Sélectionner une feuille de calcul (feuille) spécifique
data = workbook.sheet_by_name("Countries in Alpha Order")

# Charger les données géographiques
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Créer un DataFrame Pandas à partir de la feuille de calcul Excel
data_df = pd.read_excel(excel_file, sheet_name="Countries in Alpha Order")

data_df = data_df.dropna(subset=['Region Code'])

# Onglets pour la sélection
selected_tab = st.sidebar.radio("Sélectionnez un onglet :", ["Visualisation des données", "Modèle et prédiction", "Machine learning"])

theme = st.sidebar.radio("Choose your theme", ['Default', 'Dark', 'Light'])

if theme == 'Dark':
    # Custom CSS to inject dark theme
    st.markdown("""
        <style>
        .main .block-container {
            background-color: black;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)

# ... similarly for other themes


# Affichez l'onglet "Visualisation des données"
if selected_tab == "Visualisation des données":
    visualize_data()


# Affichez l'onglet "Modèle et prédiction"
if selected_tab == "Modèle et prédiction":
    model_and_prediction()
    
# Affichez l'onglet "Modèle et prédiction"
if selected_tab == "Machine learning":
    random_forest_model_and_prediction()



