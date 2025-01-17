import folium
from folium import plugins
from branca.colormap import LinearColormap
from data_manager import AgriculturalDataManager
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import webbrowser

class AgriculturalMap:
    def __init__(self, data_manager):
        """
        Initialise la carte avec le gestionnaire de données
        """
        self.data_manager = data_manager
        self.map = None
        self.yield_colormap = LinearColormap(
            colors=["red", "yellow", "green"],
            vmin=0,
            vmax=12  # rendement_estime maximal en tonnes/ha
        )
    
    def create_base_map(self):
        """
        Crée la carte de base avec les couches appropriées
        """
        try:
            self.data_manager.load_data()
            features = self.data_manager.prepare_features()
            avg_latitude = features['latitude'].mean()
            avg_longitude = features['longitude'].mean()
            self.map = folium.Map(
                location=[avg_latitude, avg_longitude],
                zoom_start=13,
                tiles='OpenStreetMap'
            )
            print("Carte de base créée avec succès.")
        except Exception as e:
            print(f"Erreur lors de la création de la carte de base : {e}")
    
    def add_yield_history_layer(self):
        """
        Ajoute une couche visualisant l'historique des rendements.
        """
        try:
            if self.map is None:
                raise ValueError("La carte de base n'est pas initialisée. Appelez create_base_map d'abord.")

            features = self.data_manager.prepare_features()

            # Vérification des colonnes nécessaires
            required_columns = ['parcelle_id', 'latitude', 'longitude', 'rendement_estime', 'date', 'culture', 'ndvi']
            for col in required_columns:
                if col not in features.columns:
                    raise KeyError(f"Colonne manquante : {col}")

            # Créer la colonne 'date' au format année si elle n'existe pas
            if 'annee' not in features.columns:
                print("Création de la colonne 'annee' à partir de 'date'.")
                features['annee'] = pd.to_datetime(features['date']).dt.year

            # Grouper par parcelle_id
            grouped = features.groupby('parcelle_id')

            for parcelle_id, group in grouped:
                # Vérification des données du groupe
                if group.empty or 'latitude' not in group.columns or 'longitude' not in group.columns:
                    print(f"Parcèle {parcelle_id} ignorée en raison de données manquantes.")
                    continue

                # Calculer le rendement moyen
                mean_yield = group['rendement_estime'].mean()

                # Calculer la tendance des rendements
                trend = self._calculate_yield_trend(parcelle_id)

                # Créer le contenu de la popup pour l'historique des rendements
                popup_content = self._create_yield_popup(group, mean_yield, trend)

                # Ajouter les cultures récentes à la popup
                recent_crops = self._format_recent_crops(group)
                popup_content += f"<h5 style='color: #2c3e50;'>Cultures récentes:</h5>{recent_crops}"

                # Créer le contenu du popup pour les données NDVI actuelles
                ndvi_popup_content = self._create_ndvi_popup(group.iloc[0])  # Utilise la première ligne de chaque groupe pour NDVI
                popup_content += f"<h5 style='color: #2c3e50;'></h5>{ndvi_popup_content}"

                # Ajouter à la carte
                lat = group['latitude'].mean()
                lon = group['longitude'].mean()

                # Vérification des coordonnées
                if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    print(f"Parcèle {parcelle_id} ignorée en raison de coordonnées invalides : ({lat}, {lon})")
                    continue

                folium.CircleMarker(
                    location=(lat, lon),
                    radius=5,
                    color=self.yield_colormap(mean_yield),
                    fill=True,
                    fill_color=self.yield_colormap(mean_yield),
                    fill_opacity=0.7,
                    popup=folium.Popup(popup_content, max_width=300)
                ).add_to(self.map)

            print("Couche d'historique des rendements ajoutée avec succès.")

        except Exception as e:
            print(f"Erreur lors de l'ajout de la couche d'historique des rendements : {e}")

    def add_current_ndvi_layer(self):
        """
        Ajoute une couche de la situation NDVI actuelle
        """
        try:
            if self.map is None:
                raise ValueError("La carte de base n'est pas initialisée. Appelez create_base_map d'abord.")
            
            # Get features data
            features = self.data_manager.prepare_features()

            # Ensure the necessary columns are present
            required_columns = ['parcelle_id', 'latitude', 'longitude', 'ndvi', 'culture', 'date']
            if not all(col in features.columns for col in required_columns):
                raise KeyError(f"Colonnes manquantes : {', '.join(required_columns)}")
            
            # Group by parcel ID
            grouped = features.groupby('parcelle_id')
            
            # Define colormap for NDVI values
            ndvi_colormap = LinearColormap(
                colors=["red", "yellow", "green"],
                vmin=features['ndvi'].min(),
                vmax=features['ndvi'].max()
            )

            # Loop through each parcel to create markers on the map
            for parcelle_id, group in grouped:
                lat = group['latitude'].mean()
                lon = group['longitude'].mean()
                ndvi = group['ndvi'].mean()
                
                # Prepare the popup content with NDVI and additional data
                popup_content = f"""
                    <div style="font-family: Arial, sans-serif; font-size: 12px;">
                        <h4 style="margin: 0; color: #2c3e50;">Parcelle ID: {parcelle_id}</h4>
                        <p style="margin: 0; color: #34495e;">Culture: {group['culture'].iloc[0]}</p>
                        <p style="margin: 0; color: #34495e;">Date: {group['date'].max().strftime('%Y-%m-%d')}</p>
                    </div>
                """

                # Add the CircleMarker with popup to the map
                folium.CircleMarker(
                    location=(lat, lon),
                    radius=5,
                    color=ndvi_colormap(ndvi),
                    fill=True,
                    fill_color=ndvi_colormap(ndvi),
                    fill_opacity=0.7,
                    popup=folium.Popup(popup_content, max_width=300)
                ).add_to(self.map)

            print("Couche NDVI actuelle ajoutée avec succès.")

        except Exception as e:
            print(f"Erreur lors de l'ajout de la couche NDVI actuelle : {e}")

    def add_risk_heatmap(self):
        """
        Ajoute une carte de chaleur des zones à risque
        """
        try:
            if self.map is None:
                raise ValueError("La carte de base n'est pas initialisée. Appelez create_base_map d'abord.")
            features = self.data_manager.prepare_features()
            # Assurer que les colonnes sont renommées si nécessaire
            features.rename(columns={
                'lat': 'latitude',
                'lon': 'longitude',
                'yield': 'rendement_estime',
                'year': 'date'
            }, inplace=True, errors='ignore')
            risk_metrics = self.data_manager.calculate_risk_metrics(features)
            if risk_metrics is None or features is None:
                raise ValueError("Les métriques de risque ou les données de caractéristiques manquent.")
            merged_data = pd.merge(risk_metrics, features[['parcelle_id', 'latitude', 'longitude']], on='parcelle_id', how='left')
            required_columns = ['latitude', 'longitude', 'avg_risk_index']
            if not all(col in merged_data.columns for col in required_columns):
                raise KeyError(f"Colonnes manquantes : {', '.join(required_columns)}")
            heatmap_data = merged_data.dropna(subset=required_columns)
            heatmap_data['latitude'] += np.random.uniform(-0.0001, 0.0001, len(heatmap_data))
            heatmap_data['longitude'] += np.random.uniform(-0.0001, 0.0001, len(heatmap_data))
            min_risk = heatmap_data['avg_risk_index'].min()
            max_risk = heatmap_data['avg_risk_index'].max()
            heatmap_data['normalized_risk'] = 0.1 + (heatmap_data['avg_risk_index'] - min_risk) / (max_risk - min_risk) * 0.9
            heat_data = heatmap_data[['latitude', 'longitude', 'normalized_risk']].values.tolist()
            plugins.HeatMap(
                heat_data,
                name='Carte de chaleur des risques',
                radius=15,
                blur=15,
                max_zoom=13,
                min_opacity=0.3
            ).add_to(self.map)
            folium.LayerControl().add_to(self.map)
            print("Carte de chaleur des risques ajoutée avec succès.")
        except Exception as e:
            print(f"Erreur lors de l'ajout de la carte de chaleur des risques : {e}")

    def _calculate_yield_trend(self, parcelle_id):
        """
        Calcule la tendance des rendement_estimes en utilisant une régression linéaire.
        Utilise l'historique des rendement_estime_estimes pour la parcelle donnée.
        """
        try:
            # Filtrer l'historique des rendement_estimes pour la parcelle donnée
            ph = self.data_manager.yield_history[self.data_manager.yield_history['parcelle_id'] == parcelle_id].copy()
            
            if ph.empty or ph['rendement_estime'].nunique() < 2:
                # Pas assez de points de données pour une régression significative
                return {'slope': 0, 'intercept': 0, 'variation_moyenne': 0}
            
            # Assurer que la colonne 'date' est au format numérique (année en entier)
            ph['date'] = ph['date'].dt.year  # Convertit en année numérique
            
            # Trier et retirer les doublons par année
            ph = ph.drop_duplicates(subset=['date']).sort_values(by='date')
            
            # Préparer les données pour la régression
            X = ph['date'].values.reshape(-1, 1)
            y = ph['rendement_estime'].values
            
            # Exécuter la régression linéaire
            model = LinearRegression().fit(X, y)
            slope = model.coef_[0]
            intercept = model.intercept_
            variation = slope / y.mean() if y.mean() != 0 else 0

            return {
                'slope': slope,
                'intercept': intercept,
                'variation_moyenne': variation
            }
        except Exception as e:
            print(f"Erreur lors du calcul de la tendance des rendement_estimes pour la parcelle {parcelle_id} : {e}")
            return {'slope': 0, 'intercept': 0, 'variation_moyenne': 0}
        
    def _create_yield_popup(self, history, mean_yield, trend):
        """
        Crée le contenu HTML pour la popup d'historique des rendements.
        """
        try:
            # Valider les entrées
            if history.empty:
                raise ValueError("Le DataFrame d'historique est vide.")
            if trend is None or not all(key in trend for key in ['slope', 'intercept', 'variation_moyenne']):
                raise ValueError("Les données de tendance sont manquantes ou incomplètes.")

            # Extraire les détails de la tendance
            slope = trend.get('slope', 0.0)
            intercept = trend.get('intercept', 0.0)
            variation = trend.get('variation_moyenne', 0.0)

            # Créer le contenu HTML de la popup
            popup_content = f"""
            <div style="font-family: Arial, sans-serif; font-size: 12px;">
                <h4 style="margin: 0; color: #2c3e50;">Parcelle ID: {history['parcelle_id'].iloc[0]}</h4>
                <p style="margin: 0; color: #34495e;">Moyenne rendement_estime: {mean_yield:.2f} t/ha</p>
                <p style="margin: 0; color: #34495e;">Tendance:</p>
                <ul style="margin: 0; padding-left: 15px;">
                    <li>Pente: {slope:.2f} t/ha/an</li>
                    <li>Intercept: {intercept:.2f}</li>
                    <li>Variation Moyenne: {variation:.2%}</li>
                </ul>
                <h5 style="margin-top: 10px; margin-bottom: 5px; color: #2c3e50;">Historique des rendements (moyenne par année):</h5>
                <ul style="margin: 0; padding-left: 15px;">
            """

            # Ajouter l'année et la moyenne des rendements
            history['annee'] = history['date'].dt.year  # Extract the year from the date

            # Group by 'annee' (year) and calculate the mean of rendement_estime for each year
            yearly_yields = history.groupby('annee')['rendement_estime'].mean().reset_index()

            # Ajouter les détails des rendements moyens par année
            for _, row in yearly_yields.iterrows():
                year = row['annee']
                yield_value = row['rendement_estime']
                popup_content += f"<li>{year}: {yield_value:.2f} t/ha</li>"

            # Fermer les balises HTML
            popup_content += """
                </ul>
            </div>
            """

            return popup_content

        except Exception as e:
            print(f"Erreur lors de la création de la popup des rendements : {e}")
            return "<div>Erreur lors de la création du contenu de la popup.</div>"

    def _format_recent_crops(self, history):
        """
        Formate la liste des cultures récentes pour le popup
        """
        try:
            # Check if the history has a valid 'culture' column
            if 'culture' not in history.columns:
                raise KeyError("La colonne 'culture' est manquante dans l'historique des données.")
            
            # Extract recent crops (assuming the most recent ones are at the top)
            recent_crops = history[['annee', 'culture']].drop_duplicates(subset=['annee']).sort_values(by='annee', ascending=False)
            
            # Format the list of recent crops
            crops_list = "<ul style='margin: 0; padding-left: 15px;'>"
            for _, row in recent_crops.iterrows():
                crop_year = row['annee']
                crop_name = row['culture']
                crops_list += f"<li>{crop_year}: {crop_name}</li>"
            
            crops_list += "</ul>"
            return crops_list

        except Exception as e:
            print(f"Erreur lors du formatage des cultures récentes : {e}")
            return "<div>Erreur lors du formatage des cultures récentes.</div>"

    def _create_ndvi_popup(self, row):
        """
        Crée le contenu HTML du popup pour les données NDVI actuelles
        """
        try:
            # Check if the row contains necessary columns
            if 'parcelle_id' not in row or 'ndvi' not in row:
                raise KeyError("Les colonnes 'parcelle_id' ou 'ndvi' sont manquantes dans la ligne de données.")
            
            # Extract the parcel ID and NDVI value
            parcel_id = row['parcelle_id']
            ndvi_value = row['ndvi']
            
            # Format the popup content for NDVI
            popup_content = f"""
            <div style="font-family: Arial, sans-serif; font-size: 12px;">
                <p style="margin: 0; color: #34495e;">NDVI actuel: {ndvi_value:.2f}</p>
            </div>
            """
            return popup_content

        except Exception as e:
            print(f"Erreur lors de la création du popup NDVI : {e}")
            return "<div>Erreur lors de la création du popup NDVI.</div>"

if __name__ == "__main__":
    # Initialiser AgriculturalDataManager et charger les données
    data_manager = AgriculturalDataManager()
    data_manager.load_data()

    # Initialiser AgriculturalMap
    agri_map = AgriculturalMap(data_manager)

    # Créer la carte de base et ajouter les couches
    agri_map.create_base_map()
    agri_map.add_yield_history_layer()
    #agri_map.add_current_ndvi_layer()
    #agri_map.add_risk_heatmap()

    # Enregistrer et ouvrir la carte
    if agri_map.map:
        map_file = "carte_agricole.html"
        agri_map.map.save(map_file)
        print(f"Carte enregistrée sous '{map_file}'.")
        webbrowser.open(map_file)
    else:
        print("La carte n'a pas pu être créée.")
