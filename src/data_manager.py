import numpy as np
import pandas as pd
import warnings

from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

class AgriculturalDataManager:

    def __init__(self):
        # Initialisation des attributs de données
        self.monitoring_data = None
        self.weather_data = None
        self.soil_data = None
        self.yield_history = None
        # Instanciation du standardiseur
        self.scalar = StandardScaler()

    def load_data(self):
        try:
            # Chargement des fichiers de données
            self.monitoring_data = pd.read_csv("../data/monitoring_cultures.csv", parse_dates=["date"])
            self.weather_data = pd.read_csv("../data/meteo_detaillee.csv", parse_dates=["date"])
            self.soil_data = pd.read_csv("../data/sols.csv")
            self.yield_history = pd.read_csv("../data/historique_rendements.csv", parse_dates=["date"])
        except FileNotFoundError as e:
            print(f"Erreur : fichier non trouvé. {e}")
        except Exception as e:
            print(f"Erreur lors du chargement des données : {e}")

    def clean_data(self):
        # Correction des valeurs négatives pour le rayonnement solaire
        if self.weather_data is not None and 'rayonnement_solaire' in self.weather_data.columns:
            self.weather_data['rayonnement_solaire'] = self.weather_data['rayonnement_solaire'].abs()

    def meteo_data_hourly_to_daily(self):
        try:
            # Conversion des dates au format datetime et rééchantillonnage en données journalières
            self.weather_data['date'] = pd.to_datetime(self.weather_data['date'], errors='coerce')
            self.weather_data = (self.weather_data
                                 .set_index('date')
                                 .resample('D')
                                 .mean()
                                 .reset_index())
        except Exception as e:
            print(f"Erreur lors de l'agrégation : {e}")

    def _setup_temporal_indices(self):
        try:
            # Définir l'index temporel pour les DataFrames
            self.monitoring_data.set_index('date', inplace=True)
            self.weather_data.set_index('date', inplace=True)
            self.yield_history.set_index('date', inplace=True)
        except Exception as e:
            print(f"Erreur lors de la configuration des indices temporels : {e}")

    def prepare_features(self):
        try:
            # Tri des données par date avant fusion
            self.monitoring_data.sort_values(by="date", inplace=True)
            self.weather_data.sort_values(by="date", inplace=True)

            # Fusion asynchrone des données de monitoring et météo
            merged_data = pd.merge_asof(self.monitoring_data,
                                        self.weather_data,
                                        on="date",
                                        direction='nearest')

            # Ajout des données de sol
            merged_data = pd.merge(merged_data, self.soil_data, how='left', on="parcelle_id")

            # Enrichissement avec l'historique des rendements
            merged_data = self._enrich_with_yield_history(merged_data)

            # Nettoyage des colonnes en double
            merged_data.drop(columns=['latitude_y', 'longitude_y'], errors='ignore', inplace=True)
            merged_data.rename(columns={'latitude_x': 'latitude', 'longitude_x': 'longitude'}, inplace=True)
            merged_data.drop(columns=['culture_y'], errors='ignore', inplace=True)
            merged_data.rename(columns={'culture_x': 'culture'}, inplace=True)

            # Sauvegarde des données fusionnées
            merged_data.to_csv("../data/features_merge.csv", index=False)
            print("Colonnes des données préparées :", merged_data.columns.tolist())

            return merged_data

        except Exception as e:
            print(f"Erreur lors de la préparation des fonctionnalités : {e}")
            return None

    def _enrich_with_yield_history(self, df):
        try:
            # Intégration de l'historique des rendements dans le jeu de données principal
            enriched_df = pd.merge(df,
                                   self.yield_history,
                                   how="left",
                                   on=["parcelle_id", "date"])
            return enriched_df
        except Exception as e:
            print(f"Erreur lors de l'enrichissement avec l'historique des rendements : {e}")
            return df

    def get_temporal_patterns(self, parcelle_id):
        try:
            features = pd.read_csv("../data/features_merge.csv", parse_dates=["date"])
            parcelle_data = features[features["parcelle_id"] == parcelle_id]

            if "ndvi" not in parcelle_data.columns:
                raise KeyError("La colonne NDVI est absente des données.")

            if parcelle_data.empty:
                raise ValueError(f"Aucune donnée trouvée pour la parcelle {parcelle_id}.")

            parcelle_data.sort_values(by="date", inplace=True)
            parcelle_data.set_index("date", inplace=True)

            ndvi_series = parcelle_data["ndvi"].dropna()
            if len(ndvi_series) < 12:
                raise ValueError("Pas assez de points de données pour la décomposition saisonnière.")

            decompo = seasonal_decompose(ndvi_series, model="additive", period=12)

            # Analyse de la tendance NDVI par régression linéaire
            dates_num = parcelle_data.index.map(datetime.toordinal).values.reshape(-1, 1)
            ndvi_values = ndvi_series.values.reshape(-1, 1)
            lin_reg = LinearRegression().fit(dates_num, ndvi_values)
            slope = lin_reg.coef_[0][0]
            intercept = lin_reg.intercept_[0]

            trend_info = {
                "pente": slope,
                "intercept": intercept,
                "variation_moyenne": slope / ndvi_series.mean() if ndvi_series.mean() != 0 else 0
            }

            history_info = {
                "ndvi_trend": decompo.trend.dropna(),
                "ndvi_seasonal": decompo.seasonal,
                "ndvi_residual": decompo.resid.dropna(),
                "ndvi_moving_avg": ndvi_series.rolling(window=30).mean().dropna(),
                "summary_stats": {
                    "mean_ndvi": ndvi_series.mean(),
                    "std_ndvi": ndvi_series.std(),
                    "min_ndvi": ndvi_series.min(),
                    "max_ndvi": ndvi_series.max(),
                }
            }

            return history_info, trend_info

        except Exception as e:
            print(f"Erreur dans get_temporal_patterns : {e}")
            return None, None

    def calculate_risk_metrics(self, data):
        try:
            # Vérification des colonnes obligatoires
            required_cols = ['parcelle_id', 'culture', 'rendement_estime', 'ph', 'matiere_organique']
            for col in required_cols:
                if col not in data.columns:
                    raise KeyError(f"La colonne requise '{col}' est absente.")

            # Standardisation des données nécessaires
            normalized = self.scalar.fit_transform(data[['rendement_estime', 'ph', 'matiere_organique']])

            # Calcul de l'indice de risque
            data['risk_index'] = (0.5 * normalized[:, 0] +
                                  0.3 * normalized[:, 1] +
                                  0.2 * normalized[:, 2])

            # Classification du risque selon l'indice
            data['risk_category'] = pd.cut(data['risk_index'],
                                           bins=[-np.inf, -1, 0, 1, np.inf],
                                           labels=['Très Bas', 'Bas', 'Modéré', 'Élevé'])

            # Agrégation des métriques de risque par parcelle et culture
            grouped = data.groupby(['parcelle_id', 'culture']).agg(
                avg_risk_index=('risk_index', 'mean'),
                most_frequent_risk_category=('risk_category', lambda x: x.mode()[0] if not x.mode().empty else None)
            ).reset_index()

            # Sauvegarde des données agrégées
            grouped.to_csv("../data/grouped_risk_metrics.csv", index=False)

            return grouped

        except Exception as e:
            print(f"Erreur lors du calcul des métriques de risque : {e}")
            return None

    def analyze_yield_patterns(self, parcelle_id):
        try:
            # Extraction de l'historique des rendements pour une parcelle spécifique
            yield_hist = self.yield_history[self.yield_history['parcelle_id'] == parcelle_id].copy()

            if yield_hist.empty:
                raise ValueError(f"Aucune donnée de rendement trouvée pour la parcelle {parcelle_id}.")

            yield_hist.sort_values(by='date', inplace=True)
            yield_series = yield_hist.set_index('date')['rendement_estime']

            # Gestion des valeurs manquantes ou constantes dans la série de rendements
            if yield_series.isnull().any():
                print("Remplissage des valeurs manquantes dans la série de rendements.")
                yield_series = yield_series.interpolate()

            if yield_series.nunique() == 1:
                print("Ajout de bruit à une série de rendements constante.")
                yield_series += np.random.normal(0, 0.1, size=len(yield_series))

            # Analyse de tendance par régression linéaire
            dates_num = yield_series.index.map(datetime.toordinal).values.reshape(-1, 1)
            yields = yield_series.values.reshape(-1, 1)
            reg_model = LinearRegression().fit(dates_num, yields)
            slope = reg_model.coef_[0][0]
            intercept = reg_model.intercept_[0]

            # Calcul des résidus
            predicted = reg_model.predict(dates_num).flatten()
            resid = yield_series - predicted

            results = {
                'tendance': {
                    'pente': slope,
                    'intercept': intercept,
                    'variation_moyenne': slope / yield_series.mean() if yield_series.mean() != 0 else 0
                },
                'residus': resid,
                'statistiques_resume': {
                    'moyenne': yield_series.mean(),
                    'ecart_type': yield_series.std(),
                    'minimum': yield_series.min(),
                    'maximum': yield_series.max()
                }
            }

            return results

        except Exception as e:
            print(f"Erreur lors de l'analyse des patterns de rendement : {e}")
            return None


if __name__ == "__main__":
    manager = AgriculturalDataManager()

    # Chargement et préparation des données
    manager.load_data()
    manager.clean_data()
    manager.meteo_data_hourly_to_daily()

    features = manager.prepare_features()

    parcelle_id = "P001"
    history, trend = manager.get_temporal_patterns(parcelle_id)
    risk_metric = manager.calculate_risk_metrics(features) if features is not None else None
    yield_analysis = manager.analyze_yield_patterns(parcelle_id)

    print("========= risk metrics =========")
    if risk_metric is not None:
        print(risk_metric.head())
    else:
        print("Aucune métrique de risque disponible.")

    print("========= Ndvi =========")
    if trend:
        print(f"Tendance de rendement : {trend['pente']:.4f} tonnes/ha/an")
        print(f"Variation moyenne : {trend['variation_moyenne'] * 100:.4f}%")
    else:
        print("Aucune donnée de tendance NDVI disponible.")

    print("========= rendement historique =========")
    if yield_analysis:
        print(f"Tendance de rendement : {yield_analysis['tendance']['pente']:.4f} tonnes/ha/an")
        print(f"Variation moyenne : {yield_analysis['tendance']['variation_moyenne'] * 100:.2f}%")
        print(f"Résidus : {yield_analysis['residus'].head()}")
    else:
        print("Aucune analyse de rendement disponible.")
