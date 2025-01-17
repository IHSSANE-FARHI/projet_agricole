import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

class AgriculturalAnalyzer:
    def __init__(self, data_manager):
        """
        Initialise l'analyseur avec le gestionnaire de données.
        """
        self.data_manager = data_manager
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def analyze_yield_factors(self, parcelle_id):
        """
        Analyse les facteurs influençant les rendements.
        """
        # Filtrer les données pour la parcelle spécifiée
        parcelle_data = self.data_manager.monitoring_data[
            self.data_manager.monitoring_data["parcelle_id"] == parcelle_id
        ]

        if parcelle_data.empty:
            raise ValueError(f"Parcelle {parcelle_id} introuvable dans les données.")

        # Vérifier que les colonnes nécessaires sont numériques
        numeric_cols = parcelle_data.select_dtypes(include=[np.number]).columns
        if "rendement_estime" not in numeric_cols:
            raise ValueError("La colonne 'rendement_estime' doit être numérique.")

        # Calcul des corrélations avec les colonnes numériques
        correlations = parcelle_data[numeric_cols].corr()
        yield_correlations = correlations["rendement_estime"].sort_values(ascending=False)

        return yield_correlations

    def _plot_yield_evolution(self, parcelle_id):
        """
        Trace l'évolution des rendements pour une parcelle donnée.
        """
        # Filtrer les données pour la parcelle spécifiée
        parcelle_data = self.data_manager.monitoring_data[
            self.data_manager.monitoring_data["parcelle_id"] == parcelle_id
        ]

        if parcelle_data.empty:
            raise ValueError(f"Parcelle {parcelle_id} introuvable dans les données.")

        # Tracer l'évolution des rendements
        plt.figure(figsize=(10, 6))
        plt.plot(parcelle_data["date"], parcelle_data["rendement_estime"], marker="o", label="Rendement")
        plt.title(f"Évolution des rendements pour la parcelle {parcelle_id}")
        plt.xlabel("Date")
        plt.ylabel("Rendement (t/ha)")
        plt.legend()
        plt.grid()
        plt.savefig(f"yield_evolution_{parcelle_id}.png")
        plt.close()
