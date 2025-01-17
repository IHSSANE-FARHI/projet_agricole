from dashboard import AgriculturalDashboard
from map_visualization import AgriculturalMap
import streamlit as st

class IntegratedDashboard:
    def __init__(self, data_manager):
        """
        Initialize the Integrated Dashboard.
        Combines Bokeh and Folium visualizations.
        """
        self.data_manager = data_manager
        self.bokeh_dashboard = AgriculturalDashboard(data_manager)
        self.map_view = AgriculturalMap(data_manager)

    def initialize_visualizations(self):
        """
        Initialize all visual components (Bokeh and Folium).
        """
        try:
            # Initialize Bokeh layout
            self.bokeh_layout = self.bokeh_dashboard.create_layout()

            # Initialize Folium map
            self.map_view.create_base_map()
            self.map_view.add_yield_history_layer()
            self.map_view.add_risk_heatmap()

            print("Visualizations initialized successfully.")
        except Exception as e:
            print(f"Error initializing visualizations: {e}")

    def create_streamlit_dashboard(self):
        """
        Create a Streamlit interface integrating all visualizations.
        """
        try:
            st.title("Tableau de Bord Agricole Intégré")

            # Initialize visualizations
            self.initialize_visualizations()

            # Display Bokeh visualizations
            st.header("Visualisations Bokeh")
            if hasattr(self, 'bokeh_layout') and self.bokeh_layout:
                st.bokeh_chart(self.bokeh_layout, use_container_width=True)
            else:
                st.warning("Bokeh layout could not be generated.")

            # Display Folium map
            st.header("Carte Interactive (Folium)")
            if hasattr(self.map_view, 'map') and self.map_view.map:
                map_file = "integrated_map.html"
                self.map_view.map.save(map_file)
                st.markdown(
                    f'<iframe src="{map_file}" width="100%" height="600px"></iframe>',
                    unsafe_allow_html=True,
                )
            else:
                st.warning("Folium map could not be generated.")

            print("Streamlit dashboard created successfully.")
        except Exception as e:
            st.error(f"Error creating Streamlit dashboard: {e}")
            print(f"Error creating Streamlit dashboard: {e}")

    def update_visualizations(self, parcelle_id):
        """
        Update all visualizations for a given parcel.
        """
        try:
            # Update Bokeh plots
            self.bokeh_dashboard.create_data_sources()

            # Update Folium map
            self.map_view.create_base_map()
            self.map_view.add_yield_history_layer()
            self.map_view.add_risk_heatmap()

            print(f"Visualizations updated for parcelle_id: {parcelle_id}")
        except Exception as e:
            print(f"Error updating visualizations: {e}")
