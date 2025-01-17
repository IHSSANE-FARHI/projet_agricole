from data_manager import AgriculturalDataManager
from integrated_dashboard import IntegratedDashboard
import streamlit as st

if __name__ == "__main__":
    # Load data
    data_manager = AgriculturalDataManager()
    data_manager.load_data()

    # Initialize dashboard
    dashboard = IntegratedDashboard(data_manager)

    # Create Streamlit interface
    dashboard.create_streamlit_dashboard()
