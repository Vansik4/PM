import streamlit as st
import pandas as pd
import pm4py
from datetime import datetime
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator



# Function to load and validate the CSV file
def load_and_validate_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        required_columns = ['CASE KEY', 'ACTIVITY', 'TIMESTAMP']
        if not all(column in df.columns for column in required_columns):
            st.error(f"The CSV file must contain the following columns: {', '.join(required_columns)}")
            return None
        return df
    except FileNotFoundError:
        st.error("File not found. Please check the file path.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the CSV file: {e}")
        return None

# Function to format and prepare the DataFrame for process mining
def prepare_dataframe(df):
    try:
        df = df.sort_values(by=['CASE KEY', 'TIMESTAMP'])
        df = pm4py.format_dataframe(df, case_id='CASE KEY', activity_key='ACTIVITY', timestamp_key='TIMESTAMP')
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
        return df
    except Exception as e:
        st.error(f"An error occurred while preparing the DataFrame: {e}")
        return None

def generate_and_display_graph(df_filtered):
    try:
        heu_net = pm4py.discover_heuristics_net(df_filtered)

        # Visualizar la red heurística y guardarla como imagen
        pm4py.save_vis_heuristics_net(heu_net, "heu_net_output.png")
        
        st.write("### Process Flow Graph")
        st.image("heu_net_output.png", caption="Process Flow Diagram")
        
        # Provide a download link for the graph
        with open("heu_net_output.png", "rb") as file:
            btn = st.download_button(
                label="Download Process Flow Graph",
                data=file,
                file_name="process_flow_graph.png",
                mime="image/png"
            )
    except Exception as e:
        st.error(f"An error occurred while generating the process flow graph: {e}")
        
# Function to calculate KPIs
def calculate_kpis(df_filtered):
    try:
        # Number of unique cases
        num_cases = df_filtered['CASE KEY'].nunique()
        
        # Total number of activities
        num_activities = df_filtered.shape[0]
        
        # Average time per case
        df_filtered['TIMESTAMP'] = pd.to_datetime(df_filtered['TIMESTAMP'])
        case_durations = df_filtered.groupby('CASE KEY')['TIMESTAMP'].apply(lambda x: (x.max() - x.min()).total_seconds() / 3600)
        avg_time_per_case = case_durations.mean()
        
        return num_cases, num_activities, avg_time_per_case
    except Exception as e:
        st.error(f"An error occurred while calculating KPIs: {e}")
        return None, None, None

# Function to send free text requisition to a webhook
def send_free_text_requisition(text):
    webhook_url = "https://ofi-partner-sandbox.us-1.celonis.cloud/ems-automation/public/api/root/aa4cb271-f1f3-49d7-a2fe-ec68cea6afb5/hook/gmn0genw1bk7k8xj2idl975m1qme7hgn"  # Reemplaza con tu URL de webhook
    payload = {
        "requisition_type": "free_text",
        "details": text
    }
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(webhook_url, data=json.dumps(payload), headers=headers)
        response.raise_for_status()  # Lanza una excepción si la solicitud no fue exitosa
        st.success("Solicitud de texto libre enviada exitosamente.")
    except requests.exceptions.RequestException as e:
        st.error(f"Error enviando la solicitud de texto libre: {str(e)}")

def load_material_data():
    file_path = "simulated_materials_table.csv"
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"An error occurred while loading material data: {e}")
        return None        
    
def send_csv_to_webhook(url, csv_path):
    try:
        # Load the CSV file and convert to JSON
        df = pd.read_csv(csv_path)
        json_data = df.where(pd.notnull(df), None).to_json(orient="records")
        
        data = {
            'key1': "Nombre del registro",  # Reemplaza con el valor adecuado
            'key2': "Usuario del registro"  # Reemplaza con el valor adecuado
        }

        # Send the JSON data to the webhook
        response = requests.post(url, json={"data": json_data, **data})
        st.success(f"ACCEPTED: {url}")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to send data to webhook: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

def create_dashboard():
    df_materials = load_material_data()
    if df_materials is None:
        return

    st.title("Material Analysis Dashboard")

    # Add buttons to send data to webhooks
    st.write("### Send Data to Webhooks")
    col1, col2 = st.columns(2)

    webhook_1_url = "https://ofi-partner-sandbox.us-1.celonis.cloud/ems-automation/public/api/root/aa4cb271-f1f3-49d7-a2fe-ec68cea6afb5/hook/ebaatc4fqiw7fnqbe7bsn2ze6o8vchok"
    webhook_2_url = "https://ofi-partner-sandbox.us-1.celonis.cloud/ems-automation/public/api/root/aa4cb271-f1f3-49d7-a2fe-ec68cea6afb5/hook/ebaatc4fqiw7fnqbe7bsn2ze6o8vchok"

   # with col1:
    #    if st.button("Send to Webhook 1"):
     #       send_csv_to_webhook(webhook_1_url, "simulated_materials_table.csv")

    #with col2:
     #   if st.button("Send to Webhook 2"):
      #      send_csv_to_webhook(webhook_2_url, "simulated_materials_table.csv")

    # Set Seaborn style to match Streamlit's background
    sns.set_style("whitegrid")
    background_color = st.get_option("theme.backgroundColor") or "white"
    sns.set(rc={"axes.facecolor": background_color, "figure.facecolor": background_color})

    # Data preparation
    df_materials['Type'] = df_materials['Material Code'].apply(lambda x: 'Valid' if pd.notna(x) else 'Free Text')
    material_counts = df_materials['Type'].value_counts()
    valid_materials = df_materials.dropna(subset=['Material Code'])
    free_text_materials = df_materials[df_materials['Material Code'].isna()]

    # Create the plots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

        
    # Plot 1: Valid vs Free Text Materials
    sns.barplot(x=material_counts.index, y=material_counts.values, ax=axs[0, 0], palette="viridis")
    axs[0, 0].set_title("Valid vs Free Text Materials", fontsize=14)
    axs[0, 0].set_ylabel("Count", fontsize=12)
    axs[0, 0].set_xlabel("Material Type", fontsize=12)

    # Plot 2: Top 5 Most Frequent Valid Materials
    top_valid = valid_materials['Material Name'].value_counts().head(5)
    sns.barplot(x=top_valid.values, y=top_valid.index, ax=axs[0, 1], palette="Blues_d")
    axs[0, 1].set_title("Top 5 Most Frequent Valid Materials", fontsize=14)
    axs[0, 1].set_xlabel("Count", fontsize=12)
    axs[0, 1].set_ylabel("Material Name", fontsize=12)

    # Plot 3: Top 5 Most Frequent Free Text Materials
    top_free_text = free_text_materials['Material Name'].value_counts().head(5)
    sns.barplot(x=top_free_text.values, y=top_free_text.index, ax=axs[1, 0], palette="Reds_d")
    axs[1, 0].set_title("Top 5 Most Frequent Free Text Materials", fontsize=14)
    axs[1, 0].set_xlabel("Count", fontsize=12)
    axs[1, 0].set_ylabel("Material Name", fontsize=12)

    # Plot 4: Pie chart of material type distribution
    axs[1, 1].pie(material_counts, labels=material_counts.index, autopct='%1.1f%%', startangle=90, colors=["#4CAF50", "#FF5722"])
    axs[1, 1].set_title("Material Type Distribution", fontsize=14)

    # Adjust layout
    plt.tight_layout()
    st.pyplot(fig)
    

    
# Main application
def main():
    st.set_page_config(layout="wide")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", ["Process Mining", "Free Text Analysis"])

    if page == "Process Mining":
        st.title("Dynamic Process Mining Interface for Credit Approval")
        st.write("Filter by activities and visualize the process flow.")

        # Load and validate the CSV file
        file_path = "corrected_process_mining_data.csv"
        df = load_and_validate_csv(file_path)
        if df is None:
            return

        # Prepare the DataFrame for process mining
        df = prepare_dataframe(df)
        if df is None:
            return

        # List of unique activities and cases
        unique_activities = df['ACTIVITY'].unique().tolist()
        unique_cases = df['CASE KEY'].unique().tolist()

        # Crear un diseño de dos columnas: una para los filtros y otra para el contenido principal
        col_main, col_joyer, col_filters = st.columns([5, 3, 2])  # Ajusta el ancho de las columnas según sea necesario

        with col_filters:
            st.header("Filters")

            # Activity selection using a multiselect widget
            selected_activities = st.multiselect(
                "Select activities to include in the analysis:",
                unique_activities,
                default=unique_activities[:2]  # Initial selection of two activities
            )

            # Case selection using a multiselect widget
            selected_cases = st.multiselect(
                "Select cases to include in the analysis:",
                unique_cases,
                default=unique_cases[:5]  # Initial selection of five cases
            )

        with col_joyer:
            # Filter DataFrame by selected activities and cases
            if selected_activities and selected_cases:
                df_filtered = df[df['ACTIVITY'].isin(selected_activities) & df['CASE KEY'].isin(selected_cases)]

                # Calculate KPIs
                num_cases, num_activities, avg_time_per_case = calculate_kpis(df_filtered)

                # Display KPIs
                st.write("### Key Performance Indicators (KPIs)")
                col1, col2, col3 = st.columns(3)
                col1.metric("Number of Unique Cases", num_cases)
                col2.metric("Total Number of Activities", num_activities)
                col3.metric("Average Time per Case (hours)", round(avg_time_per_case, 2))

                # Display the filtered DataFrame with a download option
                st.write("### Filtered Event Details")
                st.dataframe(df_filtered)

                # Provide a download link for the filtered DataFrame
                csv = df_filtered.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Filtered Data as CSV",
                    data=csv,
                    file_name="filtered_events.csv",
                    mime="text/csv"
                )
            else:
                st.warning("Please select at least one activity and one case.") 

              

        # Colocar el contenido principal en la otra columna
        with col_main:
            # Filter DataFrame by selected activities and cases
            if selected_activities and selected_cases:
                df_filtered = df[df['ACTIVITY'].isin(selected_activities) & df['CASE KEY'].isin(selected_cases)]

                # Calculate KPIs
                num_cases, num_activities, avg_time_per_case = calculate_kpis(df_filtered)
                
                # Generate and display the process flow graph
                generate_and_display_graph(df_filtered)
                
            else:
                st.warning("Please select at least one activity and one case.")
            
    elif page == "Free Text Analysis":
        st.title("Free Text Analysis Overview")
        create_dashboard()

# Run the application
if __name__ == "__main__":
    main()
