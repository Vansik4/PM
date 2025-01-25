import streamlit as st
import pandas as pd
import pm4py
import graphviz

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

# Function to generate and display the process flow graph
def generate_and_display_graph(df_filtered):
    try:
        df_log_filtered = pm4py.convert_to_event_log(df_filtered)
        dfg, start_activities, end_activities = pm4py.discover_dfg(df_log_filtered)
        pm4py.save_vis_dfg(dfg, start_activities, end_activities, "dfg_output.png")
        st.write("### Process Flow Graph")
        st.image("dfg_output.png", caption="Process Flow Diagram")
        
        # Provide a download link for the graph
        with open("dfg_output.png", "rb") as file:
            btn = st.download_button(
                label="Download Process Flow Graph",
                data=file,
                file_name="process_flow_graph.png",
                mime="image/png"
            )
    except Exception as e:
        st.error(f"An error occurred while generating the process flow graph: {e}")

# Main application
def main():
    # Crear dos columnas: una para la imagen y otra para el contenido
    col1, col2 = st.columns([1, 4])  # La primera columna es más estrecha para la imagen

    # Agregar la imagen en la primera columna (lado izquierdo)
    with col1:
        st.image("https://res.cloudinary.com/ddmifk9ub/image/upload/v1714666361/OFI/Logos/ofi-black.png", width=100)  # Ajusta el ancho según sea necesario

    # Agregar el contenido principal en la segunda columna
    with col2:
        st.title("Dynamic Process Mining Interface for Credit Approval")
        st.write("Filter by activities and visualize the process flow.")

        # Load and validate the CSV file
        file_path = "process_mining_data.csv"
        df = load_and_validate_csv(file_path)
        if df is None:
            return

        # Prepare the DataFrame for process mining
        df = prepare_dataframe(df)
        if df is None:
            return

        # List of unique activities
        unique_activities = df['ACTIVITY'].unique().tolist()

        # Activity selection using a multiselect widget
        selected_activities = st.multiselect(
            "Select activities to include in the analysis:",
            unique_activities,
            default=unique_activities[:2]  # Initial selection of two activities
        )

        # Filter DataFrame by selected activities
        if selected_activities:
            df_filtered = df[df['ACTIVITY'].isin(selected_activities)]

            # Generate and display the process flow graph
            generate_and_display_graph(df_filtered)

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
            st.warning("Please select at least one activity.")

# Run the application
if __name__ == "__main__":
    main()
