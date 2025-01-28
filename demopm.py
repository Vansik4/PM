import streamlit as st
import pandas as pd
import pm4py
from datetime import datetime

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

# Main application
def main():
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
        st.warning("Please select at least one activity and one case.")

# Run the application
if __name__ == "__main__":
    main()
