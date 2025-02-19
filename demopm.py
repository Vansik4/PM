import streamlit as st
import pandas as pd
import pm4py
from datetime import datetime
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pyodbc
import networkx as nx
from fuzzywuzzy import fuzz
from io import BytesIO
import openai
import json
from textblob import TextBlob
from difflib import get_close_matches
import plotly.express as px
import plotly.graph_objects as go



def get_exact_matches(df):
    return df[df.duplicated(subset=["case_key"], keep=False)].copy()

def find_similar_matches(df, threshold=80):
    unique_cases = df["case_key"].unique()
    duplicates = []
    
    for i, case1 in enumerate(unique_cases):
        for case2 in unique_cases[i+1:]:
            similarity = fuzz.ratio(case1, case2)
            if similarity >= threshold:
                duplicates.append({"case_key": case1, "Similar_Case": case2, "Similarity": similarity})
    
    return pd.DataFrame(duplicates)


def duplicate_invoice_analysis():
    st.title("🕵️ Duplicate Invoice Analysis")
    st.write("Filter by client, company code, and date to analyze duplicate invoices.")
    # Load and validate the CSV
    uploaded_file = "facturas_ajustadas_v1.xlsx"  # Adjust the file path as needed
    df = pd.read_excel(uploaded_file)
    # Validate required columns
    required_columns = ["ClientName", "Amount", "BKPF.BUKRS", "BKPF.GJAHR"]
    if not all(col in df.columns for col in required_columns):
        st.error(f"❌ The file must contain the following columns: {', '.join(required_columns)}")
        return
    # Preprocessing
    df["ClientName"] = df["ClientName"].fillna("UNKNOWN")
    df["Amount"] = df["Amount"].fillna(0)
    df["BKPF.BUKRS"] = df["BKPF.BUKRS"].fillna("UNKNOWN")
    df["BKPF.GJAHR"] = df["BKPF.GJAHR"].fillna(0)
    df["case_key"] = (
        df["ClientName"].astype(str) + "_" + 
        df["Amount"].astype(str) + "_" + 
        df["BKPF.BUKRS"].astype(str) + "_" + 
        df["BKPF.GJAHR"].astype(str)
    )
    # Convert BKPF.GJAHR to datetime if necessary
    df["BKPF.GJAHR"] = pd.to_datetime(df["BKPF.GJAHR"].astype(str), format='%Y', errors='coerce')
    df = df.dropna(subset=["BKPF.GJAHR"])
    # Lists of unique clients and company codes
    unique_clients = df['ClientName'].unique().tolist()
    unique_company_codes = df['BKPF.BUKRS'].unique().tolist()
    # Create the column layout
    col_main, col_joyer, col_filters = st.columns([2, 4, 1])  
    with col_filters:
        st.header("Filters")
        openai.api_key = st.secrets["OPENAI_API_KEY"] 
        # Date filter
        min_date = df["BKPF.GJAHR"].min()
        max_date = df["BKPF.GJAHR"].max()
        if pd.isnull(min_date) or pd.isnull(max_date):
            st.error("Unable to determine date limits due to missing values.")
            return
        date_range = st.date_input(
            "Select a date range",
            value=[min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )
        # Convert date_range to datetime64[ns]
        date_range = [pd.Timestamp(date) for date in date_range]
        df_filtered = df[(df["BKPF.GJAHR"] >= date_range[0]) & (df["BKPF.GJAHR"] <= date_range[1])]
        client_list = df_filtered["ClientName"].unique().tolist()
        selected_clients = st.multiselect(
            "Select a Client",
            options=client_list
        )
        df_filtered = df_filtered[df_filtered["ClientName"].isin(selected_clients)]
        company_codes = df_filtered["BKPF.BUKRS"].unique().tolist()
        selected_company_codes = st.multiselect(
            "Select Company Codes",
            options=company_codes,    
        )
        df_filtered = df_filtered[df_filtered["BKPF.BUKRS"].isin(selected_company_codes)]  
    # In the central column, KPIs and the filtered table are displayed
    with col_joyer:
        if date_range and selected_clients and selected_company_codes:
            df_filtered = df[
                (df["BKPF.GJAHR"] >= date_range[0]) &
                (df["BKPF.GJAHR"] <= date_range[1]) &
                (df["ClientName"].isin(selected_clients)) &  # Fixed
                (df["BKPF.BUKRS"].isin(selected_company_codes))  # Fixed
            ].copy()  # Create an explicit copy
            similarity_threshold = st.sidebar.slider(
                "Filter by similarity percentage",
                min_value=80, max_value=100, value=80, step=1
            )
            # Duplicate detection
            with st.spinner("Searching for duplicates..."):
                exact_duplicates = get_exact_matches(df_filtered)
                exact_duplicates["Duplicate_Type"] = "Exact"
                filtered_df = df_filtered[~df_filtered["case_key"].isin(exact_duplicates["case_key"])].copy()
                similar_duplicates = find_similar_matches(filtered_df)
                similar_duplicates["Duplicate_Type"] = "Similar"
                similar_duplicates = similar_duplicates[similar_duplicates["Similarity"] >= similarity_threshold]
                all_duplicates = pd.concat([exact_duplicates, similar_duplicates], ignore_index=True)
                # Calculate amount-based KPIs
                total_exact_amount = 0
                total_similar_amount = 0
                # Calculate total amount in exact duplicates
                if not exact_duplicates.empty:
                    exact_groups = exact_duplicates.groupby('case_key').agg(
                        count=('case_key', 'size'),
                        amount=('Amount', 'first')
                    )
                    total_exact_amount = (exact_groups['count'] - 1).dot(exact_groups['amount'])
                # Calculate total amount in similar duplicates
                if not similar_duplicates.empty:
                    similar_pairs = similar_duplicates.merge(
                        df_filtered[['case_key', 'Amount']], 
                        on='case_key'
                    ).merge(
                        df_filtered[['case_key', 'Amount']], 
                        left_on='Similar_Case', 
                        right_on='case_key', 
                        suffixes=('_original', '_similar')
                    )
                    total_similar_amount = (similar_pairs['Amount_original'] + similar_pairs['Amount_similar']).sum()
                total_duplicated_amount = total_exact_amount + total_similar_amount
                avg_duplicated_amount = total_duplicated_amount / (len(exact_duplicates) + len(similar_duplicates)) if (len(exact_duplicates) + len(similar_duplicates)) > 0 else 0    
                # Calculate KPIs
                num_exact_duplicates = len(exact_duplicates)
                num_similar_duplicates = len(similar_duplicates)
                # Display KPIs
                st.write("### Key Performance Indicators (KPIs)")
                col1, col2, col3, col4 = st.columns(4, gap="small", vertical_alignment="center", border=False)
                col1.metric("Exact Duplicates", num_exact_duplicates)
                col2.metric("Similar Duplicates", num_similar_duplicates)
                col3.metric("Total Risk Amount", f"${total_duplicated_amount:,.2f}")
                col4.metric("Average per Duplicate", f"${avg_duplicated_amount:,.2f}")
                all_duplicates_1 = pd.concat([exact_duplicates, similar_duplicates], ignore_index=True)
                st.write("### Exact Duplicate Details")
                if not exact_duplicates.empty:
                    st.dataframe(exact_duplicates[["ClientName", "Amount", "BKPF.BUKRS", "BKPF.GJAHR"]])
                else:
                    st.info("No exact duplicates found")
                st.write("### Similar Duplicate Details")
                if not similar_duplicates.empty:
                    st.dataframe(similar_duplicates[["case_key", "Similar_Case", "Similarity"]])
                else:
                    st.info("No similar duplicates found")
                csv = all_duplicates.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Full Report (CSV)",
                    data=csv,
                    file_name="duplicate_report.csv",
                    mime="text/csv"
                )
        else:
            st.warning("Please select a date range, a client, and a company code.")
        user_query = st.text_area("🤖 Ask about the dashboard:", placeholder="E.g., How many duplicates are there in total?")
        if st.button("🔍 Query LLM"):
            if all_duplicates_1.empty:
                st.warning("No data available for analysis. Adjust the filters.")
            elif user_query:
                # Convert the DataFrame `all_duplicates` to JSON to provide context to the model
                df_json = all_duplicates_1.to_json(orient="records")
                # Create a message for the model with context
                prompt = f"""
                You are an expert data analysis assistant. Below is a dataset of duplicate invoices in JSON format.
                Answer the following question based on this data:
                Data:
                {df_json}
                If asked about risk amounts, here is the risk amount for exact documents: 
                {total_exact_amount}
                And here is the risk amount for similar documents:
                {total_similar_amount}
                If a total risk amount is requested, sum the two values.
                User's question:
                {user_query}
                Respond clearly and concisely, rounding decimal values to 2 places.
                """
                # Query OpenAI
                try:
                    openai.api_key = st.secrets["OPENAI_API_KEY"] 
                    response = openai.chat.completions.create(
                        model="gpt-4o",  # Use GPT-4 or switch to "gpt-3.5-turbo" if preferred
                        messages=[{"role": "system", "content": "You are a data analysis assistant."},
                                  {"role": "user", "content": prompt}]
                    )
                    # Display the model's response
                    llm_response = response.choices[0].message.content
                    st.success("Assistant Response:")
                    st.write(llm_response)
                except Exception as e:
                    st.error(f"Error connecting to OpenAI: {e}") 
    # In the main column, charts are displayed
    with col_main:
        if date_range and selected_clients and selected_company_codes:
            df_filtered = df[
                (df["BKPF.GJAHR"] >= date_range[0]) &
                (df["BKPF.GJAHR"] <= date_range[1]) &
                (df["ClientName"].isin(selected_clients)) &  # Fixed
                (df["BKPF.BUKRS"].isin(selected_company_codes))
            ].copy()  # Create an explicit copy
            # Duplicate detection
            with st.spinner("Searching for duplicates..."):
                exact_duplicates = get_exact_matches(df_filtered)
                exact_duplicates["Duplicate_Type"] = "Exact"
                filtered_df = df_filtered[~df_filtered["case_key"].isin(exact_duplicates["case_key"])].copy()
                similar_duplicates = find_similar_matches(filtered_df)
                similar_duplicates["Duplicate_Type"] = "Similar"    
                similar_duplicates = similar_duplicates[similar_duplicates["Similarity"] >= similarity_threshold]   
                all_duplicates = pd.concat([exact_duplicates, similar_duplicates], ignore_index=True)
                # Bar chart for the distribution of duplicate types
                fig1, ax1 = plt.subplots(figsize=(10, 8))
                sns.countplot(x="Duplicate_Type", data=all_duplicates, palette="viridis")
                plt.title("Distribution of Duplicate Types")
                plt.xlabel("Duplicate Type")
                plt.ylabel("Count")
                st.pyplot(fig1)
                # Histogram for the distribution of similarity percentages
                if not similar_duplicates.empty:
                    fig2, ax2 = plt.subplots(figsize=(10, 8))   
                    sns.histplot(similar_duplicates["Similarity"], bins=15, kde=True, color="orange")
                    plt.title("Distribution of Similarity Percentages")
                    plt.xlabel("Similarity Percentage")
                    plt.ylabel("Frequency")
                    st.pyplot(fig2)
                else:
                    st.info("No similar duplicates found")
        else:
            st.warning("Please select a date range, a client, and a company code.")
        
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

def generate_and_display_graph(df_filtered, metric_type="Case Count"):
    try:
        # Descubrir la red heurística con PM4Py
        heu_net = pm4py.discover_heuristics_net(df_filtered)

        # Definir el variant según la métrica seleccionada:
        # Para "Número de Casos" se utiliza la variante "frequency" (por defecto)
        # Para "Tiempo" se utiliza la variante "performance"
        if metric_type == "Case Count":
            variant = "frequency"
        elif metric_type == "Tiempo":
            variant = "performance"
        else:
            st.error("Métrica no reconocida.")
            return

        # Guardar la visualización de la red heurística utilizando el variant correspondiente
        # Nota: asegúrate de tener una versión de PM4Py que soporte el parámetro variant.
        pm4py.save_vis_heuristics_net(heu_net, "heu_net_output.png", variant=variant)

        st.write(f"### Process Flow Graph ({metric_type})")
        st.image("heu_net_output.png", caption=f"Diagrama de Flujo del Proceso - {metric_type}")

        # Botón de descarga (si el archivo se generó correctamente)
        try:
            with open("heu_net_output.png", "rb") as file:
                st.download_button(
                    label="Download Process Flow Graph",
                    data=file,
                    file_name="process_flow_graph.png",
                    mime="image/png"
                )
        except Exception as e:
            st.error(f"Error al crear el botón de descarga: {e}")
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


def load_material_data():
    file_path = "simulated_materials_table.csv"
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"An error occurred while loading material data: {e}")
        return None        
       

    
# Main application
def main():
    st.set_page_config(layout="wide")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", ["Process Mining Credit", "Process Mining Services Desk", "Free Text Analysis", "Duplicate Invoice Analysis"])

    if page == "Process Mining Credit":
        st.title("Process Mining Interface for Credit Approval")
        st.write("Filter by activities and visualize the process flow.")

        # Cargar y validar el CSV
        file_path = "corrected_process_mining_data.csv"
        
        df = load_and_validate_csv(file_path)
        if df is None:
            return

        # Preparar el DataFrame
        df = prepare_dataframe(df)
        if df is None:
            return

        # Listas de actividades y casos únicos
        unique_activities = df['ACTIVITY'].unique().tolist()
        unique_cases = df['CASE KEY'].unique().tolist()

        # Crear el layout de columnas
        col_main, col_joyer, col_filters = st.columns([5, 3, 2])  

        with col_filters:
            st.header("Filters")
            selected_activities = st.multiselect(
                "Select activities to include in the analysis:",
                unique_activities,
                default=unique_activities[:2]
            )
            selected_cases = st.multiselect(
                "Select cases to include in the analysis:",
                unique_cases,
                default=unique_cases[:5]
            )

        # En la columna central se pueden mostrar KPIs y la tabla filtrada
        with col_joyer:
            if selected_activities and selected_cases:
                df_filtered = df[df['ACTIVITY'].isin(selected_activities) & df['CASE KEY'].isin(selected_cases)]
                # Calcular KPIs
                num_cases, num_activities, avg_time_per_case = calculate_kpis(df_filtered)
                st.write("### Key Performance Indicators (KPIs)")
                col1, col2, col3 = st.columns(3)
                col1.metric("Number of Unique Cases", num_cases)
                col2.metric("Total Number of Activities", num_activities)
                col3.metric("Average Time per Case (hours)", round(avg_time_per_case, 2))
                st.write("### Filtered Event Details")
                st.dataframe(df_filtered)
                csv = df_filtered.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Filtered Data as CSV",
                    data=csv,
                    file_name="filtered_events.csv",
                    mime="text/csv"
                )
            else:
                st.warning("Please select at least one activity and one case.") 

        # En la columna principal se muestra la gráfica
        with col_main:
            if selected_activities and selected_cases:
                df_filtered = df[df['ACTIVITY'].isin(selected_activities) & df['CASE KEY'].isin(selected_cases)]
                # Agregar el selector para elegir la métrica de la gráfica
                metric_type = st.radio(
                    "Select type:",
                    ["Case Count"]
                )
                # Generar y mostrar la gráfica según la opción seleccionada
                generate_and_display_graph(df_filtered, metric_type)
            else:
                st.warning("Please select at least one activity and one case.")

    # ... (El resto de páginas se mantienen igual)
    elif page == "Process Mining Services Desk":
        st.title("Process Mining Analysis ITSM")
        st.write("Filter by activities and visualize the process flow.")

        # Conexión a base de datos y preparación del DataFrame (se mantiene el código original)
        server = st.secrets["server"] 
        database = st.secrets["database"] 
        username = st.secrets["username"] 
        password = st.secrets["password"] 
        port = st.secrets["port"] 

        connection_string = (
            "DRIVER={ODBC Driver 17 for SQL Server};"
            f"SERVER={server},{port};"
            f"DATABASE={database};"
            f"UID={username};"
            f"PWD={password};"
            "TrustServerCertificate=yes;"
        )

        connection = pyodbc.connect(connection_string)
        cursor = connection.cursor()
        query1 = "SELECT TOP 500 * FROM dbo.WorkOrderHistory"
        rows1 = cursor.execute(query1).fetchall()
        columns1 = [column[0] for column in cursor.description]
        df1 = pd.DataFrame.from_records(rows1, columns=columns1)
        query2 = "SELECT TOP 500 * FROM dbo.WorkOrder"
        rows2 = cursor.execute(query2).fetchall()
        columns2 = [column[0] for column in cursor.description]
        df2 = pd.DataFrame.from_records(rows2, columns=columns2)

        df = df1.rename(columns={
            "WORKORDERID": "CASE KEY",
            "OPERATION": "ACTIVITY",
            "OPERATIONTIME": "TIMESTAMP"
        })
        df = df.drop(columns=["HISTORYID","OPERATIONOWNERID", "DESCRIPTION"])
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], unit='ms')
        df = df.drop_duplicates(subset=["CASE KEY", "TIMESTAMP", "ACTIVITY"])
        df = prepare_dataframe(df)
        if df is None:
            return

        unique_activities = df['ACTIVITY'].unique().tolist()
        unique_cases = df['CASE KEY'].unique().tolist()
        col_main, col_joyer, col_filters = st.columns([5, 3, 2])

        with col_filters:
            st.header("Filters")
            selected_activities = st.multiselect(
                "Select activities to include in the analysis:",
                unique_activities,
                default=unique_activities[:2]
            )
            selected_cases = st.multiselect(
                "Select cases to include in the analysis:",
                unique_cases,
                default=unique_cases[:5]
            )

        with col_joyer:
            if selected_activities and selected_cases:
                df_filtered = df[df['ACTIVITY'].isin(selected_activities) & df['CASE KEY'].isin(selected_cases)]
                num_cases, num_activities, avg_time_per_case = calculate_kpis(df_filtered)
                st.write("### Key Performance Indicators (KPIs)")
                col1, col2, col3 = st.columns(3)
                col1.metric("Number of Unique Cases", num_cases)
                col2.metric("Total Number of Activities", num_activities)
                col3.metric("Average Time per Case (hours)", round(avg_time_per_case, 2))
                st.write("### Filtered Event Details")
                st.dataframe(df_filtered)
                csv = df_filtered.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Filtered Data as CSV",
                    data=csv,
                    file_name="filtered_events.csv",
                    mime="text/csv"
                )
            else:
                st.warning("Please select at least one activity and one case.")

        with col_main:
            if selected_activities and selected_cases:
                df_filtered = df[df['ACTIVITY'].isin(selected_activities) & df['CASE KEY'].isin(selected_cases)]
                # Puedes agregar el selector de métrica aquí también si lo requieres en esta página.
                metric_type = st.radio(
                "Select Type:",
                ["Case Count"]
            )
                generate_and_display_graph(df_filtered, metric_type)
            else:
                st.warning("Please select at least one activity and one case.")

        connection.close()

    elif page == "Free Text Analysis":
                # Initial configuration       

        # Load data with caching
        @st.cache_data
        def load_data():
            inventory = pd.read_excel("test.xlsx", sheet_name="Inventory Table")
            sales = pd.read_excel("test.xlsx", sheet_name="Sales Orders")
            return inventory, sales

        inventory, sales = load_data()

        # Convert dates
        sales['order_date'] = pd.to_datetime(sales['order_date'])

        # ========== SIDEBAR FILTERS ==========
        st.sidebar.header("🎚️ Global Filters")

        # Date range filter
        date_range = st.sidebar.date_input(
            "Date Range",
            value=[sales['order_date'].min(), sales['order_date'].max()],
            min_value=sales['order_date'].min(),
            max_value=sales['order_date'].max()
        )

        # Status filter
        status_filter = st.sidebar.multiselect(
            "Order Status",
            options=sales['status'].unique(),
            default=sales['status'].unique()
        )

        # Customer filter
        customer_filter = st.sidebar.multiselect(
            "Filter by Customer",
            options=sales['customer_name'].unique(),
            default=[]
        )

        # Item type filter
        item_type_filter = st.sidebar.radio(
            "Item Type",
            options=['All', 'Free Text', 'Standard'],
            index=0
        )

        # Apply filters
        @st.cache_data
        def apply_filters(df, date_range, status_filter, customer_filter, item_type_filter):
            filtered = df[
                (df['order_date'].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]))) &
                (df['status'].isin(status_filter))
            ]
            
            if customer_filter:
                filtered = filtered[filtered['customer_name'].isin(customer_filter)]
            
            if item_type_filter == 'Free Text':
                filtered = filtered[filtered['is_free_text']]
            elif item_type_filter == 'Standard':
                filtered = filtered[~filtered['is_free_text']]
            
            return filtered

        filtered_sales = apply_filters(sales, date_range, status_filter, customer_filter, item_type_filter)

        # Filter inventory for free text items based on the same filters
        inventory_free = inventory[inventory['is_free_text']]

        # AI correction function
        @st.cache_data
        def ai_correction(text, inventory_material_names):
            corrected = TextBlob(text).correct()
            matches = get_close_matches(text, inventory_material_names, n=3)
            return str(corrected), matches

        # Generate suggestions for inventory items
        inventory_material_names = inventory['material_name'].dropna().unique()
        inventory_free[['corrected_name', 'suggestions']] = inventory_free['material_name'].apply(
            lambda x: pd.Series(ai_correction(x, inventory_material_names))
        )

        # ========== KPIs ==========
        # Calculate KPIs based on filtered data
        total_free_items = len(inventory_free)
        total_free_orders = len(filtered_sales[filtered_sales['is_free_text']])
        free_sales_value = filtered_sales[filtered_sales['is_free_text']]['total_price'].sum()
        correction_rate = np.mean([1 if x[0] in x[1] else 0 for x in zip(
            inventory_free['corrected_name'], inventory_free['suggestions'])])

        # ========== DASHBOARD LAYOUT ==========
        st.title("📈 Free Text Intelligence Dashboard")
        st.markdown("---")

        # KPI Section
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🛍️ Free Text Items", f"{total_free_items}")
        col2.metric("💰 Free Text Orders Value", f"${free_sales_value:,.0f}")
        col3.metric("📦 Free Text Orders", f"{total_free_orders}")
        col4.metric("🎯 AI Correction Rate", f"{correction_rate:.0%}")

        st.markdown("---")

        # Charts
        col1, col2 = st.columns(2)
        with col1:
            # Gráfica: Tendencia de Órdenes de Texto Libre
            sales_trend = filtered_sales[filtered_sales['is_free_text']].groupby(
                pd.to_datetime(filtered_sales['order_date']).dt.to_period('M').astype(str)
            ).size().reset_index(name='count')
            
            fig = px.line(
                sales_trend,
                x='order_date',
                y='count',
                title="📅 Free Text Orders Trend",
                labels={'count': 'Number of Orders', 'order_date': 'Month'}
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Tabla: Órdenes de Texto Libre de Alto Valor
            high_value_orders = filtered_sales[filtered_sales['is_free_text']].nlargest(5, 'total_price')[['order_id', 'free_text_name', 'quantity', 'total_price']]
            st.subheader("📋 High Value Free Text Orders")
            st.dataframe(
                high_value_orders,
                column_config={
                    "order_id": "Order ID",
                    "free_text_name": "Item Description",
                    "quantity": "Qty",
                    "total_price": "Total Value"
                },
                use_container_width=True
            )

        st.markdown("---")

        # ========== NUEVAS VISUALIZACIONES ==========
        col1, col2 = st.columns(2)

        with col1:
            # Gráfica: Distribución de Precios de Órdenes de Texto Libre
            price_distribution = filtered_sales[filtered_sales['is_free_text']]
            fig = px.histogram(
                price_distribution,
                x='total_price',
                nbins=30,
                title="📊 Distribution of Free Text Order Prices",
                labels={'total_price': 'Order Price', 'count': 'Frequency'},
                color_discrete_sequence=['#1f77b4']
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Tabla: Top Customers por Valor Total de Órdenes de Texto Libre
            top_customers = filtered_sales[filtered_sales['is_free_text']].groupby(
                'customer_name'
            )['total_price'].sum().nlargest(5).reset_index()
            top_customers.rename(columns={'total_price': 'Total Value'}, inplace=True)
            st.subheader("🏆 Top Customers by Free Text Order Value")
            st.dataframe(
                top_customers,
                column_config={
                    "customer_name": "Customer",
                    "Total Value": "Total Value ($)"
                },
                use_container_width=True
            )

        st.markdown("---")
        st.subheader("🧠 Advanced Text Analysis")
        tab1, tab2 = st.tabs(["Customer Distribution", "Lexical Analysis"])

        with tab1:
            client_dist = filtered_sales[filtered_sales['is_free_text']].groupby(
                'customer_name').size().nlargest(10)
            fig = px.bar(
                client_dist,
                orientation='h',
                title="🏆 Top Free Text Customers",
                labels={'value': 'Number of Orders', 'customer_name': 'Customer'},
                color=client_dist.values,
                color_continuous_scale='Tealgrn'
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            from wordcloud import WordCloud
            text = ' '.join(filtered_sales[filtered_sales['is_free_text']]['free_text_name'])
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            st.image(wordcloud.to_array(), caption='☁️ Word Cloud - Free Text Descriptions')

        # Final notes
        st.markdown("---")
        st.info("💡 All filters affect visualizations in real time. Use the sidebar controls to explore the data.")
        

    elif page == "Duplicate Invoice Analysis":        
        duplicate_invoice_analysis()
        

if __name__ == "__main__":
    main()
