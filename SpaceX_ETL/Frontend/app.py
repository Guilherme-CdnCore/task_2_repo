import streamlit as st
import pandas as pd
import psycopg2

st.set_page_config(page_title="SpaceX ETL Dashboard", page_icon="🚀", layout="wide")

st.title("🚀 SpaceX Launches Dashboard")

# Database connection config
DB_CONFIG = {
    "dbname": "spacex",
    "user": "postgres",
    "password": "password",
    "host": "localhost",
    "port": "5432"
}

# Load launches from DB
@st.cache_data
def load_launches(limit=20):
    conn = psycopg2.connect(**DB_CONFIG)
    query = "SELECT mission_name, launch_date, rocket_name FROM launches ORDER BY launch_date DESC LIMIT %s;"
    df = pd.read_sql(query, conn, params=(limit,))
    conn.close()
    return df

# Sidebar controls
st.sidebar.header("Filters")
limit = st.sidebar.slider("Number of launches", 5, 50, 20)

df = load_launches(limit)

# Show table
st.subheader(f"Latest {limit} launches")
st.dataframe(df)

# Show chart
st.subheader("Launches over time")
df["launch_date"] = pd.to_datetime(df["launch_date"])
launches_per_year = df.groupby(df["launch_date"].dt.year).size()
st.bar_chart(launches_per_year)




# EXAMPLE CODE #