import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="AI Traffic Monitoring Dashboard",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 AI Traffic Monitoring Dashboard")

try:
    df = pd.read_csv("traffic_data.csv")

    latest = df.iloc[-1]

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("🚗 Cars", latest["Cars"])
    col2.metric("🏍 Motorcycles", latest["Motorcycles"])
    col3.metric("🚌 Buses", latest["Buses"])
    col4.metric("🚚 Trucks", latest["Trucks"])
    col5.metric("🚦 Total", latest["Total"])

    st.markdown("---")

    density = latest["Density"]

    st.subheader("Traffic Density Status")

    if density == "LOW":
        st.success("🟢 LOW TRAFFIC")

    elif density == "MEDIUM":
        st.warning("🟡 MEDIUM TRAFFIC")

    else:
        st.error("🔴 HIGH TRAFFIC")

    st.markdown("---")

    colA, colB = st.columns(2)

    with colA:
        st.subheader("Vehicle Trends")

        fig1 = px.line(
            df,
            x="Time",
            y=["Cars", "Motorcycles", "Buses", "Trucks"],
            title="Vehicle Count Over Time"
        )

        st.plotly_chart(fig1, use_container_width=True)

    with colB:
        st.subheader("Vehicle Distribution")

        pie_df = pd.DataFrame({
            "Vehicle": ["Cars", "Motorcycles", "Buses", "Trucks"],
            "Count": [
                latest["Cars"],
                latest["Motorcycles"],
                latest["Buses"],
                latest["Trucks"]
            ]
        })

        fig2 = px.pie(
            pie_df,
            names="Vehicle",
            values="Count",
            hole=0.4
        )

        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    st.subheader("Recent Traffic Records")
    st.dataframe(df.tail(20), use_container_width=True)

except Exception as e:
    st.error(f"Error: {e}")