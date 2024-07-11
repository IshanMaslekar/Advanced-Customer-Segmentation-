import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import plotly.express as px
from fpdf import FPDF

# Define functions

@st.cache_data
def load_data(file_path):
    df = pd.read_excel(file_path)
    return df

def preprocess_data(df):
    df = df.dropna(subset=['CustomerID'])
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    df['TotalSpend'] = df['Quantity'] * df['UnitPrice']
    customer_data = df.groupby('CustomerID').agg({'TotalSpend': 'sum', 'Quantity': 'sum'}).reset_index()
    scaler = StandardScaler()
    customer_data_scaled = scaler.fit_transform(customer_data[['TotalSpend', 'Quantity']])
    return customer_data, customer_data_scaled, scaler

def find_optimal_clusters(data):
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    return wcss

def cluster_data(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
    clusters = kmeans.fit_predict(data)
    return kmeans, clusters

def generate_pdf_report(customer_data, n_clusters):
    class PDFReport(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, 'Customer Segmentation Report', 0, 1, 'C')

        def chapter_title(self, title):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, title, 0, 1, 'L')

        def chapter_body(self, body):
            self.set_font('Arial', '', 12)
            self.multi_cell(0, 10, body)

    pdf = PDFReport()
    pdf.add_page()
    pdf.chapter_title('Cluster Analysis')
    for cluster in range(n_clusters):
        cluster_data = customer_data[customer_data['Cluster'] == cluster]
        cluster_summary = cluster_data.describe().to_string()
        pdf.chapter_body(f'Cluster {cluster} Summary:\n{cluster_summary}')
    pdf_file_path = 'Customer_Segmentation_Report.pdf'
    pdf.output(pdf_file_path)
    return pdf_file_path

# Streamlit UI

st.title('Customer Segmentation Using K-Means Clustering')

uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.write("Data Preview:")
    st.write(df.head())
    
    customer_data, customer_data_scaled, scaler = preprocess_data(df)
    
    st.write("Data Preprocessing Complete.")
    
    wcss = find_optimal_clusters(customer_data_scaled)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), wcss, marker='o')
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    st.pyplot(plt)

    n_clusters = st.slider("Select Number of Clusters", 1, 10, 3)
    kmeans, clusters = cluster_data(customer_data_scaled, n_clusters)
    customer_data['Cluster'] = clusters
    
    st.write("Clustered Data Preview:")
    st.write(customer_data.head())

    fig = px.scatter(customer_data, x='TotalSpend', y='Quantity', color='Cluster', title='Customer Segmentation')
    st.plotly_chart(fig)

    if st.button("Generate PDF Report"):
        pdf_file_path = generate_pdf_report(customer_data, n_clusters)
        st.success(f"PDF Report Generated: {pdf_file_path}")
        with open(pdf_file_path, "rb") as pdf_file:
            st.download_button(label="Download Report", data=pdf_file, file_name="Customer_Segmentation_Report.pdf", mime="application/pdf")
