from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the dataset
data_path = 'APP_DEVLOPMENT_DATA.csv'
data = pd.read_csv(data_path)

# Data preprocessing
data['Age'] = pd.Timestamp.now().year - data['Year_Birth']  # Convert Year_Birth to Age

# Selecting relevant features for clustering
numeric_features = ['Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines', 
                    'MntMeatProducts', 'NumWebPurchases', 'NumStorePurchases', 'Age']

# Handling missing values
data[numeric_features] = data[numeric_features].fillna(data[numeric_features].median())

# Standard scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[numeric_features])

# Optimized KMeans model
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=50, max_iter=500, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_data)

# Assign cluster names
def assign_cluster_name(cluster_label):
    cluster_names = {
        0: "High Spenders",
        1: "Budget Buyers",
        2: "Occasional Shoppers"
    }
    return cluster_names.get(cluster_label, "Unknown")

data['Cluster_Name'] = data['Cluster'].apply(assign_cluster_name)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    try:
        # Extract input data from the form
        input_data = {}
        for key, value in request.form.items():
            if key == "Currency":
                currency = value  # Store currency separately
            else:
                input_data[key] = float(value)  # Convert only numeric inputs

        # Calculate Age and remove Year_Birth
        input_data['Age'] = pd.Timestamp.now().year - input_data.pop('Year_Birth')

        # Convert to DataFrame and scale
        input_df = pd.DataFrame([input_data])
        scaled_input = scaler.transform(input_df[numeric_features])

        # Predict cluster
        cluster = kmeans.predict(scaled_input)[0]
        cluster_name = assign_cluster_name(cluster)

        return render_template('result.html', cluster=cluster_name, input_data=input_data, currency=currency)

    except KeyError as e:
        logging.error(f"Missing input: {e}")
        return f"Missing input for: {str(e)}", 400
    except Exception as e:
        logging.error(f"Error: {e}")
        return f"An error occurred: {str(e)}", 500

@app.route('/visualize')
def visualize():
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=data['Income'], y=data['MntWines'], hue=data['Cluster_Name'], palette='Set2')
    plt.xlabel('Income')
    plt.ylabel('Money Spent on Wine')
    plt.title('Customer Segmentation Visualization')
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    encoded_img = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    
    return f'<img src="data:image/png;base64,{encoded_img}"/>'

if __name__ == '__main__':
    app.run(debug=True)
