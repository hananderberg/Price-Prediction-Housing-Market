import pandas as pd
from sklearn.cluster import KMeans

def kMeans(df, latitudes, longitides):
    df = pd.read_csv('data/cleaned_data.csv')
    latitudes = pd.read_csv('data/latitudes.csv')
    longitudes = pd.read_csv('data/longitudes.csv')

    # Create a DataFrame with the geolocation data
    data = pd.concat([latitudes, longitudes], axis=1)

    # Number of clusters (zones) to create
    num_clusters = 10

    # Initialize the KMeans model
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)

    # Fit the model to the data
    kmeans.fit(data[['Latitudes', 'Longitudes']])

    # Add the cluster labels as a new column in the DataFrame
    df['Områdeskod'] = kmeans.labels_

    df.to_csv('data/cleaned_with_geocodes.csv', index=False)