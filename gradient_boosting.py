from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
import requests
import json
import time
import matplotlib.colors as mcolors
from ipyleaflet import Map, CircleMarker
import re
from sklearn.svm import SVR

new_data = {
    'Adress':"Hälsingegatan 33", 
    'Antal rum': [2],
    'Boarea': [42],
    'Våning': [5.0],
    'Byggår': [1924],
    'Avgift/månad': [3134],
    'Driftskostnad/år': [6000],
    'Balkong_Ja':[0],
    'Balkong_Nej':[1],
    'Områdeskod': [0],
}

def kMeans(df, num_clusters):
    # Initialize the KMeans model
    kmeans_model = KMeans(n_clusters=num_clusters, random_state=0)

    # Fit the model to the data
    kmeans_model.fit(df[['Latitudes', 'Longitudes']])

    df['Områdeskod'] = kmeans_model.labels_

    visualize_clusters(df, kmeans_model, num_clusters)
        
    columns_to_drop = ['Longitudes', 'Latitudes']
    df.drop(columns=columns_to_drop, inplace=True)

    return kmeans_model, df

# Generate a list of colors
def generate_endless_colors(num_colors):
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i) for i in np.linspace(0, 1, num_colors)]
    return colors


def visualize_clusters(df, kmeans_model, num_clusters):
    latitude_center = 59.33939658166259
    longitude_center = 18.05009156898074
    m = Map(center=[latitude_center, longitude_center], zoom=12)

    cluster_colors = generate_endless_colors(num_clusters)

    for cluster_label in range(kmeans_model.n_clusters):
        cluster_data = df[df['Områdeskod'] == cluster_label]
        for _, row in cluster_data.iterrows():
            rgba_color = cluster_colors[cluster_label % len(cluster_colors)]
            html_color = mcolors.to_hex(rgba_color[:3])  # Convert RGBA to HTML color

            circle_marker = CircleMarker(
                location=(row['Latitudes'], row['Longitudes']),
                radius=5,
                color=html_color,
                fill=True,
                fill_color=html_color,
                fill_opacity=0.7,
            )
            m.add_layer(circle_marker)

    m.save("cluster_map.html")


def trainGBModel(df):
    # Split into test and training
    y = df['Slutpris']
    X = df.drop("Slutpris", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # GradientBoosting
    reg = GradientBoostingRegressor(learning_rate = 0.1, max_depth = 5, min_samples_leaf = 1, 
                                    min_samples_split = 5, n_estimators = 500, subsample = 0.8, random_state=0)
    reg.fit(X_train, y_train)

    # Predict on the test data
    y_pred = reg.predict(X_test)

    # Calculate R2 score to evaluate the model
    r2 = r2_score(y_test, y_pred)

    return reg, r2, X_train

def optimizeParams(df):
    # Define the range of hyperparameters to search over
    param_grid = {'n_estimators': [10, 50, 100, 300, 500],
                  'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 1.0],
                   'max_depth': [3, 4, 5, 6, 7],
                    'min_samples_split':  [2, 3, 4, 5],
                    'min_samples_leaf': [1, 2, 3],
                    'subsample': [0.7, 0.8, 0.9, 1.0]
                  }
    regressor = GradientBoostingRegressor()
    grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid, cv=5, n_jobs=-1)
    
    # Assuming you have your data 'X' and target 'y'
    y = df['Slutpris']
    X = df.drop("Slutpris", axis=1)

    # Perform the grid search
    grid_search.fit(X, y)

    # Print the best parameters and best score
    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)

def findZoneIdNewData(new_object, kmeans_model):
    user_agent = "Price-Prediction-Housing-Market"  
    headers = {'User-Agent': user_agent}
    adress = new_object['Adress'][0]

    addresses_with_suffix = adress + ", Stockholm, Sweden"

    url = f"https://nominatim.openstreetmap.org/search?q={addresses_with_suffix}&format=json&limit=1"
    try:
        response = requests.get(url, headers=headers)
        time.sleep(1)
        if response.status_code == 200:
            data = json.loads(response.text)
            if data and 'lat' in data[0] and 'lon' in data[0]:
                lat = data[0]['lat']
                long = data[0]['lon']
            else:
                print("No coordinates found for:", adress)
        else:
            print("Request failed for:", adress)
            print("Request failed with status code:", response.status_code)
            print("Response text:", response.text)

    except requests.exceptions.RequestException as e:
        print("Request Exception:", e)
    
    new_coordinates = [[lat, long]]

    predicted_zone_id = kmeans_model.predict(new_coordinates)[0]
    print("Predicted zone-id:", predicted_zone_id)
    new_object['Områdeskod'] = predicted_zone_id
    column_to_drop = ['Adress']
    adress = new_object['Adress']
    new_object.drop(columns=column_to_drop, inplace=True)
    
    return new_object

def saveResult(adress, prediction):
    results = {
        'Adress': adress,
        'Slutpris': prediction,
    }
    results_df = pd.DataFrame(results)
    filename = f'results/results_{adress}.csv'
    results_df.to_csv(filename, index=False)

def featureImportance(reg_model, X):
    feature_importances = reg_model.feature_importances_
    sorted_indices = np.argsort(feature_importances)[::-1]
    sorted_feature_importances = feature_importances[sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.bar(range(X.shape[1]), sorted_feature_importances, align="center")
    plt.xticks(range(X.shape[1]), sorted_indices, rotation=90)
    plt.xlabel("Feature Index")
    plt.ylabel("Feature Importance")
    plt.title("Feature Importances")
    plt.show()

def trainRidgeModel(df):
    # Split into test and training
    y = df['Slutpris']
    X = df.drop("Slutpris", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # GradientBoosting
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)  

    # Calculate R2 score to evaluate the model
    r2 = r2_score(y_test, y_pred)

    return model, r2, X_train

def trainSVMModel(df):
    # Split into test and training
    y = df['Slutpris']
    X = df.drop("Slutpris", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate R2 score to evaluate the model
    r2 = r2_score(y_test, y_pred)

    return model, r2, X_train

def find_optimal_num_clusters(df, max_clusters=10):
    best_num_clusters = 0
    best_r2_score = -1  # Initialize with a negative value

    for num_clusters in range(1, max_clusters + 1):
        print(num_clusters)

        # Initialize and fit the KMeans model
        kmeans_model = KMeans(n_clusters=num_clusters, random_state=0)
        kmeans_model.fit(df[['Latitudes', 'Longitudes']])

        # Assign cluster labels to the data
        df['Områdeskod'] = kmeans_model.labels_

        # Calculate R2 score for the KMeans clustering
        r2 = calculate_r2(df)
        
        # Check if the current R2 score is better than the previous best
        if r2 > best_r2_score:
            best_r2_score = r2
            best_num_clusters = num_clusters

    return best_num_clusters, best_r2_score

def calculate_r2(df):
    # Perform your regression modeling here
    # Example: Train a GradientBoostingRegressor on the clustered data
    # You may need to adapt this part depending on your dataset and regression approach
    reg, r2, X_train = trainGBModel(df)
    return r2

def main():
    # Read file
    df = pd.read_csv('data/data_ready.csv')
    df = df.drop('Utgångspris', axis=1) # This just adds bias
    df = df.drop(columns=df.filter(like='Hiss').columns)
    df = df.drop(columns=df.filter(like='Upplåtelseform').columns)
    df = df[(df['Antal rum'] < 6.0)]

    # Optimize number of clusters
    #best_num_clusters, best_r2_score = find_optimal_num_clusters(df, max_clusters=30)
    # Optimize parameters
    #optimizeParams(df)

    # kMeans for coordinates
    kmeans_model, df = kMeans(df, 11)

    # Train GradientBoostingModel 
    model, r2, X_train = trainGBModel(df)
    #featureImportance(model, X_train)
    #model, r2, X_train = trainSVMModel(df)
    print("R2 score:", r2)

    # New data
    new_object = pd.DataFrame(new_data)
    adress = new_object['Adress'][0]

    # Find coordinates
    new_object = findZoneIdNewData(new_object, kmeans_model)

    # Predict new object final price
    prediction = model.predict(new_object)
    
    # Create csv file with result
    saveResult(adress, prediction)

    print("Predicted final price:", prediction)
    
    '''# Sorting
    #filtered_df = df[(df['Områdeskod'] == 3) | (df['Områdeskod'] == 6) & 
                     #(df['Antal rum'] == 2.0) & (df['Våning'] > 2.0) & (df['Balkong_Ja'] == 0.0) & (df['Balkong_Nej'] == 1.0)]

    #filtered_df = filtered_df[(df['Slutpris'] / df['Boarea']) > 127000.00]
    
    filtered_df = df[(df['Upplåtelseform_Bostadsrätt'] == 1)]
    
    avg_kvm_pris = (filtered_df['Slutpris'] / filtered_df['Boarea']).mean()
    
    filtered_df_non_boratt = df[(df['Upplåtelseform_Bostadsrätt'] == 0)]
    
    avg_kvm_pris_non_boratt  = (filtered_df_non_boratt ['Slutpris'] / filtered_df_non_boratt ['Boarea']).mean()

    print(avg_kvm_pris)
    print(avg_kvm_pris_non_boratt)
    #print(filtered_df.shape[0])
    #print("Average kvadratmeterpris", avg_kvm_pris)
    #print(filtered_df.head())'''

if __name__ == "__main__":
    main()
