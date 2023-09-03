import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import requests
import json
import time
import re

def handleNaNs(df):
    # Delete rows where boarea is NaN
    df = df.dropna(subset=['Boarea'])

    # Delete rows where avgift is NaN
    df = df.dropna(subset=['Avgift/månad'])

    # Replace None values in 'Balkong' and 'Hiss' column with 'Nej'
    df['Balkong'].fillna('Nej', inplace=True)
    df['Hiss'].fillna('Nej', inplace=True)

    # Calculate the mode of the våning column
    vaning_mode = df['Våning'].mode()[0]
    
    # Replace NaN values with the mode
    df['Våning'].fillna(vaning_mode, inplace=True)

    # Calculate the median of the column
    column_median = df['Driftskostnad/år'].median()

    # Replace NaN values with the median
    df['Driftskostnad/år'].fillna(column_median, inplace=True)     

    return df

def oneHotEncoding(df):
    one_hot_columns = ['Upplåtelseform', 'Balkong', 'Hiss']
    df_one_hot = pd.get_dummies(df, columns=one_hot_columns)
    
    return df_one_hot

def split_address(address):
    match = re.search(r'(\d+)', address)  # Find the first number
    if match:
        index = match.end()  # Get the end index of the first number
        new_address = address[:index]  # Get the part of the address up to the number
        return new_address.strip()
    return address

def findGeo(df):
    df['Adress endast'] = df['Adress'].apply(split_address)

    addresses_only = df['Adress endast']
    addresses_with_suffix = addresses_only + ", Stockholm, Sweden"
    # Update the "Address" column with the new addresses
    df['Adress endast'] = addresses_with_suffix

    # Create empty lists to store latitude and longitude values
    latitudes = []
    longitudes = []

    # Define your User-Agent string
    user_agent = "Price-Prediction-Housing-Market"  

    # Add the User-Agent header to the request
    headers = {'User-Agent': user_agent}

    count = 0
    # Loop through addresses and perform geocoding
    for address in addresses_with_suffix:
        time.sleep(1)
        count = count + 1
        print(count)
        url = f"https://nominatim.openstreetmap.org/search?q={address}&format=json&limit=1"
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = json.loads(response.text)
                if data and 'lat' in data[0] and 'lon' in data[0]:
                    latitudes.append(data[0]['lat'])
                    longitudes.append(data[0]['lon'])
                else:
                    print("No coordinates found for:", address)
                    latitudes.append(None)
                    longitudes.append(None)
            else:
                print("Request failed for:", address)
                print("Request failed with status code:", response.status_code)
                print("Response text:", response.text)
                latitudes.append(None)
                longitudes.append(None)
        except requests.exceptions.RequestException as e:
            print("Request Exception:", e)
            print("Error for:", address)
            latitudes.append(None)
            longitudes.append(None)  

    longitudes_df = pd.DataFrame(longitudes, columns=['Longitudes'])
    latitudes_df = pd.DataFrame(latitudes, columns=['Latitudes'])
    
    return longitudes_df, latitudes_df

def main():
    # Read the data
    csv_file_path = 'data/cleaned_data_with_coordinates.csv'
    df = pd.read_csv(csv_file_path)

    # Drop Adress and Datum columns
    columns_to_drop = ['Adress', 'Datum', 'Bostadstyp']
    df_geo_dropped = df.drop(columns=columns_to_drop)
    
    # Handle NaN values
    df_noNaNs = handleNaNs(df_geo_dropped)

    # One-Hot-Encoding
    df_one_hot = oneHotEncoding(df_noNaNs)

    df_one_hot.to_csv('data/data_ready.csv', index=False)

if __name__ == "__main__":
    main()



'''# One-hot-encoding
one_hot_columns = ['Balkong', 'Hiss']
df = pd.get_dummies(df, columns=one_hot_columns)

# Remove adress and datum column
print(df.columns)

# Split into test and training
y = df['Slutpris']
X = df.drop("Slutpris", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Drop the 'Adress' and 'Datum' columns from X_train and X_test
X_train = X_train.drop(['Adress', 'Datum'], axis=1)
X_test = X_test.drop(['Adress', 'Datum'], axis=1)

# GradientBoosting
reg = GradientBoostingRegressor(random_state=0)
reg.fit(X_train, y_train)
pred = reg.predict(X_test)

new_data = {
    'Utgångspris': [5000000],
    'Antal rum': [2],
    'Boarea': [41],
    'Våning': [3],
    'Byggår': [1899],
    'Avgift/mån': [2322],
    'Balkong_Ja': [0],
    'Balkong_Nej': [1],
    'Hiss_Ja': [1],
    'Hiss_Nej': [0]
}
new_row = pd.DataFrame(new_data)

predi = reg.predict(new_row)
# Adress,Datum,Slutpris,Utgångspris,Antal rum,Boarea,Balkong,Våning,Byggår,Avgift/mån,Hiss
# Teknologgatan 

print(predi)

data = {'True value': y_test, 'Estimated value': pred}
result = pd.DataFrame(data)

result.to_csv('data/results.csv')'''
## SVM-Support Vector Machine, Random Forest Regressor, Linear Regressor