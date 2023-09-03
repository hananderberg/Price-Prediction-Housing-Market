from geopy.geocoders import Nominatim
import pandas as pd

df = pd.read_csv('data/cleaned_data.csv')
latitudes = pd.read_csv('data/latitudes.csv')
longitudes = pd.read_csv('data/longitudes.csv')

combined_df = pd.concat([df, latitudes, longitudes], axis=1)

combined_df.to_csv('data/cleaned_data_with_latitudes.csv', index=False)

'''
# Sample addresses from your dataset
df = pd.read_csv('data/cleaned_data.csv')
latitudes = pd.read_csv('data/latitudes.csv')
longitudes = pd.read_csv('data/longitudes.csv')

print(latitudes[latitudes.isna().any(axis=1)])
print(longitudes[longitudes.isna().any(axis=1)])

## plats 518 och 767 

print(df.iloc[516]) # 59.33494186401367 latitude 18.032901763916016 longitud
print(df.iloc[765]) # 59.3375336 latitude 18.0759703 longitud

print(latitudes.iloc[516]) 
print(latitudes.iloc[765])
      
first_lat = 59.33494186401367  # The value you want to add
first_long = 18.032901763916016 

second_lat = 59.3375336
second_long = 18.0759703

first_row_index = 516  # The row index where you want to add the value
second_row_index = 765  # The row index where you want to add the value

# Modify the value in the specified cell
latitudes.at[first_row_index, 'Latitudes'] = first_lat
longitudes.at[first_row_index, 'Longitudes'] = first_long

latitudes.at[second_row_index, 'Latitudes'] = second_lat
longitudes.at[second_row_index, 'Longitudes'] = second_long

print(latitudes[latitudes.isna().any(axis=1)])
print(longitudes[longitudes.isna().any(axis=1)])

longitudes.to_csv('data/longitudes.csv', index=False)
latitudes.to_csv('data/latitudes.csv', index=False)

geolocator = Nominatim(user_agent="price_prediction_housing_market", scheme='http')

addresses = ["Tomtebogatan 15, Stockholm, Sweden"]

coordinates = []

for address in addresses:
    location = geolocator.geocode(address)
    if location:
        coordinates.append((location.latitude, location.longitude))
    else:
        coordinates.append((None, None))

print(coordinates)'''
