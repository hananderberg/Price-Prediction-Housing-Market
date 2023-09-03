import pandas as pd

df1 = pd.read_csv('data/original_data/HemnetScraper (2).csv')
df2 = pd.read_csv('data/original_data/HemnetScraper (3).csv')
df3 = pd.read_csv('data/original_data/HemnetScraper (4).csv')

df = pd.concat([df1, df2, df3], ignore_index=True)

columns_to_drop = ['web-scraper-order', 'web-scraper-start-url', 'Links', 'Links-href','Pages']
df = df.drop(columns=columns_to_drop)

df['Adress'] = df['Adress'].astype(str)
df['Adress'] = df['Adress'].str.replace('Slutpris\n  ', '')
df['Adress'] = df['Adress'].astype(str)

# Function to clean currency strings and convert to numbers
def clean_currency_and_convert(currency_string):
    if pd.notna(currency_string):
        cleaned_string = currency_string.replace("\xa0", "").replace("kr", "")
        return int(cleaned_string)
    else:
        return currency_string

df['Slutpris'] = df['Slutpris'].apply(clean_currency_and_convert)
df['Utgångspris'] = df['Utgångspris'].fillna('0').apply(clean_currency_and_convert)

# Function to remove " rum" from strings
def clean_room_string(room_string):
    if pd.notna(room_string):
        cleaned_string = room_string.replace(" rum", "").replace(",", ".")
        return pd.to_numeric(cleaned_string, errors='coerce')
    else:
        return room_string
    
# Apply the function to the entire column
df['Antal rum'] = df['Antal rum'].apply(clean_room_string)

# Function to remove " m2" from strings
def clean_m2_string(m2_string):
    if pd.notna(m2_string):
        cleaned_string = m2_string.replace(" m²", "").replace(",", ".")
        return pd.to_numeric(cleaned_string, errors='coerce')
    else:
        return m2_string

# Apply the function to the entire column
df['Boarea'] = df['Boarea'].apply(clean_m2_string)

# Split the column values and create a new column
split_data = df['Våning'].str.split(', ', n=1, expand=True)
df['Hiss'] = split_data[1]
df['Våning'] = split_data[0]

df['Våning'] = df['Våning'].str.split(' ', n=1).str.get(0)
df['Våning'] = df['Våning'].str.replace(',', '.').astype(float)

# Function to map hiss
def map_ja_nej_to_bool_hiss(value):
    if pd.notna(value):
        if value.lower() == "hiss finns":
            return "Ja"
        else:
            return "Nej"
    else:
        return value

# Apply the function to the entire column
df['Hiss'] = df['Hiss'].apply(map_ja_nej_to_bool_hiss)

df['Avgift/månad'] = df['Avgift/månad'].str.replace(r'\D', '', regex=True).apply(pd.to_numeric) 
df['Driftskostnad/år'] = df['Driftskostnad/år'].str.replace(r'\D', '', regex=True).apply(pd.to_numeric) 

df['Byggår'] = df['Byggår'].fillna('0')
df['Byggår'] = df['Byggår'].str.split(' - ').str[0].astype(int)

# Create boolean masks for filtering
mask_bostadstyp = df['Bostadstyp'] == 'Lägenhet'
mask_upplatelseform = df['Upplåtelseform'].isin(['Bostadsrätt', 'Andel i bostadsförening'])

# Apply the masks to filter the DataFrame
filtered_df = df[mask_bostadstyp & mask_upplatelseform]

df.to_csv('data/cleaned_data.csv', index=False)
