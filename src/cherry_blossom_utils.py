import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

CLUSTERS_FOLDER = "./data/interim/representative_locations/clusters/"
LOCATIONS_FOLDER = "./data/interim/representative_locations/location/"
VALIDATION_FOLDER = "./data/interim/representative_locations/validation/"

dir_paths = ["./data/interim/representative_locations/weather/src",
             "./data/interim/main_locations/weather/src"]

OUTPUT_PATHS = ["./data/interim/representative_locations/weather/monthly_agg/",
                "./data/interim/main_locations/weather/monthly_agg/"]

VALIDATION_PATHS = ["./data/interim/representative_locations/validation/",
                    "./data/interim/main_locations/validation"]

CLEANED_PATHS = ["./data/interim/representative_locations/weather/monthly_agg/",
                 "./data/interim/main_locations/weather/monthly_agg"]

INDIV_OUT_PATH = "./data/cleaned/individual/"
COMBINED_OUT_PATH = "./data/cleaned/"

location_dict = {
    "korea":"./data/raw/south_korea.csv",
    "japan":"./data/raw/japan.csv",
    "switzerland":"./data/raw/meteoswiss.csv",
    "usa":"./data/raw/usa_formatted.csv"
}

def format_df(df):
    # Drop any rows with no Observation_Date value
    # usa_npn.dropna(subset=['Observation_Date'], inplace=True)

    # Cast Observation_Date column as datetime
    df['Observation_Date'] = pd.to_datetime(df['Observation_Date'])

    # Extract the year and create a new column
    df.insert(0, "Year", df['Observation_Date'].dt.year)

    # Add the location name
    df.insert(0, "location", df['Site_ID'])

    # Initialize empty dataframe
    full_bloom_df = pd.DataFrame()

    # Iterate through unique combinations of Species_ID and Site_ID
    for species_id, site_id in df[['Species_ID', 'Site_ID']].drop_duplicates().values:
        # Copy USA-NPN dataset for the specific Species_ID and Site_ID combination
        t_df = df[(df['Species_ID'] == species_id) & (df['Site_ID'] == site_id)]

        # Iterate through every year available in the dataset for the specific Species_ID and Site_ID combination
        for year in t_df.Year.unique():
            # Subset by selected year
            t_df_year = t_df[t_df['Year'] == year]

            # Find the first day where the Phenophase_Status was 1 (e.g. bloom date)
            t_df_year = t_df_year[t_df_year['Phenophase_Status'] == 1].sort_values(by=['Day_of_Year'],
                                                                                   ascending=True).head(1)

            # Concatenate the bloom date row with the results dataframe
            full_bloom_df = pd.concat([full_bloom_df, t_df_year])

    # Drop unecessary column
    full_bloom_df = full_bloom_df.drop(columns=['Phenophase_Status'])
    full_bloom_df = full_bloom_df.reset_index(drop=True)

    return full_bloom_df


def get_representative_locations(file_path, output_name):
    # Read the CSV file
    data = pd.read_csv(file_path)

    # Calculate the count of observations for each location
    num_obs_per_location = data.groupby('location').size().reset_index(name='num_obs')

    # Remove duplicate points
    data_no_duplicates = data.drop_duplicates(subset=['lat', 'long'])

    # Feature Scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_no_duplicates[['lat', 'long']])

    # Clustering
    num_clusters = min(len(data_no_duplicates), 5)  # Adjusting number of clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    data_no_duplicates = data_no_duplicates.copy()
    data_no_duplicates['cluster'] = kmeans.fit_predict(scaled_data)

    # Select Representative Locations by which ones have the most observations
    representative_locations = data_no_duplicates.groupby('cluster').apply(lambda x: x.max())

    representative_locations = representative_locations[['cluster', 'location', 'lat', 'long']]

    # Merge representative locations with the count of observations
    representative_locations = pd.merge(representative_locations, num_obs_per_location, on='location', how='left')

    # Get bloom data for each location
    location_data = [data[data['location'] == row['location']] for _, row in representative_locations.iterrows()]

    return data_no_duplicates[['cluster', 'location', 'lat', 'long']], representative_locations, location_data


def clean_weather_df(file_path):
    df = pd.read_csv(file_path)

    # Get name for the dataframe
    df_name_str = df.iloc[0]['name']

    # Calculate the total seconds of sunlight in the day
    df['sunrise'] = pd.to_datetime(df['sunrise'], format='%Y-%m-%dT%H:%M:%S')
    df['sunset'] = pd.to_datetime(df['sunset'], format='%Y-%m-%dT%H:%M:%S')
    df['day_length'] = (df['sunset'] - df['sunrise']).dt.total_seconds()

    # Convert 'datetime' column to datetime datatype
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Extract year and month
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month

    # Group by year and calculate average values for each year
    yearly_avg = df.groupby('year').agg({
        'day_length': 'mean',
        'tempmax': 'mean',
        'tempmin': 'mean',
        'temp': 'mean',
        'dew': 'mean',
        'humidity': 'mean',
        'precip': 'mean',
        'precipprob': 'mean',
        'precipcover': 'mean',
        'snowdepth': 'mean',
        'windgust': 'mean',
        'windspeed': 'mean',
        'windspeedmax': 'mean',
        'windspeedmean': 'mean',
        'windspeedmin': 'mean',
        'sealevelpressure': 'mean',
        'cloudcover': 'mean',
        'solarradiation': 'mean',
        'solarenergy': 'mean',
        'uvindex': 'mean'
    }).reset_index()

    # Calculate monthly averages for each feature
    monthly_avg = df.groupby(['year', 'month']).mean(numeric_only=True).unstack()

    # Calculate yearly averages for each feature
    yearly_avg = df.groupby('year').mean(numeric_only=True)

    # Drop redundant columns
    yearly_avg.drop(
        columns=['day_length', 'month', 'tempmax', 'tempmin', 'temp', 'dew', 'humidity', 'precip', 'precipprob',
                 'precipcover', 'snowdepth', 'windgust', 'windspeed', 'windspeedmax', 'windspeedmean', 'windspeedmin',
                 'sealevelpressure', 'cloudcover', 'solarradiation', 'solarenergy', 'uvindex'], inplace=True)

    # Combine monthly and yearly averages
    final_df = pd.concat([yearly_avg, monthly_avg], axis=1)

    # Rename columns for clarity
    final_df.columns = [f'{col[0]}_{col[1]:02d}' if isinstance(col, tuple) else col for col in final_df.columns]

    # Turn the index column into a real column
    final_df.reset_index(inplace=True)

    # Add the name of the dataframe
    final_df.insert(0, 'location', df_name_str)

    return final_df


def process_weather_files(dir_path):
    # Check if the path exists
    if not os.path.exists(dir_path):
        print(f"Path '{dir_path}' does not exist.")
        return

    df_dict = {}

    # Iterate through files in the directory
    for filename in os.listdir(dir_path):
        if filename.endswith('.csv'):
            # Construct the full file path
            file_path = os.path.join(dir_path, filename)

            # Read the CSV file into a pandas DataFrame
            df = clean_weather_df(file_path)

            # Append the cleaned df to the list
            df_dict[filename] = df

    # Return the list of cleaned dfs
    return df_dict


def clean_df(v_path, c_path):
    # Rearrange the columns in the validation dataset
    validation_columns_format = ['location', 'bloom_doy', 'year', 'lat', 'long', 'bloom_date', 'alt']

    # Load in validation df
    v_df = pd.read_csv(v_path)
    v_df = v_df[validation_columns_format]

    # Make extra column to ease merge
    v_df['pred_year'] = v_df['year'] - 1

    # Drop unused column
    v_df = v_df.drop(columns=['year'])

    # Load in cleaned df
    c_df = pd.read_csv(c_path)

    # Drop unused column
    c_df = c_df.drop(columns=["location"])

    # Combine them on the 'pred_year'='year'
    combined_df = pd.merge(v_df, c_df, left_on='pred_year', right_on='year', how='inner')

    # Drop unused columns
    combined_df = combined_df.drop(columns=["pred_year", 'year'])

    # Cast column to datetime
    combined_df['bloom_date'] = pd.to_datetime(combined_df['bloom_date'])

    # Recreate year column
    combined_df.insert(1, 'year', combined_df['bloom_date'].dt.year)

    return combined_df


def combine_files(validation_path, cleaned_path):
    # Check if the path exists
    if not os.path.exists(validation_path):
        print(f"Path '{validation_path}' does not exist.")
        return

    if not os.path.exists(cleaned_path):
        print(f"Path '{cleaned_path}' does not exist.")
        return

    df_dict = {}

    # Iterate through files in the directory
    for filename in os.listdir(validation_path):
        if filename.endswith('.csv'):
            # Construct the full file path
            v_file_path = os.path.join(validation_path, filename)
            c_file_path = os.path.join(cleaned_path, filename)

            # Read the CSV file into a pandas DataFrame
            df = clean_df(v_file_path, c_file_path)

            # Append the cleaned df to the list
            df_dict[filename] = df

    # Return the list of cleaned dfs
    return df_dict