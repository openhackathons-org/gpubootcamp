import cudf
import os
import pandas as pd
import urllib
import tqdm
from zipfile import ZipFile 

pbar = None

def show_progress(block_num, block_size, total_size):
    global pbar
    if pbar is None:
        pbar = tqdm.tqdm(total=total_size / 1024, unit='kB')

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(block_size / 1024)
    else:
        pbar.close()
        pbar = None
        

def fetch_bike_dataset(years, data_dir="data"):
    """ Dowload bike dataset for a given year and return the list of files.
    """
    base_url = "https://s3.amazonaws.com/capitalbikeshare-data/"
    files = []
    for year in years:
        filename = str(year) + "-capitalbikeshare-tripdata.zip"
        filepath = os.path.join(data_dir, filename)

        if not os.path.isfile(filepath):
            urllib.request.urlretrieve(base_url+filename, filepath, reporthook=show_progress)

        with ZipFile(filepath) as myzip:
            files += [os.path.join(data_dir, name) for name in myzip.namelist()]
            myzip.extractall(data_dir)
    
    print("Files extracted: "+ str(files))
    return files


def fetch_weather_dataset(data_dir='data'):
    base_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00275/'
    fn = 'Bike-Sharing-Dataset.zip'
    
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    filepath = os.path.join(data_dir, fn)    
    
    if not os.path.isfile(filepath):
        print(f'Downloading {base_url+fn} to {filepath}')
        urllib.request.urlretrieve(base_url+fn, filepath)
    
    files = []
    with ZipFile(filepath) as myzip:
        files = [os.path.join(data_dir, name) for name in myzip.namelist()]
        myzip.extractall(data_dir)
    
    # Extract weather features from the dataset
    # Note this weather dataset is already preprocessed.
    # We reverse the steps to provide a more interesting exercise.
    weather = cudf.read_csv(files[2], parse_dates=[1])
    out = cudf.DataFrame();
    out['Hour'] = weather['dteday'] + cudf.Series(pd.to_timedelta(weather['hr'].to_pandas(), unit='h'))
    out['Temperature'] = weather['temp'] * 47.0 -8
    out['Relative Temperature'] = weather['atemp'] * 66.0 - 16
    out['Rel. humidity'] = (weather['hum'] * 100).astype('int')
    out['Wind'] = weather['windspeed'] * 67
    
    # Spell out weather categories
    # - 1: Clear, Few clouds, Partly cloudy, Partly cloudy
    #- 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
    #- 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
    # - 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog 
    out['Weather'] = 'Clear or Partly cloudy'
    out['Weather'][weather['weathersit']==2] = 'Mist or Cloudy'
    out['Weather'][weather['weathersit']==3] = 'Light Rain or Snow, Thunderstorm'
    out['Weather'][weather['weathersit']==4] = 'Heavy Rain, Snow + Fog, Ice'
    
    filepath = os.path.join(data_dir, 'weather2011-2012.csv')
    out.to_csv(filepath, index=False)
    print("Weather file saved at ", filepath)
    return filepath


def read_bike_data_pandas(files):
    # Reads a list of files and concatenates them
    tables = []
    for filename in files:
        tmp_df = pd.read_csv(filename, usecols=[1], parse_dates=['Start date'])
        tables.append(tmp_df)

    merged_df = pd.concat(tables, ignore_index=True)