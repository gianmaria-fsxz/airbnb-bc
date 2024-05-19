import math
import numpy as np
from typing import Mapping, Tuple
import pandas as pd

def euclidean_distance(p1: Mapping[int, float], p2: Mapping[int, float]) -> float:
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def convert_to_cartesian(lat:float, lon:float) -> Tuple[float, float]:
        R = 6371  # Radius of the Earth in kilometers
        x = R * math.cos(math.radians(lat)) * math.cos(math.radians(lon))
        y = R * math.cos(math.radians(lat)) * math.sin(math.radians(lon))
        return (x, y)

def get_dist_from_bc(df: pd.DataFrame) -> pd.DataFrame:
    """Add column 'dist_from_bc' as the distance in km between the property and the baricenter

    Args:
        df (pd.DataFrame): input Dataframe with coordinates 'latitude' and 'longitude' to provide as float nums

    Returns:
        pd.DataFrame: equal to input with the addition of the column 'dist_from_bc'
    """

    # Function to convert latitude and longitude to Cartesian coordinates
    

    geo_df = df[['airbnb_property_id','latitude', 'longitude']].drop_duplicates()
    geo_df["cartesian_coordinates"] = [convert_to_cartesian(*a) for a in zip(geo_df["latitude"], geo_df["longitude"])]
    #we assume earth is locally flat
    barycenter = np.mean(np.vstack(geo_df.cartesian_coordinates.values), axis=0)
    geo_df['dist_from_bc'] = geo_df['cartesian_coordinates'].map(lambda x : euclidean_distance(barycenter,x))

    #hardcoding variables (key) is not the best, but in this case we would just get a less readable code 
    return pd.merge(df, geo_df[['airbnb_property_id', 'dist_from_bc']], on = 'airbnb_property_id', how='left'
             )

def get_num_neighbours(df: pd.DataFrame) -> pd.DataFrame:
    """Add column 'num_neighbours' as the number of neighbours in the radius of 0.1 km

    Args:
        df (pd.DataFrame): input Dataframe with coordinates 'latitude' and 'longitude' to provide as float nums

    Returns:
        pd.DataFrame: equal to input with the addition of the column 'num_neighbours'
    """
    
    geo_df = df[['airbnb_property_id','latitude', 'longitude']].drop_duplicates()
    geo_df["cartesian_coordinates"] = [convert_to_cartesian(*a) for a in zip(geo_df["latitude"], geo_df["longitude"])]
    coord_set = np.vstack(geo_df['cartesian_coordinates'].drop_duplicates().values)
    geo_df['num_neighbours'] = geo_df['cartesian_coordinates'].apply(lambda x: len([d for d in [euclidean_distance(x, cs) for cs in coord_set] if d < .1]) - 1)

    #hardcoding variables (key) is not the best, but in this case we would just get a less readable code 
    return pd.merge(df, geo_df[['airbnb_property_id', 'num_neighbours']], on = 'airbnb_property_id', how='left'
             )