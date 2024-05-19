import sys
from utils.geo_processing_utils import get_dist_from_bc, get_num_neighbours
from utils.io_utils import read_and_rename

if __name__ == "__main__":
    print('reading file')
    df = read_and_rename("./BrightonPerformanceData.csv")
    
    print('computing dist_from_bc')
    df = get_dist_from_bc(df)
    print('computing num_neighbours')
    df = get_num_neighbours(df)