import torch
import pandas as pd
from scipy.spatial import distance_matrix
from math import sqrt

def list_with_coordinates_to_two_lists_with_x_and_y(coords):
    #[x1,y1],[x2,y2]... -> [x1,x2,x3..][y1,y2,y3...]
    xs = []
    ys = []
    for i in range(len(coords[0])):
        coordinates_x_y = coords[0][i]
        xs.append(coordinates_x_y[0])
        ys.append(coordinates_x_y[1])
    return [xs, ys]

'''
inputs:   
    coordinates = [[5, 7], [7, 3], [8, 1]]
    ctys = ['Boston', 'Phoenix', 'New York']
output:
              xcord ycord
    Boston      5   7
    Phoenix     7   3
    New York    8   1
'''
def get_distance_matrix_from_coordinate_pairs(coordinates, ctys):
    df = pd.DataFrame(coordinates, columns=['xcord', 'ycord'], index=ctys)
    result = pd.DataFrame(distance_matrix(df.values, df.values), index=df.index, columns=df.index)
    return result

def get_distance_from_coordinate_pairs(coordinates, cities, tour):
    dist_matrix = get_distance_matrix_from_coordinate_pairs(coordinates, cities)
    tour_len =0
    for i in range(len(tour)-1):
        tour_len += dist_matrix[i][i+1]
    return tour_len
#
# coordinates = [[5, 7], [7, 3], [8, 1]]
# tour = [1, 2,0]
# ctys = [0, 1, 2]
# result =  get_distance_from_coordinate_pairs(coordinates, ctys,tour)
# print(result)

"""
        Parameters
        ----------
        static: torch.FloatTensor containing static (e.g. x, y) data
        static: [[x1,x2,x3..xnumOfNodes][y1..ynumOfNodes]]
        tour_indices: torch.IntTensor of size (batch_size, num_cities)
        Returns
        -------
        Euclidean distance between consecutive nodes on the route. of size
        (batch_size, num_cities)
"""
def reward(static, tour_indices):
    # Convert the indices back into a tour
    if static[0].shape[0]!=static[0].shape[1]:
        static = list_with_coordinates_to_two_lists_with_x_and_y(static)
    #idx = tour_indices.unsqueeze(1).expand_as(static)
    idx = tour_indices.expand_as(torch.tensor(static))
    tour = torch.gather(static.data, 2, idx).permute(0, 2, 1)
    # Make a full tour by returning to the start
    tour_returning_to_start = torch.cat((tour, tour[:, :1]), dim=1)
    # Euclidean distance between each consecutive point
    tour_len = torch.sqrt(torch.sum(torch.pow(tour_returning_to_start[:, :-1] - tour_returning_to_start[:, 1:], 2), dim=2))
    return tour_len.sum(1).detach()