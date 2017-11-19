import numpy as np
import sys
import random
import scipy.io as sp

class Clustering:

  def __init__(self, num_clusters, num_rand_intialization, num_iterations, input_data_file_name):
    self.num_clusters           = num_clusters
    self.num_rand_intialization = num_rand_intialization
    self.num_iterations         = num_iterations
    self.input_data_file_name   = input_data_file_name

  def read_data_from_file(self):
    with open(self.input_data_file_name) as f:
      data_points = f.read().splitlines()
    x = np.genfromtxt(data_points, delimiter = '')
    return x

  def randomly_initialize_centroids(self, num_clusters, data_set):
    return (random.sample(data_set, num_clusters)) 

  def find_the_closest_cluster_centroid(self, data_point, cluster_centriods, num_clusters):
    difference          = np.tile(data_point, (num_clusters, 1)) - cluster_centriods
    euclidian_dist      = np.sqrt(np.sum(difference**2, axis = 1))
    closest_index       = np.argmin(euclidian_dist)
    closest_centroid    = cluster_centriods[closest_index]
    return [closest_index, closest_centroid]

  def start(self):
    data_set           = self.read_data_from_file()
    m, n               = data_set.shape
    cost_clusters_hash = {}
    for _ in range(self.num_rand_intialization):
      cluster_centriods, data_p_c_centroids = self.converge_cluster_centroids(data_set, m ,n)
      cost                     = (1.0/m) * np.sum((data_set - data_p_c_centroids)**2)
      cost_clusters_hash[cost] = cluster_centriods
    lowest_cost_clusters = cost_clusters_hash[min(cost_clusters_hash.keys())]
    self.write_output_to_file(lowest_cost_clusters)

  def write_output_to_file(self, lowest_cost_clusters):
    np.savetxt('clusters.txt', lowest_cost_clusters, delimiter = ',', fmt = '%f')
    print("Successfully written cluster centroids to clusters.txt file!")

    
  def converge_cluster_centroids(self, data_set, m, n):
    cluster_centriods = self.randomly_initialize_centroids(self.num_clusters, data_set)
    for _ in range(self.num_iterations):
      c, data_p_c_centroids = self.assign_datapoints_to_clusters(data_set, cluster_centriods, self.num_clusters, m)
      cluster_centriods     = self.move_cluster_centroids(c, cluster_centriods, data_set, self.num_clusters)
    return [cluster_centriods, data_p_c_centroids]
  
  def assign_datapoints_to_clusters(self, data_set, cluster_centriods, num_clusters, num_rows):
    c                 = np.empty(num_rows)
    data_p_c_centroid = np.empty(data_set.shape)
    for i in range(num_rows):
      closest_index, closest_centroid = self.find_the_closest_cluster_centroid(data_set[i], cluster_centriods, num_clusters)
      c[i]                            = closest_index
      data_p_c_centroid[i]            = closest_centroid
    return [c, data_p_c_centroid]

  def move_cluster_centroids(self, c, cluster_centriods, data_set, num_clusters):
    for i in range(num_clusters):
      index_of_points_assigned_to_cluster = np.where(c == i)[0] 
      data_points_assigned_to_cluster     = data_set[index_of_points_assigned_to_cluster]
      cluster_centriods[i]                = (1.0/len(data_points_assigned_to_cluster)) * np.sum(data_points_assigned_to_cluster, axis = 0)
    return cluster_centriods





try:
  if (len(sys.argv) == 3):
    input_data_file = str(sys.argv[1])
    num_clusters    = int(sys.argv[2])
    if num_clusters < 1:
      raise ValueError('The number of clusters cannot be less than 1 ')
    else:
      num_iterations          = 100
      num_rand_intialization  = 10
      Clustering(num_clusters, num_rand_intialization, num_iterations, input_data_file).start()
  else:
    raise ValueError('Need two command line arguments')
except Exception as e:
  print('Error: ' + repr(e))
