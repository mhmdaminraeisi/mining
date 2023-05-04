import pandas as pd
import numpy as np
import time
import random
from utils import *
from data_handler import *
from assignment import *


def run_assignments_with_bags(method, u_points: List[UncertainPoint], centers: List[Center], bag_size: int) -> Tuple:
    start_time = time.time()
    assignments = method(u_points, centers, bag_size)
    cost = ecost_of_an_assignment(u_points, centers, assignments)
    run_time = time.time() - start_time
    return cost, run_time
    

def run_assignments(method, u_points: List[UncertainPoint], centers: List[Center]) -> Tuple:
    start_time = time.time()
    assignments = method(u_points, centers)
    cost = ecost_of_an_assignment(u_points, centers, assignments)
    run_time = time.time() - start_time
    return cost, run_time
    

ep_c, prob_c, ex_bag_c, opt_bag_c, ep_r, prob_r, ex_bag_r, opt_bag_r = [], [], [], [], [], [], [], []
nn, kk, zz, bb = [], [], [], []

for z_sqrt in [2, 3]:
    print('reading data z = {} ...'.format(z_sqrt * z_sqrt))
    u_points = transfer_csv_to_desired_format('data/pokemon-spawns.csv', 's2_token', [z_sqrt, z_sqrt])
    for n in [10, 50, len(u_points)]:
        sample = random.sample(u_points, n)
        for k in [3, 4]:
            expected_coordinates = list(map(lambda up: get_expected_coordinate_of_uncertain_point(up), sample))
            centers = generate_k_center_with_kmeans(expected_coordinates, k) 
            # for bag_size in [2, int(np.ceil(np.log2(len(sample))))]:
            for bag_size in [2, 5]:
                print('for n = {}, k = {}, z = {}, bag_size = {} ...'.format(n, k, z_sqrt * z_sqrt, bag_size))
                nn.append(n)
                kk.append(k)
                zz.append(z_sqrt * z_sqrt)
                bb.append(bag_size)
                print('running expected_point_assignments ...')
                cost, run_time = run_assignments(get_expected_point_assignments, sample, centers)
                ep_c.append(round(cost, 4))
                ep_r.append(round(run_time, 4))
                print('running probable_center_assignments ...')
                cost, run_time = run_assignments(get_probable_center_assignment, sample, centers)
                prob_c.append(round(cost, 4))
                prob_r.append(round(run_time, 4))
                print('running bagging_assignments ...')
                cost, run_time = run_assignments_with_bags(get_bagging_assignments, sample, centers, bag_size)
                ex_bag_c.append(round(cost, 4))
                ex_bag_r.append(round(run_time, 4))
                print('running bagging_assignments_with_opts ...')
                cost, run_time = run_assignments_with_bags(get_bagging_assignments_with_fixed_prev_opts, sample, centers, bag_size)
                opt_bag_c.append(round(cost, 4))
                opt_bag_r.append(round(run_time, 4))
                
                
save_data_to_csv('results/pokemon_result.csv', nn, zz, kk, bb, ep_c, prob_c, ex_bag_c, opt_bag_c, ep_r, prob_r, ex_bag_r, opt_bag_r)

