import pandas as pd
import numpy as np
import time
import random
from utils import *
from data_handler import *
from assignment import *
from config import *
import sys

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
    


for data_name in sys.argv[1:]:
    ep_c, prob_c, ex_bag_c, opt_bag_c, ep_r, prob_r, ex_bag_r, opt_bag_r = [], [], [], [], [], [], [], []
    nn, kk, zz, bb = [], [], [], []
    prev_z_sqrt = -1

    for p in parameters.get(data_name):
        n, z_sqrt, k, bag_size = p[0], p[1], p[2], p[3]
        z = z_sqrt * z_sqrt
        if prev_z_sqrt < z_sqrt:
            print('reading {} data z = {} ...'.format(data_name, z))
            u_points = transfer_csv_to_desired_format('data/{}.csv'.format(data_name), 's2_token', [z_sqrt, z_sqrt])
            prev_z_sqrt = z_sqrt
        
        if not check_params_is_valid_for_bagging_assignments(n, z, k, bag_size):
            continue

        sample = u_points[0:n]
        expected_coordinates = list(map(lambda up: get_expected_coordinate_of_uncertain_point(up), sample))
        centers = generate_k_center_with_k_center(expected_coordinates, k) 

        print('for n = {}, k = {}, z = {}, bag_size = {} ...'.format(n, k, z, bag_size))
        nn.append(n)
        kk.append(k)
        zz.append(z)
        bb.append(bag_size)
        print('running expected_point_assignments ...')
        cost, run_time = run_assignments(get_expected_point_assignments, sample, centers)
        ep_c.append(round(cost, 4))
        ep_r.append(round(run_time, 4))
        print(cost, run_time)
        print('running probable_center_assignments ...')
        cost, run_time = run_assignments(get_probable_center_assignment, sample, centers)
        prob_c.append(round(cost, 4))
        prob_r.append(round(run_time, 4))
        print(cost, run_time)
        print('running bagging_assignments ...')
        cost, run_time = run_assignments_with_bags(get_bagging_assignments, sample, centers, bag_size)
        ex_bag_c.append(round(cost, 4))
        ex_bag_r.append(round(run_time, 4))
        print(cost, run_time)
        print('running bagging_assignments_with_opts ...')
        cost, run_time = run_assignments_with_bags(get_bagging_assignments_with_fixed_prev_opts, sample, centers, bag_size)
        opt_bag_c.append(round(cost, 4))
        opt_bag_r.append(round(run_time, 4))
        print(cost, run_time)
        save_data_to_csv('results/{}.csv'.format(data_name), nn, zz, kk, bb, ep_c, prob_c, ex_bag_c, opt_bag_c, ep_r, prob_r, ex_bag_r, opt_bag_r)

