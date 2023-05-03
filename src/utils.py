import numpy as np
import random
from typing import *
from entities import *
from functools import reduce


def euclidean_distance(coordinate: Coordinate, center: Center) -> float:
    return np.sqrt(np.sum(np.power(np.array(coordinate) - np.array(center), 2)))


def ecost_of_an_assignment(u_points: List[UncertainPoint], centers: List[Center],
                           assignments: Assignments) -> float:

    objects = [{'i': i, 'prob': point.prob,
                'distance_to_center': euclidean_distance(point.coordinate, centers[assignments[i]])}
                for (i, up) in enumerate(u_points)
                for point in up]

    objects.sort(key=lambda x: x.get('distance_to_center'))

    sum_probs = [0] * len(u_points)
    number_of_zero_sum_probs = len(u_points)
    mult, ecost = 1, 0

    for obj in objects:
        i, prob, dist = obj.get('i'), obj.get('prob'), obj.get('distance_to_center')

        if prob == 0:
            continue
        
        if sum_probs[i] == 0:
            number_of_zero_sum_probs -= 1

        if sum_probs[i] != 0:
            mult /= sum_probs[i]

        sum_probs[i] += prob
        mult *= sum_probs[i]

        if number_of_zero_sum_probs > 0:
            continue

        ecost += prob * (mult / sum_probs[i]) * dist

    return ecost


def generate_k_center_with_kmeans(coordinates: List[Coordinate], k: int) -> List[Center]:
    start_index = random.randint(0, len(coordinates) - 1)
    centers = [coordinates[start_index]]

    while len(centers) < k:
        dists = list(map(lambda coord: get_distance_to_nearest_center(coord, centers), coordinates))
        index = int(np.argmax(dists))   
        centers.append(coordinates[index])
        
    return centers



def get_expected_coordinate_of_uncertain_point(up:UncertainPoint) -> Coordinate:
    return list(np.dot([p.prob for p in up], [p.coordinate for p in up]))
    
    
def get_index_of_nearest_center_to_coordinate(coordinate: Coordinate, centers: List[Center]) -> Center:
    return int(np.argmin(list(map(lambda c: euclidean_distance(coordinate, c), centers))))


def get_distance_to_nearest_center(coordinate: Coordinate, centers: List[Center]) -> Center:
    return [euclidean_distance(coordinate, c) for c in centers]
    

def get_expected_distance_of_uncertain_point_to_center(up: UncertainPoint, center: Center) -> float:
    return np.sum(np.dot([p.prob for p in up], [euclidean_distance(p.coordinate, center) for p in up]))


def get_expected_distance_assignment_for_one_uncertain_point(u_point:UncertainPoint, centers:List[Center]) -> int:
    return int(np.argmin(list(map(lambda c: get_expected_distance_of_uncertain_point_to_center(u_point, c), centers))))


def get_one_center_of_uncertain_point(up: UncertainPoint, num_of_intervals=10) -> Center:
    mins = np.min(list(map(lambda p: p.coordinate, up)), axis=0)
    maxs = np.max(list(map(lambda p: p.coordinate, up)), axis=0)
    interval_lengths = np.array((maxs - mins) / num_of_intervals)
    permutations = get_all_permutations(len(interval_lengths), num_of_intervals)
    one_centers = list(map(lambda per: [mins[i] + (p+0.5)*interval_lengths[i]
                                        for (i, p) in enumerate(per)], permutations))

    return one_centers[get_expected_distance_assignment_for_one_uncertain_point(up, one_centers)]


def get_all_complete_assignments(temp_assignment: Assignments, k:int) -> List[Assignments]:
    def get_assignments_for_one_permutation(per: List[int], h_indices: List[int]):
        np_assignments = np.array(temp_assignment)
        np_assignments[h_indices] = per
        return list(np_assignments)

    hole_indices = list(filter(lambda i: temp_assignment[i] == -1, range(len(temp_assignment))))
    permutations = get_all_permutations(len(hole_indices), k)
    return list(map(lambda per:get_assignments_for_one_permutation(per, hole_indices), permutations))    


def get_best_assignments_in_list(u_points: List[UncertainPoint], centers: List[Center],
                                 assignments_list: List[Assignments]) -> Assignments:
    return assignments_list[np.argmin(list(map(lambda a: ecost_of_an_assignment(u_points, centers, a), assignments_list)))]


def get_all_permutations(n: int, k: int) -> List[List[int]]:
    def convert_base(number: int, base: int) -> List[int]:
        if base < 2:
            return False
        remainders = []
        while number > 0:
            remainders.append(number % base)
            number //= base
        remainders.reverse()
        return remainders
   
    assignments = []
    for i in range (k**n):
        a = convert_base(i, k)
    
        while len(a) < n:
            a.insert(0, 0)

        assignments.append(a)

    return assignments
