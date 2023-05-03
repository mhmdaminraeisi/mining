import random
import numpy as np
from typing import List
from entities import *
from config import *

def generate_random_coordinate(d: int) -> Coordinate:
    return list(np.random.uniform(MIN_POINT_RANGE, MAX_POINT_RANGE, d))


def generate_n_uncertain_points_in_Rd(n: int, d: int) -> List[UncertainPoint]:
    def generate_uncertain_point_inRd(d: int) -> UncertainPoint:
        z = random.randint(MIN_POINT_COUNT, MAX_POINT_COUNT)
        probs = list(list(np.random.dirichlet(np.ones(z), size=1))[0])
        return [Point(generate_random_coordinate(d), prob) for prob in probs]
        
    return [generate_uncertain_point_inRd(d) for _ in range(0, n)]


def generate_k_centers_in_Rd(k: int, d: int) -> List[Center]:
    return [generate_random_coordinate(d) for _ in range(0, k)]
