from jmetal.core.operator import Neighbours
from jmetal.core.solution import Solution
from typing import List, Tuple, Optional
import random


class RandomNeighbours(Neighbours[Solution]):
    def __init__(self, seed: float = None):
        self.seed = seed

    def execute(self, source: Tuple[Solution, List[Solution]]) -> Optional[Solution]:
        individual, rest = source

        if not rest:
            return None

        random.seed(self.seed)
        return random.sample(rest, 1)[0]

    def get_name(self) -> str:
        "Random neighbours operator"
