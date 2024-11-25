from typing import Dict, List

import numpy as np
from pddl.action import Action
from pddl.core import Domain, Problem
from pddl.logic import Predicate, constants, variables
from pddl.requirements import Requirements


class NeuralToPDDL:
    def __init__(self):
        self.threshold = 0.5
        self.domain = self.create_domain()

    def create_domain(self) -> Domain:
        """Create PDDL domain programmatically"""
        # Define types
        types = {"location": None, "robot": None}

        # Define variables individually
        r = variables("r", types=["robot"])[0]
        from_loc = variables("from_loc", types=["location"])[0]
        to_loc = variables("to_loc", types=["location"])[0]

        # Define predicates with individual variables
        at = Predicate("at", r, from_loc)
        holding = Predicate("holding", r)
        near = Predicate("near", r)
        medium_distance = Predicate("medium-distance", r)
        far = Predicate("far", r)
        connected = Predicate("connected", from_loc, to_loc)

        # Define move action
        move = Action(
            "move",
            parameters=[r, from_loc, to_loc],
            precondition=at(r, from_loc) & connected(from_loc, to_loc),
            effect=~at(r, from_loc) & at(r, to_loc),
        )

        # Create domain
        requirements = [Requirements.STRIPS, Requirements.TYPING]
        return Domain(
            "robot-domain",
            requirements=requirements,
            types=types,
            predicates=[at, holding, near, medium_distance, far, connected],
            actions=[move],
        )

    def convert_nn_output_to_predicates(self, nn_output: Dict) -> List:
        """Convert neural network outputs to PDDL predicates"""
        predicates = []

        # Get robot constant
        robot = constants("robot1", type_="robot")[0]

        # Convert location probabilities
        location_idx = np.argmax(nn_output["location"])
        locations = constants("locA locB locC", type_="location")
        current_loc = locations[location_idx]

        # Create predicates
        at = Predicate("at", robot, current_loc)
        predicates.append(at)

        if nn_output["holding"] > self.threshold:
            holding = Predicate("holding", robot)
            predicates.append(holding)

        distance = nn_output["distance"]
        if distance < 1.0:
            near = Predicate("near", robot)
            predicates.append(near)
        elif distance < 3.0:
            medium = Predicate("medium-distance", robot)
            predicates.append(medium)
        else:
            far = Predicate("far", robot)
            predicates.append(far)

        return predicates

    def create_problem(
        self, initial_predicates: List, goal_predicates: List
    ) -> Problem:
        """Create PDDL problem"""
        # Define objects
        robot = constants("robot1", type_="robot")[0]
        locations = constants("locA locB locC", type_="location")

        # Create variables first
        l1 = variables("l1", types=["location"])[0]  # Get single variable
        l2 = variables("l2", types=["location"])[0]  # Get single variable

        # Create connected predicate with individual variables
        connected = Predicate("connected", l1, l2)
        init_connected = [
            connected(locations[0], locations[1]),
            connected(locations[1], locations[2]),
        ]

        return Problem(
            "robot-problem",
            domain=self.domain,
            requirements=self.domain.requirements,
            objects=[robot] + locations,
            init=initial_predicates + init_connected,
            goal=goal_predicates,
        )


def main():
    # Create converter
    converter = NeuralToPDDL()

    # Example neural network output
    nn_output = {
        "location": np.array([0.9, 0.1, 0.1]),  # High probability at location A
        "holding": 0.8,  # Probably holding something
        "distance": 2.5,  # Medium distance
    }

    # Convert to predicates
    initial_predicates = converter.convert_nn_output_to_predicates(nn_output)

    # Define goal
    robot = constants("robot1", type_="robot")[0]
    loc_c = constants("locC", type_="location")[0]
    goal_at = Predicate("at", robot, loc_c)
    goal_predicates = [goal_at]

    # Create problem
    problem = converter.create_problem(initial_predicates, goal_predicates)

    # Print results
    print("\nGenerated PDDL Domain:")
    print(converter.domain)
    print("\nGenerated PDDL Problem:")
    print(problem)


if __name__ == "__main__":
    main()
