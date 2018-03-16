import numpy as np
from enum import Enum
from queue import PriorityQueue
import operator as op
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from shapely.geometry import Polygon, Point, LineString


class Action2D(Enum):
    """
    Actions when doing 2.5D A* search.

    Actions are only stored as how the will alter the location.
    Costs are inferred from them.
    """

    WEST = (0, -1)
    EAST = (0, 1)
    NORTH = (-1, 0)
    SOUTH = (1, 0)
    NORTH_EAST = (1, 1)
    SOUTH_EAST = (1, -1)
    SOUTH_WEST = (-1, -1)
    NORTH_WEST = (-1, 1)

    @property
    def delta(self):
        return self.value[0], self.value[1]

    @property
    def cost(self):
        return np.linalg.norm(np.array(self.value))


class Action3D(Enum):
    """
    Actions when performing 3D A* search.

    Just like Action2D, only the alternation of location are stored as 3-element tuple.
    Costs are inferred from them.
    """
    WEST = (0, -1, 0)
    EAST = (0, 1, 0)
    NORTH = (-1, 0, 0)
    SOUTH = (1, 0, 0)

    NORTH_EAST = (1, 1, 0)
    SOUTH_EAST = (1, -1, 0)
    SOUTH_WEST = (-1, -1, 0)
    NORTH_WEST = (-1, 1, 0)

    UP = (0, 0, 1)
    DOWN = (0, 0, -1)

    @property
    def delta(self):
        return self.value[0], self.value[1], self.value[2]

    @property
    def cost(self):
        return np.linalg.norm(np.array(self.value))


def create_grid_2_5d(data, safe_distance):
    """
    Create a 2.5D grid from given obstacle data.

    :param data: obstacle data
    :param safe_distance: safe distance added to the surrounding of obstacle
    :return: grid-based 2.5D configuration space
    """
    north_min = int(np.floor(np.min(data[:, 0] - data[:, 3] - safe_distance)))
    north_max = int(np.ceil(np.max(data[:, 0] + data[:, 3] + safe_distance)))
    east_min = int(np.floor(np.min(data[:, 1] - data[:, 4] - safe_distance)))
    east_max = int(np.ceil(np.max(data[:, 1] + data[:, 4] + safe_distance)))

    north_size = north_max - north_min
    east_size = east_max - east_min

    grid = np.zeros((north_size, east_size))

    for i in range(data.shape[0]):
        n, e, a, dn, de, da = data[i]
        n_min = int(n - dn - safe_distance - north_min)
        n_max = int(n + dn + safe_distance - north_min)
        e_min = int(e - de - safe_distance - east_min)
        e_max = int(e + de + safe_distance - east_min)

        grid[n_min:n_max, e_min:e_max] = a + dn + safe_distance

    return grid, int(north_min), int(east_min)


def heuristic(position, goal):
    """
    Heuristic function used for A* planning. Simply return the euclidean distance of the two points given.
    """
    return np.linalg.norm(np.array(position) - np.array(goal))


def valid_actions(grid, current_node):
    """
    Returns a list of valid actions given a grid and current node.
    """
    max_allowed_altitude_diff = 15
    actions = list(Action2D)
    north_max, east_max = grid.shape[0] - 1, grid.shape[1] - 1
    n, e, a = current_node

    # check if the node is off the grid or
    # it's an obstacle
    valid = []
    for action in actions:
        dn, de = action.delta
        nn, ne = n + dn, e + de
        # theoretically, a drone can climb up as high as the obstacles then fly over them.
        # in reality, climbing up always requires more thrust power, so it is not always a better
        # choice to climb up when facing with obstacles
        #
        # here I made a simplification: when the drone need to go up 15 meters more than it's current
        # altitude, then going up will be ignored.
        if not (nn < 0 or nn > north_max or
                ne < 0 or ne > east_max or
                grid[nn, ne] - a > max_allowed_altitude_diff):
            valid.append(action)

    return valid


def reconstruct_path(goal, branch, waypoint_fn):
    """
    Reconstruct a path from the goal state and branch information
    """
    current_node = goal
    path = [current_node]
    while current_node is not None:
        previous_node = branch[waypoint_fn(current_node)]
        path.append(previous_node)
        current_node = previous_node
    path.pop()
    path.reverse()
    return path


def waypoint_fn_2_5d(node):
    """
    Return the waypoint used in 2.5D A* search planning
    """
    return tuple(node[:2])


def points_collinear_2d_xy(p1, p2, p3):
    """
    Test if given 3 points are collinear if projected onto XY-plane.

    Given points can be in any dimensions greater or equal to 2, but only the first 2 dimensions will be used
    for collinear test.
    """
    x1, y1 = p1[:2]
    x2, y2 = p2[:2]
    x3, y3 = p3[:2]
    return x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2) == 0


def prune_path(path, collinear_fn):
    """
    Remove unnecessary intermediate waypoints in the path.

    :param path: path to be pruned
    :param collinear_fn: collinearity testing function used for determine if points are collinear
    """
    new_path = [path[0]]
    line_start = path[0]
    line_end = path[1]
    for i in range(2, len(path)):
        next_end = path[i]
        if collinear_fn(line_start, line_end, next_end):
            line_end = next_end
        else:
            new_path.append(line_end)
            line_start = line_end
            line_end = next_end
    new_path.append(line_end)
    return new_path


def a_star_2_5d(grid, h, start, goal, flight_altitude, waypoint_fn=lambda n: tuple(n[:2])):
    """
    Perform 2.5D A* search

    :param grid: The 2.5D grid map
    :param h: heuristic function
    :param start: start node in the grid. shall be a 3-element tuple (north, east, altitude) specified in local grid
    coordinates. altitudes shall be specified as positive up.
    :param goal: goal node in the grid.
    :param flight_altitude: target flight altitude
    :param waypoint_fn: a function extracting 2D representation of nodes.
    :return: A path from start to goal in grid coordinate.
    """
    start_2d = waypoint_fn(start)
    goal_2d = waypoint_fn(goal)

    final_plan = None
    visited = set()
    queue = PriorityQueue()

    queue.put((0, start))
    visited.add(start_2d)
    branch = {start_2d: None}
    found = False
    while not queue.empty() and not found:
        current_cost, current_node = queue.get()
        for action in valid_actions(grid, current_node):
            if found:
                break

            cost = action.cost
            next_node = tuple(map(op.add, waypoint_fn(current_node), action.delta))
            # we want to keep the drone flying in relatively low altitude because that's power saving,
            # on the other hand, the drone shall fly above certain altitude to avoid risk of hitting
            # pedestrians, cars or other objects in low altitudes.
            #
            # limit the drone so that it will at least flying at the lowest flight altitude we specified.
            lowest_alt = int(np.ceil(max(np.ceil(grid[next_node]), flight_altitude)))
            new_node = (next_node + (lowest_alt,))

            new_node_2d = waypoint_fn(new_node)
            if new_node_2d not in visited:
                new_cost = current_cost + cost + h(new_node, goal)
                branch[new_node_2d] = current_node
                visited.add(new_node_2d)
                queue.put((new_cost, new_node))

                if new_node_2d == goal_2d:
                    final_plan = new_cost, reconstruct_path(goal, branch, waypoint_fn)
                    found = True

    if found:
        print("Found a plan. Total cost: {}".format(final_plan[0]))
        return final_plan[1]
    else:
        print("Path not found")
        return None


def visualize_grid_and_pickup_goal(grid, start, callback):
    """
    Visualize 2.5D grid
    """
    im = plt.imshow(grid, cmap='gray_r', picker=True)
    plt.axis((0, grid.shape[1], 0, grid.shape[0]))
    plt.xlabel("EAST")
    plt.ylabel("NORTH")
    plt.scatter(start[1], start[0], marker='x', c='red')
    fig = plt.gcf()
    fig.colorbar(im)
    fig.canvas.mpl_connect('pick_event', callback)
    plt.gca().set_title("Pickup the goal on the map")
    plt.show()
