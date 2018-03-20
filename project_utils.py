import numpy as np
from enum import Enum
from queue import PriorityQueue
import operator as op
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString
from functools import reduce


class Action2D(Enum):
    """
    Actions when doing 2.5D A* search.

    Actions are only stored as how the will alter the location.
    Costs are inferred from them.
    """

    # I've used a step size of 2 when performing 2.5D A* search
    # This will surely speed up the searching, with drawback that
    # the algorithm may not find a valid way to the goal because of
    # overshooting
    #
    # to handle with this, A* search will stop when the algorithm found
    # a location near to the goal (see `a_star_2_5d` below)
    WEST = (0, -4)
    EAST = (0, 4)
    NORTH = (-4, 0)
    SOUTH = (4, 0)

    # NORTH_EAST = (3, 3)
    # SOUTH_EAST = (3, -3)
    # SOUTH_WEST = (-3, -3)
    # NORTH_WEST = (-3, 3)

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
    # minimum and maximum north coordinates
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil((north_max - north_min + 1)))
    east_size = int(np.ceil((east_max - east_min + 1)))

    # Initialize an empty grid
    grid = np.zeros((north_size, east_size))

    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        obstacle = [
            int(np.clip(north - d_north - safe_distance - north_min, 0, north_size - 1)),
            int(np.clip(north + d_north + safe_distance - north_min, 0, north_size - 1)),
            int(np.clip(east - d_east - safe_distance - east_min, 0, east_size - 1)),
            int(np.clip(east + d_east + safe_distance - east_min, 0, east_size - 1)),
        ]
        obs = grid[obstacle[0]:obstacle[1] + 1, obstacle[2]:obstacle[3] + 1]
        np.maximum(obs, np.ceil(alt + d_alt + safe_distance), obs)

    return grid, int(north_min), int(east_min)
    # north_min = int(np.floor(np.min(data[:, 0] - data[:, 3] - safe_distance)))
    # north_max = int(np.ceil(np.max(data[:, 0] + data[:, 3] + safe_distance)))
    # east_min = int(np.floor(np.min(data[:, 1] - data[:, 4] - safe_distance)))
    # east_max = int(np.ceil(np.max(data[:, 1] + data[:, 4] + safe_distance)))
    #
    # north_size = north_max - north_min + 1
    # east_size = east_max - east_min + 1
    #
    # grid = np.zeros((north_size, east_size))
    #
    # for i in range(data.shape[0]):
    #     n, e, a, dn, de, da = data[i]
    #     n_min = int(n - dn - safe_distance - north_min)
    #     n_max = int(n + dn + safe_distance - north_min)
    #     e_min = int(e - de - safe_distance - east_min)
    #     e_max = int(e + de + safe_distance - east_min)
    #     print(n_min, n_max, int(n - dn - north_min), int(n + dn - north_min))
    #
    #     grid[n_min:n_max+1, e_min:e_max+1] = a + dn + safe_distance
    #
    # return grid, int(north_min), int(east_min)


def heuristic(position, goal):
    """
    Heuristic function used for A* planning. Simply return the euclidean distance of the two points given.
    """
    return np.linalg.norm(np.array(position) - np.array(goal))


def heuristic_manhattan_dist_2d(position, goal):
    """
    Heuristic function used for calculating manhattan distance between given 2D points.
    """
    return abs(position[0] - goal[0]) + abs(position[1] - goal[1])


def valid_actions(grid, current_node):
    """
    Returns a list of valid actions given a grid and current node.
    """
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
        # here I made a simplification: when the drone need to go up 10 meters more than it's current
        # altitude, then going up will be ignored.
        if not (nn < 0 or nn > north_max or
                ne < 0 or ne > east_max):
            # altitude cost. going up will always cost more
            altitude_cost = max(grid[nn, ne] - a, 0) * 100
            valid.append((altitude_cost, action))

    return valid


def valid_actions_3d(grid, current_node):
    actions = list(Action3D)
    north_max, east_max, alt_max = grid.shape[0] - 1, grid.shape[1] - 1, grid.shape[2] - 1
    n, e, a = current_node

    # check if the node is off the grid or
    # it's an obstacle
    valid = []
    for action in actions:
        dn, de, da = action.delta
        nn, ne, na = n + dn, e + de, a + da
        if not (nn < 0 or nn > north_max or
                ne < 0 or ne > east_max or
                na < 0 or na > alt_max or
                grid[nn, ne, min(na, alt_max)] > 0):
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


def waypoint_fn_3d(node):
    """
    Return the waypoint used in 3D A* search planning
    """
    return tuple(node[:3])


def point_projection_xy(point):
    return point[0], point[1]


def point_projection_xz(point):
    return point[0], point[2]


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


def points_collinear_3d(p1, p2, p3):
    proj_xy = tuple(map(point_projection_xy, (p1, p2, p3)))
    proj_xz = tuple(map(point_projection_xz, (p1, p2, p3)))
    return points_collinear_2d_xy(*proj_xy) and points_collinear_2d_xy(*proj_xz)


def prune_path(path, collinear_fn):
    """
    Remove unnecessary intermediate waypoints in the path.

    :param path: path to be pruned
    :param collinear_fn: collinearity testing function used for determine if points are collinear
    """
    if len(path) <= 1:
        return path[:]
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


def get_waypoint_ahead_idx(loc, path, tree):
    n, e = loc[:2]
    nearest_waypoint_idx = tree.query(((n, e),), k=1, return_distance=False)[0][0]

    p1_idx = max(nearest_waypoint_idx - 1, 0)
    p2_idx = nearest_waypoint_idx

    while True:
        vec_12 = np.array(path[p2_idx][:2]) - np.array(path[p1_idx][:2])
        vec_10 = np.array((n, e)) - np.array(path[p1_idx][:2])
        vec_20 = np.array((n, e)) - np.array(path[p2_idx][:2])
        cos_angle_1 = np.dot(vec_12, vec_10)
        cos_angle_2 = np.dot(vec_12, vec_20)
        if cos_angle_1 >= 0 and cos_angle_2 < 0:
            return p2_idx
        p1_idx = p2_idx
        p2_idx += 1
        if p2_idx >= len(path):
            return p1_idx


def create_local_path_planning_grid_and_endpoints(grid, path, tree, start, north_span, east_span, altitude_span):
    center_n, center_e, center_a = start
    north_size, east_size = grid.shape

    north_min = max(0, center_n - north_span)
    north_max = min(center_n + north_span, north_size)

    east_min = max(0, center_e - east_span)
    east_max = min(center_e + east_span, east_size)

    alt_min = max(center_a - altitude_span, 0)
    alt_max = center_a + altitude_span

    grid3d = np.zeros((north_max - north_min + 1, east_max - east_min + 1, alt_max - alt_min + 1))
    for iter_n in np.arange(north_min, north_max + 1):
        for iter_e in np.arange(east_min, east_max + 1):
            alt = min(int(np.ceil(grid[iter_n, iter_e])) + 1, alt_max + 1)
            grid3d[iter_n - north_min, iter_e - east_min, 0:max(alt - alt_min, 0)] = 1

    local_polygon = Polygon(((east_max, north_max),
                             (east_min, north_max),
                             (east_min, north_min),
                             (east_max, north_min)))

    horizon_waypoint_idx = get_waypoint_ahead_idx((center_n, center_e), path, tree)

    start_point = (center_n, center_e, center_a)
    end_point = path[horizon_waypoint_idx]

    print("S-E:", start_point, end_point)
    while local_polygon.contains(Point(end_point[1], end_point[0])) and \
            grid[end_point[1], end_point[0]] <= alt_max:
        horizon_waypoint_idx += 1
        if horizon_waypoint_idx >= len(path):
            break
        start_point = end_point
        end_point = path[horizon_waypoint_idx]
        print("S-E:", start_point, end_point)

    p1, p2 = start_point, end_point

    print("Finding intersection of line {}-{} with local polygon to get local planning's goal".format(p1, p2))
    p1p2 = LineString((p1[1::-1], p2[1::-1]))
    intersection = local_polygon.intersection(p1p2)
    if intersection:
        pie, pin = intersection.coords.xy
        pie, pin = pie[1], pin[1]
    else:
        pie, pin = path[-1][1], path[-1][0]
    dist = np.linalg.norm(np.array([center_n, center_e]) - np.array([pin, pie]))
    if dist < 1.0:
        return None, None, None, None
    e, n = int(pie), int(pin)
    # dist_all = np.linalg.norm(np.array(horizon_waypoint[:2]) - np.array(path[previous_waypoint_idx][:2]))
    # dist = np.linalg.norm(np.array((n, e)) - np.array(path[previous_waypoint_idx][:2]))
    # a = int(np.ceil(max((path[horizon_waypoint_idx][2] - path[previous_waypoint_idx][2]) * (dist / dist_all) +
    #                     path[previous_waypoint_idx][2],
    #                     grid[n, e])))
    a = int(np.ceil(max(p2[2], grid[n, e] + 1)))
    return (grid3d,
            (center_n - north_min, center_e - east_min, center_a - alt_min),
            (n - north_min, e - east_min, a - alt_min),
            a < alt_max)


def local_path_to_global_path(path, start_local, north_span, east_span, altitude_span):
    center_n, center_e, center_a = start_local
    north_min = max(center_n - north_span, 0)
    east_min = max(center_e - east_span, 0)
    alt_min = max(center_a - altitude_span, 0)
    return [(n + north_min, e + east_min, a + alt_min) for n, e, a in path]


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
        for alt_cost, action in valid_actions(grid, current_node):
            if found:
                break

            cost = action.cost + alt_cost
            next_node = tuple(map(op.add, waypoint_fn(current_node), action.delta))
            # we want to keep the drone flying in relatively low altitude because that's power saving,
            # on the other hand, the drone shall fly above certain altitude to avoid risk of hitting
            # pedestrians, cars or other objects in low altitudes.
            #
            # limit the drone so that it will at least flying at the lowest flight altitude we specified.
            lowest_alt = int(np.ceil(max(np.ceil(grid[next_node]) + 1, flight_altitude)))
            new_node = (next_node + (lowest_alt,))

            new_node_2d = waypoint_fn(new_node)
            if new_node_2d not in visited:
                new_cost = current_cost + cost + h(new_node, goal)
                branch[new_node_2d] = current_node
                visited.add(new_node_2d)
                queue.put((new_cost, new_node))

                # beware: since the step size of actions are set to 2, the algorithm
                # may overshoot the goal and finally report no paths is found
                #
                # so here instead of exact equal, I use a range for determine if
                # the goal is reached
                if goal_2d[0] - 2 <= new_node_2d[0] <= goal_2d[0] + 2 and \
                        goal_2d[1] - 2 <= new_node_2d[1] <= goal_2d[1] + 2:
                    branch[goal_2d] = current_node
                    goal_loc = (goal[0], goal[1], new_node[2])
                    final_plan = new_cost, reconstruct_path(goal_loc, branch, waypoint_fn)
                    found = True

    if found:
        print("Found a plan. Total cost: {}".format(final_plan[0]))
        return final_plan[1]
    else:
        print("Path not found")
        return None


def a_star_3d(grid, h, start, goal, flight_altitude):
    print("Performing 3D A* from {} to {}".format(start, goal))
    final_plan = None
    visited = set()
    queue = PriorityQueue()

    queue.put((0, start))
    visited.add(start)
    branch = {start: None}
    found = False
    while not queue.empty() and not found:
        current_cost, current_node = queue.get()
        for action in valid_actions_3d(grid, current_node):
            if found:
                break
            action_cost, new_node = action.cost, tuple(map(op.add, current_node, action.delta))
            # penalty for flying too low
            if new_node[2] < flight_altitude:
                action_cost += (flight_altitude - new_node[2]) * 10

            if new_node not in visited:
                new_cost = current_cost + action_cost + h(new_node, goal)
                branch[new_node] = current_node
                visited.add(new_node)
                queue.put((new_cost, new_node))

                if new_node == goal:
                    final_plan = new_cost, reconstruct_path(goal, branch, waypoint_fn_3d)
                    found = True

    if found:
        print("Found a local plan. Total cost: {}".format(final_plan[0]))
        print(final_plan[1])
        return final_plan[1]
    else:
        print("Local path not found.")
        return None


def visualize_grid_and_pickup_goal(grid, start, callback):
    """
    Visualize 2.5D grid and wait for the goal being picked up
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


def bresenham(start, end):
    n1, e1 = start[:2]
    n2, e2 = end[:2]

    if abs(e2 - e1) < 1e-5:
        return [(n, e1) for n in range(min(n1, n2), max(n1, n2) + 1)]

    slope = (n2 - n1) / (e2 - e1)

    cells = []

    if e1 < e2:
        n, e = n1, e1
        ne, ee = n2, e2
    else:
        n, e = n2, e2
        ne, ee = n1, e1

    f = n
    if slope >= 0:
        while e < ee and n < ne:
            cells.append((n, e))
            f_new = f + slope
            if f_new > n + 1:
                n += 1
            else:
                e += 1
                f = f_new
    else:
        while e < ee and n > ne:
            cells.append((n, e))
            f_new = f + slope
            if f_new < n - 1:
                n -= 1
            else:
                e += 1
                f = f_new

    return cells


def simplify_path(grid, path):
    """
    Check against path[0] --- path[-1], path[0] --- path[-2], ... path[0] --- path[1],
    see whether we have a direct path among them. Returns the longest path once we have found one.
    """
    if len(path) <= 2:
        return path
    print("Simplifying path:", path)
    start_idx = 0
    end_idx = len(path) - 1
    result_path = [path[0]]
    while start_idx < end_idx:
        start = path[start_idx]
        end = path[end_idx]
        min_height = min(start[2], end[2])
        cells = bresenham(start, end)

        has_obs = False
        for n, e in cells:
            if grid[n, e] >= min_height:
                has_obs = True
                break

        if has_obs:
            end_idx -= 1
        else:
            result_path.append(end)
            start_idx = end_idx
            end_idx = len(path) - 1

    print("Result path:", result_path)
    return result_path


def get_length_of_path(path, waypoint_fn=waypoint_fn_3d):
    """
    Given a path, return the total length (Euclidean distance) of it
    """
    return reduce(op.add,
                  map(lambda x: np.linalg.norm(np.array(waypoint_fn(x[0]))
                                               -
                                               np.array(waypoint_fn(x[1]))),
                      zip(path[:-1], path[1:])),
                  0)
