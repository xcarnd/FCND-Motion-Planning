import argparse
import time
import msgpack
from enum import Enum, auto

import numpy as np

from project_utils import create_grid_2_5d, a_star_2_5d, prune_path, heuristic, points_collinear_2d_xy, \
    visualize_grid_and_pickup_goal, create_local_path_planning_grid_and_endpoints, \
    a_star_3d, points_collinear_3d, local_path_to_global_path, simplify_path, \
    get_length_of_path, heuristic_manhattan_dist_2d, \
    path_2_5d_to_3d_path
from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local, local_to_global
from sklearn.neighbors import KDTree

TARGET_ALTITUDE = 5
SAFETY_DISTANCE = 5


class States(Enum):
    MANUAL = auto()
    ARMING = auto()
    TAKEOFF = auto()
    WAYPOINT = auto()
    LANDING = auto()
    DISARMING = auto()
    PLANNING = auto()


class MotionPlanning(Drone):

    def __init__(self, connection):
        super().__init__(connection)

        self.target_position = np.array([0.0, 0.0, 0.0])
        self.waypoints = []
        self.in_mission = True
        self.check_state = {}
        self.full_path = []

        #self.interactive_goal = (205, 814)
        self.interactive_goal = (705, 84)
        self.temporary_scatter = None
        self.previous_location = None
        self.map_grid = None
        self.north_offset = None
        self.east_offset = None
        self.path = None
        self.path_kdtree = None

        # initial state
        self.flight_state = States.MANUAL

        # register all your callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    def local_position_callback(self):
        if self.flight_state == States.TAKEOFF:
            if -1.0 * self.local_position[2] > 0.95 * self.target_position[2]:
                self.waypoint_transition()
        elif self.flight_state == States.WAYPOINT:
            self.plan_next_waypoints_if_needed()
            if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < 3.0 \
                    and abs(self.target_position[2] - (-self.local_position[2])) < 2.0:
                if len(self.waypoints) > 0:
                    self.waypoint_transition()
                else:
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        self.landing_transition()

    def velocity_callback(self):
        if self.flight_state == States.LANDING:
            if self.global_position[2] - self.global_home[2] < 0.1:
                if abs(self.local_position[2]) < 0.01:
                    self.disarming_transition()

    def state_callback(self):
        if self.in_mission:
            if self.flight_state == States.MANUAL:
                self.arming_transition()
            elif self.flight_state == States.ARMING:
                if self.armed:
                    self.plan_path()
            elif self.flight_state == States.PLANNING:
                self.takeoff_transition()
            elif self.flight_state == States.DISARMING:
                if ~self.armed & ~self.guided:
                    self.manual_transition()

    def arming_transition(self):
        self.flight_state = States.ARMING
        print("arming transition")
        self.arm()
        self.take_control()

    def takeoff_transition(self):
        self.flight_state = States.TAKEOFF
        print("takeoff transition")
        self.takeoff(self.target_position[2])

    def waypoint_transition(self):
        print("Full path:", self.full_path)
        self.flight_state = States.WAYPOINT
        print("waypoint transition")
        print(self.waypoints)
        self.target_position = self.waypoints.pop(0)
        print('target position', self.target_position)
        self.cmd_position(self.target_position[0], self.target_position[1], self.target_position[2],
                          self.target_position[3])

    def landing_transition(self):
        print("Full path:", self.full_path)
        self.flight_state = States.LANDING
        print("landing transition")
        self.land()

    def disarming_transition(self):
        self.flight_state = States.DISARMING
        print("disarm transition")
        self.disarm()
        self.release_control()

    def manual_transition(self):
        self.flight_state = States.MANUAL
        print("manual transition")
        self.stop()
        self.in_mission = False

    def send_waypoint(self, waypoint):
        data = msgpack.dumps([waypoint])
        self.connection._master.write(data)

    def send_waypoints2(self, waypoints):
        print("Sending waypoints to simulator ...")
        self.full_path += [(int(n - self.north_offset),
                          int(e - self.east_offset)) for n, e, a, o in waypoints]
        data = msgpack.dumps(waypoints)
        self.connection._master.write(data)
        print("Done.")

    def send_waypoints(self):
        print("Sending waypoints to simulator ...")
        data = msgpack.dumps(self.waypoints)
        self.connection._master.write(data)

    def pick_goal(self, event):
        evt = event.mouseevent
        east = int(evt.xdata)
        north = int(evt.ydata)
        alt = self.map_grid[north, east]
        self.interactive_goal = local_to_global(self.grid_coord_to_local_position((north, east, alt)), self.global_home)

        if self.temporary_scatter is not None:
            self.temporary_scatter.remove()
        fig = event.artist.figure
        self.temporary_scatter = fig.gca().scatter(east, north, marker='o', c='g')
        fig.canvas.draw()
        print("You've pick up (lat, lon, alt) {} as the goal. "
              "Close the figure to continue.".format(self.interactive_goal))

    def grid_coord_to_local_position(self, grid_coord):
        lat = grid_coord[0] + self.north_offset
        lon = grid_coord[1] + self.east_offset
        return lat, lon, -grid_coord[2]

    def local_position_to_grid_coord(self, position):
        north = int(position[0] - self.north_offset)
        east = int(position[1] - self.east_offset)
        alt = int(-position[2])
        return north, east, alt

    def plan_path(self):
        self.flight_state = States.PLANNING
        print("Searching for a path ...")

        self.target_position[2] = TARGET_ALTITUDE

        with open('colliders.csv', 'r') as f:
            header_line = f.readline()
            lat_str, lon_str = header_line.split(',')
            lat = float(lat_str.strip().split(' ')[1])
            lon = float(lon_str.strip().split(' ')[1])
            print("Map home location: ({}, {})".format(lat, lon))

        home_position = (lon, lat, 0)
        self.set_home_position(*home_position)

        global_position = self.global_position

        local_position = global_to_local(global_position, home_position)

        print('global home {0}, position {1}, local position {2}'.format(self.global_home, self.global_position,
                                                                         self.local_position))
        # Read in obstacle map
        data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=2)

        # Define a grid for a particular altitude and safety margin around obstacles
        grid, north_offset, east_offset = create_grid_2_5d(data, SAFETY_DISTANCE)
        self.map_grid = grid
        self.north_offset = north_offset
        self.east_offset = east_offset

        print("North offset = {0}, east offset = {1}".format(north_offset, east_offset))
        # starting point on the grid
        grid_start_north = int(local_position[0] - north_offset)
        grid_start_east = int(local_position[1] - east_offset)
        grid_start = (grid_start_north,
                      grid_start_east,
                      int(max(TARGET_ALTITUDE, grid[grid_start_north, grid_start_east] + 1, -local_position[2] + 1)))

        # visualize grid
        visualize_grid_and_pickup_goal(grid, grid_start, self.pick_goal)
        # goal will be picked up interactively. But if the user (or, you the reviewer lol)
        # just simply close the grid map, then I've also set a default goal I chose beforehand.

        # the goal is specified in (x, y), where x means easting and y means northing
        # the target altitude is read from the 2.5D map
        goal = self.interactive_goal
        goal_local = global_to_local(goal, self.global_home)
        goal_grid = self.local_position_to_grid_coord(goal_local)
        goal_north, goal_east, goal_alt = goal_grid
        grid_goal = (goal_north,
                     goal_east,
                     int(max(grid[int(goal_north), int(goal_east)], TARGET_ALTITUDE, goal_alt)))

        print('Start and goal in the grid', grid_start, grid_goal)
        print("Searching path ... Please be patient")
        t0 = time.time()
        path = a_star_2_5d(grid, heuristic, grid_start, grid_goal, TARGET_ALTITUDE)
        print("Path planned by 2.5D A* planner:", path)
        path = path_2_5d_to_3d_path(path)
        print("Path in 3D:", path)
        path = prune_path(path, points_collinear_3d)
        print("Path after prunning:", path)
        path = simplify_path(grid, path)
        print("Search done. Take {} seconds in total".format(time.time() - t0))
        print(path)
        self.path = path
        # build KDTree for querying
        # self.path_kdtree = KDTree(tuple(p[:2] for p in path))
        #
        # self.waypoints = self.path_to_waypoints([self.path[0]])
        # self.send_waypoints2(self.waypoints)
        # self.plan_next_waypoints_if_needed()
        # # build KDTree for the coarse path
        # self.path_kdtree = KDTree(tuple(p[:2] for p in path))
        # path = self.plan_local_path(grid_start)
        waypoints = self.path_to_waypoints(path)
        self.waypoints = waypoints
        self.send_waypoints2(waypoints)

    def plan_local_path(self, local_position):
        grid3d, start_3d, goal_3d, feasible = \
            create_local_path_planning_grid_and_endpoints(self.map_grid, self.path,
                                                          self.path_kdtree, local_position,
                                                          20, 20, 20)

        if grid3d is None:
            return []

        local_start = local_position

        grid_start, grid_goal = local_path_to_global_path([start_3d, goal_3d], local_position, 20, 20, 20)
        print("Finding local path from {} ({}) to {} ({})".format(start_3d, grid_start, goal_3d, grid_goal))
        print("Path feasible? ", feasible)
        pre_path = []
        post_path = []
        if not feasible:
            print("Local planning not feasible.")
            minimum_flyable = max(self.map_grid[grid_goal[0], grid_goal[1]], TARGET_ALTITUDE)
            print("The goal found is inside the building so the proposed plan is not feasible. "
                  "Adjust the drone's position first")
            print("Goal: {}, target location minimum flyable altitude: {}"
                  .format(goal_3d, minimum_flyable))
            start_alt, goal_alt = grid_start[2], grid_goal[2]

            if goal_alt > start_alt:
                new_start = (grid_start[0], grid_start[1], goal_alt)
                pre_path = [grid_start, new_start]
                print("Goal altitude > start altitude "
                      "Lifting and search for local path from {} to {} instead".format(new_start, goal_3d))
                grid3d, start_3d, goal_3d, feasible = \
                    create_local_path_planning_grid_and_endpoints(self.map_grid, self.path,
                                                                  self.path_kdtree, new_start,
                                                                  20, 20, 20)
                local_start = new_start
            elif start_alt > goal_alt:
                new_goal = (grid_goal[0], grid_goal[1], start_alt)
                post_path = [new_goal, grid_goal]
                print("Start altitude > goal altitude. "
                      "First reaching the goal then landing to the specific altitude."
                      "Then try to search path from {} to {} first".format(new_goal, grid_goal))
                goal_3d = (goal_3d[0], goal_3d[1], start_3d[2])
        t0 = time.time()

        # if altitude difference is too large, try to reach that altitude first

        local_path = a_star_3d(grid3d, heuristic, start_3d, goal_3d, TARGET_ALTITUDE)
        #print("Local path found. Time cost: {}".format(time.time() - t0))
        local_path = prune_path(local_path, points_collinear_3d)

        #print("Start & goal in local path:", local_path[0], local_path[-1])
        final_path = local_path_to_global_path(local_path, local_start, 20, 20, 20)
        final_path = pre_path + final_path + post_path
        print("Local path:", final_path)

        return final_path

    def path_to_waypoints(self, path):
        # Convert path to waypoints
        waypoints = []
        for i in range(len(path)):
            p = path[i]
            p_next = path[i + 1] if i < len(path) - 1 else None
            orientation = 0
            if p_next is not None:
                orientation = np.arctan2(p_next[1] - p[1], p_next[0] - p[0])
            waypoints.append([p[0] + self.north_offset, p[1] + self.east_offset, p[2], orientation])
        # Set self.waypoints
        return waypoints

    def plan_next_waypoints_if_needed(self):
        if len(self.waypoints) == 0:
            return
        waypoints_length = get_length_of_path(self.waypoints)
        if 0 <= waypoints_length < 40:
            last_wp = self.waypoints[-1]
            next_north, next_east, next_alt, _ = last_wp
            grid_start = (int(next_north - self.north_offset),
                          int(next_east - self.east_offset),
                          int(max(0, next_alt)))
            next_path = [grid_start]
            while get_length_of_path(next_path) < 40 and grid_start != self.path[-1]:
                path = self.plan_local_path(grid_start)
                if path is None:
                    break
                next_path += path[1:]
                next_path = simplify_path(self.map_grid, next_path)
                grid_start = next_path[-1]

            next_waypoints = self.path_to_waypoints(next_path)

            if len(next_waypoints) > 1:
                self.send_waypoints2(next_waypoints[1:])
                self.waypoints += next_waypoints[1:]

        # if 0 < len(self.waypoints) < 5:
        #     next_north, next_east, next_alt, _ = self.waypoints[-1]
        #     grid_start = (int(next_north - self.north_offset),
        #                   int(next_east - self.east_offset),
        #                   int(max(0, next_alt)))
        #
        #     waypoints = self.plan_local_path(grid_start)
        #     if len(waypoints) > 1:
        #         self.send_waypoints2(waypoints[1:])
        #     self.waypoints += waypoints[1:]
        #     print(self.waypoints)

    #            self.waypoints = prune_path(self.waypoints, points_collinear_3d)

    def start(self):
        self.start_log("Logs", "NavLog.txt")

        print("starting connection")
        self.connection.start()

        # Only required if they do threaded
        # while self.in_mission:
        #    pass

        self.stop_log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    args = parser.parse_args()

    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=60)
    drone = MotionPlanning(conn)
    time.sleep(1)

    drone.start()
