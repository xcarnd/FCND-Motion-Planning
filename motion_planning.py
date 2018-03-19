import argparse
import time
import msgpack
from enum import Enum, auto

import numpy as np

from project_utils import create_grid_2_5d, a_star_2_5d, prune_path, heuristic, points_collinear_2d_xy, \
    visualize_grid_and_pickup_goal, create_local_path_planning_grid_and_endpoints, \
    a_star_3d, points_collinear_3d, local_path_to_global_path
from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local
from sklearn.neighbors import KDTree

TARGET_ALTITUDE = 5
SAFETY_DISTANCE = 7


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

        self.interactive_goal = (504, 615)
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
            if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < 2.0:
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
        self.flight_state = States.WAYPOINT
        print("waypoint transition")
        print(self.waypoints)
        self.target_position = self.waypoints.pop(0)
        print('target position', self.target_position)
        self.cmd_position(self.target_position[0], self.target_position[1], self.target_position[2],
                          self.target_position[3])
        if 0 < len(self.waypoints) < 3:
            next_north, next_east, next_alt, _ = self.waypoints[-1]
            grid_start = (int(next_north - self.north_offset),
                          int(next_east - self.east_offset),
                          int(max(0, next_alt)))

            waypoints = self.plan_local_path(grid_start)
            if len(waypoints) > 1:
                self.send_waypoints2(waypoints[1:])
            self.waypoints += waypoints
            self.waypoints = prune_path(self.waypoints, points_collinear_3d)

    def landing_transition(self):
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
        data = msgpack.dumps(waypoints)
        self.connection._master.write(data)

    def send_waypoints(self):
        print("Sending waypoints to simulator ...")
        data = msgpack.dumps(self.waypoints)
        self.connection._master.write(data)

    def pick_goal(self, event):
        evt = event.mouseevent
        x = int(evt.xdata)
        y = int(evt.ydata)
        self.interactive_goal = (x, y)

        if self.temporary_scatter is not None:
            self.temporary_scatter.remove()
        fig = event.artist.figure
        self.temporary_scatter = fig.gca().scatter(x, y, marker='o', c='g')
        fig.canvas.draw()
        print("You've pick up {} as in the grid as your goal. "
              "Close the figure to continue.".format(self.interactive_goal))

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
        grid_start = (int(local_position[0] - north_offset),
                      int(local_position[1] - east_offset),
                      int(TARGET_ALTITUDE))

        # visualize grid
        visualize_grid_and_pickup_goal(grid, grid_start, self.pick_goal)
        # goal will be picked up interactively. But if the user (or, you the reviewer lol)
        # just simply close the grid map, then I've also set a default goal I chose beforehand.

        # the goal is specified in (x, y), where x means easting and y means northing
        # the target altitude is read from the 2.5D map
        goal_east, goal_north = self.interactive_goal
        grid_goal = (goal_north,
                     goal_east,
                     int(max(grid[int(goal_north), int(goal_east)], TARGET_ALTITUDE)))

        print('Local Start and Goal: ', grid_start, grid_goal)
        print("Searching path ... Please be patient")
        t0 = time.time()
        path = a_star_2_5d(grid, heuristic, grid_start, grid_goal, TARGET_ALTITUDE)
        path = prune_path(path, points_collinear_2d_xy)
        print("Search done. Take {} seconds in total".format(time.time() - t0))
        print(path)
        self.path = path
        # build KDTree for the coarse path
        self.path_kdtree = KDTree(tuple(p[:2] for p in path))
        waypoints = self.plan_local_path(grid_start)
        self.waypoints = waypoints
        self.send_waypoints2(waypoints)

    def plan_local_path(self, local_position):
        grid3d, start_3d, goal_3d = create_local_path_planning_grid_and_endpoints(self.map_grid, self.path,
                                                                                  self.path_kdtree, local_position,
                                                                                  20, 20, 20)
        if grid3d is None:
            return []
        t0 = time.time()
        s, g = local_path_to_global_path([start_3d, goal_3d], local_position, 20, 20, 20)
        print("Finding local path from {} ({}) to {} ({})".format(start_3d, s, goal_3d, g))
        local_path = a_star_3d(grid3d, heuristic, start_3d, goal_3d, TARGET_ALTITUDE)
        #print("Local path found. Time cost: {}".format(time.time() - t0))
        local_path = prune_path(local_path, points_collinear_3d)

        #print("Start & goal in local path:", local_path[0], local_path[-1])
        final_path = local_path_to_global_path(local_path, local_position, 20, 20, 20)
        print("Local path:", final_path)

        # Convert path to waypoints
        waypoints = [[p[0] + self.north_offset, p[1] + self.east_offset, p[2], 0] for p in final_path]
        # Set self.waypoints
        return waypoints

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
