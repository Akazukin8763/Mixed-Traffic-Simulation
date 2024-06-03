class Config:
    DRAW_DEBUG = False

    SPAWN_RATE_PEDESTRIAN = 100
    SPAWN_RATE_VEHICLE = 180
    SPAWN_RATE_BICYCLE = 250
    SIMULATE_FRAME = 1000

    #
    Inf_position = [99999, 99999]

    Epsilon = 0.2
    Epsilon_lane_changing = 0.5

    Pedestrian_radius = 0.33

    Pedestrian_desired_speed_mean = 1.2
    Pedestrian_desired_speed_std = 0.05

    Vehicle_desired_speed_mean = 3.3
    Vehicle_desired_speed_std = 0.05
    Vehicle_com_decel_mean = 5.8
    Vehicle_com_decel_std = 0.05
    Vehicle_min_accel_mean = 1.5
    Vehicle_min_accel_std = 0.05
    Vehicle_space_headway_mean = 2.5
    Vehicle_space_headway_std = 0.05

    Vehicle_lane_changing_cooldown = 150  # Need to wait N frame to execute a new lane changing process
    Vehicle_lane_changing_safe_distance = 4.0

    Bicycle_desired_speed_mean = 1.6
    Bicycle_desired_speed_std = 0.15
    Bicycle_com_decel_mean = 2.7
    Bicycle_com_decel_std = 0.05
    Bicycle_min_accel_mean = 1.2
    Bicycle_min_accel_std = 0.05
    Bicycle_space_headway_mean = 3.5
    Bicycle_space_headway_std = 0.05

    Minimum_forward_speed_for_side_movement = 2.3

    #
    Tau = 0.5
    T_c = 0.5

    # Vehicle Lane Changing
    P = 0  # Determin which degree these successors influence the lane changing decision of vehicle
    A_th = 1  # The lane changing threshold

    # Bicycle Overtaking
    Alpha = 12

    # Pedestrian replusive to Vehicle and Bicycle
    Beta = 4.5

    # Adjacent lane road user, Lane Boundary
    U_cq = 1.5
    R_cq = 0.47

    # Collision Avoidance
    U_kj = 2.8
    R_kj = 0.35
