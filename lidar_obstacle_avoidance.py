import carla
import random
import time
import math
import numpy as np


latest_lidar = {
    "points": None
}


def lidar_callback(data):
    points = np.frombuffer(data.raw_data, dtype=np.float32)
    points = np.reshape(points, (-1, 4))

    # vehicle local coordinate 기준 x, y만 사용
    xy = points[:, :2]
    latest_lidar["points"] = xy


def detect_front_obstacle(points, front_dist=6.0, half_width=1.5):
    """
    차량 전방 영역에 장애물이 있는지 판단
    x > 0: 차량 앞쪽
    |y| < half_width: 차량 폭 근처
    """

    if points is None or len(points) == 0:
        return False, None

    front_points = points[
        (points[:, 0] > 0.5) &
        (points[:, 0] < front_dist) &
        (np.abs(points[:, 1]) < half_width)
    ]

    if len(front_points) == 0:
        return False, None

    distances = np.linalg.norm(front_points, axis=1)
    min_dist = float(np.min(distances))

    return True, min_dist


def choose_avoid_direction(points):
    """
    좌/우 중 더 빈 공간이 많은 방향 선택
    y > 0: 왼쪽
    y < 0: 오른쪽
    """

    if points is None or len(points) == 0:
        return 0.4  # 기본 오른쪽

    near_points = points[
        (points[:, 0] > 0.5) &
        (points[:, 0] < 8.0)
    ]

    left_count = np.sum(near_points[:, 1] > 0)
    right_count = np.sum(near_points[:, 1] < 0)

    # 점이 적은 쪽이 더 비어 있음
    if left_count < right_count:
        return 0.4   # 왼쪽 조향
    else:
        return -0.4  # 오른쪽 조향


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def move_to_target_with_lidar(vehicle, target_loc):
    transform = vehicle.get_transform()
    loc = transform.location
    yaw = math.radians(transform.rotation.yaw)

    dx = target_loc.x - loc.x
    dy = target_loc.y - loc.y

    distance_to_goal = math.sqrt(dx * dx + dy * dy)

    if distance_to_goal < 3.0:
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        print("Arrived at target")
        return True

    target_yaw = math.atan2(dy, dx)
    yaw_error = normalize_angle(target_yaw - yaw)

    base_steer = np.clip(yaw_error * 0.8, -0.6, 0.6)

    points = latest_lidar["points"]
    obstacle, min_dist = detect_front_obstacle(points)

    if obstacle:
        avoid_steer = choose_avoid_direction(points)

        print(f"[LiDAR] Obstacle detected: {min_dist:.2f} m → avoid")

        control = carla.VehicleControl(
            throttle=0.25,
            steer=float(avoid_steer),
            brake=0.0
        )
    else:
        print(f"[Move] distance_to_goal={distance_to_goal:.2f} m")

        control = carla.VehicleControl(
            throttle=0.35,
            steer=float(base_steer),
            brake=0.0
        )

    vehicle.apply_control(control)
    return False


def main():
    actor_list = []

    try:
        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)

        world = client.get_world()

        # 동기 모드 설정
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()

        # 차량 생성
        vehicle_bp = blueprint_library.filter("vehicle.tesla.cybertruck")[0]
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)

        ego_vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(ego_vehicle)

        print("Ego vehicle spawned")

        spectator = world.get_spectator()

        # 자동 주행은 끔. 직접 제어할 것이기 때문.
        ego_vehicle.set_autopilot(False)

        # 2D LiDAR 생성
        lidar_bp = blueprint_library.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("channels", "1")
        lidar_bp.set_attribute("range", "30.0")
        lidar_bp.set_attribute("points_per_second", "7200")
        lidar_bp.set_attribute("rotation_frequency", "10.0")
        lidar_bp.set_attribute("upper_fov", "0.0")
        lidar_bp.set_attribute("lower_fov", "0.0")
        lidar_bp.set_attribute("sensor_tick", "0.05")

        lidar_transform = carla.Transform(
            carla.Location(x=0.0, y=0.0, z=1.8),
            carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
        )

        lidar = world.spawn_actor(
            lidar_bp,
            lidar_transform,
            attach_to=ego_vehicle
        )
        actor_list.append(lidar)
        lidar.listen(lidar_callback)

        print("2D LiDAR attached")

        # 장애물 차량 생성
        forward_vector = spawn_point.get_forward_vector()
        obstacle_location = spawn_point.location + forward_vector * 25.0
        obstacle_location.z += 0.3

        obstacle_bp = blueprint_library.filter("vehicle.audi.tt")[0]
        obstacle_tf = carla.Transform(
            obstacle_location,
            spawn_point.rotation
        )

        obstacle_vehicle = world.try_spawn_actor(obstacle_bp, obstacle_tf)
        if obstacle_vehicle:
            actor_list.append(obstacle_vehicle)
            obstacle_vehicle.set_autopilot(False)
            obstacle_vehicle.apply_control(carla.VehicleControl(hand_brake=True))
            print("Obstacle vehicle spawned")
        else:
            print("Obstacle spawn failed")

        # 목표 좌표: 시작 위치에서 전방 60m
        target_location = spawn_point.location + forward_vector * 60.0
        target_location.z = spawn_point.location.z

        print(
            f"Target: x={target_location.x:.2f}, "
            f"y={target_location.y:.2f}"
        )

        for step in range(2000):
            world.tick()

            tf = ego_vehicle.get_transform()
            loc = tf.location
            yaw = tf.rotation.yaw

            # 차량 뒤쪽 위치 계산
            forward = tf.get_forward_vector()

            camera_loc = loc - forward * 6 + carla.Location(z=3)

            spectator.set_transform(
                carla.Transform(
                    camera_loc,
                    carla.Rotation(pitch=-15, yaw=yaw)
                )
            )

            arrived = move_to_target_with_lidar(
                ego_vehicle,
                target_location
            )

            if arrived:
                break

            time.sleep(0.01)

    finally:
        print("Cleaning up...")

        for actor in actor_list:
            if actor is not None:
                actor.destroy()

        try:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
        except Exception:
            pass

        print("Done")


if __name__ == "__main__":
    main()