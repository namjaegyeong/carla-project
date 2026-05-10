import carla
import random
import time
import math
import numpy as np
from sklearn.cluster import DBSCAN


latest_lidar = {
    "points": None
}

def lidar_callback(data):
    points = np.frombuffer(data.raw_data, dtype=np.float32)
    points = np.reshape(points, (-1, 4))

    # vehicle local coordinate 기준 x, y만 사용
    xy = points[:, :2]
    latest_lidar["points"] = xy
    
    # lidar 좌표 확인 (차량 기준 좌표계)
    # distances = np.linalg.norm(points, axis=1)

    # sorted_idx = np.argsort(distances)

    # nearest_points = points[sorted_idx[:10]]

    # print(f"nearest_points : {nearest_points[:, :2]}")
    
    # for p in nearest_points[:, :2]:
    #     x, y = p

    #     dist = math.sqrt(x*x + y*y)
    #     angle = math.degrees(math.atan2(y, x))

    #     print(
    #         f"x={x:.2f}, y={y:.2f}, "
    #         f"dist={dist:.2f}, angle={angle:.2f}"
    #     )
    
    angles = np.degrees(np.arctan2(points[:,1], points[:,0]))
    distances = np.linalg.norm(points, axis=1)

    mask = (
        (np.abs(angles) < 90) &
        (distances < 20)
    )

    filtered = points[mask]

    nearest_idx = np.argsort(
        np.linalg.norm(filtered, axis=1)
    )

    nearest_points = filtered[nearest_idx[:10]]

    print("\n=== FRONT NEAREST ===")
    print(f"frone nearest : {nearest_points[:, :2]}")

# 거리 기반 회피 판단
def choose_avoid_direction(cluster):
    """
    obstacle 중심 기준 회피 방향 결정
    """

    if cluster is None or len(cluster) == 0:
        return 0.0

    center = np.mean(cluster, axis=0)

    x = center[0]
    y = center[1]

    print(f"[CLUSTER CENTER] x={x:.2f}, y={y:.2f}")

    # 장애물이 왼쪽 → 오른쪽 회피
    if y > 0:
        return -0.8

    # 장애물이 오른쪽 → 왼쪽 회피
    else:
        return 0.8

def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))

# LiDAR 전처리
def preprocess_lidar(points):
    if points is None or len(points) == 0:
        return points

    # Downsample
    # points = points[::5]

    # 너무 먼 점 제거
    angles = np.degrees(np.arctan2(points[:,1], points[:,0]))
    distances = np.linalg.norm(points, axis=1)

    mask = (
        (np.abs(angles) < 60) &
        (distances < 20)
    )

    filtered = points[mask]
    
    return filtered

def cluster_obstacles(points):
    """
    LiDAR point들을 obstacle 단위로 clustering
    """

    if points is None or len(points) == 0:
        return []

    clustering = DBSCAN(
        eps=1.2,       # point 간 최대 거리
        min_samples=4  # 최소 point 개수
    ).fit(points)

    labels = clustering.labels_

    clusters = []

    for label in set(labels):

        # noise 제거
        if label == -1:
            continue

        cluster_points = points[labels == label]

        clusters.append(cluster_points)

    return clusters

def find_closest_cluster(clusters):

    closest_cluster = None
    min_dist = 9999

    for cluster in clusters:

        # cluster 내부 point 중 가장 가까운 point
        distances = np.linalg.norm(cluster, axis=1)

        nearest_dist = np.min(distances)

        if nearest_dist < min_dist:
            min_dist = nearest_dist
            closest_cluster = cluster

    return closest_cluster, min_dist

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

    points = preprocess_lidar(latest_lidar["points"])

    clusters = cluster_obstacles(points)

    closest_cluster, cluster_dist = find_closest_cluster(clusters)

    obstacle = False
    min_dist = None

    if closest_cluster is not None:

        center = np.mean(closest_cluster, axis=0)

        x = center[0]
        y = center[1]

        dist = np.linalg.norm(center)

        print(f"[OBSTACLE] center=({x:.2f}, {y:.2f}), dist={dist:.2f}")

        # 전방 obstacle만 사용
        if (
            x > 0.5 and
            x < 10.0 and
            abs(y) < 4.0
        ):
            obstacle = True
            min_dist = dist

    print(points)

    print(obstacle, min_dist)

    if obstacle:
        avoid_steer = choose_avoid_direction(closest_cluster)

        print(f"[LiDAR] Obstacle detected: {min_dist:.2f} m → avoid")

        steer = 0.3 * base_steer + 0.7 * avoid_steer
        steer = float(np.clip(steer, -1.0, 1.0))

        if min_dist < 2.5:
            throttle = 0.0
            brake = 0.8
        else:
            throttle = 0.2
            brake = 0.0

    else:
        print(f"[Move] distance_to_goal={distance_to_goal:.2f} m")
        
        steer = base_steer
        throttle = 0.35
        brake = 0.0
        
    control = carla.VehicleControl(
        throttle=throttle,
        steer=float(steer),
        brake=brake
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
            f"Ego Vehicle: x={spawn_point.location.x:.2f}, "
            f"y={spawn_point.location.y:.2f}"
        )
        
        print(
            f"Obstacle: x={obstacle_location.x:.2f}, "
            f"y={obstacle_location.y:.2f}"
        )

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

            # 차량 좌표 확인하기 (world 좌표계)
            # print(f"car x={loc.x:.2f}, y={loc.y:.2f}, ")

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