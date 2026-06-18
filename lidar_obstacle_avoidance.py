import carla
import random
import time
import math
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
import open3d as o3d
from sklearn.cluster import DBSCAN
from ros2.carla_publisher import CarlaPublisher

vis = None
pcd = None
latest_xyz = None

latest_lidar = {
    "points": None
}

latest_frame = None

# YOLO 모델 서버 아이피, 주소
HOST = '192.168.0.10'
PORT = '1234'

# YOLO 모델 Load
# model = YOLO("yolov8n.pt")

# Ros2 Publisher Node
node: CarlaPublisher = None

# RGB 카메라 콜백
def camera_callback(image):

    global latest_frame

    node.publish_carla_image(image)

    # array = np.frombuffer(
    #     image.raw_data,
    #     dtype=np.uint8
    # )

    # array = np.reshape(
    #     array,
    #     (image.height, image.width, 4)
    # )

    # frame = array[:, :, :3].copy()      # BGRA Raw 데이터

    # # Object Detection
    # results = model(frame)
    
    # # Bounding Box 가져오기
    # for result in results:
    #     for box in result.boxes:

    #         cls = int(box.cls[0])

    #         conf = float(box.conf[0])

    #         x1, y1, x2, y2 = map(
    #             int,
    #             box.xyxy[0]
    #         )

    #         print(cls, conf)
            
    #         names = model.names

    #         label = f"{names[cls]} {conf:.2f}"
            
    #         cv2.rectangle(
    #             frame,
    #             (x1, y1),
    #             (x2, y2),
    #             (0,255,0),
    #             2
    #         )
            
    #         cv2.putText(
    #             frame,
    #             label,
    #             (x1, y1 - 10),
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             0.5,
    #             (0,255,0),
    #             2
    #         )

    # # Ultralytics YOLO 는 Bounding Box, Class Name, Confidence 전부 그려주는 기능 존재
    # latest_frame = results[0].plot()

# Lidar 콜백
def lidar_callback(data):

    global latest_xyz

    points = np.frombuffer(
        data.raw_data,
        dtype=np.float32
    )

    points = np.reshape(points, (-1, 4))

    # =========================
    # ego vehicle body mask
    # =========================
    not_self = ~(
        (points[:,0] > -3.0) &
        (points[:,0] < 4.0) &
        (np.abs(points[:,1]) < 2.5)
    )

    points = points[not_self]

    # =========================
    # ground / curb filtering
    # =========================
    points = points[points[:,2] > -0.5]

    xyz = points[:, :3]

    # =========================
    # distance filtering
    # =========================
    dist = np.linalg.norm(
        xyz,
        axis=1
    )

    xyz = xyz[dist > 2.5]

    # =========================
    # Open3D visualization
    # =========================
    latest_xyz = xyz.copy()

    # =========================
    # 2D obstacle detection
    # =========================
    xy = xyz[:, :2]

    latest_lidar["points"] = xy

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

    # =========================
    # downsample
    # =========================
    points = points[::2]

    # =========================
    # distance
    # =========================
    distances = np.linalg.norm(
        points,
        axis=1
    )

    # =========================
    # angle
    # =========================
    angles = np.degrees(
        np.arctan2(
            points[:,1],
            points[:,0]
        )
    )

    # =========================
    # forward sector
    # =========================
    mask = (
        (np.abs(angles) < 40) &
        (distances > 2.0) &
        (distances < 40.0)
    )

    filtered = points[mask]

    return filtered

def cluster_obstacles(points):

    if points is None or len(points) == 0:
        return []

    clustering = DBSCAN(
        eps=1.8,
        min_samples=3
    ).fit(points)

    labels = clustering.labels_

    filtered_clusters = []

    for label in set(labels):

        if label == -1:
            continue

        cluster = points[labels == label]
        
        # print(f"cluster 좌표 확인 : {cluster}")

        width = (
            np.max(cluster[:,0]) -
            np.min(cluster[:,0])
        )

        height = (
            np.max(cluster[:,1]) -
            np.min(cluster[:,1])
        )

        # =========================
        # 너무 긴 벽 제거
        # =========================
        if width > 6 or height > 6:
            continue

        # =========================
        # 너무 작은 noise 제거
        # =========================
        if len(cluster) < 3:
            continue

        filtered_clusters.append(cluster)

    return filtered_clusters

def find_closest_cluster(clusters):

    closest_cluster = None
    min_dist = 9999

    for cluster in clusters:

        distances = np.linalg.norm(
            cluster,
            axis=1
        )

        nearest_dist = np.min(distances)

        if nearest_dist < min_dist:
            min_dist = nearest_dist
            closest_cluster = cluster

    return closest_cluster, min_dist

def find_front_vehicle_like_cluster(clusters):

    best_cluster = None
    best_dist = 9999

    for cluster in clusters:

        # =========================
        # cluster 중심
        # =========================
        center = np.mean(cluster, axis=0)

        x, y = center

        cluster_dist = np.linalg.norm(center)

        width = (
            np.max(cluster[:,0]) -
            np.min(cluster[:,0])
        )

        height = (
            np.max(cluster[:,1]) -
            np.min(cluster[:,1])
        )

        # 너무 가까운 자기 차체 제거
        if cluster_dist < 3.0:
            continue

        # 전방 후보
        if not (3.0 < x < 30.0 and abs(y) < 5.0):
            continue

        # 긴 벽 제거
        if width > 8.0 or height > 8.0:
            continue

        if cluster_dist < best_dist:
            best_dist = cluster_dist
            best_cluster = cluster

    return best_cluster, best_dist

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

    print("yaw:", math.degrees(yaw))
    print("target:", math.degrees(target_yaw))
    print("error:", math.degrees(yaw_error))

    base_steer = np.clip(yaw_error * 0.8, -0.6, 0.6)

    points = preprocess_lidar(latest_lidar["points"])

    clusters = cluster_obstacles(points)

    closest_cluster, cluster_dist = find_front_vehicle_like_cluster(clusters)

    obstacle = False
    min_dist = None

    if closest_cluster is not None:

        distances = np.linalg.norm(
            closest_cluster,
            axis=1
        )

        nearest_idx = np.argmin(distances)

        nearest_point = closest_cluster[nearest_idx]

        x = nearest_point[0]
        y = nearest_point[1]

        dist = distances[nearest_idx]

        print(
            f"[OBSTACLE] "
            f"x={x:.2f}, y={y:.2f}, dist={dist:.2f}"
        )

        if (
            x > 0.5 and
            x < 20.0 and
            abs(y) < 3.5
        ):
            obstacle = True
            min_dist = dist

    # print(points)

    # print(obstacle, min_dist)

    if obstacle:
        avoid_steer = choose_avoid_direction(closest_cluster)

        print(f"[LiDAR] Obstacle detected: {min_dist:.2f} m → avoid")

        steer = 0.3 * base_steer + 0.7 * avoid_steer
        steer = float(np.clip(steer, -1.0, 1.0))
        
        print(f"[STEER] {steer}")

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
    global vis
    global pcd
    global latest_xyz

    # Open3D Visualizer 생성
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="CARLA LiDAR",
        width=960,
        height=540
    )

    pcd = o3d.geometry.PointCloud()

    vis.add_geometry(pcd)
    
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=3.0
    )

    vis.add_geometry(axis)
    
    render_option = vis.get_render_option()

    render_option.point_size = 3.0
    
    ctr = vis.get_view_control()

    ctr.set_front([0, 0, 1])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([1, 0, 0])
    ctr.set_zoom(0.7)
    
    actor_list = []

    # ros2 publisher node 코드
    rclpy.init()
    node = CarlaPublisher()

    try:
        # ros2 node callback 실행
        rclpy.spin_once(node, timeout_sec=0.0)

        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)

        print(client.get_client_version())
        print(client.get_server_version())

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
        lidar_bp.set_attribute("range", "60.0")
        lidar_bp.set_attribute("rotation_frequency", "10.0")
        lidar_bp.set_attribute("sensor_tick", "0.05")
        lidar_bp.set_attribute("channels", "16")
        lidar_bp.set_attribute("points_per_second", "50000")
        lidar_bp.set_attribute("upper_fov", "10.0")
        lidar_bp.set_attribute("lower_fov", "-10.0")

        lidar_transform = carla.Transform(
            carla.Location(
                x=2.5,
                y=0.0,
                z=2.0
            ),
            carla.Rotation(
                pitch=-5.0,
                yaw=0.0,
                roll=0.0
            )
        )

        lidar = world.spawn_actor(
            lidar_bp,
            lidar_transform,
            attach_to=ego_vehicle
        )
        actor_list.append(lidar)
        lidar.listen(lidar_callback)

        print("2D LiDAR attached")
        
        # RGB 카메라 생성
        camera_bp = blueprint_library.find("sensor.camera.rgb")

        camera_bp.set_attribute("image_size_x", "1280")     # 이미지 너비(픽셀)
        camera_bp.set_attribute("image_size_y", "720")      # 이미지 높이(픽셀)
        camera_bp.set_attribute("fov", "90")                # 수평 시야각(도)

        camera_transform = carla.Transform(
            carla.Location(x=1.5, z=2.4)
        )

        camera = world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=ego_vehicle
        )
        actor_list.append(camera)
        camera.listen(camera_callback)

        print("camera attached")

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

            # Bounding Box 시각화
            if latest_frame is not None:
                cv2.imshow("RGB", latest_frame)
                cv2.waitKey(1)

            # open3D 좌표 업데이트
            if latest_xyz is not None:
                xyz = np.copy(latest_xyz)

                xyz = np.stack([
                    xyz[:,0],
                    -xyz[:,1],
                    xyz[:,2]
                ], axis=1)
                
                pcd.points = o3d.utility.Vector3dVector(xyz)

                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()

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

        vis.destroy_window()

        # ros2 node 정리
        node.destroy_node()
        rclpy.shutdown()

        print("Done")


if __name__ == "__main__":
    main()