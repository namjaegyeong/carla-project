import carla
import random
import math
import time
import numpy as np


# =========================
# UWB Anchor 설정
# =========================
ANCHORS = {
    0: carla.Location(x=0.0, y=0.0, z=2.0),
    1: carla.Location(x=30.0, y=0.0, z=2.0),
    2: carla.Location(x=0.0, y=30.0, z=2.0),
    3: carla.Location(x=30.0, y=30.0, z=2.0),
}


def simulate_uwb(ego_vehicle):
    """
    CARLA ground truth 위치를 이용해서 UWB 거리 측정값 생성
    """
    ego_loc = ego_vehicle.get_transform().location

    measurements = []

    for anchor_id, anchor_loc in ANCHORS.items():
        dx = ego_loc.x - anchor_loc.x
        dy = ego_loc.y - anchor_loc.y
        dz = ego_loc.z - anchor_loc.z

        true_range = math.sqrt(dx * dx + dy * dy + dz * dz)

        # UWB noise
        noise = random.gauss(0.0, 0.10)  # sigma = 10cm

        # NLOS bias 예시
        nlos_bias = 0.0
        # 필요하면 특정 구간에서 bias 추가 가능
        # if ego_loc.x > 10 and ego_loc.y > 10:
        #     nlos_bias = random.uniform(0.5, 2.0)

        measured_range = true_range + noise + nlos_bias

        measurements.append({
            "anchor_id": anchor_id,
            "true_range": true_range,
            "measured_range": measured_range,
        })

    return measurements


def lidar_callback(data):
    """
    LiDAR point cloud callback
    CARLA LiDAR data는 float32 배열: x, y, z, intensity 반복
    """
    points = np.frombuffer(data.raw_data, dtype=np.float32)
    points = np.reshape(points, (-1, 4))

    # 2D LiDAR로 사용할 것이므로 x, y만 사용
    xy = points[:, :2]

    print(f"[LiDAR] frame={data.frame}, points={len(xy)}")


def main():
    actor_list = []

    try:
        # =========================
        # CARLA 서버 연결
        # =========================
        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)

        world = client.get_world()

        # 맵 변경하고 싶으면 사용
        # world = client.load_world("Town03")

        # 맵 로딩까지 대기
        time.sleep(2.0)

        # =========================
        # Synchronous Mode 설정
        # =========================
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 Hz
        world.apply_settings(settings)

        # =========================
        # Blueprint 가져오기
        # =========================
        blueprint_library = world.get_blueprint_library()

        vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]

        spawn_points = world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)

        # =========================
        # 차량 생성
        # =========================
        ego_vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(ego_vehicle)

        print("Ego vehicle spawned")

        # 카메라 시점을 차량으로 옮기기
        spectator = world.get_spectator()

        transform = ego_vehicle.get_transform()
        spectator.set_transform(
            carla.Transform(
                transform.location + carla.Location(z=10),
                carla.Rotation(pitch=-90)
            )
        )

        # 자동 주행 ON
        ego_vehicle.set_autopilot(True)

        # =========================
        # 2D LiDAR 생성
        # =========================
        lidar_bp = blueprint_library.find("sensor.lidar.ray_cast")

        lidar_bp.set_attribute("channels", "1")               # 2D LiDAR 핵심
        lidar_bp.set_attribute("range", "30.0")
        lidar_bp.set_attribute("points_per_second", "7200")
        lidar_bp.set_attribute("rotation_frequency", "10.0")
        lidar_bp.set_attribute("upper_fov", "0.0")
        lidar_bp.set_attribute("lower_fov", "0.0")
        lidar_bp.set_attribute("sensor_tick", "0.1")

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

        print("2D LiDAR spawned")

        # =========================
        # 메인 루프
        # =========================
        for step in range(1000):
            world.tick()

            tf = ego_vehicle.get_transform()
            loc = tf.location
            yaw = tf.rotation.yaw

            uwb_data = simulate_uwb(ego_vehicle)

            print("=" * 60)
            print(f"[EGO] step={step}, x={loc.x:.3f}, y={loc.y:.3f}, z={loc.z:.3f}, yaw={yaw:.2f}")

            for m in uwb_data:
                print(
                    f"[UWB] anchor={m['anchor_id']}, "
                    f"true={m['true_range']:.3f}, "
                    f"measured={m['measured_range']:.3f}"
                )

            time.sleep(0.01)

    finally:
        # =========================
        # 종료 처리
        # =========================
        print("Cleaning up actors...")

        for actor in actor_list:
            if actor is not None:
                actor.destroy()

        # synchronous mode 해제
        try:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
        except:
            pass

        print("Done")


if __name__ == "__main__":
    main()