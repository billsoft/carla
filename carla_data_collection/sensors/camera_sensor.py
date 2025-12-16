"""Camera Sensor Module

Responsibilities:
1. Create CARLA camera sensor
2. Compute camera intrinsic matrix
3. Compute camera extrinsic matrix
4. Receive image data callback
"""

import carla
import numpy as np
import weakref
import queue


class CameraSensor:
    """Single camera sensor wrapper"""

    def __init__(self, world, vehicle, config):
        """
        Initialize camera sensor

        Args:
            world: CARLA World object
            vehicle: Vehicle to attach camera to
            config: Camera configuration dict (contains id, transform, width, height, fov)
        """
        self.world = world
        self.vehicle = vehicle
        self.config = config
        self.camera_id = config['id']
        self.sensor = None
        self.data_queue = queue.Queue()

        # Image dimensions
        self.width = config['width']
        self.height = config['height']
        self.fov = config['fov']  # Horizontal FOV (degrees)

        # Compute camera intrinsic and extrinsic
        self.intrinsic_matrix = self._compute_intrinsic_matrix()
        self.extrinsic_matrix = self._compute_extrinsic_matrix()

        # Create sensor
        self._setup_sensor()

    def _compute_intrinsic_matrix(self):
        """
        Compute camera intrinsic matrix K (3x3)

        Based on pinhole camera model:
            fx = fy = width / (2 * tan(fov/2))
            cx = width / 2
            cy = height / 2

        Returns:
            K: (3, 3) numpy array
                [[fx,  0, cx],
                 [ 0, fy, cy],
                 [ 0,  0,  1]]
        """
        fov_rad = np.radians(self.fov)
        focal_length = self.width / (2.0 * np.tan(fov_rad / 2.0))

        K = np.array([
            [focal_length, 0,            self.width / 2.0],
            [0,            focal_length, self.height / 2.0],
            [0,            0,            1.0]
        ], dtype=np.float32)

        return K

    def _compute_extrinsic_matrix(self):
        """
        Compute camera extrinsic matrix [R|t] (4x4)

        Transform from vehicle coordinate system to camera coordinate system

        Returns:
            E: (4, 4) numpy array
                [[r11, r12, r13, tx],
                 [r21, r22, r23, ty],
                 [r31, r32, r33, tz],
                 [  0,   0,   0,  1]]
        """
        transform = self.config['transform']
        loc = transform.location
        rot = transform.rotation

        # Convert Euler angles to radians
        pitch = np.radians(rot.pitch)
        yaw = np.radians(rot.yaw)
        roll = np.radians(rot.roll)

        # Compute rotation matrix R (ZYX order)
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)

        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr           ]
        ], dtype=np.float32)

        # Combine into 4x4 transformation matrix
        E = np.eye(4, dtype=np.float32)
        E[:3, :3] = R
        E[:3, 3] = [loc.x, loc.y, loc.z]

        return E

    def _setup_sensor(self):
        """Create and configure CARLA camera sensor"""
        bp_lib = self.world.get_blueprint_library()

        # Use RGB camera
        camera_bp = bp_lib.find('sensor.camera.rgb')

        # Set attributes
        camera_bp.set_attribute('image_size_x', str(self.width))
        camera_bp.set_attribute('image_size_y', str(self.height))
        camera_bp.set_attribute('fov', str(self.fov))

        # 12-bit RAW configuration (extended dynamic range)
        camera_bp.set_attribute('enable_postprocess_effects', 'False')
        camera_bp.set_attribute('gamma', '2.2')  # Remove gamma correction

        # Apply physical lens distortion (fisheye/wide-angle simulation)
        if 'lens_distortion' in self.config and self.config['lens_distortion'] is not None:
            distortion = self.config['lens_distortion']

            for attr_name, attr_value in distortion.items():
                if camera_bp.has_attribute(attr_name):
                    camera_bp.set_attribute(attr_name, str(attr_value))

            print(f"  [{self.camera_id}] Lens distortion applied: "
                  f"k={distortion.get('lens_k', 0):.3f}, "
                  f"circle_mult={distortion.get('lens_circle_multiplier', 0):.1f}")

        # Spawn sensor
        self.sensor = self.world.spawn_actor(
            camera_bp,
            self.config['transform'],
            attach_to=self.vehicle
        )

        # Set callback function
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: CameraSensor._on_image_received(weak_self, image))

    @staticmethod
    def _on_image_received(weak_self, image):
        """
        Camera data callback function

        Args:
            weak_self: Weak reference to CameraSensor
            image: CARLA Image object
        """
        self = weak_self()
        if not self:
            return

        # Put image data into queue
        data = {
            'frame': image.frame,
            'timestamp': image.timestamp,
            'raw_data': image.raw_data,
            'width': image.width,
            'height': image.height,
            'camera_id': self.camera_id
        }

        self.data_queue.put(data)

    def get_intrinsic(self):
        """Return camera intrinsic matrix (3, 3)"""
        return self.intrinsic_matrix.copy()

    def get_extrinsic(self):
        """Return camera extrinsic matrix (4, 4)"""
        return self.extrinsic_matrix.copy()

    def destroy(self):
        """Destroy sensor"""
        if self.sensor is not None:
            self.sensor.stop()
            self.sensor.destroy()
            self.sensor = None
