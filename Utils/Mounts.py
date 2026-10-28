import numpy as np
from scipy.spatial.transform import Rotation

def kabsch(M, S):
    """
    M : Nx3 source vectors
    S : Nx3 destination vectors

    Find R such that:

        S ≈ R @ M

    """

    H = M.T @ S

    U, _, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    # prevent reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    return R

def rotX(a):
    c = np.cos(a)
    s = np.sin(a)

    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])

def rotY(a):
    c = np.cos(a)
    s = np.sin(a)

    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])

def rotZ(a):
    c = np.cos(a)
    s = np.sin(a)

    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])


def rotationMatrix(pitch, roll, yaw):

    return rotZ(np.radians(0-yaw)) @ rotY(np.radians(pitch)) @ rotX(np.radians(roll))


def recoverAngles(rotation_matrix):

    r = Rotation.from_matrix(rotation_matrix)

    # Extract in the same order:
    # R = Rz(angle1) @ Ry(angle2) @ Rx(angle3)
    x, y, z = r.as_euler('xyz', degrees=True)

    roll_deg = x
    pitch_deg = y
    yaw_deg = -z

    return pitch_deg, roll_deg, yaw_deg


class MountModel:


    def __init__(self):
        self.installation_rotation_matrix = rotationMatrix(0,0,0)

    def setInstallation(self,pitch_deg,roll_deg,yaw_deg):

        self.installation_rotation_matrix = rotationMatrix(pitch_deg, roll_deg, yaw_deg)

    def mountVector(self, pan_deg, tilt_deg):

        p, t = np.radians(pan_deg), np.radians(tilt_deg)

        # Return in an ENU frame
        return np.array([np.cos(t)*np.sin(p), np.cos(t)*np.cos(p), np.sin(t)])

    def computeSkyVector(self, pan, tilt):

        return self.installation_rotation_matrix @ self.mountVector(pan, tilt)


mount = MountModel()

# Configure mount rotations around the three local axes
pitch_mount, roll_mount, yaw_mount = 10, 15, 20
pan, tilt = 0, 0
mount.setInstallation(pitch_mount, roll_mount, yaw_mount)
print(f"Set Initial angles {pitch_mount, roll_mount, yaw_mount}")
print(f"Recovered Initial angles {recoverAngles(mount.installation_rotation_matrix)}")
sky_vector = mount.computeSkyVector(pan, tilt)

print(f"Installation characteristics pitch {pitch_mount}, roll {roll_mount}, yaw {yaw_mount}")
print(f"Pan {pan}, tilt {tilt}")
print(f"Achieved vector {sky_vector[0]:.2f},{sky_vector[1]:.2f},{sky_vector[2]:.2f} ")
observation_count = 50
pan_arr, tilt_arr = np.random.uniform(0,350, size=observation_count), np.random.uniform(0,90, size=observation_count)
sky_vector_list, ideal_mount_vector_list = [], []
ideal_mount = MountModel()
ideal_mount.setInstallation(0, 0, 0)

for observation, (pan, tilt) in enumerate(zip(pan_arr, tilt_arr)):

    print(f"Setting mount to {pan:.2f},{tilt:.2f}")
    ideal_mount_vector_list.append(ideal_mount.mountVector(pan, tilt))
    pan_modified = pan + np.random.uniform(-2,2)
    tilt_modified = tilt + np.random.uniform(-2,2)
    sky_vector_list.append(mount.computeSkyVector(pan_modified, tilt_modified))


ideal_vectors, sky_vectors = np.array(ideal_mount_vector_list), np.array(sky_vector_list)

fit_transformation = kabsch(ideal_vectors, sky_vectors)
print(fit_transformation)
print(mount.installation_rotation_matrix)
pitch_recovered, roll_recovered, yaw_recovered = recoverAngles(fit_transformation)

print("Installed\n")
print(f"Pitch {pitch_mount:.2f}")
print(f"Roll {roll_mount:.2f}")
print(f"Yaw {yaw_mount:.2f}")

print(f"\nRecovered after {observation_count} observations\n")
print(f"Pitch {pitch_recovered:.2f}")
print(f"Roll {roll_recovered:.2f}")
print(f"Yaw {yaw_recovered:.2f}")