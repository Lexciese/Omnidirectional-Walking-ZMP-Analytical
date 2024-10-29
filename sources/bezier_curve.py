import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def cubic_swing_foot_path(str_pt, end_pt, swing_height, t):
    p0 = str_pt.copy()
    p1 = str_pt.copy()
    p1[2,0] = swing_height + (0.25 * swing_height)
    p2 = end_pt.copy()
    p2[2,0] = swing_height + (0.25 * swing_height)
    p3 = end_pt.copy()
    path = (np.power((1-t), 3) * p0 +
            3 * np.power((1-t), 2) * t * p1 +
            3 * (1-t) * np.power(t, 2) * p2 +
            np.power(t, 3) * p3)
    return path

def quintic_swing_foot_path(str_pt, end_pt, swing_height, t):
    p0 = str_pt.copy()
    p1 = str_pt.copy()
    p1[2,0] = swing_height * 0.5  # Initial upward adjustment

    p2 = str_pt.copy()
    p2[2,0] = swing_height  # Peak height in the first half of the swing

    p3 = end_pt.copy()
    p3[2,0] = swing_height  # Peak height in the second half of the swing

    p4 = end_pt.copy()
    p4[2,0] = swing_height * 0.5  # Final downward adjustment

    p5 = end_pt.copy()

    # Quintic Bézier curve calculation
    path = (np.power((1-t), 5) * p0 +
            5 * np.power((1-t), 4) * t * p1 +
            10 * np.power((1-t), 3) * np.power(t, 2) * p2 +
            10 * np.power((1-t), 2) * np.power(t, 3) * p3 +
            5 * (1-t) * np.power(t, 4) * p4 +
            np.power(t, 5) * p5)
    return path

# Define starting and ending points, and swing height
str_pt = np.array([[0.0], [0.0], [0.0]])
end_pt = np.array([[1.0], [0.0], [0.0]])
swing_height = 0.1

# Generate points for the swing path
t_values = np.linspace(0, 1, 100)
path_points_cubic = np.array([cubic_swing_foot_path(str_pt, end_pt, swing_height, t) for t in t_values]).squeeze()
path_points_quintic = np.array([quintic_swing_foot_path(str_pt, end_pt, swing_height, t) for t in t_values]).squeeze()


# Plotting Cubic Bézier Curve
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(path_points_cubic[:, 0], path_points_cubic[:, 1], path_points_cubic[:, 2], label="Swing Foot Path", color="blue")
ax.scatter([str_pt[0, 0], end_pt[0, 0]], [str_pt[1, 0], end_pt[1, 0]], [str_pt[2, 0], end_pt[2, 0]], color="red", label="Start/End Points")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Swing Foot Path Visualization")
ax.legend()
plt.show()

# Plotting Quintic Bézier Curve
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(path_points_quintic[:, 0], path_points_quintic[:, 1], path_points_quintic[:, 2], label="Quintic Swing Foot Path", color="purple")
ax.scatter([str_pt[0, 0], end_pt[0, 0]], [str_pt[1, 0], end_pt[1, 0]], [str_pt[2, 0], end_pt[2, 0]], color="red", label="Start/End Points")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Quintic Swing Foot Path Visualization")
ax.legend()
plt.show()
