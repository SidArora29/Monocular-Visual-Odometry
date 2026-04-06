import open3d as o3d
import numpy as np

import os

os.environ["XDG_SESSION_TYPE"]="x11"
os.environ["GDK_BACKEND"]="x11"
os.environ["MESA_D3D12_DEFAULT_ADAPTER_NAME"]="NVIDIA"

def visualize_map():
    # 1. Load the data as a string first to handle errors safely
    try:
        data = np.loadtxt("/home/sid/mono-vo/build/stereo_map.xyz")
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    if data.size == 0:
        print("File is empty!")
        return

    print(f"Loaded {len(data)} points.")

    # 2. Filter out any remaining NaNs or Infs just in case
    data = data[~np.isnan(data).any(axis=1)]
    data = data[~np.isinf(data).any(axis=1)]

    # 3. Create Open3D Point Cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)

    # Optional: Color them slightly nicely (Height-based coloring)
    # This helps you see the "floor" vs "walls"
    colors = np.zeros_like(data)
    # Simple trick: map Y (height) to Green channel
    min_y, max_y = np.min(data[:, 1]), np.max(data[:, 1])
    colors[:, 1] = (data[:, 1] - min_y) / (max_y - min_y + 0.001) # Green gradient
    colors[:, 0] = 0.5 # Add some red
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 4. Create a coordinate frame to see where the origin is
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])

    # 5. Visualize
    o3d.visualization.draw_geometries([pcd, axes], window_name="My First SLAM Map")

if __name__ == "__main__":
    visualize_map()