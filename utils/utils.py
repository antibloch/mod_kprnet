import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
def back_project(cl_map_2d, rows, cols):
    return cl_map_2d[rows, cols]
def spherical_projection(points,
                         reflectance,
                         labels,
                         fov_up_deg: float,
                         fov_down_deg: float,
                         H: int,
                         W: int
                         ):
    orcale_indices = np.arange(len(points))
    # based on salsa's projection function
    # 1) convert FOV to radians
    fov_up   = np.deg2rad(fov_up_deg)
    fov_down = np.deg2rad(fov_down_deg)
    fov = abs(fov_down) + abs(fov_up)
    # 2) depths and angles
    x, y, z = points[:,0], points[:,1], points[:,2]
    depth = np.linalg.norm(points, axis=1) + 1e-6  # avoid zero
    yaw   = -np.arctan2(y, -x)                # [-π, +π]
    pitch = np.arcsin(z / depth)            # [-π/2, +π/2]
    # 3) normalized projection coords in [0..1]
    proj_x = 0.5 * (yaw/np.pi + 1.0)         # azimuth → [0..1]
    proj_x = (proj_x - np.min(proj_x))/ (np.max(proj_x) - np.min(proj_x))
    proj_y_sub = (pitch + abs(fov_down))/fov
    proj_y_sub = (proj_y_sub - np.min(proj_y_sub)) / (np.max(proj_y_sub) - np.min(proj_y_sub))
    proj_y = 1.0 - proj_y_sub  # elevation → [0..1]
    # 4) scale to image size
    proj_x = proj_x * W
    proj_y = proj_y * (H-1)
    pee_x = proj_x.copy()
    # 5) integer pixel indices
    px = np.floor(proj_x).astype(np.int32)
    py = np.floor(proj_y).astype(np.int32)
    pee_y = py.copy()
    pee_y = pee_y.astype(np.float32)
    # clamp into valid range
    np.clip(px, 0, W-1, out=px)
    np.clip(py, 0, H-1, out=py)
    # 6) prepare output maps
    depth_map = np.zeros((H, W), dtype=np.float32)
    refl_map  = np.zeros((H, W), dtype=np.float32)
    # 7) sort points far→near so nearer overwrite farther
    order = np.argsort(depth)[::-1]
    px_ord = px[order]
    py_ord = py[order]
    depth_ord = depth[order]
    refl_ord  = reflectance[order]
    points_ord = points[order]
    labels_ord = labels[order]
    indices_ord = orcale_indices[order]
    # 8) fill
    depth_map[py_ord, px_ord] = depth_ord
    refl_map [py_ord, px_ord] = refl_ord
    # indices of points to image indices
    points_2_image_indices = {}
    for i in range(len(indices_ord )):
        points_2_image_indices[indices_ord [i]] = [py_ord[i], px_ord[i]]
        
    #============================================
    # 1) flatten labels and indices
    labels_flat = labels_ord.squeeze()                # shape (N,)
    indices     = indices_ord                         # shape (N,)
    # 2) build a (N,2) array of pixel coordinates
    coords = np.array([ points_2_image_indices[i]      # (row, col)
                        for i in indices ])            # shape (N,2)
    rows, cols = coords[:,0], coords[:,1]
    # 3) allocate and assign
    cl_map_2d = np.zeros((H, W), dtype=np.int32) + 18
    cl_map_2d[ rows, cols ] = labels_flat
    labels_ord = np.expand_dims(back_project(cl_map_2d, rows, cols), axis = -1)

    return depth_map, refl_map, cl_map_2d, rows, cols, points_ord, refl_ord, labels_ord