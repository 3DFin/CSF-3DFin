from turtledemo.clock import current_day
from .CSF_3DFin_ext import CSF, CSFParams, __doc__

import fcpw
import time
import numpy as np


def extract_exact_z0(pointcloud, bSloopSmooth=True, cloth_resolution=0.5):
    """Extract the exact Z0 from a point cloud using mesh-ray intersection.

    Uses CSF to return a mesh of the DRM of the point cloud and then computes the exact vertical distance (Z0)
    from each point to this surface via ray-triangle intersections. This is a WIP method to serve
    as a benchmark vs.  the 3-nearest-neighbors (3NN) interpolation method currently used in Dendromatics.

    Parameters:
    ----------
    pointcloud : np.ndarray
        A Nx3 numpy matrix representing the 3D point cloud [X, Y, Z].
    bSloopSmooth : bool, optional
        If True, enables slope smoothing in CSF.
    cloth_resolution : float
        The resolution of the cloth mesh in the CSF filter.

    Returns:
    -------
    exact_z0 : np.ndarray
        A 1D numpy array of length N containing the exact Z0 value
        for each input point.
    """

    csf = CSF()
    csf.params.smooth_slope = bSloopSmooth
    csf.params.cloth_resolution = cloth_resolution
    csf.params.verbose = True
    csf.set_point_cloud(pointcloud)
    print("Running exact classification (cloth simulation)...")
    cloth_points, tri_id = csf.cloth_and_mesh()
    start = time.time()

    print("Exact Z0 computation by mesh intersection")
    # initialize a 3d scene
    scene = fcpw.scene_3D()

    # load positions and indices of a single triangle mesh
    scene.set_object_count(1)
    scene.set_object_vertices(cloth_points, 0)
    scene.set_object_triangles(tri_id, 0)

    # Compute the min/max Z0 for bounds
    print("Max bound computation")
    zmin = np.min(pointcloud[:, 2])
    zmax = np.max(pointcloud[:, 2])
    max_bound = abs(zmax - zmin) + 0.01
    print(f"max bound {max_bound}")

    # build acceleration structure
    print("Building the BVH")
    aggregate_type = fcpw.aggregate_type.bvh_surface_area
    build_vectorized_bvh = True
    scene.build(aggregate_type, build_vectorized_bvh, True)
    print("End BVH computation")

    # This create ray pointing downward
    dirs = np.zeros((pointcloud.shape[0], 3))
    dirs[:, 2] = -1.0
    # Create bounds to limit the computation time
    bounds = np.ones(pointcloud.shape[0]) * max_bound
    exact_z0 = np.zeros(pointcloud.shape[0])

    print("Do the intersections (positive)")
    # do the intersection
    interactions = fcpw.interaction_3D_list()
    scene.intersect(pointcloud, dirs, bounds, interactions)

    # extract the distance from the interactions
    # this part is more time consuming than the intersection itself
    # see if we can improve the bindings batch retrieve these data
    for id, interaction in enumerate(interactions):
        exact_z0[id] = interaction.d

    # Now we flag points that do not have intersections
    # these are likely to be "under" the cloth mesh
    # so we have to compute an intersection with an upward
    # pointing ray
    negative_mask = np.where(exact_z0 > max_bound + 0.01)
    exact_num_negative = negative_mask[0].shape[0]
    print(f"Exact: number of negative points {exact_num_negative}")

    print("Do the intersections (negative)")
    interactions = fcpw.interaction_3D_list()
    negative_pc = pointcloud[negative_mask]
    bounds = bounds[negative_mask]
    dirs = dirs[negative_mask] * -1.0
    scene.intersect(negative_pc, dirs, bounds, interactions)

    for id, interaction in enumerate(interactions):
        exact_z0[negative_mask[0][id]] = -interaction.d
    print(f"Exact computation time {time.time() - start}")

    return exact_z0
