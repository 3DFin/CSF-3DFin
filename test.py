from CSF_3DFin import CSF
import numpy as np
from scipy.spatial import KDTree
import laspy

def clean_cloth(dtm_points):
    """This function takes a Digital Terrain Model (DTM) and denoises it. This
    denoising is done via a 2 MADs criterion from the median height value of a
    neighborhood of size 15.

    Parameters
    ----------
    dtm_points : numpy.ndarray
        Matrix containing (x, y, z) coordinates of the DTM points.

    Returns
    -------
    clean_points : numpy.ndarray
        Matrix containing (x, y, z) coordinates of the denoised DTM points.
    """

    if dtm_points.shape[0] < 15:
        raise ValueError("input DTM is too small (less than 15 points). Denoising cannot be done.")
    tree = KDTree(dtm_points[:, :2])
    _, indexes = tree.query(dtm_points[:, :2], 15, workers=-1)
    abs_devs = np.abs(dtm_points[:, 2] - np.median(dtm_points[:, 2][indexes], axis=1)) # abs(coord - median)
    mads = np.median(abs_devs)
    clean_points = dtm_points[abs_devs <= 2 * mads]

    return clean_points


las = laspy.read("/Users/romainjanvier/data/3DFin/plot_03_splits.las")
pointcloud = las.xyz
csf = CSF()
csf.set_point_cloud(pointcloud)
csf.params.rigidness = 1
csf.params.cloth_resolution = 0.1
csf.params.smooth_slope = True
cloth_points = csf.run_cloth_simulation()

clean_cloth_dtm = clean_cloth(cloth_points)

las_cloth = laspy.create(file_version="1.4", point_format=2)
las_cloth.xyz = cloth_points
las_cloth.write(
    "/Users/romainjanvier/data/3DFin/experimentations/base_cloth_3points.las")

clean_cloth = laspy.create(file_version="1.4", point_format=2)
clean_cloth.xyz = clean_cloth_dtm
las_cloth.write("/Users/romainjanvier/data/3DFin/experimentations/clean_cloth_3points.las")
