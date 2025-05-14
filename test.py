from CSF_3DFin import CSF
import laspy

las = laspy.read("/Users/romainjanvier/data/3DFin/plot_03_splits.las")
pointcloud = las.xyz
csf = CSF()
csf.set_point_cloud(pointcloud)
csf.params.rigidness = 1
csf.params.cloth_resolution = 0.1
csf.params.smooth_slope = True
cloth_points = csf.do_cloth()

las_cloth = laspy.create(file_version="1.4", point_format=2)
las_cloth.xyz = cloth_points
las_cloth.write(
    "/Users/romainjanvier/data/3DFin/experimentations/base_cloth_3points.las")
