![csf1](https://github.com/jianboqi/CSF/blob/master/CSFDemo/CSF1.png) ![csf2](https://github.com/jianboqi/CSF/blob/master/CSFDemo/CSF2.png)
# CSF
Airborne LiDAR filtering method based on Cloth Simulation.
This is the code for the article:

W. Zhang, J. Qi*, P. Wan, H. Wang, D. Xie, X. Wang, and G. Yan, “An Easy-to-Use Airborne LiDAR Data Filtering Method Based on Cloth Simulation,” Remote Sens., vol. 8, no. 6, p. 501, 2016.
(http://www.mdpi.com/2072-4292/8/6/501/htm)


**This is a modified version of the original Code for 3DFin purposes**

For now this is only intended to be used inside [3DFin](https://github.com/3DFin/3Dfin)/[dendromatics](https://github.com/3DFin/dendromatics).
But this repository also serve as a playground to bootstrap a planned full rewrite of CSF.

List of changes:

- Better handling of numpy arrays in the python bindings (avoid expensive copies)
- Port to nanobind (instead of SWIG)
- Bug fixes and improvement (mostly backport from CloudCompare version of CSF and original ones)
- Better (but yet to be improved) documentation and code structure.
- Removed ascii handling in favor of pure numpy interface

### How to use CSF in Python
```python
pip install csf-3dfin
```

#### Python examples

```python
import laspy
from CSF_3DFin import CSF
import numpy as np

las_file = laspy.read(r"sample.las") # read a las file
csf = CSF()

# prameter settings
csf.params.smooth_slope = False
csf.params.cloth_resolution = 0.5
# more details about parameter: http://ramm.bnu.edu.cn/projects/CSF/download/

csf.set_point_cloud(las_file.xyz)
ground, _ = csf.do_filtering() # do actual filtering.

out_file = laspy.LasData(las_file.header)
out_file.points = las_file.points[np.array(ground)] # extract ground points, and save it to a las file.
out_file.write(r"out.las")
```

### How to use CSF in C++
Now, CSF is built by CMake, it produces a static library, which can be used by other c++ programs.
#### linux
To build the library, run:
```bash
mkdir build #or other name
cd build
cmake ..
make
sudo make install
```
or if you want to build the library and the demo executable `csfdemo`

```bash
mkdir build #or other name
cd build
cmake -DBUILD_DEMO=ON ..
make
sudo make install
```

### License
CSF is maintained and developed by Jianbo QI. It is now released under Apache 2.0.
