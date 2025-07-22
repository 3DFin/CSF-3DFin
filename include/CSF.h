#pragma once
// ======================================================================================
// Copyright 2017 State Key Laboratory of Remote Sensing Science,
// Institute of Remote Sensing Science and Engineering, Beijing Normal
// University

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ======================================================================================

// #######################################################################################
// # # #            CSF: Airborne LiDAR filtering based on Cloth Simulation # #
// # #  Please cite the following paper, If you use this software in your work.
// # # # #  Zhang W, Qi J, Wan P, Wang H, Xie D, Wang X, Yan G. An Easy-to-Use
// Airborne LiDAR  # #  Data Filtering Method Based on Cloth Simulation. Remote
// Sensing. 2016; 8(6):501.   # # (http://ramm.bnu.edu.cn/) # # # # Wuming
// Zhang; Jianbo Qi; Peng Wan; Hongtao Wang                # # # # contact us:
// 2009zwm@gmail.com; wpqjbzwm@126.com                # # #
// #######################################################################################

// cloth simulation filter for airborne lidar filtering

#include <string>
#include <vector>

#include "Cloth.h"
#include "PointCloud.h"

#ifdef _CSF_DLL_EXPORT_
#ifdef DLL_IMPLEMENT
#define DLL_API __declspec(dllexport)
#else  // ifdef DLL_IMPLEMENT
#define DLL_API __declspec(dllimport)
#endif  // ifdef DLL_IMPLEMENT
#endif  // ifdef _CSF_DLL_EXPORT_

#ifdef _CSF_DLL_EXPORT_
class DLL_API Params
#else
struct Params
#endif
{
    Params()  = default;
    ~Params() = default;

    // refer to the website:http://ramm.bnu.edu.cn/projects/CSF/ for th`e setting
    // of these paramters
    double   time_step{0.65};
    double   class_threshold{0.5};
    double   cloth_resolution{1.0};
    uint32_t rigidness{3};
    uint32_t iterations{500};
    double   iter_tolerance{0.005};
    bool     smooth_slope{true};
    bool     verbose{true};
};

#ifdef _CSF_DLL_EXPORT_
class DLL_API CSF
#else  // ifdef _CSF_DLL_EXPORT_
class CSF
#endif  // ifdef _CSF_DLL_EXPORT_
{
   public:
    CSF()  = default;
    ~CSF() = default;

    // PointCloud set pointcloud
    void setPointCloud(const csf::PointCloud& pc);

    // read pointcloud from txt file: (X Y Z) for each line
    void readPointsFromFile(const std::string& filename);

    csf::PointCloud& getPointCloud() { return point_cloud; }

    const csf::PointCloud& getPointCloud() const { return point_cloud; }

    // save points to file
    // The results are index of ground points in the original point cloud
    void savePoints(const std::vector<int>& grp, const std::string& path) const;

    // pointcloud and write the cloth particles coordinates
    void classifyGround(
        std::vector<int>& groundIndexes, std::vector<int>& offGroundIndexes, const bool exportCloth = true);

    // Do the filtering and return the Cloth object
    Cloth runClothSimulation();

   public:
    Params params;

   private:
    csf::PointCloud point_cloud;
};
