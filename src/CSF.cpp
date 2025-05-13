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
#include "CSF.h"

#include <chrono>
#include <fstream>
#include <iomanip>

#include "Rasterization.h"
#include "Vec3.h"
#include "XYZReader.h"
#include "c2cdist.h"

void CSF::setPointCloud(const csf::PointCloud& pc) { point_cloud = pc; }

void CSF::readPointsFromFile(const std::string& filename)
{
    point_cloud.resize(0);
    read_xyz(filename, point_cloud);
}

Cloth CSF::runClothSimulation()
{
    std::streambuf* old_buffer = nullptr;
    if (!params.verbose) old_buffer = std::cout.rdbuf(nullptr);

    // Terrain
    std::cout << "Configuring terrain..." << std::endl;
    csf::Point bbMin, bbMax;
    point_cloud.computeBoundingBox(bbMin, bbMax);
    std::cout << " - bbMin: " << bbMin.x << " " << bbMin.y << " " << bbMin.z << std::endl;
    std::cout << " - bbMax: " << bbMax.x << " " << bbMax.y << " " << bbMax.z << std::endl;

    const double   cloth_y_height = 0.05;
    const uint32_t cloth_margin  = 2;

    // origin is shifted by clothbuffer_d * params.cloth_resolution
    const Vec3 origin_pos(
        bbMin.x - cloth_margin * params.cloth_resolution, bbMax.y + cloth_y_height,
        bbMin.z - cloth_margin * params.cloth_resolution);

    const uint32_t width_num =
        static_cast<uint32_t>(std::floor((bbMax.x - bbMin.x) / params.cloth_resolution)) + 2 * cloth_margin;
    const uint32_t height_num =
        static_cast<uint32_t>(std::floor((bbMax.z - bbMin.z) / params.cloth_resolution)) + 2 * cloth_margin;

    std::cout << "Configuring cloth..." << std::endl;
    std::cout << " - width: " << width_num << " "
              << "height: " << height_num << std::endl;

    Cloth cloth(
        origin_pos, width_num, height_num, params.cloth_resolution, params.cloth_resolution, 0.3, 9999,
        params.rigidness, params.time_step);

    auto start_raster = std::chrono::system_clock::now();
    std::cout << "Rasterizing..." << std::endl;
    Rasterization::Rasterize(cloth, point_cloud, cloth.getHeightvals());
    auto stop_raster    = std::chrono::system_clock::now();
    auto elapsed_raster = std::chrono::duration_cast<std::chrono::milliseconds>(stop_raster - start_raster);
    std::cout << "-> time raster " << elapsed_raster.count() << " ms" << std::endl;

    auto start_simul = std::chrono::system_clock::now();
    std::cout << "Simulating..." << std::endl;
    for (uint32_t i = 0; i < params.iterations; i++)
    {
        auto         start_timestep   = std::chrono::system_clock::now();
        const double max_diff         = cloth.timeStep();
        auto         stop_timestep    = std::chrono::system_clock::now();
        auto         elapsed_timestep = std::chrono::system_clock::now();
        cloth.terrCollision();
        if (max_diff != 0.0 && max_diff < params.iter_tolerance)
        {
            std::cout << " - early stop at iteration " << i << " (max_diff: " << max_diff << ")" << std::endl;
            break;
        }
    }


    auto stop_simul    = std::chrono::system_clock::now();
    auto elapsed_simul = std::chrono::duration_cast<std::chrono::milliseconds>(stop_simul - start_simul);
    std::cout << "-> time cloth simulation " << elapsed_simul.count() << " ms" << std::endl;

    if (params.smooth_slope)
    {
        auto start_slope = std::chrono::system_clock::now();
        std::cout << "Slope post processing..." << std::endl;
        cloth.movableFilter();
        auto stop_slope    = std::chrono::system_clock::now();
        auto elapsed_slope = std::chrono::duration_cast<std::chrono::milliseconds>(stop_slope - start_slope);
        std::cout << "-> time slope " << elapsed_slope.count() << " ms" << std::endl;
    }

    if (!params.verbose) std::cout.rdbuf(old_buffer);

    return cloth;
}

void CSF::classifyGround(std::vector<int>& groundIndexes, std::vector<int>& offGroundIndexes, const bool exportCloth)
{
    auto cloth = runClothSimulation();
    if (exportCloth) cloth.saveToFile();
    c2cdist c2c(params.class_threshold);
    c2c.calCloud2CloudDist(cloth, point_cloud, groundIndexes, offGroundIndexes);
}

void CSF::savePoints(const std::vector<int>& grp, const std::string& path) const
{
    if (path == "") { return; }

    std::ofstream f1(path.c_str(), std::ios::out);

    if (!f1) return;

    for (size_t i = 0; i < grp.size(); i++)
    {
        f1 << std::fixed << std::setprecision(8) << point_cloud[grp[i]].x << "	" << point_cloud[grp[i]].z << "	"
           << -point_cloud[grp[i]].y << std::endl;
    }

    f1.close();
}
