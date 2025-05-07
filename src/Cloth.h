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

/*
 * This source code is about a ground filtering algorithm for airborn LiDAR data
 * based on physical process simulations, specifically cloth simulation.
 *
 * this code is based on a Cloth Simulation Tutorial at the cg.alexandra.dk
 * blog. Thanks to Jesper Mosegaard (clothTutorial@jespermosegaard.dk)
 *
 *
 *
 * When applying the cloth simulation to LIDAR point filtering. A lot of
 * features have been added to the original source code, including configuration
 * file management point cloud data read/write point-to-point collsion detection
 * nearest point search structure from CGAL
 * addding a terrain class
 *
 *
 */
// using discrete steps (drop and pull) to approximate the physical process


#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif  // ifdef _WIN32

#include <vector>
#ifdef CSF_USE_OPENMP
#include <omp.h>
#endif
#include <cstdint>
#include <string>

#include "Particle.h"
#include "Vec3.h"

struct XY
{
    XY(uint32_t x1, uint32_t y1) : x(x1), y(y1) {}
    ~XY() = default;

    uint32_t x;
    uint32_t y;
};

class Cloth
{
   public:
    /* This is a important constructor for the entire system of
     * particles and constraints */
    Cloth(
        const Vec3& origin_pos, uint32_t num_particles_width, uint32_t num_particles_height, double step_x,
        double step_y, double smoothThreshold, double heightThreshold, uint32_t rigidness, double time_step);

    ~Cloth() = default;

    /* this is an important methods where the time is progressed one
     * time step for the entire cloth.  This includes calling
     * satisfyConstraint() for every constraint, and calling
     * timeStep() for all particles
     */
    double timeStep();

    void terrCollision();

    void movableFilter();

    std::vector<uint32_t> findUnmovablePoint(const std::vector<XY>& connected);

    void handle_slope_connected(
        const std::vector<uint32_t>& edgePoints, const std::vector<XY>& connected,
        const std::vector<std::vector<uint32_t>>& neighbors);

    void saveToFile(std::string path = "");

    void saveMovableToFile(std::string path = "");

    Particle& getParticle(uint32_t x, uint32_t y) { return particles[y * num_particles_width + x]; }

    const Particle& getParticle(uint32_t x, uint32_t y) const { return particles[y * num_particles_width + x]; }

    Particle& getParticle(uint32_t index) { return particles[index]; }

    const std::vector<Particle>& getParticles() const { return particles; }

    void makeConstraint(Particle* p1, Particle* p2)
    {
        p1->neighborsList.push_back(p2);
        p2->neighborsList.push_back(p1);
    }

    uint32_t getSize() const { return num_particles_width * num_particles_height; }

    std::pair<uint32_t, uint32_t> getGridSize() const
    {
        return std::make_pair(num_particles_width, num_particles_height);
    }

    std::vector<double>& getHeightvals() { return height_values; }

   public:
    const Vec3          origin_pos;
    const double        step_x, step_y;
    std::vector<double> height_values;  // height values

   private:
    // post processing is only for connected component which is large than 50
    static constexpr uint32_t max_particle_for_post_processing = 50;
    static constexpr double   gravity                          = 0.2;  // TODO: make it a parameter

    const uint32_t constraint_iterations;  // rigidness

    const uint32_t num_particles_width;  // number of particles in width direction
    const uint32_t num_particles_height;  // number of particles in height direction

    const double smoothThreshold;
    const double heightThreshold;

    std::vector<Particle> particles;  // all particles that are part of this cloth
};
