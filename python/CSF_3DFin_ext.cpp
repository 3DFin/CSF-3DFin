#include <CSF.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>

#include <cstddef>

namespace nb = nanobind;

using namespace nb::literals;

NB_MODULE(CSF_3DFin_ext, m)
{
    m.doc() = "Improved binding for CSF";

    nb::class_<Params>(m, "CSFParams")
        .def(nb::init<>())
        .def_rw("smooth_slope", &Params::smooth_slope)
        .def_rw("time_step", &Params::time_step)
        .def_rw("class_threshold", &Params::class_threshold)
        .def_rw("cloth_resolution", &Params::cloth_resolution)
        .def_rw("rigidness", &Params::rigidness)
        .def_rw("iterations", &Params::iterations)
        .def_rw("iter_tolerance", &Params::iter_tolerance)
        .def_rw("verbose", &Params::verbose);

    nb::class_<CSF>(m, "CSF")
        .def(nb::init<>())
        .def(
            "set_point_cloud",
            [](CSF& csf, nb::ndarray<double, nb::numpy, nb::shape<-1, 3>> point_cloud)
            {
                auto& csf_pc = csf.getPointCloud();
                csf_pc.clear();
                csf_pc.resize(point_cloud.shape(0));
                auto v = point_cloud.view();

                for (size_t i = 0; i < v.shape(0); ++i) { csf_pc[i] = {v(i, 0), -v(i, 2), v(i, 1)}; }
            },
            "point_cloud"_a.noconvert())
        .def(
            "run_cloth_simulation",
            [](CSF& csf)
            {
                auto cloth = csf.runClothSimulation();

                const auto& particles      = cloth.getParticles();
                size_t      num_particles  = particles.size();
                size_t      size_arr       = num_particles * 3;
                double*     raw_cloth_data = new double[size_arr];

                for (size_t particle_id = 0; particle_id < num_particles; ++particle_id)
                {
                    size_t id              = particle_id * 3;
                    raw_cloth_data[id]     = particles[particle_id].initial_pos.f[0];
                    raw_cloth_data[id + 1] = particles[particle_id].initial_pos.f[2];
                    raw_cloth_data[id + 2] = -particles[particle_id].height;
                }

                nb::capsule capsule(raw_cloth_data, [](void* p) noexcept { delete[] (double*)p; });
                return nb::ndarray<double, nb::numpy, nb::shape<-1, 3>>(raw_cloth_data, {num_particles, 3}, capsule);
            })
        .def(
            "cloth_with_mesh",
            [](CSF& csf)
            {
                struct ReturnValues
                {
                    std::vector<double> particles_coordinates;
                    std::vector<int>    tri_indices;
                };
                ReturnValues* result = new ReturnValues();

                nb::capsule capsule(result, [](void* p) noexcept { delete (ReturnValues*)p; });

                auto cloth = csf.runClothSimulation();

                const auto& particles     = cloth.getParticles();
                size_t      num_particles = particles.size();
                size_t      size_arr      = num_particles * 3;

                int width, height;
                std::tie(width, height) = cloth.getGridSize();

                // Step 1: Build 2D height map from particles
                std::vector<std::vector<double>> height_map(height, std::vector<double>(width));

                for (int y = 0; y < height; ++y)
                {
                    for (int x = 0; x < width; ++x)
                    {
                        int idx          = y * width + x;
                        height_map[y][x] = -particles[idx].height;
                    }
                }

                // Step 2: Apply 3x3 median filter to inner cells
                std::vector<std::vector<double>> filtered_map = height_map;

                for (int y = 1; y < height - 1; ++y)
                {
                    for (int x = 1; x < width - 1; ++x)
                    {
                        std::vector<double> window;
                        for (int dy = -1; dy <= 1; ++dy)
                        {
                            for (int dx = -1; dx <= 1; ++dx) { window.push_back(height_map[y + dy][x + dx]); }
                        }
                        std::sort(window.begin(), window.end());
                        filtered_map[y][x] = window[4];  // median of 9 values
                    }
                }

                result->particles_coordinates.resize(size_arr, 0);

                for (size_t particle_id = 0; particle_id < num_particles; ++particle_id)
                {
                    size_t id                             = particle_id * 3;
                    result->particles_coordinates[id]     = particles[particle_id].initial_pos.f[0];
                    result->particles_coordinates[id + 1] = particles[particle_id].initial_pos.f[2];
                    result->particles_coordinates[id + 2] = filtered_map[particle_id / width][ particle_id % width];
                }


                nb::print(nb::str("{} x {}").format(width, height));

                size_t num_triangles = (width - 1) * 2 * (height - 1) * 2;
                result->tri_indices.resize(num_triangles * 3, 0);

                // mesh export code taken from CC.
                // A---D
                // | / |
                // B---C
                for (int x = 0; x < width - 1; ++x)
                {
                    for (int y = 0; y < height - 1; ++y)
                    {
                        int A = y * width + x;
                        int D = iA + width;
                        int B = iA + 1;
                        int C = iD + 1;

                        size_t base_id                   = 6 * x * width + 6 * y;
                        result->tri_indices[base_id]     = A;
                        result->tri_indices[base_id + 1] = B;
                        result->tri_indices[base_id + 2] = D;
                        result->tri_indices[base_id + 3] = D;
                        result->tri_indices[base_id + 4] = B;
                        result->tri_indices[base_id + 5] = C;
                    }
                }

                return std::make_pair(
                    nb::ndarray<double, nb::numpy, nb::shape<-1, 3>>(
                        result->particles_coordinates.data(), {num_particles, 3}, capsule),
                    nb::ndarray<int, nb::numpy, nb::shape<-1, 3>>(
                        result->tri_indices.data(), {num_triangles, 3}, capsule));
            })
        .def(
            "classify_ground",
            [](CSF& csf)
            {
                struct ReturnValues
                {
                    std::vector<int> ground_indices;
                    std::vector<int> off_ground_indices;
                };

                ReturnValues* result = new ReturnValues();

                nb::capsule capsule(result, [](void* p) noexcept { delete (ReturnValues*)p; });

                csf.classifyGround(result->ground_indices, result->off_ground_indices, false);

                size_t size_ground = result->ground_indices.size();
                size_t size_off    = result->off_ground_indices.size();

                return std::make_pair(
                    nb::ndarray<nb::numpy, int>(result->ground_indices.data(), {size_ground}, capsule),
                    nb::ndarray<nb::numpy, int>(result->off_ground_indices.data(), {size_off}, capsule));
            })
        .def_rw("params", &CSF::params);
}
