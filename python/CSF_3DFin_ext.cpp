#include <CSF.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>

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
                auto & csf_pc = csf.getPointCloud();
                csf_pc.clear();
                csf_pc.resize(point_cloud.shape(0));
                auto v = point_cloud.view();

                for (size_t i = 0; i < v.shape(0); ++i)
                {
                    csf_pc[i] = {v(i, 0), -v(i, 2), v(i, 1)};
                }
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
