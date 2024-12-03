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
        .def_rw("smooth_slope", &Params::bSloopSmooth)
        .def_rw("time_step", &Params::time_step)
        .def_rw("class_threshold", &Params::class_threshold)
        .def_rw("cloth_resolution", &Params::cloth_resolution)
        .def_rw("rigidness", &Params::rigidness)
        .def_rw("iterations", &Params::interations);

    nb::class_<CSF>(m, "CSF")
        .def(nb::init<>())
        .def(
            "set_point_cloud", [](CSF& csf, nb::ndarray<double, nb::numpy, nb::shape<-1, 3>> point_cloud)
            { csf.setPointCloud(point_cloud.data(), point_cloud.shape(0)); }, "point_cloud"_a.noconvert())
        .def(
            "do_cloth",
            [](CSF& csf, bool verbose)
            {
                std::streambuf* old_buffer = nullptr;
                if (!verbose) old_buffer = std::cout.rdbuf(nullptr);

                auto cloth = csf.do_cloth();

                if (!verbose) std::cout.rdbuf(old_buffer);

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
            },
            "verbose"_a = true)
        .def(
            "do_filtering",
            [](CSF& csf, bool verbose)
            {
                struct ReturnValues
                {
                    std::vector<int> ground_indices;
                    std::vector<int> off_ground_indices;
                };

                ReturnValues* result = new ReturnValues();

                nb::capsule capsule(result, [](void* p) noexcept { delete (ReturnValues*)p; });
                

                std::streambuf* old_buffer = nullptr;
                if (!verbose) old_buffer = std::cout.rdbuf(nullptr);

                csf.do_filtering(result->ground_indices, result->off_ground_indices, false);
               
                if (!verbose) std::cout.rdbuf(old_buffer);
                
                size_t size_ground = result->ground_indices.size();
                size_t size_off    = result->off_ground_indices.size();

                return std::make_pair(
                    nb::ndarray<nb::numpy, int>(result->ground_indices.data(), {size_ground}, capsule),
                    nb::ndarray<nb::numpy, int>(result->off_ground_indices.data(), {size_off}, capsule));
            }, "verbose"_a = true)
        .def_rw("params", &CSF::params);
}