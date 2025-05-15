set(CSF_SOURCES
    c2cdist.cpp
    Cloth.cpp
    Constraint.cpp
    CSF.cpp
    Particle.cpp
    PointCloud.cpp
    Rasterization.cpp
    XYZReader.cpp
)

set(CSF_HEADERS
    c2cdist.h
    Cloth.h
    Constraint.h
    CSF.h
    Particle.h
    PointCloud.h
    Rasterization.h
    Vec3.h
    XYZReader.h
)

add_library(CSF STATIC ${CSF_SOURCES} ${CSF_HEADERS})
target_include_directories(CSF PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/csf>)

set_property(TARGET CSF PROPERTY CXX_STANDARD 17)
set_property(TARGET CSF PROPERTY POSITION_INDEPENDENT_CODE ON)

if(OpenMP_CXX_FOUND)
    target_link_libraries(CSF PUBLIC OpenMP::OpenMP_CXX)
endif()

if(NOT COMPILE_PYTHON_MODULE)
    install(TARGETS CSF LIBRARY DESTINATION csf)
    install(FILES ${CSF_HEADERS} DESTINATION include/csf)
endif()
