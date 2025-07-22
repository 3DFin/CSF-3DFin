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


add_library(CSF STATIC ${CSF_SOURCES} ${CSF_HEADERS} ${CSF_API_HEADERS})
add_library(CSF::CSF ALIAS CSF)

set_property(TARGET CSF PROPERTY CXX_STANDARD 17)
set_property(TARGET CSF PROPERTY POSITION_INDEPENDENT_CODE ON)


# Handle // computation libraries
find_package(taskflow QUIET)
if(taskflow_FOUND)
    message(STATUS "CSF: Taskflow found")
    target_compile_definitions(CSF PRIVATE CSF_USE_TASKFLOW)
    target_link_libraries(CSF PRIVATE Taskflow::Taskflow)
    # TODO add thread lib as public link library
else()
    # Else fallback to OpenMP
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        message(STATUS "CSF: OpenMP found")
        target_compile_definitions(CSF PUBLIC CSF_USE_OPENMP)
        target_link_libraries(CSF PUBLIC OpenMP::OpenMP_CXX)
    else()
        message(STATUS "CSF: configured as single-threaded")
    endif()
endif()

target_include_directories(CSF PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include/>
    $<INSTALL_INTERFACE:include/csf>)


if(NOT COMPILE_PYTHON_MODULE)
    include(CMakePackageConfigHelpers)

    # Install the target and export it
    install(TARGETS CSF
        EXPORT CSFTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include/csf
    )

    # Install headers
    install(FILES ${CSF_HEADERS} DESTINATION include/csf)

    # Export the target to make it available via find_package
    export(EXPORT CSFTargets
        FILE "${CMAKE_CURRENT_BINARY_DIR}/CSFTargets.cmake"
        NAMESPACE CSF::
    )

    install(EXPORT CSFTargets
        FILE CSFTargets.cmake
        NAMESPACE CSF::
        DESTINATION lib/cmake/CSF
    )

    # Create a build-tree config file
    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/CSFConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/CSFConfig.cmake"
        INSTALL_DESTINATION lib/cmake/CSF
        PATH_VARS CMAKE_CURRENT_SOURCE_DIR
    )

    # Generate CSFConfigVersion.cmake
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/CSFConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY AnyNewerVersion
    )

    # Install the config files
    install(
        FILES
            "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/CSFConfig.cmake.in"
            "${CMAKE_CURRENT_BINARY_DIR}/CSFConfigVersion.cmake"
        DESTINATION lib/cmake/CSF
    )
endif()
