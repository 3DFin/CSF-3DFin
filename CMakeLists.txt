cmake_minimum_required(VERSION 3.15...3.27)
project(CSF LANGUAGES CXX)

option(COMPILE_PYTHON_MODULE "Compile Python module" OFF)
option(BUILD_DEMO "Build csfdemo executable" OFF)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    add_compile_definitions("CSF_USE_OPENMP")
endif()

add_subdirectory(src)

if(COMPILE_PYTHON_MODULE)
if (NOT SKBUILD)
  message(ERROR "Python module is meant to be compiled via sk-build-core")
endif()

if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.8 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(
  # Name of the extension
  CSF_3DFin_ext
  # Target the stable ABI for Python 3.12+, which reduces
  # the number of binary wheels that must be built. This
  # does nothing on older Python versions
  STABLE_ABI
  NB_STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/python/CSF_3DFin_ext.cpp
)

# Stub generation
nanobind_add_stub(
  CSF_3DFin_ext_stub
  MODULE CSF_3DFin_ext
  OUTPUT CSF_3DFin_ext.pyi
  PYTHON_PATH $<TARGET_FILE_DIR:CSF_3DFin_ext>
  DEPENDS CSF_3DFin_ext
)

target_link_libraries(CSF_3DFin_ext PRIVATE CSF)
install(TARGETS CSF_3DFin_ext DESTINATION CSF_3DFin)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CSF_3DFin_ext.pyi DESTINATION CSF_3DFin)
endif()

if(BUILD_DEMO)
  add_subdirectory(CSFDemo)
endif()
