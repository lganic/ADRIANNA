cmake_minimum_required(VERSION 3.16)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(evodev LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(PkgConfig REQUIRED)
pkg_check_modules(GTK3 REQUIRED IMPORTED_TARGET gtk+-3.0)

# Builds the plotato library
add_subdirectory(src/Plotato)
add_subdirectory(src/gui)

# Builds the CDN library. Only the library itself is needed here, not its examples and tests.
set(CDN_BUILD_EXAMPLES OFF)
set(CDN_BUILD_TESTS OFF)
add_subdirectory(src/cdn)

# This is the hot loop of fast forward, which runs about 6x faster optimized. So it stays optimized
# even in Debug builds, while the GUI code keeps the normal debug settings.
if(NOT MSVC)
    target_compile_options(codeepneat PRIVATE -O3 -march=native -fno-math-errno -fno-signed-zeros -fno-trapping-math -fassociative-math)
endif()

# Builds Box2D. Same deal as CDN above, since every physics step of fast forward goes through it.
add_subdirectory(src/box2d)

if(NOT MSVC)
    target_compile_options(box2d PRIVATE -O2)
endif()

add_subdirectory(src/sim)

# Every folder in projects/ gets the GUI compiled for it, as its own executable. See projects/README.md
include(cmake/EvoDevProject.cmake)

set(EVODEV_PROJECTS "" CACHE STRING "The projects to compile, separated by semicolons. Empty means all of them.")

if(NOT EVODEV_PROJECTS)
    file(GLOB project_files CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/projects/*/CMakeLists.txt)

    foreach(project_file ${project_files})
        get_filename_component(project_dir ${project_file} DIRECTORY)
        get_filename_component(project_name ${project_dir} NAME)

        list(APPEND EVODEV_PROJECTS ${project_name})
    endforeach()
endif()

foreach(project_name ${EVODEV_PROJECTS})
    add_subdirectory(projects/${project_name})
endforeach()
