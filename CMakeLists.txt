cmake_minimum_required(VERSION 3.16)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(evodev LANGUAGES CXX)

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
    target_compile_options(codeepneat PRIVATE -O2)
endif()

add_subdirectory(src/sim)

add_executable(evodev_test src/gui/test.cpp)

target_link_libraries(evodev_test
    PRIVATE
        plotato
        gui_utils
        evosim
)

target_link_libraries(evodev_test PRIVATE PkgConfig::GTK3)
