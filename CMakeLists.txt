cmake_minimum_required(VERSION 3.16)

project(evodev LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(PkgConfig REQUIRED)
pkg_check_modules(GTK3 REQUIRED IMPORTED_TARGET gtk+-3.0)

# Builds the plotato library
# add_subdirectory(Plotato)

add_executable(evodev_test src/gui/test.cpp)

# target_link_libraries(evodev_test
#     PRIVATE
#         plotato
# )

target_link_libraries(evodev_test PRIVATE PkgConfig::GTK3)
