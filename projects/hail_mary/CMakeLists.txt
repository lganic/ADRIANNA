evodev_add_project(hail_mary
    TITLE "Hail Mary: packs, herds, and calls"
    SEED 125
    SOURCES
        hail_mary.cpp
        continent.cpp
        brain.cpp
        pool_threads.cpp
)

# The brains are plain loops over floats, so let the compiler vectorise them.
if(NOT MSVC)
    target_compile_options(hail_mary_sim PRIVATE -O3 -march=native -fno-math-errno)
endif()

# Runs the simulation with no window, for experiments. See headless.cpp.
add_executable(hail_mary_headless headless.cpp)
target_link_libraries(hail_mary_headless PRIVATE hail_mary_sim)
set_target_properties(hail_mary_headless PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
