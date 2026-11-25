evodev_add_project(food_web
    TITLE "Food web: grass, herds, and packs on four continents"
    SEED 125
    SOURCES
        food_web.cpp
        world.cpp
        terrain.cpp
        brain.cpp
        pool_threads.cpp
)

# The brains are plain loops over floats, so let the compiler vectorise them.
if(NOT MSVC)
    target_compile_options(food_web_sim PRIVATE -O3 -march=native -fno-math-errno)
endif()

# Runs the simulation with no window, for experiments. See headless.cpp.
add_executable(food_web_headless headless.cpp)
target_link_libraries(food_web_headless PRIVATE food_web_sim)
set_target_properties(food_web_headless PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
