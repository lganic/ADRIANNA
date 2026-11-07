add_library(evosim STATIC
    src/physics.cpp
    src/models.cpp
    src/world.cpp
    src/sim_thread.cpp
)

target_include_directories(evosim PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
)

# No GTK in here on purpose. The simulation only ever talks to the GUI through plain data.
target_link_libraries(evosim PUBLIC codeepneat box2d)

if(MSVC)
    target_compile_options(evosim PRIVATE /W4 /permissive-)
else()
    # Always optimized, even in Debug builds. See the note on codeepneat in the top level CMakeLists.
    target_compile_options(evosim PRIVATE -Wall -Wextra -Wpedantic -O2)
endif()
