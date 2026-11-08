# The framework that projects hook in to. See projects/README.md
add_library(evosim STATIC
    src/canvas.cpp
    src/models.cpp
    src/sim_thread.cpp
    src/ui.cpp
)

target_include_directories(evosim PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
)

# Plotato is here because projects draw with it. The simulation thread still never makes a GTK call.
target_link_libraries(evosim PUBLIC codeepneat plotato PkgConfig::GTK3)

if(MSVC)
    target_compile_options(evosim PRIVATE /W4 /permissive-)
else()
    # Always optimized, even in Debug builds. See the note on codeepneat in the top level CMakeLists.
    target_compile_options(evosim PRIVATE -Wall -Wextra -Wpedantic -Wno-reorder -O2) # Reorder is for the Plotato headers.
endif()
