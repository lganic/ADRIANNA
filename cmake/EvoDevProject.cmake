# Compile the GUI for a project. Every project calls this from its own CMakeLists.txt:
#
#   evodev_add_project(my_project
#       TITLE "Goes on the window"
#       SEED 125
#       SOURCES my_project.cpp physics.cpp
#       LIBRARIES box2d
#   )
#
# That makes two targets:
#   my_project      The executable, which ends up at build/my_project.exe
#   my_project_sim  Just the simulation, with no GUI in it.
function(evodev_add_project name)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" "TITLE;SEED" "SOURCES;LIBRARIES")

    if(NOT ARG_SOURCES)
        message(FATAL_ERROR "evodev_add_project(${name}) needs some SOURCES")
    endif()

    if(NOT ARG_TITLE)
        set(ARG_TITLE ${name})
    endif()

    if(NOT ARG_SEED)
        set(ARG_SEED 1)
    endif()

    add_library(${name}_sim STATIC ${ARG_SOURCES})

    target_link_libraries(${name}_sim PUBLIC evosim ${ARG_LIBRARIES})

    if(MSVC)
        target_compile_options(${name}_sim PRIVATE /W4 /permissive-)
    else()
        # Always optimized, even in Debug builds. See the note on codeepneat in the top level CMakeLists.
        target_compile_options(${name}_sim PRIVATE -Wall -Wextra -Wpedantic -O2)
    endif()

    add_executable(${name} ${CMAKE_SOURCE_DIR}/src/gui/src/main.cpp)

    target_compile_definitions(${name} PRIVATE EVODEV_TITLE="${ARG_TITLE}" EVODEV_SEED=${ARG_SEED})

    target_link_libraries(${name} PRIVATE evodev_gui ${name}_sim)

    set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
endfunction()
