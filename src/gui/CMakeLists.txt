add_library(gui_utils STATIC
    src/util/widget_location.cpp
)

target_include_directories(gui_utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GTK3 REQUIRED IMPORTED_TARGET gtk+-3.0)

target_link_libraries(gui_utils PUBLIC PkgConfig::GTK3)

# The GUI itself. It doesn't know anything about any one project, it builds itself out of whatever
# the project asked for. The project gets linked in by evodev_add_project.
add_library(evodev_gui STATIC
    src/app.cpp
)

target_link_libraries(evodev_gui PUBLIC gui_utils plotato evosim)

if(MSVC)
    target_compile_options(gui_utils PRIVATE /W4 /permissive-)
    target_compile_options(evodev_gui PRIVATE /W4 /permissive-)
else()
    target_compile_options(gui_utils PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(evodev_gui PRIVATE -Wall -Wextra -Wpedantic -Wno-reorder) # Reorder is for the Plotato headers.
endif()
