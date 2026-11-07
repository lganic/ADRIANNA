add_library(gui_utils STATIC
    src/util/widget_location.cpp
)

target_include_directories(gui_utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GTK3 REQUIRED IMPORTED_TARGET gtk+-3.0)

target_link_libraries(gui_utils PUBLIC PkgConfig::GTK3)

if(MSVC)
    target_compile_options(gui_utils PRIVATE /W4 /permissive-)
else()
    target_compile_options(gui_utils PRIVATE -Wall -Wextra -Wpedantic)
endif()