#include <gui/app.hpp>

// This gets compiled into each project's executable. The title and seed come from evodev_add_project.
int main(int argc, char* argv[])
{
    return gui::run(argc, argv, EVODEV_TITLE, EVODEV_SEED);
}
