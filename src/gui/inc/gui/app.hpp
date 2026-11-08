#ifndef GUI_APP_HPP
#define GUI_APP_HPP

#include <cstdint>

namespace gui {

// Open the GUI on whichever project this got compiled with, and run until it closes.
int run(int argc, char* argv[], const char* title, uint64_t seed);

}

#endif // GUI_APP_HPP
