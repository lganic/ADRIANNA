cmake --preset msys2-ucrt64 --fresh
cmake --build --preset msys2-ucrt64

call "build/evodev_test.exe"