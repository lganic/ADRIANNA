@echo off

rem Builds and runs a project from projects/, like: dumb.bat find_food
set PROJECT=%1
if "%PROJECT%"=="" set PROJECT=chase_target

cmake --preset msys2-ucrt64 --fresh
cmake --build --preset msys2-ucrt64 --target %PROJECT%

call "build/%PROJECT%.exe"
