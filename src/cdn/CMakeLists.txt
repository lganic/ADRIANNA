cmake_minimum_required(VERSION 3.16)
project(codeepneat VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(CDN_BUILD_EXAMPLES "Build CoDeepNEAT examples" ON)
option(CDN_BUILD_TESTS "Build CoDeepNEAT tests" ON)

find_package(Threads REQUIRED)

add_library(codeepneat
  src/tensor.cpp
  src/hyperparams.cpp
  src/spaces.cpp
  src/graph.cpp
  src/layers.cpp
  src/network.cpp
  src/optim.cpp
  src/genome.cpp
  src/population.cpp
  src/codeepneat.cpp
  src/rl.cpp
  src/export.cpp
)
target_include_directories(codeepneat PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(codeepneat PUBLIC Threads::Threads)
target_compile_features(codeepneat PUBLIC cxx_std_17)

if(MSVC)
  target_compile_options(codeepneat PRIVATE /W4 /permissive-)
else()
  target_compile_options(codeepneat PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(CDN_BUILD_EXAMPLES)
  foreach(ex cartpole_codeepneat memory_lstm catch_conv inspection_demo deepneat_supervised export_pytorch)
    add_executable(${ex} examples/${ex}.cpp)
    target_link_libraries(${ex} PRIVATE codeepneat)
  endforeach()
endif()

if(CDN_BUILD_TESTS)
  enable_testing()
  add_executable(cdn_tests
    tests/test_main.cpp
    tests/test_tensor.cpp
    tests/test_network.cpp
    tests/test_genome.cpp
    tests/test_evolution.cpp
    tests/test_export.cpp)
  target_link_libraries(cdn_tests PRIVATE codeepneat)
  add_test(NAME cdn_tests COMMAND cdn_tests)

  # End-to-end check of the PyTorch export (runs only if Python with torch is found).
  option(CDN_TEST_PYTORCH "Verify PyTorch export when Python + torch are available" ON)
  if(CDN_TEST_PYTORCH AND CDN_BUILD_EXAMPLES)
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
      execute_process(COMMAND ${Python3_EXECUTABLE} -c "import torch"
                      RESULT_VARIABLE CDN_TORCH_RC OUTPUT_QUIET ERROR_QUIET)
      if(CDN_TORCH_RC EQUAL 0)
        set(CDN_EXPORT_DIR ${CMAKE_CURRENT_BINARY_DIR}/torch_export)
        add_test(NAME torch_export_generate COMMAND export_pytorch ${CDN_EXPORT_DIR})
        add_test(NAME torch_export_verify
                 COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/verify_torch_export.py
                         --dir ${CDN_EXPORT_DIR} --write-back)
        add_test(NAME torch_export_import COMMAND export_pytorch ${CDN_EXPORT_DIR} --check-import)
        set_tests_properties(torch_export_generate PROPERTIES FIXTURES_SETUP cdn_torch_files)
        set_tests_properties(torch_export_verify PROPERTIES FIXTURES_REQUIRED cdn_torch_files
                                                            FIXTURES_SETUP cdn_torch_trained)
        set_tests_properties(torch_export_import PROPERTIES FIXTURES_REQUIRED cdn_torch_trained)
        message(STATUS "PyTorch export tests enabled (${Python3_EXECUTABLE})")
      else()
        message(STATUS "PyTorch export tests disabled: 'import torch' failed for ${Python3_EXECUTABLE}")
      endif()
    endif()
  endif()
endif()

install(TARGETS codeepneat EXPORT codeepneatTargets ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
