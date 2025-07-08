cmake_minimum_required(VERSION 3.16)
project(LibTorchTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Setze vcpkg toolchain
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif()

# Finde Torch
find_package(Torch REQUIRED)

# Erstelle executable
add_executable(LibTorchTest main.cpp)

# Link libraries
target_link_libraries(LibTorchTest ${TORCH_LIBRARIES})

# Compiler flags
target_compile_features(LibTorchTest PRIVATE cxx_std_17)

# Windows spezifisch
if(MSVC)
    target_compile_options(LibTorchTest PRIVATE /bigobj)
endif()