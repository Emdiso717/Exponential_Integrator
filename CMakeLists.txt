project(arborescence)
cmake_minimum_required(VERSION 3.5)

include(cmake/CPM.cmake)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/;${CMAKE_MODULE_PATH}")


#O3 optimization
message("cmake build type: ${CMAKE_BUILD_TYPE}")
if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 ")
endif()
#enable all instructions
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Wno-c++11-narrowing -Wno-unused-variable -Wno-unused-parameter -Wno-array-bounds -Wno-dangling-reference -Wno-tautological-compare")


set(INCLUDES
  eigen
  libigl
)

foreach(INCLUDE IN LISTS INCLUDES)
  include(cmake/include_${INCLUDE}.cmake)
  message("Load ${INCLUDE} done.")
endforeach()

igl_include(glfw)
include_directories(src)
include_directories(phy_system)
add_subdirectory(src)
add_subdirectory(phy_system)
add_executable(ei main.cpp)
target_link_libraries(ei solver phy_system)

# Test executable for Krylov subspace method
add_executable(test_krylov test_krylov.cpp)
target_link_libraries(test_krylov solver)













