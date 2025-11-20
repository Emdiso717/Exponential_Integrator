if(TARGET igl::core)
  return()
endif()

include(FetchContent)
FetchContent_Declare(
  libigl
  GIT_REPOSITORY https://github.com/libigl/libigl.git
  GIT_TAG v2.4.0
)
# Enable required libigl components before fetching/building
FetchContent_MakeAvailable(libigl)
