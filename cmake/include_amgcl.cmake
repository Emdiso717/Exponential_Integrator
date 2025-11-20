set(amgcl_patch ${PROJECT_SOURCE_DIR}/cmake/amgcl_enable_examples.patch)
CPMAddPackage(
  NAME amgcl
  GIT_REPOSITORY "https://github.com/ddemidov/amgcl.git"
  GIT_TAG "1.4.4"
  # PATCH_COMMAND
  # git restore . && git apply ${amgcl_patch}
  # OPTIONS
  # "AMGCL_BUILD_EXAMPLES ON"

)
# target_link_libraries(amgcl INTERFACE
#   VexCL::CUDA
# )
