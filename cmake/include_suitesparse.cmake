 CPMAddPackage(
   NAME SuiteSparse
   GIT_REPOSITORY https://github.com/sergiud/SuiteSparse.git
   # GIT_TAG v7.4.0
   GIT_TAG 5.13.0-cmake.4
   OPTIONS "WITH_CUDA OFF"
 )


# set (SUITESPARSE_ENABLE_PROJECTS  "suitesparse_config;cholmod;umfpack;spqr")
# set(suitesparse_patch "${PROJECT_SOURCE_DIR}/cmake/suitesparse_expose_metis.patch")
# CPMAddPackage(
#    NAME SuiteSparse
#    GIT_REPOSITORY https://github.com/DrTimothyAldenDavis/SuiteSparse.git
#    GIT_TAG v7.8.2
#    OPTIONS
#    "SUITESPARSE_USE_CUDA OFF"
#    "SUITESPARSE_DEMOS OFF"
#    "SUITESPARSE_ENABLE_PROJECTS=${SUITESPARSE_ENABLE_PROJECTS}"
#    "CHOLMOD_PARTITION ON"
#    "CHOLMOD_SUPERNODAL ON"
#    "CHOLMOD_USE_OPENMP ON"
#  )
