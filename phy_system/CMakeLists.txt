aux_source_directory(. source)
file(GLOB header *.hpp)

add_library(phy_system STATIC ${source} ${header})
target_link_libraries(phy_system
  eigen
)
target_include_directories(phy_system PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(phy_system PUBLIC cxx_std_20)