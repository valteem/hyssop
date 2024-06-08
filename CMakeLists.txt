cmake_minimum_required(VERSION 3.22)

project(hyssop)

#---------------------------------------------------------------------------------
# GTest bootstrapping
#---------------------------------------------------------------------------------
enable_testing()
add_subdirectory(googletest)
include_directories(googletest/include)

#---------------------------------------------------------------------------------
# Package files
#---------------------------------------------------------------------------------
add_executable(hyssop src/get-started.cc)
#add_subdirectory(src)
include_directories($(PROJECT_SOURCE_DIR)/include)

#---------------------------------------------------------------------------------
# Eigen
#---------------------------------------------------------------------------------
#find_package(Eigen3 3.4 REQUIRED NO_MODULE)
#target_link_libraries(hyssop Eigen3::Eigen)
include_directories(eigen)
#---------------------------------------------------------------------------------

#---------------------------------------------------------------------------------
# Test files
#---------------------------------------------------------------------------------
add_subdirectory(tests)

target_link_libraries(hyssop GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(hyssop)