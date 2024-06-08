cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(hyssop)

#---------------------------------------------------------------------------------
# GTest bootstrapping
#---------------------------------------------------------------------------------
enable_testing()
add_subdirectory(googletest)
include_directories(googletest/include)

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
add_executable(hyssop_test)
add_subdirectory(tests)

target_link_libraries(hyssop_test GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(hyssop_test)

#---------------------------------------------------------------------------------
# Package files
# Placed after test target, gtest fails to discover any tests otherwise
#---------------------------------------------------------------------------------
add_executable(hyssop)
add_subdirectory(src)
