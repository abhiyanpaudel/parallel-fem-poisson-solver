cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(Assignment2)
include(FetchContent)
include(GNUInstallDirs)

################## OPTIONS ##################
option(Assignment_ENABLE_TESTING        "Enable testing"        ON)

#################### Dependencies ####################
find_package(Kokkos REQUIRED)

################## Add Tests ##################
if(Assignment_ENABLE_TESTING)
    message(STATUS "TESTS ENABLED")
    include(CTest)
    enable_testing()
    add_subdirectory(tests)
endif()

####################### Source Subdirectory ######################
add_subdirectory(src)