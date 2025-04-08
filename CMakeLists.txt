cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(Assignment2)
include(FetchContent)
include(GNUInstallDirs)

################## OPTIONS ##################
option(Assignment_ENABLE_TESTING        "Enable testing"        ON)


################## Add Tests ##################
if(Assignment_ENABLE_TESTING)
    message(STATUS "TESTS ENABLED")
    include(CTest)
    enable_testing()
    add_subdirectory(tests)
endif()


#################### Dependencies ####################
find_package(Kokkos REQUIRED)

####################### Source Subdirectory ######################
add_subdirectory(src)