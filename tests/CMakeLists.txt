################################## Catch2 ########################
find_package(Catch2 3 QUIET)
if(Catch2_FOUND)
    message(STATUS "Found Catch2: ${Catch2_DIR}")
else()
    message(STATUS "Catch2 not found. Fetching Catch2...")
    FetchContent_Declare(
            Catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2.git
            GIT_TAG        v3.4.0 # or a later release
    )
    FetchContent_MakeAvailable(Catch2)
endif()
###################################################################

##################### Catch2 Integration ##########################
include(CTest)
include(Catch)


########################### Tests ############################

# Unit tests
add_executable(unit_tests unit_tests.cpp ${CMAKE_SOURCE_DIR}/src/Mesh.cpp)
target_link_libraries(unit_tests PRIVATE Catch2::Catch2WithMain Kokkos::kokkos)
target_include_directories(unit_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)
catch_discover_tests(unit_tests WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})