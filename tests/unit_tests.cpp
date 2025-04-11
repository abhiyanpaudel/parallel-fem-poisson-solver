//
// Created by Fuad Hasan on 4/7/25.
//

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <iostream>

#include "Mesh.h"

TEST_CASE("Test check_file_existence") 
{
  std::string filename = "assets/small_square.msh";
  check_file_existence(filename);
  printf("If it's failing, try running the test from the tests directory.\n");
}

TEST_CASE("Test mesh constructor for triangle mesh.")
{
    Kokkos::initialize();
    {
        
        std::string meshfile = "assets/TriMesh.msh";
        auto mesh = Mesh(meshfile);
        auto host_data = Kokkos::create_mirror_view(mesh.GetData());
        Kokkos::deep_copy(host_data, mesh.GetData());
        double tol = 1e-12;

        std::vector<double> correct_x_coords({0., 1., 1., 0., 0.5});
        std::vector<double> correct_y_coords({0., 0., 1., 1., 0.5});

        // Element 1 (nodes 2, 5, 1)
        REQUIRE_THAT(host_data(0,0,1), Catch::Matchers::WithinAbs(correct_x_coords[1], tol));
        REQUIRE_THAT(host_data(0,0,2), Catch::Matchers::WithinAbs(correct_y_coords[1], tol));
        REQUIRE_THAT(host_data(0,1,1), Catch::Matchers::WithinAbs(correct_x_coords[4], tol));
        REQUIRE_THAT(host_data(0,1,2), Catch::Matchers::WithinAbs(correct_y_coords[4], tol));
        REQUIRE_THAT(host_data(0,2,1), Catch::Matchers::WithinAbs(correct_x_coords[0], tol));
        REQUIRE_THAT(host_data(0,2,2), Catch::Matchers::WithinAbs(correct_y_coords[0], tol));

        // Element 2 (nodes 1, 5, 4)
        REQUIRE_THAT(host_data(1,0,1), Catch::Matchers::WithinAbs(correct_x_coords[0], tol));
        REQUIRE_THAT(host_data(1,0,2), Catch::Matchers::WithinAbs(correct_y_coords[0], tol));
        REQUIRE_THAT(host_data(1,1,1), Catch::Matchers::WithinAbs(correct_x_coords[4], tol));
        REQUIRE_THAT(host_data(1,1,2), Catch::Matchers::WithinAbs(correct_y_coords[4], tol));
        REQUIRE_THAT(host_data(1,2,1), Catch::Matchers::WithinAbs(correct_x_coords[3], tol));
        REQUIRE_THAT(host_data(1,2,2), Catch::Matchers::WithinAbs(correct_y_coords[3], tol));
        
        // Element 3 (nodes 3, 5, 2)
        REQUIRE_THAT(host_data(2,0,1), Catch::Matchers::WithinAbs(correct_x_coords[2], tol));
        REQUIRE_THAT(host_data(2,0,2), Catch::Matchers::WithinAbs(correct_y_coords[2], tol));
        REQUIRE_THAT(host_data(2,1,1), Catch::Matchers::WithinAbs(correct_x_coords[4], tol));
        REQUIRE_THAT(host_data(2,1,2), Catch::Matchers::WithinAbs(correct_y_coords[4], tol));
        REQUIRE_THAT(host_data(2,2,1), Catch::Matchers::WithinAbs(correct_x_coords[1], tol));
        REQUIRE_THAT(host_data(2,2,2), Catch::Matchers::WithinAbs(correct_y_coords[1], tol));
        
        // Element 4 (nodes 4, 5, 3)
        REQUIRE_THAT(host_data(3,0,1), Catch::Matchers::WithinAbs(correct_x_coords[3], tol));
        REQUIRE_THAT(host_data(3,0,2), Catch::Matchers::WithinAbs(correct_y_coords[3], tol));
        REQUIRE_THAT(host_data(3,1,1), Catch::Matchers::WithinAbs(correct_x_coords[4], tol));
        REQUIRE_THAT(host_data(3,1,2), Catch::Matchers::WithinAbs(correct_y_coords[4], tol));
        REQUIRE_THAT(host_data(3,2,1), Catch::Matchers::WithinAbs(correct_x_coords[2], tol));
        REQUIRE_THAT(host_data(3,2,2), Catch::Matchers::WithinAbs(correct_y_coords[2], tol));
    }
    Kokkos::finalize();
}

TEST_CASE("Test mesh constructor for quadrilateral mesh.")
{
    Kokkos::initialize();
    {
        std::string meshfile = "assets/QuadMesh.msh";
        auto mesh = Mesh(meshfile);
        auto host_data = Kokkos::create_mirror_view(mesh.GetData());
        Kokkos::deep_copy(host_data, mesh.GetData());
        double tol = 1e-12;

        std::vector<double> correct_x_coords({0., 1., 1., 0., 0.5, 1., 0.5, 0., 0.5});
        std::vector<double> correct_y_coords({0., 0., 1., 1., 0., 0.5, 1., 0.5, 0.5});

        // Element 1 (nodes 1, 5, 9, 8)
        REQUIRE_THAT(host_data(0,0,1), Catch::Matchers::WithinAbs(correct_x_coords[0], tol));
        REQUIRE_THAT(host_data(0,0,2), Catch::Matchers::WithinAbs(correct_y_coords[0], tol));
        REQUIRE_THAT(host_data(0,1,1), Catch::Matchers::WithinAbs(correct_x_coords[4], tol));
        REQUIRE_THAT(host_data(0,1,2), Catch::Matchers::WithinAbs(correct_y_coords[4], tol));
        REQUIRE_THAT(host_data(0,2,1), Catch::Matchers::WithinAbs(correct_x_coords[8], tol));
        REQUIRE_THAT(host_data(0,2,2), Catch::Matchers::WithinAbs(correct_y_coords[8], tol));
        REQUIRE_THAT(host_data(0,3,1), Catch::Matchers::WithinAbs(correct_x_coords[7], tol));
        REQUIRE_THAT(host_data(0,3,2), Catch::Matchers::WithinAbs(correct_y_coords[7], tol));

        // Element 2 (nodes 8, 9, 7, 4)
        REQUIRE_THAT(host_data(1,0,1), Catch::Matchers::WithinAbs(correct_x_coords[7], tol));
        REQUIRE_THAT(host_data(1,0,2), Catch::Matchers::WithinAbs(correct_y_coords[7], tol));
        REQUIRE_THAT(host_data(1,1,1), Catch::Matchers::WithinAbs(correct_x_coords[8], tol));
        REQUIRE_THAT(host_data(1,1,2), Catch::Matchers::WithinAbs(correct_y_coords[8], tol));
        REQUIRE_THAT(host_data(1,2,1), Catch::Matchers::WithinAbs(correct_x_coords[6], tol));
        REQUIRE_THAT(host_data(1,2,2), Catch::Matchers::WithinAbs(correct_y_coords[6], tol));
        REQUIRE_THAT(host_data(1,3,1), Catch::Matchers::WithinAbs(correct_x_coords[3], tol));
        REQUIRE_THAT(host_data(1,3,2), Catch::Matchers::WithinAbs(correct_y_coords[3], tol));
        
        // Element 3 (nodes 5, 2, 6, 9)
        REQUIRE_THAT(host_data(2,0,1), Catch::Matchers::WithinAbs(correct_x_coords[4], tol));
        REQUIRE_THAT(host_data(2,0,2), Catch::Matchers::WithinAbs(correct_y_coords[4], tol));
        REQUIRE_THAT(host_data(2,1,1), Catch::Matchers::WithinAbs(correct_x_coords[1], tol));
        REQUIRE_THAT(host_data(2,1,2), Catch::Matchers::WithinAbs(correct_y_coords[1], tol));
        REQUIRE_THAT(host_data(2,2,1), Catch::Matchers::WithinAbs(correct_x_coords[5], tol));
        REQUIRE_THAT(host_data(2,2,2), Catch::Matchers::WithinAbs(correct_y_coords[5], tol));
        REQUIRE_THAT(host_data(2,3,1), Catch::Matchers::WithinAbs(correct_x_coords[8], tol));
        REQUIRE_THAT(host_data(2,3,2), Catch::Matchers::WithinAbs(correct_y_coords[8], tol));
        
        // Element 4 (nodes 9, 6, 3, 7)
        REQUIRE_THAT(host_data(3,0,1), Catch::Matchers::WithinAbs(correct_x_coords[8], tol));
        REQUIRE_THAT(host_data(3,0,2), Catch::Matchers::WithinAbs(correct_y_coords[8], tol));
        REQUIRE_THAT(host_data(3,1,1), Catch::Matchers::WithinAbs(correct_x_coords[5], tol));
        REQUIRE_THAT(host_data(3,1,2), Catch::Matchers::WithinAbs(correct_y_coords[5], tol));
        REQUIRE_THAT(host_data(3,2,1), Catch::Matchers::WithinAbs(correct_x_coords[2], tol));
        REQUIRE_THAT(host_data(3,2,2), Catch::Matchers::WithinAbs(correct_y_coords[2], tol));
        REQUIRE_THAT(host_data(3,3,1), Catch::Matchers::WithinAbs(correct_x_coords[6], tol));
        REQUIRE_THAT(host_data(3,3,2), Catch::Matchers::WithinAbs(correct_y_coords[6], tol));
    }
    Kokkos::finalize();
}