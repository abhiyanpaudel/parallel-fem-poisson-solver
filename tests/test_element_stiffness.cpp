//
// Created by Fuad Hasan on 4/16/25.
//

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "CalculateStiffnessMatrixAndLoadVector.hpp"
#include "StiffnessMatrix.h"

void test_mesh(std::string filename,
               std::vector<double> expected_stiffness_matrix, int n_nodes);

TEST_CASE("Test Element Stiffness Matrix") {
  Kokkos::initialize();
  {
    std::vector<double> tri_expected_dense_stiffness_matrix = {
        1, 0,  0, 0, -1, 0, 1,  0,  0,  -1, 0,  0, 1,
        0, -1, 0, 0, 0,  1, -1, -1, -1, -1, -1, 4};

    std::vector<double> quad_expected_dense_stiffness_matrix = {
        0.666667,  0,         0,         0,         -0.166667, 0,
        0,         -0.166667, -0.333333, 0,         0.666667,  0,
        0,         -0.166667, -0.166667, 0,         0,         -0.333333,
        0,         0,         0.666667,  0,         0,         -0.166667,
        -0.166667, 0,         -0.333333, 0,         0,         0,
        0.666667,  0,         0,         -0.166667, -0.166667, -0.333333,
        -0.166667, -0.166667, 0,         0,         1.33333,   -0.333333,
        0,         -0.333333, -0.333333, 0,         -0.166667, -0.166667,
        0,         -0.333333, 1.33333,   -0.333333, 0,         -0.333333,
        0,         0,         -0.166667, -0.166667, 0,         -0.333333,
        1.33333,   -0.333333, -0.333333, -0.166667, 0,         0,
        -0.166667, -0.333333, 0,         -0.333333, 1.33333,   -0.333333,
        -0.333333, -0.333333, -0.333333, -0.333333, -0.333333, -0.333333,
        -0.333333, -0.333333, 2.66667};

    test_mesh("assets/TriMesh.msh", tri_expected_dense_stiffness_matrix, 5);
    test_mesh("assets/QuadMesh.msh", quad_expected_dense_stiffness_matrix, 9);
  }
  Kokkos::finalize();
}

void test_mesh(std::string filename,
               std::vector<double> expected_stiffness_matrix, int n_nodes) {
  printf("Testing for mesh: %s\n", filename.c_str());
  Mesh mesh(filename);
  StiffnessMatrix stiffnessMatrix(mesh);

  auto el_stiffness_load =
      calculateAllElementStiffnessMatrixAndLoadVector(mesh, 1);
  auto el_stiffness = el_stiffness_load.allElementStiffnessMatrix;
  REQUIRE(el_stiffness.size() == stiffnessMatrix.getElementStiffnessSize());

  stiffnessMatrix.sortDataByRowCol(el_stiffness);
  stiffnessMatrix.assemble(el_stiffness);

  printf("=>--------- Dense Matrix for TriMesh ---------<=\n");
  stiffnessMatrix.printDenseMatrix();
  auto dense_stiffness_matrix = stiffnessMatrix.getDenseMatrix();

  REQUIRE(dense_stiffness_matrix.size() == n_nodes);
  REQUIRE(dense_stiffness_matrix[0].size() == n_nodes);
  for (int i = 0; i < n_nodes; ++i) {
    for (int j = 0; j < n_nodes; ++j) {
      double expected = expected_stiffness_matrix[i * n_nodes + j];
      double actual = dense_stiffness_matrix[i][j];
      REQUIRE(actual == Catch::Approx(expected).epsilon(0.0001));
    }
  }
}