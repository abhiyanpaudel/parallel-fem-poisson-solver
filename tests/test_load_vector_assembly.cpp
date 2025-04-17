//
// Created by Fuad Hasan on 4/17/25.
//

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "CalculateStiffnessMatrixAndLoadVector.hpp"
#include "Mesh.h"
#include "StiffnessMatrix.h"

TEST_CASE("Test Load Assembly") {
  Kokkos::initialize();
  {
    std::string mesh_filename = "assets/TriMesh.msh";
    Mesh mesh(mesh_filename);

    auto el_stiffness_load =
        calculateAllElementStiffnessMatrixAndLoadVector(mesh, 1.0);
    auto load_vector =
        assembleLoadVector(el_stiffness_load.allElementLoadVector, mesh);

    REQUIRE(load_vector.size() == mesh.GetNumVertices());

    auto load_vector_host = Kokkos::create_mirror_view(load_vector);
    Kokkos::deep_copy(load_vector_host, load_vector);

    std::vector<double> expected_load_vector = {0.166667, 0.166667, 0.166667,
                                                0.166667, 0.333333};

    for (size_t i = 0; i < expected_load_vector.size(); ++i) {
      REQUIRE(load_vector_host(i) ==
              Catch::Approx(expected_load_vector[i]).epsilon(0.0001));
    }
  }
  Kokkos::finalize();
}