//
// Created by Fuad Hasan on 4/11/25.
//

#include <StiffnessMatrix.h>

#include <catch2/catch_test_macros.hpp>
#include <vector>

TEST_CASE("Test StiffnessMatrix Construction") {
  Kokkos::initialize();
  {
    std::string mesh_filename = "assets/TriMesh.msh";
    Mesh mesh(mesh_filename);
    printf("Loaded %s mesh with %zu nodes and %zu elements\n",
           mesh_filename.c_str(), mesh.GetNumVertices(), mesh.GetNumElements());
    StiffnessMatrix stiffnessMatrix(mesh);

    auto oor_host =
        Kokkos::create_mirror_view(stiffnessMatrix.elementStiffnessMatrix.oor_);
    auto ooc_host =
        Kokkos::create_mirror_view(stiffnessMatrix.elementStiffnessMatrix.ooc_);
    Kokkos::deep_copy(oor_host, stiffnessMatrix.elementStiffnessMatrix.oor_);
    Kokkos::deep_copy(ooc_host, stiffnessMatrix.elementStiffnessMatrix.ooc_);

    auto expected_size =
        mesh.GetNumElements() * mesh.GetMeshType() * mesh.GetMeshType();
    printf("Created OOR and OOC matrices of size %zu and %zu\n",
           oor_host.size(), ooc_host.size());
    REQUIRE(oor_host.size() == expected_size);
    REQUIRE(ooc_host.size() == expected_size);

    std::vector<int> expected_oor = {1, 1, 1, 4, 4, 4, 0, 0, 0, 0, 0, 0,
                                     4, 4, 4, 3, 3, 3, 2, 2, 2, 4, 4, 4,
                                     1, 1, 1, 3, 3, 3, 4, 4, 4, 2, 2, 2};
    std::vector<int> expected_ooc = {1, 4, 0, 1, 4, 0, 1, 4, 0, 0, 4, 3,
                                     0, 4, 3, 0, 4, 3, 2, 4, 1, 2, 4, 1,
                                     2, 4, 1, 3, 4, 2, 3, 4, 2, 3, 4, 2};
    printf("\nOOR and OOC matrices:\n");
    printf("(row, col)\n");
    for (auto i = 0; i < oor_host.size(); i++) {
      printf("(%d, %d)\n", oor_host(i), ooc_host(i));
      REQUIRE(oor_host(i) == expected_oor[i]);
      REQUIRE(ooc_host(i) == expected_ooc[i]);
    }
  }
  Kokkos::finalize();
}