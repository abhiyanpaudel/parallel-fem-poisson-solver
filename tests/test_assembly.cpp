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

    auto expected_size =
        mesh.GetNumElements() * mesh.GetMeshType() * mesh.GetMeshType();

    Kokkos::View<double*> elem_stiffness_data("elem_stiffness_data",
                                              expected_size);
    Kokkos::parallel_for(
        "fill_elem_dummy_data", expected_size,
        KOKKOS_LAMBDA(const int i) { elem_stiffness_data(i) = i; });

    StiffnessMatrix stiffnessMatrix(mesh);

    auto rowColIndex_host = Kokkos::create_mirror_view(
        stiffnessMatrix.elementStiffnessMatrix.rowColIndex_);
    Kokkos::deep_copy(rowColIndex_host,
                      stiffnessMatrix.elementStiffnessMatrix.rowColIndex_);

    REQUIRE(rowColIndex_host.size() == expected_size);

    std::vector<int> expected_oor = {1, 1, 1, 4, 4, 4, 0, 0, 0, 0, 0, 0,
                                     4, 4, 4, 3, 3, 3, 2, 2, 2, 4, 4, 4,
                                     1, 1, 1, 3, 3, 3, 4, 4, 4, 2, 2, 2};
    std::vector<int> expected_ooc = {1, 4, 0, 1, 4, 0, 1, 4, 0, 0, 4, 3,
                                     0, 4, 3, 0, 4, 3, 2, 4, 1, 2, 4, 1,
                                     2, 4, 1, 3, 4, 2, 3, 4, 2, 3, 4, 2};
    printf("\nOOR and OOC matrices:\n");
    printf("(row, col)\n");
    for (auto i = 0; i < rowColIndex_host.size(); i++) {
      printf("(%d, %d)\n", rowColIndex_host(i).r, rowColIndex_host(i).c);
      REQUIRE(rowColIndex_host(i).r == expected_oor[i]);
      REQUIRE(rowColIndex_host(i).c == expected_ooc[i]);
    }

    stiffnessMatrix.sortDataByRowCol(elem_stiffness_data);
    Kokkos::deep_copy(rowColIndex_host,
                      stiffnessMatrix.elementStiffnessMatrix.rowColIndex_);
    auto elem_stiffness_data_host =
        Kokkos::create_mirror_view(elem_stiffness_data);
    Kokkos::deep_copy(elem_stiffness_data_host, elem_stiffness_data);

    expected_oor = {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
                    3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
    expected_ooc = {0, 0, 1, 3, 4, 4, 0, 1, 1, 2, 4, 4, 1, 2, 2, 3, 4, 4,
                    0, 2, 3, 3, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4};
    std::vector<int> expected_data = {
        8,  9,  6,  11, 7,  10, 2, 0,  26, 24, 1,  25, 20, 35, 18, 33, 34, 19,
        15, 29, 17, 27, 16, 28, 5, 12, 23, 3,  21, 32, 30, 14, 4,  22, 31, 13};

    printf("\nSorted OOR and OOC matrices:\n");
    printf("(row, col): data\n");
    for (auto i = 0; i < rowColIndex_host.size(); i++) {
      printf("(%d, %d): %.1f\n", rowColIndex_host(i).r, rowColIndex_host(i).c,
             elem_stiffness_data_host(i));
      REQUIRE(rowColIndex_host(i).r == expected_oor[i]);
      REQUIRE(rowColIndex_host(i).c == expected_ooc[i]);
      REQUIRE(elem_stiffness_data_host(i) == expected_data[i]);
    }
  }
  Kokkos::finalize();
}
