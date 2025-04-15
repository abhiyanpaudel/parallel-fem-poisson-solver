//
// Created by Fuad Hasan on 4/11/25.
//

#include <StiffnessMatrix.h>

bool matches_any(const std::vector<int>& vec, int value) {
  return std::any_of(vec.begin(), vec.end(),
                     [value](const int& v) { return v == value; });
}

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
        stiffnessMatrix.elementStiffnessMatrix.rowColCOO_);
    Kokkos::deep_copy(rowColIndex_host,
                      stiffnessMatrix.elementStiffnessMatrix.rowColCOO_);

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
                      stiffnessMatrix.elementStiffnessMatrix.rowColCOO_);
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
    std::vector<std::vector<int>> expected_data_combination(
        expected_data.size() + 1);
    expected_data_combination[0] = {8, 9};
    expected_data_combination[1] = {8, 9};
    expected_data_combination[2] = {6};
    expected_data_combination[3] = {11};
    expected_data_combination[4] = {7, 10};
    expected_data_combination[5] = {7, 10};
    expected_data_combination[6] = {2};
    expected_data_combination[7] = {0, 26};
    expected_data_combination[8] = {0, 26};
    expected_data_combination[9] = {24};
    expected_data_combination[10] = {1, 25};
    expected_data_combination[11] = {1, 25};
    expected_data_combination[12] = {20};
    expected_data_combination[13] = {35, 18};
    expected_data_combination[14] = {35, 18};
    expected_data_combination[15] = {33};
    expected_data_combination[16] = {34, 19};
    expected_data_combination[17] = {34, 19};
    expected_data_combination[18] = {15};
    expected_data_combination[19] = {29};
    expected_data_combination[20] = {17, 27};
    expected_data_combination[21] = {17, 27};
    expected_data_combination[22] = {16, 28};
    expected_data_combination[23] = {16, 28};
    expected_data_combination[24] = {5, 12};
    expected_data_combination[25] = {5, 12};
    expected_data_combination[26] = {23, 3};
    expected_data_combination[27] = {23, 3};
    expected_data_combination[28] = {21, 32};
    expected_data_combination[29] = {21, 32};
    expected_data_combination[30] = {30, 14};
    expected_data_combination[31] = {30, 14};
    expected_data_combination[32] = {4, 22, 31, 13};
    expected_data_combination[33] = {4, 22, 31, 13};
    expected_data_combination[34] = {4, 22, 31, 13};
    expected_data_combination[35] = {4, 22, 31, 13};
    expected_data_combination[36] = {4, 22, 31, 13};

    printf("\nSorted OOR and OOC matrices:\n");
    printf("(row, col): data\n");
    for (auto i = 0; i < rowColIndex_host.size(); i++) {
      printf("(%d, %d): %.1f\n", rowColIndex_host(i).r, rowColIndex_host(i).c,
             elem_stiffness_data_host(i));
      REQUIRE(rowColIndex_host(i).r == expected_oor[i]);
      REQUIRE(rowColIndex_host(i).c == expected_ooc[i]);
      REQUIRE(matches_any(expected_data_combination[i],
                          int(elem_stiffness_data_host(i))));
    }

    printf("\n");
    printf("=>=============== Test Assembly ===============<=\n");
    printf("\n");

    stiffnessMatrix.assemble(elem_stiffness_data);
    auto row_id = stiffnessMatrix.GetRowIndex();
    auto row_id_host = Kokkos::create_mirror_view(row_id);
    Kokkos::deep_copy(row_id_host, row_id);

    REQUIRE(row_id_host.size() == stiffnessMatrix.GetDim() + 1);

    printf("CSR Row Index:\n");
    for (auto i = 0; i < row_id_host.size(); i++) {
      printf("%d ", row_id_host(i));
    }
    printf("\n");

    std::vector<int> expected_row_id = {0, 4, 8, 12, 16, 21};
    for (auto i = 0; i < row_id_host.size(); i++) {
      REQUIRE(row_id_host(i) == expected_row_id[i]);
    }

    printf("=>========== Stiffness Matrix ===========<=\n");
    stiffnessMatrix.printStiffnessMatrix();

    auto col_id = stiffnessMatrix.GetColIndex();
    auto col_id_host = Kokkos::create_mirror_view(col_id);
    Kokkos::deep_copy(col_id_host, col_id);
    REQUIRE(col_id_host.size() == 21);
    std::vector<int> expected_col_ids = {0, 1, 3, 4, 0, 1, 2, 4, 1, 2, 3,
                                         4, 0, 2, 3, 4, 0, 1, 2, 3, 4};
    for (auto i = 0; i < col_id_host.size(); i++) {
      REQUIRE(col_id_host(i) == expected_col_ids[i]);
    }

    auto assembled_data = stiffnessMatrix.GetValues();
    auto assembled_data_host = Kokkos::create_mirror_view(assembled_data);
    Kokkos::deep_copy(assembled_data_host, assembled_data);
    REQUIRE(assembled_data_host.size() == 21);
    std::vector<int> expected_assembled_data = {17, 6,  11, 17, 2,  26, 24,
                                                26, 20, 53, 33, 53, 15, 29,
                                                44, 44, 17, 26, 53, 44, 70};
    for (auto i = 0; i < assembled_data_host.size(); i++) {
      REQUIRE(assembled_data_host(i) == int(expected_assembled_data[i]));
    }
  }
  Kokkos::finalize();
}
