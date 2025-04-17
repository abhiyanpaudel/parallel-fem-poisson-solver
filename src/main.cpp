//
// Created by Fuad Hasan on 4/7/25.
//

#include <CalculateStiffnessMatrixAndLoadVector.hpp>
#include <Kokkos_Core.hpp>
#include <chrono>
#include <iostream>

#include "StiffnessMatrix.h"

int main(int argc, char** argv) {
  auto start = std::chrono::steady_clock::now();
#ifdef KOKKOS_ENABLE_OPENMP
  printf("Using Kokkos OpenMP backend\n");
#endif
#ifdef KOKKOS_ENABLE_CUDA
  printf("Using Kokkos CUDA backend\n");
#endif

  Kokkos::initialize(argc, argv);
  {
    if (argc != 2) {
      std::string usage = "Usage: " + std::string(argv[0]) + " <meshFileName>";
      throw std::runtime_error(usage);
    }
    std::string meshFileName = argv[1];
    printf("Reading mesh file: %s\n", meshFileName.c_str());
    Mesh mesh(meshFileName);
    printf("Mesh loaded of type %d with %zu elements and %zu nodes\n",
           mesh.GetMeshType(), mesh.GetNumElements(), mesh.GetNumVertices());

    printf("Creating stiffness matrix\n");
    StiffnessMatrix stiffnessMatrix(mesh);
    auto el_stiff_size = stiffnessMatrix.getElementStiffnessSize();
    printf("Element stiffness size: %zu\n", el_stiff_size);

    auto el_stiffness_and_load =
        calculateAllElementStiffnessMatrixAndLoadVector(mesh, 1);
    auto element_stiffness = el_stiffness_and_load.allElementStiffnessMatrix;
    printf("The element stiffness matrix size: %ld\n",
           element_stiffness.size());

#ifndef NDEBUG
    auto element_stiffness_host = Kokkos::create_mirror_view(element_stiffness);
    Kokkos::deep_copy(element_stiffness_host, element_stiffness);
    printf("Element stiffness matrices:\n");
    for (int i = 0; i < element_stiffness_host.size(); i++) {
      printf("%f, \n", element_stiffness_host(i));
    }
    printf("\n");
#endif

    stiffnessMatrix.sortDataByRowCol(element_stiffness);
    stiffnessMatrix.assemble(element_stiffness);

    auto load_vector =
        assembleLoadVector(el_stiffness_and_load.allElementLoadVector, mesh);

#ifndef NDEBUG
    stiffnessMatrix.printStiffnessMatrix();
    printf("=>----------- Dense Matrix -----------<=\n");
    stiffnessMatrix.printDenseMatrix();

    printf("=>----------- Load Vector -----------<=\n");
    auto load_vector_host = Kokkos::create_mirror_view(load_vector);
    Kokkos::deep_copy(load_vector_host, load_vector);
    printf("Load vector:\n");
    for (int i = 0; i < load_vector_host.size(); i++) {
      printf("%f, \n", load_vector_host(i));
    }
#endif
  }
  Kokkos::finalize();

  auto end = std::chrono::steady_clock::now();
  auto elapsed_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  printf("Total execution time: %lld ms\n", elapsed_time.count());

  return 0;
}