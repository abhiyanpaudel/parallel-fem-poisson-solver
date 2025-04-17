//
// Created by Fuad Hasan on 4/7/25.
//

#include <CalculateStiffnessMatrixAndLoadVector.hpp>
#include <Kokkos_Core.hpp>
#include <iostream>

#include "StiffnessMatrix.h"

int main(int argc, char** argv) {
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

    stiffnessMatrix.printStiffnessMatrix();

#ifndef NDEBUG
    printf("=>----------- Dense Matrix -----------<=\n");
    stiffnessMatrix.printDenseMatrix();
#endif
  }
  Kokkos::finalize();

  return 0;
}