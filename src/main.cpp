//
// Created by Fuad Hasan on 4/7/25.
//

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
    printf("Element stiffness size: %d\n", el_stiff_size);

    Kokkos::View<double*> element_stiffness("element_stiffness", el_stiff_size);
    auto element_stiffness_h = Kokkos::create_mirror_view(element_stiffness);
    // fill with random values
    for (int i = 0; i < el_stiff_size; i++) {
      element_stiffness_h(i) = static_cast<double>(rand()) / RAND_MAX;
    }
    Kokkos::deep_copy(element_stiffness, element_stiffness_h);

    stiffnessMatrix.sortDataByRowCol(element_stiffness);
    stiffnessMatrix.assemble(element_stiffness);
  }
  Kokkos::finalize();

  return 0;
}