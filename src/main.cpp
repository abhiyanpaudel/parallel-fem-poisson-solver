//
// Created by Fuad Hasan on 4/7/25.
//

#include <Kokkos_Core.hpp>
#include <iostream>

#include "Mesh.h"

int main(int argc, char* argv[]) {
#ifdef KOKKOS_ENABLE_OPENMP
  printf("Using Kokkos OpenMP backend\n");
#endif
#ifdef KOKKOS_ENABLE_CUDA
  printf("Using Kokkos CUDA backend\n");
#endif

  return 0;
}