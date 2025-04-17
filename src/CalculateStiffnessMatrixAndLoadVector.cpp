#include "CalculateStiffnessMatrixAndLoadVector.hpp"

Results calculateAllElementStiffnessMatrixAndLoadVector(const Mesh& mesh,
                                                        double k) {
  int numElements = mesh.GetNumElements();

  int numNodes = mesh.GetNumNodesPerElement();

  int sizePerElement = numNodes * numNodes;

  Kokkos::Profiling::pushRegion(
      "Allocate Element Stiffness Matrix and Load Vector");
  View1D allElementStiffnessMatrix("stores all element stiffness matrix",
                                   numElements * sizePerElement);
  View1D allElementLoadVector("stores all element load vector",
                              numElements * numNodes);
  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::pushRegion(
      "Compute Element Stiffness Matrix and Load Vector");
  Kokkos::parallel_for(
      "CalculateStiffness", numElements, KOKKOS_LAMBDA(const int elemIdx) {
        double stiffnessMatrixPerElement[MAX_STIFFNESS_MATRIX_SIZE] = {};
        double loadVectorPerElement[MAX_LOAD_VECTOR_SIZE] = {};
        if (numNodes == 3) {  // Triangle element
          TriElement triElem(mesh, elemIdx);
          triElem.setMaterialProperty(k);
          triElem.computeElementStiffnessMatrix(stiffnessMatrixPerElement);
          triElem.computeElementLoadVector(loadVectorPerElement);
        } else {  // Quad element
          QuadElement quadElem(mesh, elemIdx);
          quadElem.setMaterialProperty(k);
          quadElem.computeElementStiffnessMatrix(stiffnessMatrixPerElement);
          quadElem.computeElementLoadVector(loadVectorPerElement);
        }

        int base_stiffness_idx = elemIdx * sizePerElement;
        for (int i = 0; i < sizePerElement; ++i) {
          allElementStiffnessMatrix(base_stiffness_idx + i) =
              stiffnessMatrixPerElement[i];
        }

        int base_load_idx = elemIdx * numNodes;

        for (int i = 0; i < numNodes; ++i) {
          allElementLoadVector(base_load_idx + i) = loadVectorPerElement[i];
        }
      });
  Kokkos::Profiling::popRegion();

  return Results{allElementStiffnessMatrix, allElementLoadVector};
}
