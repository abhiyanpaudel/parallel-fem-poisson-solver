//
// Created by Fuad Hasan on 4/7/25.
//

#ifndef ASSIGNMENT2_MESH_H
#define ASSIGNMENT2_MESH_H

#include <Kokkos_Core.hpp>
#include <string>

//********************************** Helper Functions
//****************************//
void check_file_existence(const std::string filename);

//********************************** Mesh Class
//**********************************//
enum MeshType {
  TRIANGLE = 3,
  QUAD = 4,
};

class Mesh {
 public:
  // ***************************** Public Functions
  // ***************************** //
  Mesh(const std::string filename);

  // Get the physical x or y coordinate of an element node on the host
  // Kokkos::View.
  [[nodiscard]] KOKKOS_INLINE_FUNCTION double GetCoordinate(int element,
                                                            int node,
                                                            int dim) const {
    return data_(element, node, dim + 1);
  }

  // Get the global degree of freedom on the host Kokkos::View.
  [[nodiscard]] KOKKOS_INLINE_FUNCTION int GetGlobalDof(int element,
                                                        int node) const {
    return data_(element, node, 0);
  }

  // Get the device view data_ containing the dofs and physical coordinates
  [[nodiscard]] KOKKOS_INLINE_FUNCTION
      Kokkos::View<double***, Kokkos::DefaultExecutionSpace>
      GetData() const {
    return data_;
  }

  // Get the number of vertices in the mesh
  [[nodiscard]] KOKKOS_INLINE_FUNCTION size_t GetNumVertices() const {
    return numVertices_;
  }

  // Get the number of elements in the mesh
  [[nodiscard]] KOKKOS_INLINE_FUNCTION size_t GetNumElements() const {
    return numElements_;
  }

  KOKKOS_INLINE_FUNCTION
  MeshType GetMeshType() const { return meshType_; }

  [[nodiscard]] size_t GetNumNodesPerElement() const {
	return static_cast<int>(GetMeshType()); 
  }

 private:
  // ***************************** Private Attributes
  // ***************************** //
  size_t numVertices_ = 0;
  size_t numElements_ = 0;
  MeshType meshType_;

  // Mesh data in GPU memory space
  Kokkos::View<double***, Kokkos::DefaultExecutionSpace> data_;
};

#endif  // ASSIGNMENT2_MESH_H
