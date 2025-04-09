//
// Created by Fuad Hasan on 4/7/25.
//

#ifndef ASSIGNMENT2_MESH_H
#define ASSIGNMENT2_MESH_H

#include <string>
#include <Kokkos_Core.hpp>

//********************************** Helper Functions
//****************************//
void check_file_existence(const std::string filename);

//********************************** Mesh Class
//**********************************//
enum MeshType 
{
  TRIANGLE = 3,
  QUAD = 4,
};

class Mesh 
{
    public:
    Mesh(const std::string filename);

    [[nodiscard]]
    inline double GetCoordinate(int element, int node, int dim) {return data_(element,node,dim+1);}

    [[nodiscard]]
    inline int GetGlobalDof(int element, int node) {return data_(element,node,0);}

    private:
    size_t numVertices_ = 0;
    size_t numElements_ = 0;
    MeshType meshType_;
    Kokkos::View<double***, Kokkos::DefaultHostExecutionSpace> data_; // CHANGE FROM HOST SPACE LATER

    // data arrays depending on cuda flag
};

#endif  // ASSIGNMENT2_MESH_H
