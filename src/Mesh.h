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

    /**
     * Get the physical x or y coordinate of an element node on the host Kokkos::View.
     */
    [[nodiscard]]
    inline double GetCoordinate(int element, int node, int dim) const {return data_(element,node,dim+1);}

    /**
     * Get the global degree of freedom on the host Kokkos::View.
     */
    [[nodiscard]]
    inline int GetGlobalDof(int element, int node) const {return data_(element,node,0);}

    #ifdef KOKKOS_ENABLE_CUDA
        /**
         * Get the physical x or y coordinate of an element node on the device Kokkos::View.
         */
        [[nodiscard]] 
        inline double GetDeviceCoordinate(int element, int node, int dim) const {return device_data_(element,node,dim+1);}
        /**
         * Get the global degree of freedom (DOF) number on the device Kokkos::View.
         */
        [[nodiscard]]
        inline int GetDeviceGlobalDof(int element, int node) const {return device_data_(element,node,0);}
    #endif

    private:
    size_t numVertices_ = 0;
    size_t numElements_ = 0;
    MeshType meshType_;
    
    // Mesh data in CPU memory space
    Kokkos::View<double***, Kokkos::LayoutRight, Kokkos::HostSpace> data_; 
    
    // Mesh data in GPU memory space if using CUDA
    #ifdef KOKKOS_ENABLE_CUDA
        Kokkos::View<double***, Kokkos::LayoutRight, Kokkos::CudaSpace> device_data_; 
    #endif
};

#endif  // ASSIGNMENT2_MESH_H
