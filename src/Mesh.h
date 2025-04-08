//
// Created by Fuad Hasan on 4/7/25.
//

#ifndef ASSIGNMENT2_MESH_H
#define ASSIGNMENT2_MESH_H

#include <string>


//********************************** Helper Functions ****************************//
void check_file_existence(const std::string filename);



//********************************** Mesh Class **********************************//
enum class MeshType {
    TRIANGLE = 0,
    QUAD = 1,
};

class Mesh {
public:
    Mesh(const std::string filename);
private:
    size_t numVertices;
    size_t numElements;

    MeshType meshType;


    // data arrays depending on cuda flag
};


#endif //ASSIGNMENT2_MESH_H
