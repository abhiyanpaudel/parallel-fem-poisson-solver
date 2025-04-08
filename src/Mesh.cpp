//
// Created by Fuad Hasan on 4/7/25.
//

#include "Mesh.h"
#include <filesystem>
#include <string>

void check_file_existence(const std::string filename){
    if(!std::filesystem::exists(filename)){
       throw std::runtime_error("File does not exist: " + filename);
    }
}

Mesh::Mesh(const std::string filename) {
    check_file_existence(filename);
}