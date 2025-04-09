//
// Created by Fuad Hasan on 4/7/25.
//

#include "Mesh.h"
#include <filesystem>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>

void check_file_existence(const std::string filename) 
{
  if (!std::filesystem::exists(filename)) 
  {
    throw std::runtime_error("File does not exist: " + filename);
  }
}

Mesh::Mesh(const std::string filename) 
{ 
    check_file_existence(filename); // Check if file exists before allocating

    std::vector<double> x_coords, y_coords; // x and y coordinate values for each node
    std::vector<int> element_nodes; // Flattened element nodes
    
    std::ifstream input_file;
    input_file.open(filename); 
    if (input_file.is_open())
    {
        std::string line;
        bool getting_nodes = false, getting_elements = false, known_mesh_type = false;
        int i = -1;
        while (std::getline(input_file, line))
        {
            if (line == "$Nodes")
            {
                getting_nodes = true;
            }
            else if (line == "$EndNodes")
            {
                getting_nodes = false;
            }
            else if (line == "$Elements")
            {
                getting_elements = true;
            }
            else if (line == "$EndElements")
            {
                getting_elements = false;
            }
            else if (getting_nodes && i == -1)
            {
                x_coords.resize(std::stoi(line));
                y_coords.resize(std::stoi(line));
                i++;
            }
            else if (getting_nodes)
            {
                std::stringstream ss(line);
                std::string coord_as_string;
                int k = 0;
                while (std::getline(ss, coord_as_string, ' '))
                {
                    if (k == 1)
                    {
                        x_coords[i] = std::stod(coord_as_string);
                    }
                    else if (k == 2)
                    {
                        y_coords[i] = std::stod(coord_as_string);
                    }
                    k++;
                }
                i++;
            }
            else if (getting_elements)
            {
                std::stringstream ss(line);
                std::string idx_as_string;
                int k = 0;
                while (std::getline(ss, idx_as_string, ' '))
                {
                    if (k == 1)
                    {
                        int geom_type = std::stoi(idx_as_string);
                        if (geom_type != 2 && geom_type != 3)
                        {
                            break; // If element is not a quad or triangle go to next line
                        }
                        else if (!known_mesh_type)
                        {
                            if (geom_type == 2)
                            {
                                meshType_ = MeshType::TRIANGLE;

                            }
                            else if (geom_type == 3)
                            {
                                meshType_ = MeshType::QUAD;
                            }
                            known_mesh_type = true;
                            numElements_++;
                        }
                        else
                        {
                            numElements_++;
                        }
                    }
                    else if (k >= 5) 
                    {
                        element_nodes.push_back(std::stoi(idx_as_string) - 1);

                    }
                    k++;
                }
            }
        }
        input_file.close();
    }
    else 
    {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    // Resize view and populate with data from the file
    Kokkos::resize(data_, static_cast<int>(numElements_), static_cast<int>(meshType_), 3);
    for (int i=0; i<numElements_; i++) 
    {
        for (int j=0; j<meshType_; j++)
        {
            data_(i,j,0) = element_nodes[i*meshType_+j]; 
            data_(i,j,1) = x_coords[element_nodes[i*meshType_+j]];
            data_(i,j,2) = y_coords[element_nodes[i*meshType_+j]];
        }
    }
}