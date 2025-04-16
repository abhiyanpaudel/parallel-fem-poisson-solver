#ifndef COMPUTING_AT_SCALE_ASSIGNMENT_CALCULATE_STIFFNESS_MATRIX_AND_LOAD_VECTOR_HPP 
#define COMPUTING_AT_SCALE_ASSIGNMENT_CALCULATE_STIFFNESS_MATRIX_AND_LOAD_VECTOR_HPP 

#include "TriElement.hpp"
#include "QuadElement.hpp"

constexpr static int MAX_STIFFNESS_MATRIX_SIZE = 16;
constexpr static int MAX_LOAD_VECTOR_SIZE = 4;


struct Results {
	View1D allElementStiffnessMatrix;
	View1D allElementLoadVector;

};


Results calculateAllElementStiffnessMatrixAndLoadVector(const Mesh& mesh, double k = 1.0);

#endif 
