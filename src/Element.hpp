// created by Abhiyan Paudel 

#ifndef COMPUTING_AT_SCALE_ASSIGNMENT_ELEMENT_HPP
#define COMPUTING_AT_SCALE_ASSIGNMENT_ELEMENT_HPP

#include <Kokkos_Core.hpp>
#include "Mesh.h"

using View1D = Kokkos::View<double*>;

class Element {
protected:
    const Mesh& mesh_;
    int elemIdx_;
	double k_;

public:
	KOKKOS_INLINE_FUNCTION
    Element(const Mesh& mesh, int elemIdx, double k = 1.0) : mesh_(mesh), elemIdx_(elemIdx), k_(k) {}
   
	KOKKOS_INLINE_FUNCTION
	virtual ~Element() {}

    // Get number of nodes per element
    KOKKOS_INLINE_FUNCTION
    virtual int getNumNodes() const = 0;
    
	// Compute local basis function
    KOKKOS_INLINE_FUNCTION
    virtual double computeLocalBasisFunction(const int node, const double xi, const double eta) const = 0;

    // Compute jacobian for an element
    KOKKOS_INLINE_FUNCTION
    virtual double computeJacobian(const double xi, const double eta) const = 0;

    // Compute element stiffness matrix
    KOKKOS_INLINE_FUNCTION
    virtual void computeElementStiffnessMatrix(double* stiffness) const = 0;

    // Compute element load vector
    KOKKOS_INLINE_FUNCTION
    virtual void computeElementLoadVector(double* load) const = 0;

	// Set material property k
	KOKKOS_INLINE_FUNCTION
	void setMaterialProperty(double k) {k_ = k;}

	// Get material property k
	KOKKOS_INLINE_FUNCTION
	void getMaterialProperty(double k) {k_ = k;}
	
    
};

#endif // COMPUTING_AT_SCALE_ASSIGNMENT_ELEMENT_HPP
