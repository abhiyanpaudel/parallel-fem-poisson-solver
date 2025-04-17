#ifndef COMPUTING_AT_SCALE_ASSIGNMENT_QUAD_ELEMENT_HPP
#define COMPUTING_AT_SCALE_ASSIGNMENT_QUAD_ELEMENT_HPP

#include "Element.hpp"

class QuadElement : public Element {
 private:
  // Gauss quadrature points and weights for quads
  static constexpr int numQuadPoints_ = 4;
  static constexpr int numNodes_ = 4;
  static constexpr double gaussPoint_ = 0.57735026919;  // 1/sqrt(3)

  // Get quadrature point coordinates and weight
  KOKKOS_INLINE_FUNCTION
  void getQuadPoint(int q, double& xi, double& eta, double& weight) const {
    if (q == 0) {
      xi = -gaussPoint_;
      eta = -gaussPoint_;
      weight = 1.0;
    } else if (q == 1) {
      xi = gaussPoint_;
      eta = -gaussPoint_;
      weight = 1.0;
    } else if (q == 2) {
      xi = gaussPoint_;
      eta = gaussPoint_;
      weight = 1.0;
    } else if (q == 3) {
      xi = -gaussPoint_;
      eta = gaussPoint_;
      weight = 1.0;
    }
  }

 public:
  KOKKOS_INLINE_FUNCTION
  QuadElement(const Mesh& mesh, int elemIdx) : Element(mesh, elemIdx) {}

  KOKKOS_INLINE_FUNCTION
  int getNumNodes() const override { return numNodes_; }

  KOKKOS_INLINE_FUNCTION
  double computeLocalBasisFunction(const int node, const double xi,
                                   const double eta) const override {
    switch (node) {
      case 0:
        return 0.25 * (1.0 - xi) * (1.0 - eta);
      case 1:
        return 0.25 * (1.0 + xi) * (1.0 - eta);
      case 2:
        return 0.25 * (1.0 + xi) * (1.0 + eta);
      case 3:
        return 0.25 * (1.0 - xi) * (1.0 + eta);
      default:
        return -10000.0;  // Error case
    }
  }

  KOKKOS_INLINE_FUNCTION
  void computeBasisGradient(const int node, const double xi, const double eta,
                            double& dN_dxi, double& dN_deta) const {
    switch (node) {
      case 0:
        dN_dxi = -0.25 * (1.0 - eta);
        dN_deta = -0.25 * (1.0 - xi);
        break;
      case 1:
        dN_dxi = 0.25 * (1.0 - eta);
        dN_deta = -0.25 * (1.0 + xi);
        break;
      case 2:
        dN_dxi = 0.25 * (1.0 + eta);
        dN_deta = 0.25 * (1.0 + xi);
        break;
      case 3:
        dN_dxi = -0.25 * (1.0 + eta);
        dN_deta = 0.25 * (1.0 - xi);
        break;
    }
  }

  KOKKOS_INLINE_FUNCTION
  double computeJacobian(const double xi, const double eta) const override {
    // For quads, the Jacobian varies by position
    double x[4], y[4];
    for (int i = 0; i < 4; i++) {
      x[i] = mesh_.GetCoordinate(elemIdx_, i, 0);
      y[i] = mesh_.GetCoordinate(elemIdx_, i, 1);
    }

    // Compute derivatives of x and y w.r.t. local coordinates
    double dxdxi = 0.0, dxdeta = 0.0, dydxi = 0.0, dydeta = 0.0;

    for (int i = 0; i < 4; i++) {
      double dN_dxi, dN_deta;
      computeBasisGradient(i, xi, eta, dN_dxi, dN_deta);

      dxdxi += x[i] * dN_dxi;
      dxdeta += x[i] * dN_deta;
      dydxi += y[i] * dN_dxi;
      dydeta += y[i] * dN_deta;
    }

    return dxdxi * dydeta - dxdeta * dydxi;
  }

  KOKKOS_INLINE_FUNCTION
  void computeElementStiffnessMatrix(double* stiffness) const override {
    // Initialize stiffness matrix
    for (int i = 0; i < numNodes_ * numNodes_; i++) {
      stiffness[i] = 0.0;
    }

    // Get coordinates of quadrilateral vertices
    double x[4], y[4];
    for (int i = 0; i < 4; i++) {
      x[i] = mesh_.GetCoordinate(elemIdx_, i, 0);
      y[i] = mesh_.GetCoordinate(elemIdx_, i, 1);
    }

    // Integrate using Gauss quadrature
    for (int q = 0; q < numQuadPoints_; q++) {
      double xi, eta, weight;
      getQuadPoint(q, xi, eta, weight);

      // Compute Jacobian at this quadrature point
      double dxdxi = 0.0, dxdeta = 0.0, dydxi = 0.0, dydeta = 0.0;

      for (int n = 0; n < numNodes_; n++) {
        double dN_dxi, dN_deta;
        computeBasisGradient(n, xi, eta, dN_dxi, dN_deta);

        dxdxi += x[n] * dN_dxi;
        dxdeta += x[n] * dN_deta;
        dydxi += y[n] * dN_dxi;
        dydeta += y[n] * dN_deta;
      }

      double det_J = dxdxi * dydeta - dxdeta * dydxi;

      // compute inverse of the jacobian
      double invJ = 1 / det_J;

      // Compute contribution to stiffness matrix
      for (int i = 0; i < numNodes_; i++) {
        double dNi_dxi, dNi_deta;
        computeBasisGradient(i, xi, eta, dNi_dxi, dNi_deta);

        double dNi_dx = dydeta * dNi_dxi - dydxi * dNi_deta;
        double dNi_dy = -dxdeta * dNi_dxi + dxdxi * dNi_deta;

        for (int j = 0; j < numNodes_; j++) {
          double dNj_dxi, dNj_deta;
          computeBasisGradient(j, xi, eta, dNj_dxi, dNj_deta);

          double dNj_dx = dydeta * dNj_dxi - dydxi * dNj_deta;
          double dNj_dy = -dxdeta * dNj_dxi + dxdxi * dNj_deta;

          stiffness[i * numNodes_ + j] +=
              (dNi_dx * dNj_dx + dNi_dy * dNj_dy) * invJ * weight;
        }
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void computeElementLoadVector(double* load) const override {
    // Create load vector (4 entries)

    // Initialize load vector
    for (int i = 0; i < numNodes_; i++) {
      load[i] = 0.0;
    }

    double f = 1.0;

    // Integrate load using quadrature
    for (int q = 0; q < numQuadPoints_; q++) {
      double xi, eta, weight;
      getQuadPoint(q, xi, eta, weight);

      double det_J = computeJacobian(xi, eta);
      double abs_det_J = det_J > 0 ? det_J : -det_J;

      for (int i = 0; i < numNodes_; i++) {
        double phi = computeLocalBasisFunction(i, xi, eta);
        load[i] += phi * f * weight * abs_det_J;
      }
    }
  }
};

#endif  // COMPUTING_AT_SCALE_ASSIGNMENT_QUAD_ELEMENT_HPP
