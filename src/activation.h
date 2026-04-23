#pragma once
#include "matrix.h"

#define CE_EPS 1e-8

/* A = relu(Z),  element-wise */
void relu_forward(const Matrix *Z, Matrix *A);

/* dZ = dA * (Z > 0),  element-wise (ReLU backward) */
void relu_backward(const Matrix *Z, const Matrix *dA, Matrix *dZ);

/* A = softmax(Z),  numerically stable row-wise */
void softmax_forward(const Matrix *Z, Matrix *A);

/* mean cross-entropy: -mean_i sum_j Y[i][j]*log(A[i][j]+eps) */
double cross_entropy_loss(const Matrix *A, const Matrix *Y);
