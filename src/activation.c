#include "activation.h"
#include <math.h>

void relu_forward(const Matrix *Z, Matrix *A) {
    assert(Z->rows == A->rows && Z->cols == A->cols);
    int n = Z->rows * Z->cols;
    for (int i = 0; i < n; i++)
        A->data[i] = Z->data[i] > 0.0 ? Z->data[i] : 0.0;
}

void relu_backward(const Matrix *Z, const Matrix *dA, Matrix *dZ) {
    assert(Z->rows == dA->rows && Z->cols == dA->cols);
    assert(dZ->rows == Z->rows && dZ->cols == Z->cols);
    int n = Z->rows * Z->cols;
    for (int i = 0; i < n; i++)
        dZ->data[i] = dA->data[i] * (Z->data[i] > 0.0 ? 1.0 : 0.0);
}

void softmax_forward(const Matrix *Z, Matrix *A) {
    assert(Z->rows == A->rows && Z->cols == A->cols);
    int rows = Z->rows, cols = Z->cols;
    for (int i = 0; i < rows; i++) {
        const double *z = Z->data + i*cols;
        double       *a = A->data + i*cols;

        double max_val = z[0];
        for (int j = 1; j < cols; j++)
            if (z[j] > max_val) max_val = z[j];

        double sum = 0.0;
        for (int j = 0; j < cols; j++) {
            a[j] = exp(z[j] - max_val);
            sum += a[j];
        }
        for (int j = 0; j < cols; j++)
            a[j] /= sum;
    }
}

double cross_entropy_loss(const Matrix *A, const Matrix *Y) {
    assert(A->rows == Y->rows && A->cols == Y->cols);
    int rows = A->rows, cols = A->cols;
    double loss = 0.0;
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            if (Y->data[i*cols + j] > 0.0)
                loss -= log(A->data[i*cols + j] + CE_EPS);
    return loss / rows;
}
