#include "nn.h"
#include "activation.h"
#include <math.h>
#include <string.h>

Params params_create(int hidden) {
    Params p;
    p.hidden = hidden;
    p.W1 = mat_create(784,    hidden);
    p.b1 = mat_create(1,      hidden);
    p.W2 = mat_create(hidden, 10);
    p.b2 = mat_create(1,      10);
    return p;
}

void params_free(Params *p) {
    mat_free(&p->W1); mat_free(&p->b1);
    mat_free(&p->W2); mat_free(&p->b2);
}

WorkSpace ws_create(int bs, int hidden) {
    WorkSpace ws;
    ws.X   = mat_create(bs, 784);
    ws.Y   = mat_create(bs, 10);
    ws.Z1  = mat_create(bs, hidden);
    ws.A1  = mat_create(bs, hidden);
    ws.Z2  = mat_create(bs, 10);
    ws.A2  = mat_create(bs, 10);
    ws.dZ2 = mat_create(bs, 10);
    ws.dA1 = mat_create(bs, hidden);
    ws.dZ1 = mat_create(bs, hidden);
    ws.dW1 = mat_create(784,    hidden);
    ws.db1 = mat_create(1,      hidden);
    ws.dW2 = mat_create(hidden, 10);
    ws.db2 = mat_create(1,      10);
    return ws;
}

void ws_free(WorkSpace *ws) {
    mat_free(&ws->X);  mat_free(&ws->Y);
    mat_free(&ws->Z1); mat_free(&ws->A1);
    mat_free(&ws->Z2); mat_free(&ws->A2);
    mat_free(&ws->dZ2);mat_free(&ws->dA1);mat_free(&ws->dZ1);
    mat_free(&ws->dW1);mat_free(&ws->db1);
    mat_free(&ws->dW2);mat_free(&ws->db2);
}

void params_he_init(Params *p, RNG *rng) {
    double std1 = sqrt(2.0 / 784);
    for (int i = 0; i < p->W1.rows * p->W1.cols; i++)
        p->W1.data[i] = rng_normal(rng) * std1;
    mat_zero(&p->b1);

    double std2 = sqrt(2.0 / p->hidden);
    for (int i = 0; i < p->W2.rows * p->W2.cols; i++)
        p->W2.data[i] = rng_normal(rng) * std2;
    mat_zero(&p->b2);
}

/* Z1 = X*W1 + b1,  A1 = relu(Z1)
   Z2 = A1*W2 + b2, A2 = softmax(Z2) */
void forward(const Params *p, WorkSpace *ws) {
    mat_mul(&ws->X, &p->W1, &ws->Z1);
    mat_add_bias(&ws->Z1, &p->b1);
    relu_forward(&ws->Z1, &ws->A1);

    mat_mul(&ws->A1, &p->W2, &ws->Z2);
    mat_add_bias(&ws->Z2, &p->b2);
    softmax_forward(&ws->Z2, &ws->A2);
}

/* dZ2 = (A2-Y)/bs
   dW2 = A1^T * dZ2,  db2 = colsum(dZ2)
   dA1 = dZ2 * W2^T
   dZ1 = dA1 .* relu'(Z1)
   dW1 = X^T * dZ1,   db1 = colsum(dZ1) */
void backward(const Params *p, WorkSpace *ws) {
    int bs = ws->X.rows;

    mat_sub(&ws->A2, &ws->Y, &ws->dZ2);
    mat_scale(&ws->dZ2, 1.0 / bs);

    mat_mul_ta(&ws->A1,  &ws->dZ2, &ws->dW2);
    mat_col_sum(&ws->dZ2, &ws->db2);

    mat_mul_tb(&ws->dZ2, &p->W2,  &ws->dA1);
    relu_backward(&ws->Z1, &ws->dA1, &ws->dZ1);

    mat_mul_ta(&ws->X,   &ws->dZ1, &ws->dW1);
    mat_col_sum(&ws->dZ1, &ws->db1);
}

void params_update(Params *p, const WorkSpace *ws, double lr) {
    mat_axpy(&p->W1, -lr, &ws->dW1);
    mat_axpy(&p->b1, -lr, &ws->db1);
    mat_axpy(&p->W2, -lr, &ws->dW2);
    mat_axpy(&p->b2, -lr, &ws->db2);
}

double test_accuracy(const Params *p, const double *images, const int *labels,
                     int n, int chunk_size, int hidden) {
    Matrix Xc = mat_create(chunk_size, 784);
    Matrix Z1 = mat_create(chunk_size, hidden);
    Matrix A1 = mat_create(chunk_size, hidden);
    Matrix Z2 = mat_create(chunk_size, 10);
    Matrix A2 = mat_create(chunk_size, 10);

    int correct = 0;
    for (int start = 0; start < n; start += chunk_size) {
        int end = start + chunk_size;
        if (end > n) end = n;
        int cs = end - start;

        /* copy chunk into Xc (may be smaller than chunk_size at last batch) */
        memcpy(Xc.data, images + (size_t)start * 784, (size_t)cs * 784 * sizeof(double));

        /* temporary smaller view — avoid realloc by just setting rows */
        Matrix Xv = { cs, 784,    Xc.data };
        Matrix Z1v= { cs, hidden, Z1.data };
        Matrix A1v= { cs, hidden, A1.data };
        Matrix Z2v= { cs, 10,     Z2.data };
        Matrix A2v= { cs, 10,     A2.data };

        mat_mul(&Xv, &p->W1, &Z1v);
        mat_add_bias(&Z1v, &p->b1);
        relu_forward(&Z1v, &A1v);
        mat_mul(&A1v, &p->W2, &Z2v);
        mat_add_bias(&Z2v, &p->b2);
        softmax_forward(&Z2v, &A2v);

        for (int i = 0; i < cs; i++) {
            double *row = A2v.data + i * 10;
            int pred = 0;
            for (int j = 1; j < 10; j++)
                if (row[j] > row[pred]) pred = j;
            if (pred == labels[start + i]) correct++;
        }
    }

    mat_free(&Xc); mat_free(&Z1); mat_free(&A1);
    mat_free(&Z2); mat_free(&A2);
    return (double)correct / n;
}
