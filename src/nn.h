#pragma once
#include "matrix.h"
#include "rng.h"

/* Network parameters: 784 -> hidden -> 10 */
typedef struct {
    Matrix W1, b1;  /* (784,H) (1,H) */
    Matrix W2, b2;  /* (H,10)  (1,10) */
    int hidden;
} Params;

/* Scratch space pre-allocated for one mini-batch */
typedef struct {
    Matrix X, Y;            /* (bs,784)  (bs,10) — batch copies */
    Matrix Z1, A1, Z2, A2; /* forward intermediates */
    Matrix dZ2, dA1, dZ1;  /* backward intermediates */
    Matrix dW1, db1, dW2, db2; /* gradients */
} WorkSpace;

/* Allocate / free */
Params    params_create(int hidden);
void      params_free(Params *p);
WorkSpace ws_create(int batch_size, int hidden);
void      ws_free(WorkSpace *ws);

/* He initialisation: W ~ N(0, sqrt(2/fan_in)) */
void params_he_init(Params *p, RNG *rng);

/* Forward: fills ws->Z1,A1,Z2,A2 from ws->X and p */
void forward(const Params *p, WorkSpace *ws);

/* Backward: fills ws->dW1,db1,dW2,db2 from ws->Y,Z1,A1,A2,W2 */
void backward(const Params *p, WorkSpace *ws);

/* SGD update: p -= lr * gradients */
void params_update(Params *p, const WorkSpace *ws, double lr);

/* Test-set accuracy, evaluated in chunks of chunk_size */
double test_accuracy(const Params *p, const double *images, const int *labels,
                     int n, int chunk_size, int hidden);
