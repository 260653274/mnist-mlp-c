#pragma once

typedef struct {
    double *images;    /* [n * 784]  normalized [0,1], row-major */
    double *labels_oh; /* [n * 10]   one-hot float64 */
    int    *labels;    /* [n]        integer 0-9 */
    int     n;
} MNISTData;

/* Load MNIST IDX files.  Normalizes images to [0,1], builds one-hot labels. */
MNISTData mnist_load(const char *img_path, const char *lbl_path);
void      mnist_free(MNISTData *d);
