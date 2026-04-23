#include "data.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* MNIST IDX magic numbers */
#define MAGIC_IMAGES 0x00000803
#define MAGIC_LABELS 0x00000801

static uint32_t read_be32(FILE *f) {
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4) { fputs("IDX read error\n", stderr); exit(1); }
    return ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16)
         | ((uint32_t)b[2] <<  8) |  (uint32_t)b[3];
}

MNISTData mnist_load(const char *img_path, const char *lbl_path) {
    MNISTData d;

    /* --- images --- */
    FILE *fi = fopen(img_path, "rb");
    if (!fi) { fprintf(stderr, "cannot open %s\n", img_path); exit(1); }

    uint32_t magic = read_be32(fi);
    if (magic != MAGIC_IMAGES) { fprintf(stderr, "bad image magic: %u\n", magic); exit(1); }
    uint32_t n_img = read_be32(fi);
    uint32_t rows  = read_be32(fi);
    uint32_t cols  = read_be32(fi);
    if (rows != 28 || cols != 28) { fputs("unexpected image dimensions\n", stderr); exit(1); }

    unsigned char *raw = (unsigned char *)malloc(n_img * 784);
    if (!raw || fread(raw, 1, n_img * 784, fi) != n_img * 784)
        { fputs("image read error\n", stderr); exit(1); }
    fclose(fi);

    d.n      = (int)n_img;
    d.images = (double *)malloc((size_t)n_img * 784 * sizeof(double));
    for (size_t i = 0; i < (size_t)n_img * 784; i++)
        d.images[i] = raw[i] / 255.0;
    free(raw);

    /* --- labels --- */
    FILE *fl = fopen(lbl_path, "rb");
    if (!fl) { fprintf(stderr, "cannot open %s\n", lbl_path); exit(1); }

    magic = read_be32(fl);
    if (magic != MAGIC_LABELS) { fprintf(stderr, "bad label magic: %u\n", magic); exit(1); }
    uint32_t n_lbl = read_be32(fl);
    if (n_lbl != n_img) { fputs("image/label count mismatch\n", stderr); exit(1); }

    unsigned char *lraw = (unsigned char *)malloc(n_lbl);
    if (!lraw || fread(lraw, 1, n_lbl, fl) != n_lbl)
        { fputs("label read error\n", stderr); exit(1); }
    fclose(fl);

    d.labels    = (int *)malloc(n_lbl * sizeof(int));
    d.labels_oh = (double *)calloc((size_t)n_lbl * 10, sizeof(double));
    for (uint32_t i = 0; i < n_lbl; i++) {
        d.labels[i] = (int)lraw[i];
        d.labels_oh[(size_t)i * 10 + lraw[i]] = 1.0;
    }
    free(lraw);

    return d;
}

void mnist_free(MNISTData *d) {
    free(d->images);   d->images    = NULL;
    free(d->labels);   d->labels    = NULL;
    free(d->labels_oh);d->labels_oh = NULL;
    d->n = 0;
}
