#pragma once

/* All kernel launchers operate on device pointers (float32). */

/* element-wise: A[i] = max(0, Z[i]) */
void launch_relu_forward(const float *Z, float *A, int n);
/* element-wise: dZ[i] = (Z[i] > 0) ? dA[i] : 0 */
void launch_relu_backward(const float *Z, const float *dA, float *dZ, int n);
/* row-wise stable softmax. cols must be <= 32 (we use one warp per row). */
void launch_softmax_forward(const float *Z, float *A, int rows, int cols);
/* A[i,j] += b[j]  for every i */
void launch_add_bias(float *A, const float *b, int rows, int cols);
/* out[j] = sum_i A[i,j] */
void launch_col_sum(const float *A, float *out, int rows, int cols);
/* A[i] += alpha * B[i] */
void launch_axpy(float *A, float alpha, const float *B, int n);
/* C[i] = (A[i] - B[i]) * scale */
void launch_sub_scale(const float *A, const float *B, float scale, float *C, int n);
/* atomicAdd to *loss_sum the value -sum_i sum_j Y[i,j] * log(A[i,j] + eps) */
void launch_ce_loss(const float *A, const float *Y, float *loss_sum,
                    int rows, int cols, float eps);
/* Row gather: dst[b, :] = src[idx[b], :], dst (bs, cols) */
void launch_gather_rows(const float *src, const int *idx, float *dst,
                        int bs, int cols);
/* Count rows whose argmax equals label[offset+i], atomicAdd into *correct */
void launch_argmax_count(const float *A, const int *labels, int *correct,
                         int rows, int cols, int offset);
