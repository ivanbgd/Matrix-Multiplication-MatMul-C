/* Matrices are represented as 1-D arrays in memory.
 * That means they are contiguous in memory.
 * Minimum dimension is 1, not 0, and internal dimensions must match. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Initializes vector or matrix, sequentially, with indices. */
void init_seq(double *a, const int n_rows_a, const int n_cols_a) {
    for (size_t i = 0; i < n_rows_a; i++) {
        for (size_t j = 0; j < n_cols_a; j++) {
            a[i*n_cols_a + j] = i*n_cols_a + j;
        }
    }
}

/* Initializes vector or matrix, randomly. */
void init_rand(double *a, const int n_rows_a, const int n_cols_a) {
    for (size_t i = 0; i < n_rows_a; i++) {
        for (size_t j = 0; j < n_cols_a; j++) {
            a[i*n_cols_a + j] = rand() / (double)RAND_MAX;
        }
    }
}

/* Dot product of two arrays, or matrix product
 * Allocates and returns an array. */
double *dot(const double *a, const int n_rows_a, const int n_cols_a, const double *b, const int n_rows_b, const int n_cols_b) {
    if (n_cols_a != n_rows_b) {
        printf("#columns A must be equal to #rows B!\n");
        system("pause");
        exit(-2);
    }

    double *c = malloc(n_rows_a * n_cols_b * sizeof(*c));
    if (c == NULL) {
        printf("Couldn't allocate memory!\n");
        system("pause");
        exit(-1);
    }

    for (size_t i = 0; i < n_rows_a; i++) {
        for (size_t k = 0; k < n_cols_b; k++) {
            double sum = 0.0;
            for (size_t j = 0; j < n_cols_a; j++) {
                sum += a[i*n_cols_a + j] * b[j*n_cols_b + k];
            }
            c[i*n_cols_b + k] = sum;
        }
    }

    return c;
}

/* Prints vector, or matrix. */
void print(const double *a, const int n_rows_a, const int n_cols_a) {
    for (size_t i = 0; i < n_rows_a; i++) {
        for (size_t j = 0; j < n_cols_a; j++) {
            printf("%8.3f ", a[i*n_cols_a + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char *argv[]) {
    /* Intializes random number generator */
    time_t t;
    srand((unsigned)time(&t));

    const int n_rows_a = 4;
    const int n_cols_a = 3;
    const int n_rows_b = 3;
    const int n_cols_b = 2;

    double *a = malloc(n_rows_a * n_cols_a * sizeof(*a));
    double *b = malloc(n_rows_b * n_cols_b * sizeof(*b));
    double *c = NULL;

    if (!a || !b) {
        printf("Couldn't allocate memory!\n");
        system("pause");
        exit(-1);
    }

    init_rand(a, n_rows_a, n_cols_a);
    init_rand(b, n_rows_b, n_cols_b);

    c = dot(a, n_rows_a, n_cols_a, b, n_rows_b, n_cols_b);

    printf("Matrix A:\n");
    print(a, n_rows_a, n_cols_a);
    printf("Matrix B:\n");
    print(b, n_rows_b, n_cols_b);
    printf("Matrix C:\n");
    print(c, n_rows_a, n_cols_b);

    free(a);
    free(b);
    free(c);

    system("pause");
    return(0);
}
