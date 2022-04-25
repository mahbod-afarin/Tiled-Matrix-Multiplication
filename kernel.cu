/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>

#define TILE_SIZE 16

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     * Use shared memory for tiling
     *
     ********************************************************************/

    // INSERT KERNEL CODE HERE

    __shared__ float Ashared[TILE_SIZE][TILE_SIZE];
    __shared__ float Bshared[TILE_SIZE][TILE_SIZE];
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * TILE_SIZE + ty;
    int Col = bx * TILE_SIZE + tx;
    float temp = 0.0;

    for (int x = 0; x < (k-1 + TILE_SIZE)/TILE_SIZE ; ++x) 
    {
        if (Row < m && (x*TILE_SIZE+tx)<k )
        {
            Ashared[ty][tx] = A[Row*k + x*TILE_SIZE+tx];
        }
        else
        {
            Ashared[ty][tx] = 0;
        }

        if ((x*TILE_SIZE +ty)< k && Col< n )
        {
            Bshared[ty][tx] = B[(x*TILE_SIZE +ty)*n + Col];
        }
        else
        {
            Bshared[ty][tx] = 0;
        }

        __syncthreads();
        for (int y = 0; y < TILE_SIZE; ++y)
            temp += Ashared[ty][y] * Bshared[y][tx];
        __syncthreads();

    }

    if (Row<m && Col<n)
    {
        C[Row*n+Col] = temp;
    }
}

void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
    if ((transa != 'N') && (transa != 'n')) {
	printf("unsupported value of 'transa'\n");
    	return;
    }

    if ((transb != 'N') && (transb != 'n')) {
	printf("unsupported value of 'transb'\n");
	return;
    }

    if ((alpha - 1.0f > 1e-10) || (alpha - 1.0f < -1e-10)) {
	printf("unsupported value of alpha\n");
	return;
    }

    if ((beta - 0.0f > 1e-10) || (beta - 0.0f < -1e-10)) {
	printf("unsupported value of beta\n");
	return;
    }

    // Initialize thread block and kernel grid dimensions ---------------------

    const unsigned int BLOCK_SIZE = TILE_SIZE;

    //INSERT CODE HERE

    dim3 dimGrid((n+BLOCK_SIZE-1)/BLOCK_SIZE,(m+BLOCK_SIZE-1)/BLOCK_SIZE,1);
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE,1);

    // Invoke CUDA kernel -----------------------------------------------------

    //INSERT CODE HERE
    mysgemm<<<dimGrid,dimBlock>>>(m, n, k, A, B, C);
}


