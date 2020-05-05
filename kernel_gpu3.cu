
#include <assert.h>
#include <stdio.h>

#include "kernel.h"
#include "matrix.h"
#include "timer.h"

#define THRESHOLD 0.000001
#define YMAX 32
#define BLOCKDIM 32




__global__ void spmspm(COOMatrix *result, CSRTiledMatrix *A, CSCTiledMatrix *B, float bias, unsigned int K, unsigned int N, unsigned int M) {
    __shared__ unsigned int As_rowPtrs[BLOCKDIM + 1];
    __shared__ unsigned int As_colIdxs[BLOCKDIM*BLOCKDIM];
    __shared__ float        As_values[BLOCKDIM*BLOCKDIM];
    
    __shared__ unsigned int Bs_colPtrs[BLOCKDIM + 1];
    __shared__ unsigned int Bs_rowIdxs[BLOCKDIM*BLOCKDIM];
    __shared__ float        Bs_values[BLOCKDIM*BLOCKDIM];

    unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int totalRows = A->numRows; 
    unsigned int totalCols = B->numCols;

    unsigned int tilesPerDim = K / BLOCKDIM;
    float sum = 0.0f;
    
    for(unsigned int tile = 0; tile < tilesPerDim; ++tile) {
        // 1 thread loads a row and a col.
        if(threadIdx.x == threadIdx.y){
            unsigned int nnz_row = 0;
            unsigned int nnz_col = 0;

            if(row < totalRows){
                As_rowPtrs[threadIdx.y] = A->rowPtrsBlock[row*tilesPerDim + tile];
                for(unsigned int i = A->rowPtrs[row*tilesPerDim + tile]; i < A->rowPtrs[row*tilesPerDim + tile + 1]; ++i){
                    As_colIdxs[nnz_row] = A->colIdxs[i];
                    As_values[nnz_row]  = A->values[i];
                    nnz_row += 1;
                }
            }
            
            if(col < totalCols){
                Bs_colPtrs[threadIdx.x] = B->colPtrsBlock[row*tilesPerDim + tile];
                for(unsigned int i = B->colPtrs[col*tilesPerDim + tile]; i < B->colPtrs[col*tilesPerDim + tile + 1]; ++i){
                    Bs_rowIdxs[nnz_col] = B->rowIdxs[i];
                    Bs_values[nnz_col]  = B->values[i];
                    nnz_col += 1;
                }
            }
            
        } else if(threadIdx.x == 0 && threadIdx.y == 1) {
            if(blockDim.y * (blockIdx.y + 1) > totalRows) {
                As_rowPtrs[totalRows % BLOCKDIM] = A->rowPtrsBlock[(totalRows - 1)*tilesPerDim + tile] + (A->rowPtrs[(totalRows - 1)*tilesPerDim + tile + 1] - A->rowPtrs[(totalRows - 1)*tilesPerDim + tile]);
            } else {
                As_rowPtrs[BLOCKDIM] = 
                A->rowPtrsBlock[(blockDim.y * (blockIdx.y + 1) - 1)*tilesPerDim + tile] + 
                (
                    A->rowPtrs[(blockDim.y * (blockIdx.y + 1) - 1)*tilesPerDim + tile + 1] -
                    A->rowPtrs[(blockDim.y * (blockIdx.y + 1) - 1)*tilesPerDim + tile]
                );
            }
            
            if(blockDim.x * (blockIdx.x + 1) > totalCols) {
                Bs_colPtrs[totalCols % BLOCKDIM] = B->colPtrsBlock[(totalCols - 1)*tilesPerDim + tile] + (B->colPtrs[(totalCols - 1)*tilesPerDim + tile + 1] - B->colPtrs[(totalCols - 1)*tilesPerDim + tile]);
            } else {
                Bs_colPtrs[BLOCKDIM] = 
                B->colPtrsBlock[(blockDim.x * (blockIdx.x + 1) - 1)*tilesPerDim + tile] + 
                (
                    B->colPtrs[(blockDim.x * (blockIdx.x + 1) - 1)*tilesPerDim + tile + 1] - 
                    B->colPtrs[(blockDim.x * (blockIdx.x + 1) - 1)*tilesPerDim + tile]
                );
            }      
        }
        __syncthreads();

        // Compute with tile
        if(row < M && col < N){
            unsigned int nnzA = As_rowPtrs[threadIdx.y + 1] - A->rowPtrs[threadIdx.y];
            unsigned int nnzB = Bs_colPtrs[threadIdx.x + 1] - B->colPtrs[threadIdx.x];
            if(nnzA > 0 && nnzB > 0) {
                unsigned int rowPtrA = As_rowPtrs[threadIdx.y];
                unsigned int colPtrB = Bs_colPtrs[threadIdx.x];
    
                unsigned int* colIdxsA = As_colIdxs + rowPtrA; 
                unsigned int* rowIdxsB = Bs_rowIdxs + colPtrB;
    
                float* valueA = As_values + rowPtrA; //Ptr to first value in A 
                float* valueB = Bs_values + colPtrB; //Ptr to first value in B
    
                // Loop and find intersection
                unsigned int ia = 0;
                unsigned int ib = 0;
                while(ia < nnzA && ib < nnzB) {
                    unsigned int colIdx = colIdxsA[ia]; 
                    unsigned int rowIdx = rowIdxsB[ib]; 
                    if(colIdx < rowIdx) {
                        ia++;
                    } else if(colIdx > rowIdx) {
                        ib++;
                    } else {
                        sum += valueA[ia] * valueB[ib];
                        ia++;
                        ib++;
                    }
                }      
            }
        }
        __syncthreads(); 
    }

    //Relu
    if(sum > THRESHOLD || sum < -THRESHOLD) {
        sum += bias;
        if(sum > 0) {
            unsigned int nnzIdx = atomicAdd(&result->nnz, 1);
            if(sum > YMAX) {
                sum = YMAX;
            }
            if(nnzIdx >= result->capacity) {
                printf("WE RAN OUT OF CAPACITY\n");
            }
            result->colIdxs[nnzIdx] = col;
            result->rowIdxs[nnzIdx] = row;
            result->values[nnzIdx]  = sum;
        }    
    }  
}




void findNonzeroRows(Vector* v, CSRTiledMatrix* A) {
    unsigned int nnz = 0;
    for(unsigned int r = 0; r < A->numRows; ++r) {
        unsigned int rowPtrA = A->rowPtrs[r];
        unsigned int nnzA = A->rowPtrs[r + 1] - rowPtrA;
        if(nnzA > 0) {
            if(nnz >= v->capacity) {
                expandVectorCapacity(v, 2 * v->capacity);
            }
            v->data[nnz] = r;
            ++nnz;
        }
    }
    v->nnz = nnz;
}




// COO Functions
COOMatrix* createEmptyCOO_d(unsigned int numRows, unsigned int numCols, unsigned int capacity) {
    COOMatrix cooShadow;

    cooShadow.numRows = numRows;
    cooShadow.numCols = numCols;
    cooShadow.nnz = 0;
    cooShadow.capacity = capacity;

    cudaMalloc((void**) &cooShadow.rowIdxs, capacity * sizeof(unsigned int));
    cudaMalloc((void**) &cooShadow.colIdxs, capacity * sizeof(unsigned int));
    cudaMalloc((void**) &cooShadow.values,  capacity * sizeof(float));

    COOMatrix* coo_d;
    cudaMalloc((void**) &coo_d, sizeof(COOMatrix));
    cudaMemcpy(coo_d, &cooShadow, sizeof(COOMatrix), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    return coo_d;
}

void copyCOOfromGPU(COOMatrix* coo_d, COOMatrix* coo) {
    COOMatrix cooShadow;
    cudaMemcpy(&cooShadow, coo_d, sizeof(COOMatrix), cudaMemcpyDeviceToHost);
    
    assert(coo->numRows  == cooShadow.numRows);
    assert(coo->numCols  == cooShadow.numCols);
    assert(coo->capacity >= cooShadow.nnz);
    coo->nnz = cooShadow.nnz;

    cudaMemcpy(coo->rowIdxs, cooShadow.rowIdxs, cooShadow.nnz * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(coo->colIdxs, cooShadow.colIdxs, cooShadow.nnz * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(coo->values,  cooShadow.values,  cooShadow.nnz * sizeof(float),        cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
}

COOMatrix* createEmptyCOO_pinned(unsigned int numRows, unsigned int numCols, unsigned int capacity) {
    COOMatrix *coo; cudaMallocHost((void**)&coo,sizeof(COOMatrix));

    coo->numRows = numRows;
    coo->numCols = numCols;
    coo->nnz = 0;
    coo->capacity = capacity;

    cudaMallocHost((void**) &coo->colIdxs, capacity * sizeof(unsigned int));
    cudaMallocHost((void**) &coo->values,  capacity * sizeof(float));
    cudaMallocHost((void**) &coo->rowIdxs, capacity * sizeof(unsigned int));

    for(int i = 0; i<capacity; ++i){
        coo->rowIdxs[i] = 0;
    }

    return coo;
}

void freeCOO_pinned(COOMatrix* coo) {
    cudaFreeHost(coo->rowIdxs);
    cudaFreeHost(coo->colIdxs);
    cudaFreeHost(coo->values);
    cudaFreeHost(coo);
}




// CSR Functions
CSRTiledMatrix* createEmptyCSRTiled_d(unsigned int numRows, unsigned int numCols, unsigned int capacity) {
    CSRTiledMatrix csrShadow;

    csrShadow.numRows = numRows;
    csrShadow.numCols = numCols;
    csrShadow.nnz = 0;
    csrShadow.capacity = capacity;
    unsigned int tilesPerRow = (numCols+BLOCKDIM-1)/BLOCKDIM;

    cudaMalloc((void**) &csrShadow.rowPtrs,      ((numRows * tilesPerRow) + 1) * sizeof(unsigned int));
    cudaMalloc((void**) &csrShadow.rowPtrsBlock, ((numRows * tilesPerRow) + 1) * sizeof(unsigned int));
    cudaMalloc((void**) &csrShadow.colIdxs,      capacity * sizeof(unsigned int));
    cudaMalloc((void**) &csrShadow.values,       capacity * sizeof(float));

    CSRTiledMatrix* csr_d;
    cudaMalloc((void**) &csr_d,   sizeof(CSRTiledMatrix));
    cudaMemcpy(csr_d, &csrShadow, sizeof(CSRTiledMatrix), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    return csr_d;
}

void copyCSRTiledtoGPU(CSRTiledMatrix* csr, CSRTiledMatrix* csr_d) {
    unsigned int tilesPerDim = (csr->numCols+BLOCKDIM-1)/BLOCKDIM;

    CSRTiledMatrix csrShadow;
    cudaMemcpy(&csrShadow, csr_d, sizeof(CSRTiledMatrix), cudaMemcpyDeviceToHost);

    assert(csrShadow.numRows  == csr->numRows);
    assert(csrShadow.numCols  == csr->numCols);
    assert(csrShadow.capacity >= csr->nnz);
    csrShadow.nnz = csr->nnz;

    cudaMemcpy(csrShadow.rowPtrs, csr->rowPtrs,      ((csr->numRows * tilesPerDim) + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(csrShadow.rowPtrs, csr->rowPtrsBlock, ((csr->numRows * tilesPerDim) + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(csrShadow.colIdxs, csr->colIdxs,      csr->nnz * sizeof(unsigned int),                           cudaMemcpyHostToDevice);
    cudaMemcpy(csrShadow.values,  csr->values,       csr->nnz * sizeof(float),                                  cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
}

CSRTiledMatrix* createEmptyCSRTiled_pinned(unsigned int numRows, unsigned int numCols, unsigned int capacity) {
    CSRTiledMatrix *csr; 
    cudaMallocHost((void**) &csr, sizeof(CSRTiledMatrix));

    csr->numRows = numRows;
    csr->numCols = numCols;
    csr->nnz = 0;
    csr->capacity = capacity;
    unsigned int tilesPerRow = (numCols+BLOCKDIM-1)/BLOCKDIM;

    cudaMallocHost((void**) &csr->rowPtrs,      (numRows * tilesPerRow + 1) * sizeof(unsigned int));
    cudaMallocHost((void**) &csr->rowPtrsBlock, (numRows * tilesPerRow + 1) * sizeof(unsigned int));
    cudaMallocHost((void**) &csr->colIdxs,      capacity * sizeof(unsigned int));
    cudaMallocHost((void**) &csr->values,       capacity * sizeof(float));
    for(int i = 0; i < numRows * tilesPerRow + 1; ++i){
        csr->rowPtrsBlock[i] = 0;
        csr->rowPtrs[i]      = 0;
    }
    
    return csr;
}

void freeCSRTiled_pinned(CSRTiledMatrix* csr) {
    cudaFreeHost(csr->rowPtrs);
    cudaFreeHost(csr->rowPtrsBlock);
    cudaFreeHost(csr->colIdxs);
    cudaFreeHost(csr->values);
    cudaFreeHost(csr);
}




// CSC Functions
CSCTiledMatrix* createCSCfromCSCTiled_d(CSCTiledMatrix* csc) {
    CSCTiledMatrix cscShadow;

    cscShadow.numRows = csc->numRows;
    cscShadow.numCols = csc->numCols;
    cscShadow.nnz = csc->nnz;
    cscShadow.capacity = csc->capacity;

    unsigned int tilesPerCol = (csc->numRows + BLOCKDIM - 1) /BLOCKDIM;

    cudaMalloc((void**) &cscShadow.colPtrs,      ((csc->numCols * tilesPerCol) + 1) * sizeof(unsigned int));
    cudaMalloc((void**) &cscShadow.colPtrsBlock, ((csc->numCols * tilesPerCol) + 1) * sizeof(unsigned int));
    cudaMalloc((void**) &cscShadow.rowIdxs,      csc->capacity * sizeof(unsigned int));
    cudaMalloc((void**) &cscShadow.values,       csc->capacity * sizeof(float));

    cudaMemcpy(cscShadow.colPtrs, csc->colPtrs,       ((csc->numCols * tilesPerCol) + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(cscShadow.colPtrs, csc->colPtrsBlock,  ((csc->numCols * tilesPerCol) + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(cscShadow.rowIdxs, csc->rowIdxs,       csc->capacity * sizeof(unsigned int),                                cudaMemcpyHostToDevice);
    cudaMemcpy(cscShadow.values, csc->values,         csc->capacity * sizeof(float),                                       cudaMemcpyHostToDevice);

    CSCTiledMatrix* csc_d;
    cudaMalloc((void**) &csc_d, sizeof(CSCTiledMatrix));
    cudaMemcpy(csc_d, &cscShadow, sizeof(CSCTiledMatrix), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    return csc_d;
}

CSCTiledMatrix* createEmptyCSCTiled_pinned(unsigned int numRows, unsigned int numCols, unsigned int capacity) {
    CSCTiledMatrix *csc; 
    cudaMallocHost((void**) &csc, sizeof(CSCTiledMatrix));

    csc->numRows = numRows;
    csc->numCols = numCols;
    csc->nnz = 0;
    csc->capacity = capacity;
    unsigned int tilesPerCol = (numRows+BLOCKDIM-1)/BLOCKDIM;

    cudaMallocHost((void**) &csc->colPtrs,      (numCols * tilesPerCol + 1) * sizeof(unsigned int));
    cudaMallocHost((void**) &csc->colPtrsBlock, (numCols * tilesPerCol + 1) * sizeof(unsigned int));
    cudaMallocHost((void**) &csc->rowIdxs,      capacity * sizeof(unsigned int));
    cudaMallocHost((void**) &csc->values,       capacity * sizeof(float));
    for(int i = 0; i < numCols * tilesPerCol + 1; ++i){
        csc->colPtrsBlock[i] = 0;
        csc->colPtrs[i]      = 0;
    }
    
    return csc;
}

void freeCSCTiled_pinned(CSCTiledMatrix* csc) {
    cudaFreeHost(csc->colPtrs);
    cudaFreeHost(csc->colPtrsBlock);
    cudaFreeHost(csc->rowIdxs);
    cudaFreeHost(csc->values);
    cudaFreeHost(csc);
}




void sparseNN(Vector* result, COOMatrix* featureVectors, COOMatrix** layerWeights, float bias, unsigned int numLayers) {

    Timer timer;
    
    // Convert featureVectors to CSR
    startTime(&timer);
    CSRTiledMatrix* Y0   = createEmptyCSRTiled_pinned(featureVectors->numRows, featureVectors->numCols, 4 * featureVectors->nnz);        // Assuming 4*nnz is enough for all Y vectors
    convertCOOtoCSRTiled(featureVectors, Y0, BLOCKDIM);

    CSRTiledMatrix* Y0_d = createEmptyCSRTiled_d(featureVectors->numRows, featureVectors->numCols, 4 * featureVectors->nnz);             // Assuming 4*nnz is enough for all Y vectors
    stopTimeAndPrint(&timer, "Convert feature vectors to CSR");


    // Convert layer weights to CSC
    startTime(&timer);
    CSCTiledMatrix* W[numLayers];
    CSCTiledMatrix* W_d[numLayers];
    for(unsigned int layer = 0; layer < numLayers; ++layer) {
        W[layer] = createCSCTiledfromCOO(layerWeights[layer]);
        

        // W[layer] = createEmptyCSCTiled_pinned(layerWeights[layer]->numRows, layerWeights[layer]->numCols, 4 * layerWeights[layer]->nnz);
        // convertCOOtoCSCTiled(layerWeights[layer], W[layer], BLOCKDIM);
        // W_d[layer] = createCSCfromCSCTiled_d(W[layer]);
    }
    stopTimeAndPrint(&timer, "Convert weights to CSR");


    // Temporary buffer
    startTime(&timer);
    COOMatrix *tmp   = createEmptyCOO_pinned(Y0->numRows, Y0->numCols, Y0->capacity);
    COOMatrix *tmp_d = createEmptyCOO_d(Y0->numRows, Y0->numCols, Y0->capacity);
    stopTimeAndPrint(&timer, "Allocate temporary buffer");


    // Loop over layers
    CSRTiledMatrix *Yin    = Y0;
    COOMatrix      *Yout   = tmp;
    CSRTiledMatrix *Yin_d  = Y0_d;
    COOMatrix      *Yout_d = tmp_d;
    for(unsigned int layer = 0; layer < numLayers; ++layer) {

        printf("Computing layer %u (SpMSpM)\n", layer);

        // Copy to GPU
        startTime(&timer);
        copyCSRTiledtoGPU(Yin, Yin_d);
        cudaMemset(&Yout_d->nnz, 0, sizeof(unsigned int));
        stopTimeAndPrint(&timer, "    Copy CSR to GPU and clear COO");

        // SpMSpM
        startTime(&timer);    
        dim3 gridSize( (W[layer]->numCols + BLOCKDIM - 1) / BLOCKDIM,  (Yin->numRows + BLOCKDIM - 1) / BLOCKDIM);
        dim3 blockSize(BLOCKDIM, BLOCKDIM); 

        spmspm<<<gridSize, blockSize>>> (Yout_d, Yin_d , W_d[layer], bias, W[layer]->numRows, W[layer]->numCols, Yin->numRows);

        cudaDeviceSynchronize();
        stopTimeAndPrint(&timer, "    SpMSpM");

        // Copy from GPU
        startTime(&timer);
        copyCOOfromGPU(Yout_d, Yout);
        stopTimeAndPrint(&timer, "    Copy COO from GPU");
        printf("    Output matrix number of nonzeros: %d\n", Yout->nnz);

        // Convert COO to CSR
        startTime(&timer);
        convertCOOtoCSRTiled(Yout, Yin, BLOCKDIM);
        stopTimeAndPrint(&timer, "    Converting COO to CSR");

    }

    // Find nonzero rows
    startTime(&timer);
    findNonzeroRows(result, Yin);
    stopTimeAndPrint(&timer, "Find nonzero rows");

    // Free buffers
    startTime(&timer);
    freeCSRTiled_pinned(Y0);
    for(unsigned int layer = 0; layer < numLayers; ++layer) {
        freeCSCTiled(W[layer]);
    }
    freeCOO_pinned(tmp);
    stopTimeAndPrint(&timer, "Deallocate memory");
}
