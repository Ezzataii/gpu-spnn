
#include <assert.h>
#include <stdio.h>

#include "kernel.h"
#include "matrix.h"
#include "timer.h"

#define THRESHOLD 0.000001
#define YMAX 32
#define BLOCKDIM 32

__global__ void spmspm(COOMatrix *result, CSRMatrix *A, CSCMatrix *B, float bias, unsigned int K, unsigned int N, unsigned int M) {
    __shared__ unsigned int As_rowPtrs[BLOCKDIM + 1];
    __shared__ unsigned int As_colIdxs[BLOCKDIM*BLOCKDIM];
    __shared__ float        As_values[BLOCKDIM*BLOCKDIM];
    
    __shared__ unsigned int Bs_colPtrs[BLOCKDIM + 1];
    __shared__ unsigned int Bs_rowIdxs[BLOCKDIM*BLOCKDIM];
    __shared__ float        Bs_values[BLOCKDIM*BLOCKDIM];

    float sum = 0.0f;

    //Thread
    unsigned int outRow = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int outCol = blockDim.x * blockIdx.x + threadIdx.x;

    //OutCell
    unsigned int totalRows = A->numRows; 
    unsigned int totalCols = B->numCols;

    //A (CSR)
    unsigned int tilesPerRowCSR = (K+BLOCKDIM-1)/BLOCKDIM;

    //B (CSC)
    unsigned int tilesPerColCSC = (K+BLOCKDIM-1)/BLOCKDIM; //For readability only
    
    for(unsigned int tile = 0; tile < ((K+BLOCKDIM-1)/BLOCKDIM); ++tile) {  
        //CSR Vars
        unsigned int tileOffsetCSR = (blockIdx.y * tilesPerRowCSR + tile) * BLOCKDIM; //Index of first rowPtr of the tile.
        unsigned int tileRowPtrCSR = A->rowPtrs[tileOffsetCSR]; //RowPtr value of first row in the tile
        unsigned int nnzCSRTile = A->rowPtrs[tileOffsetCSR+BLOCKDIM] - tileRowPtrCSR; //Row Pointer of first row of next tile - RowPtr of first row for this tile

        //CSC Vars
        unsigned int tileOffsetCSC = (blockIdx.x * tilesPerColCSC + tile) * BLOCKDIM; //Index of first colPtr of the tile.
        unsigned int tileColPtrCSC = B->colPtrs[tileOffsetCSC]; //ColPtr value of first col in the tile
        unsigned int nnzCSCTile = B->colPtrs[tileOffsetCSC+BLOCKDIM] - tileColPtrCSC; //Row Pointer of first row of next tile - RowPtr of first row for this tile

        unsigned int threadCount = threadIdx.x + threadIdx.y*BLOCKDIM; //Since 2d tile we need a way to index the threads

        //First thread per row loads a row ptr
        if(threadIdx.x == 0 && outRow < totalRows){
            As_rowPtrs[threadIdx.y] = A->rowPtrs[tileOffsetCSR+threadIdx.y] - tileRowPtrCSR;
        }

        //First thread per col loads a col ptr
        if(threadIdx.y == 0 && outCol < totalCols){
            Bs_colPtrs[threadIdx.x] = B->colPtrs[tileOffsetCSC+threadIdx.x] - tileColPtrCSC;
        }

        //Each thread loads one nnz csc and csr
        if(threadCount < nnzCSRTile){
            As_colIdxs[threadCount] = A->colIdxs[tileRowPtrCSR+threadCount]; 
            As_values[threadCount]  = A->values[tileRowPtrCSR+threadCount];
        }
        
        if(threadCount< nnzCSCTile){
            Bs_rowIdxs[threadCount] = B->rowIdxs[tileColPtrCSC + threadCount];
            Bs_values[threadCount]  = B->values[tileColPtrCSC + threadCount];
        }
        __syncthreads();

        // Compute with tile
        if(outRow < M && outCol < N){
            unsigned int nnzA = nnzCSRTile;
            unsigned int nnzB = nnzCSCTile;

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

    // Relu
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
            result->colIdxs[nnzIdx] = outCol;
            result->rowIdxs[nnzIdx] = outRow;
            result->values[nnzIdx]  = sum;
        }    
    }  
}

void quicksort_2(float *data, unsigned int *key, unsigned int start, unsigned int end) {
    if((end - start + 1) > 1) {
        unsigned int left = start, right = end;
        unsigned int pivot = key[right];
        while(left <= right) {
            while(key[left] < pivot) {
                left = left + 1;
            }
            while(key[right] > pivot) {
                right = right - 1;
            }
            if(left <= right) {
                unsigned int tmpKey = key[left]; key[left] = key[right]; key[right] = tmpKey;
                float tmpData = data[left]; data[left] = data[right]; data[right] = tmpData;
                left = left + 1;
                right = right - 1;
            }
        }
        quicksort_2(data, key, start, right);
        quicksort_2(data, key, left, end);
    }
}
void findNonzeroRows(Vector* v, CSRMatrix* A) {
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
CSRMatrix* createEmptyCSR_modified(unsigned int numRows, unsigned int numCols, unsigned int capacity) {
    unsigned int tilesPerRow = (numCols+BLOCKDIM-1)/BLOCKDIM ;
    CSRMatrix *csr = (CSRMatrix *)malloc(sizeof(CSRMatrix));
    csr->rowPtrs= (unsigned int *)calloc(1, (numRows*tilesPerRow+1) * sizeof(unsigned int));
    csr->colIdxs= (unsigned int *)malloc( capacity * sizeof(unsigned int));
    csr->values = (float *)malloc( capacity * sizeof(float));
    csr->numRows = numRows;
    csr->numCols = numCols;
    csr->nnz = 0;
    csr->capacity = capacity;
    return csr;
}
CSRMatrix* createEmptyCSR_d_modified(unsigned int numRows, unsigned int numCols, unsigned int capacity) {
    unsigned int tilesPerRow = (numCols+BLOCKDIM-1)/BLOCKDIM;
    CSRMatrix csrShadow;
    csrShadow.numRows = numRows;
    csrShadow.numCols = numCols;
    csrShadow.nnz = 0;
    csrShadow.capacity = capacity;
    cudaMalloc((void**) &csrShadow.rowPtrs, (numRows*tilesPerRow+1)*sizeof(unsigned int));
    cudaMalloc((void**) &csrShadow.colIdxs, capacity*sizeof(unsigned int));
    cudaMalloc((void**) &csrShadow.values, capacity*sizeof(float));
    CSRMatrix* csr_d;
    cudaMalloc((void**) &csr_d, sizeof(CSRMatrix));
    cudaMemcpy(csr_d, &csrShadow, sizeof(CSRMatrix), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    return csr_d;
}
CSCMatrix* convertCSCfromCOO_modified(COOMatrix* A) {
    unsigned int tilesPerColCSC = (A->numRows + BLOCKDIM - 1) / BLOCKDIM;

    // Allocate
    unsigned int *colPtrs= (unsigned int *) calloc(A->numCols*tilesPerColCSC + 1, sizeof(unsigned int));
    unsigned int *rowIdxs = (unsigned int *) malloc(A->nnz*sizeof(unsigned int));
    float *values = (float *) malloc(A->nnz*sizeof(float));
    
    CSCMatrix* B = (CSCMatrix*) malloc(sizeof(CSCMatrix));
    B->colPtrs = colPtrs;
    B->rowIdxs = rowIdxs;
    B->values = values;
    B->nnz = A->nnz;
    B->numRows = A->numRows;
    B->numCols = A->numCols;
    B->capacity = A->nnz;

    
    // Histogram
    for(unsigned int i = 0; i < A->nnz; ++i) {
        unsigned int row  = A->rowIdxs[i];
        unsigned int col = A->colIdxs[i];
        unsigned int blockIdx = ((col/BLOCKDIM) * tilesPerColCSC) + (row/BLOCKDIM);
        unsigned int localBlockIdx = col%BLOCKDIM; 
        B->colPtrs[blockIdx*BLOCKDIM + localBlockIdx]++;
    }

    // Prefix sum
    unsigned int sum = 0;
    for(unsigned int col = 0; col < A->numCols*tilesPerColCSC; ++col) {
        unsigned int val = colPtrs[col];
        B->colPtrs[col] = sum;
        sum += val;
    }
    B->colPtrs[A->numCols*tilesPerColCSC] = sum;

    // Binning
    for(unsigned int index = 0; index < A->nnz; ++index) {
        unsigned int row = A->rowIdxs[index];
        unsigned int col = A->colIdxs[index];
        unsigned int blockIdx = ((col/BLOCKDIM) * tilesPerColCSC) + (row/BLOCKDIM);
        unsigned int localBlockIdx = col%BLOCKDIM; 
        unsigned int i = colPtrs[blockIdx*BLOCKDIM + localBlockIdx]++;
        rowIdxs[i] = A->rowIdxs[index];
        values[i] = A->values[index];
    }

    // Restore column pointers
    for(unsigned int col = A->numCols*tilesPerColCSC - 1; col > 0; --col) {
        colPtrs[col] = colPtrs[col - 1];
    }
    colPtrs[0] = 0;
    

    // Sort nonzeros within each row
    for(unsigned int colPtrIdx = 0; colPtrIdx < B->numCols*tilesPerColCSC; ++colPtrIdx) {
        unsigned int start = B->colPtrs[colPtrIdx];
        unsigned int end = B->colPtrs[colPtrIdx + 1] - 1;
        quicksort_2(B->values, B->rowIdxs, start, end);
    }


    return B;

}
void convertCOOtoCSR_modified(COOMatrix* A, CSRMatrix* B) {

    unsigned int tilesPerRowCSR = (B->numCols + BLOCKDIM - 1) / BLOCKDIM;
    // Check compatibility
    if(B->numRows != A->numRows || B->numCols != A->numCols) {
        fprintf(stderr, "%s: matrices have incompatible dimensions!\n", __func__);
        exit(1);
    }
    if(B->capacity < A->nnz) {
        fprintf(stderr, "%s: CSR matrix has insufficient capacity!\n", __func__);
        exit(1);
    }

    // Set nonzeros
    B->nnz = A->nnz;

    // Histogram
    memset(B->rowPtrs, 0, (B->numRows*tilesPerRowCSR + 1)*sizeof(unsigned int));
    for(unsigned int i = 0; i < A->nnz; ++i) {
        unsigned int row  = A->rowIdxs[i];
        unsigned int col  = A->colIdxs[i];
        unsigned int blockIdx = ((row/BLOCKDIM) * tilesPerRowCSR) + (col/BLOCKDIM);
        unsigned int localBlockIdx = row%BLOCKDIM;
        B->rowPtrs[blockIdx*BLOCKDIM + localBlockIdx]++;
    }

    // Prefix sum
    unsigned int sum = 0;
    for(unsigned int row = 0; row < (A->numRows * tilesPerRowCSR); ++row) {
        unsigned int val = B->rowPtrs[row];
        B->rowPtrs[row] = sum;
        sum += val;
    }
    B->rowPtrs[A->numRows * tilesPerRowCSR] = sum;

    // Binning
    for(unsigned int index = 0; index < A->nnz; ++index) {
        unsigned int row = A->rowIdxs[index];
        unsigned int col = A->colIdxs[index];
        unsigned int blockIdx = ((row/BLOCKDIM) * tilesPerRowCSR) + (col/BLOCKDIM);
        unsigned int localBlockIdx = row%BLOCKDIM;
        unsigned int i = B->rowPtrs[blockIdx*BLOCKDIM + localBlockIdx]++;
        B->colIdxs[i] = A->colIdxs[index];
        B->values[i] = A->values[index];
    }

    // Restore row pointers
    for(unsigned int row = A->numRows*tilesPerRowCSR - 1; row > 0; --row) {
        B->rowPtrs[row] = B->rowPtrs[row - 1];
    }
    B->rowPtrs[0] = 0;

    // Sort nonzeros within each row
    for(unsigned int rowPtrIdx = 0; rowPtrIdx < B->numRows*tilesPerRowCSR; ++rowPtrIdx) {
        unsigned int start = B->rowPtrs[rowPtrIdx];
        unsigned int end = B->rowPtrs[rowPtrIdx + 1] - 1;
        quicksort_2(B->values, B->colIdxs, start, end);
    }

}
CSCMatrix* createCSCfromCSC_d_modified(CSCMatrix* csc) {
    unsigned int tilesPerCol = (csc->numRows + BLOCKDIM - 1)/BLOCKDIM;
    CSCMatrix cscShadow;
    cscShadow.numRows = csc->numRows;
    cscShadow.numCols = csc->numCols;
    cscShadow.nnz = csc->nnz;
    cscShadow.capacity = csc->capacity;
    cudaMalloc((void**) &cscShadow.colPtrs, (csc->numCols * tilesPerCol + 1)*sizeof(unsigned int));
    cudaMalloc((void**) &cscShadow.rowIdxs, csc->capacity*sizeof(unsigned int));
    cudaMalloc((void**) &cscShadow.values, csc->capacity*sizeof(float));
    cudaMemcpy(cscShadow.colPtrs, csc->colPtrs, (csc->numCols * tilesPerCol + 1)*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(cscShadow.rowIdxs, csc->rowIdxs, csc->capacity*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(cscShadow.values, csc->values, csc->capacity*sizeof(float), cudaMemcpyHostToDevice);
    CSCMatrix* csc_d;
    cudaMalloc((void**) &csc_d, sizeof(CSCMatrix));
    cudaMemcpy(csc_d, &cscShadow, sizeof(CSCMatrix), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    return csc_d;
}
COOMatrix* createEmptyCOO_d(unsigned int numRows, unsigned int numCols, unsigned int capacity) {
    COOMatrix cooShadow;
    cooShadow.numRows = numRows;
    cooShadow.numCols = numCols;
    cooShadow.nnz = 0;
    cooShadow.capacity = capacity;
    cudaMalloc((void**) &cooShadow.rowIdxs, capacity*sizeof(unsigned int));
    cudaMalloc((void**) &cooShadow.colIdxs, capacity*sizeof(unsigned int));
    cudaMalloc((void**) &cooShadow.values, capacity*sizeof(float));
    COOMatrix* coo_d;
    cudaMalloc((void**) &coo_d, sizeof(COOMatrix));
    cudaMemcpy(coo_d, &cooShadow, sizeof(COOMatrix), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    return coo_d;
}
void copyCOOfromGPU(COOMatrix* coo_d, COOMatrix* coo) {
    COOMatrix cooShadow;
    cudaMemcpy(&cooShadow, coo_d, sizeof(COOMatrix), cudaMemcpyDeviceToHost);
    assert(coo->numRows == cooShadow.numRows);
    assert(coo->numCols == cooShadow.numCols);
    assert(coo->capacity >= cooShadow.nnz);
    coo->nnz = cooShadow.nnz;
    cudaMemcpy(coo->rowIdxs, cooShadow.rowIdxs, cooShadow.nnz*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(coo->colIdxs, cooShadow.colIdxs, cooShadow.nnz*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(coo->values, cooShadow.values, cooShadow.nnz*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}
void copyCSRtoGPU(CSRMatrix* csr, CSRMatrix* csr_d) {
    unsigned int tilesPerRow = (csr->numCols + BLOCKDIM - 1)/ BLOCKDIM;
    CSRMatrix csrShadow;
    cudaMemcpy(&csrShadow, csr_d, sizeof(CSRMatrix), cudaMemcpyDeviceToHost);
    assert(csrShadow.numRows == csr->numRows);
    assert(csrShadow.numCols == csr->numCols);
    assert(csrShadow.capacity >= csr->nnz);
    csrShadow.nnz = csr->nnz;
    cudaMemcpy(csrShadow.rowPtrs, csr->rowPtrs, (csr->numRows*tilesPerRow + 1)*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(csrShadow.colIdxs, csr->colIdxs, csr->nnz*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(csrShadow.values, csr->values, csr->nnz*sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}




void sparseNN(Vector* result, COOMatrix* featureVectors, COOMatrix** layerWeights, float bias, unsigned int numLayers) {

    Timer timer;

    // Convert featureVectors to CSR
    startTime(&timer);
    CSRMatrix* Y0 = createEmptyCSR_modified(featureVectors->numRows, featureVectors->numCols, 16*featureVectors->nnz); // Assuming 4*nnz is enough for all Y vectors
    convertCOOtoCSR_modified(featureVectors, Y0);
    CSRMatrix* Y0_d = createEmptyCSR_d_modified(featureVectors->numRows, featureVectors->numCols, 16*featureVectors->nnz); // Assuming 4*nnz is enough for all Y vectors
    stopTimeAndPrint(&timer, "Convert feature vectors to CSR");

    // Convert layer weights to CSC
    startTime(&timer);
    CSCMatrix* W[numLayers];
    CSCMatrix* W_d[numLayers];
    for(unsigned int layer = 0; layer < numLayers; ++layer) {
        W[layer]   = convertCSCfromCOO_modified(layerWeights[layer]);
        W_d[layer] = createCSCfromCSC_d_modified(W[layer]);
    }
    stopTimeAndPrint(&timer, "Convert weights to CSR");

    // Temporary buffer
    startTime(&timer);
    COOMatrix *tmp   = createEmptyCOO(Y0->numRows, Y0->numCols, Y0->capacity);   //This func doesn't require change because of structure change
    COOMatrix *tmp_d = createEmptyCOO_d(Y0->numRows, Y0->numCols, Y0->capacity); //This func doesn't require change because of structure change
    stopTimeAndPrint(&timer, "Allocate temporary buffer");

    // Loop over layers
    CSRMatrix *Yin    = Y0;
    COOMatrix *Yout   = tmp;
    CSRMatrix *Yin_d  = Y0_d;
    COOMatrix *Yout_d = tmp_d;
    for(unsigned int layer = 0; layer < numLayers; ++layer) {

        if(layer == 1){
            break;
        }

        printf("Computing layer %u (SpMSpM)\n", layer);

        // Copy to GPU
        startTime(&timer);
        copyCSRtoGPU(Yin, Yin_d);
        cudaMemset(&Yout_d->nnz, 0, sizeof(unsigned int));
        stopTimeAndPrint(&timer, "    Copy CSR to GPU and clear COO");

        // SpMSpM
        startTime(&timer);  
        dim3 gridSize( (W[layer]->numCols + BLOCKDIM - 1) / BLOCKDIM,  (Yin->numRows + BLOCKDIM - 1) / BLOCKDIM);
        dim3 blockSize(BLOCKDIM, BLOCKDIM); 

        printf("nnzA:%d nnzB:%d  \n",Yin->nnz,W[layer]->nnz);
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
        convertCOOtoCSR_modified(Yout, Yin);
        stopTimeAndPrint(&timer, "    Converting COO to CSR");

    }

    // Find nonzero rows
    startTime(&timer);
    findNonzeroRows(result, Yin);
    stopTimeAndPrint(&timer, "Find nonzero rows");

    // Free buffers
    startTime(&timer);
    freeCSR(Y0);
    for(unsigned int layer = 0; layer < numLayers; ++layer) {
        freeCSC(W[layer]);
    }
    freeCOO(tmp);
    stopTimeAndPrint(&timer, "Deallocate memory");

}
