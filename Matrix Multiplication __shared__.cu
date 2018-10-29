
*************
Device Query
*************
Device 0: "GeForce 9400 GT"
  CUDA Driver Version:                           	3.0
  CUDA Runtime Version:                          	2.30
  CUDA Capability Major revision number:    	    1
  CUDA Capability Minor revision number:         	1
  Total amount of global memory:                 	1073414144 bytes
  Number of multiprocessors:                     	2
  Number of cores:                               	16
  Total amount of constant memory:           		65536 bytes
  Total amount of shared memory per block:	        16384 bytes
  Total number of registers available per block: 	8192
  Warp size:                                     	32
  Maximum number of threads per block:   	        512
  Maximum sizes of each dimension of a block:	    512 x 512 x 64
  Maximum sizes of each dimension of a grid:   	    65535 x 65535 x 1
  Maximum memory pitch:                         	262144 bytes
  Texture alignment:                             	256 bytes
  Clock rate:                                    	1.35 GHz
  Concurrent copy and execution: 		            Yes
  Run time limit on kernels:                     	Yes
  Integrated:                                    	No
  Support host page-locked memory mapping:	        No
  Compute mode:                                  	Default (multiple host threads can use this device simultaneously)

*****************
Problem Statement
*****************
Program to calculate Matrix Multiplication of two Matrices using shared memory

****************************************************************************************/

#include<stdio.h>
#include<conio.h>

__global__ void matrix_mul_shared(float *ad,float *bd,float *cd,int N)
{
	float pvalue=0;
    int TILE=blockDim.x;
    int ty=threadIdx.y;
    int tx=threadIdx.x;
    
    //allocate shared memory per block
    __shared__ float ads[16][16];
    __shared__ float bds[16][16];

    //find Row and Column corresponding to a data element for each thread
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int Col = blockIdx.x * blockDim.x + threadIdx.x;

    //iterate through TILEs to traverse whole WIDTH
	for(int i=0;i< N/TILE;++i)
	{
		//copy values of data TILE into shared memory
        ads[ty][tx] = ad[Row * N + (i * TILE) + tx];
        bds[ty][tx] = bd[(i * TILE + ty) * N + Col];
        
        __syncthreads();                            //synchronize to confirm that whole TILE has been copied

        //calculate partial dot-product
        for(int k=0;k<TILE;k++)
                pvalue += ads[ty][k] * bds[k][tx];
        
        __syncthreads();                            //synchronize to confirm that whole partial product corresponding to all threads of the block has been calculated
	}

    //store dot product at corresponding positon in resultant Matrix
	cd[Row * N + Col] = pvalue;
}

int main()
{
	int N = 1024,i,j;				//N == size of square matrix
	
	float *a,*b;
	float *ad,*bd,*cd,*c;

    //open a file for outputting the result
    FILE *f;
	f=fopen("Parallel Multiply.txt","w");

	size_t size=sizeof(float)* N * N;

    //allocate host side memory
	a=(float*)malloc(size);
	b=(float*)malloc(size);
	c=(float*)malloc(size);

	for(i=0;i<N;i++)
	{
		for(j=0;j<N;j++)
		{
			a[i*N+j]=1.0;   //(float)(i*N+j);		//initializing each value with its own index
            b[i*N+j]=1.0;   //(float)(i*N+j);		//random functions can be used alternatively
		}
	}

    //allocate device memory
	cudaMalloc(&ad,size);
	//printf("\nAfter cudaMalloc for ad\n%s\n",cudaGetErrorString(cudaGetLastError()));
	cudaMalloc(&bd,size);
	//printf("\nAfter cudaMalloc bd\n%s\n",cudaGetErrorString(cudaGetLastError()));
	cudaMalloc(&cd,size);
	//printf("\nAfter cudaMalloc cd\n%s\n",cudaGetErrorString(cudaGetLastError()));
	
	//copy value from host to device
    cudaMemcpy(ad,a,size,cudaMemcpyHostToDevice);
	cudaMemcpy(bd,b,size,cudaMemcpyHostToDevice);
	printf("\nAfter HostToDevice Memcpy\n%s\n",cudaGetErrorString(cudaGetLastError()));

    //calculate execution configuration
    dim3 blocksize(16,16);		        //each block contains 16 * 16 (=256) threads 
	dim3 gridsize(N/16,N/16);			//creating just sufficient no of blocks
    
    //GPU timer code
    float time;
    cudaEvent_t start,stop;			
	cudaEventCreate(&start);		
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);

	matrix_mul_shared <<< gridsize, blocksize >>> (ad, bd, cd, N);
	
    cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time,start,stop);			//time taken in kernel call calculated
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

    //copy back results
	cudaMemcpy(c,cd,sizeof(float)* N*N,cudaMemcpyDeviceToHost);
	printf("\nAfter DeviceToHost Memcpy\n%s\n",cudaGetErrorString(cudaGetLastError()));
	
    //output results in output_file
	fprintf(f,"Array A was---\n");
	for(i=0;i<N;i++)
	{
		for(j=0;j<N;j++)
			fprintf(f,"%f ",a[i*N+j]);
		fprintf(f,"\n");
	}
	fprintf(f,"\nArray B was---\n");
	for(i=0;i<N;i++)
	{
		for(j=0;j<N;j++)
			fprintf(f,"%f ",b[i*N+j]);
		fprintf(f,"\n");
	}
	fprintf(f,"\nMultiplication of A and B gives C----\n");
	for(i=0;i<N;i++)
	{
		for(j=0;j<N;j++)
			fprintf(f,"%f ",c[i*N+j]);              //if correctly computed, then all values must be N
		fprintf(f,"\n");
	}
	printf("\nYou can see output in Parallel Mutiply.txt file in project directory");
    printf("\n\nTime taken is %f (ms)\n",time);
    fprintf(f,"\n\nTime taken is %f (ms)\n",time);
	fclose(f);

	cudaThreadExit();
	//cudaFree(ad); cudaFree(bd); cudaFree (cd);
	free(a);free(b);free(c);
	_getch();
    return 1;
}