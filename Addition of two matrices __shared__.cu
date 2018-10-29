*****************
Problem Statement
*****************
Program to Add two Matrices using shared memory.

****************************************************************************************/

#include<stdio.h>
#include<conio.h>

//size of square matrix
#define N 20

__global__ void add_matrices(float *ad,float *bd, float *cd)
{
    //allocate shared memory for the two matrices
    __shared__ float ads [N][N];
    __shared__ float bds [N][N];

    //copy values from global memory into shared memory
    ads[threadIdx.y][threadIdx.x] = ad[threadIdx.y * N + threadIdx.x];
    bds[threadIdx.y][threadIdx.x] = bd[threadIdx.y * N + threadIdx.x];

    cd[threadIdx.y * N + threadIdx.x] = ads[threadIdx.y][threadIdx.x] + bds[threadIdx.y][threadIdx.x];
}
int main()
{
	unsigned int i,j;
	float *a,*b;
	float *ad,*bd,*cd,*c;
	size_t size=sizeof(float)* N * N;

    //allocate memory on host
	a=(float*)malloc(size);
	b=(float*)malloc(size);
    c=(float*)malloc(size);
	
    //allocate memory on device
	cudaMalloc(&ad,size);
	//printf("\nAfter cudaMalloc for ad\n%s\n",cudaGetErrorString(cudaGetLastError()));
	cudaMalloc(&bd,size);
	//printf("\nAfter cudaMalloc for bd\n%s\n",cudaGetErrorString(cudaGetLastError()));
    cudaMalloc(&cd,size);
	//printf("\nAfter cudaMalloc for cd\n%s\n",cudaGetErrorString(cudaGetLastError()));
	
    //initialize host memory with its own indices
    for(i=0;i<N;i++)
	{
		for(j=0;j<N;j++)
        {
			a[i * N + j]=(float)(i * N + j);
            b[i * N + j]= -(float)(i * N + j);
        }
    }

	//copy data from host memory to device memory
	cudaMemcpy(ad,a,size,cudaMemcpyHostToDevice);
    //printf("\nAfter HostToDevice Memcpy for ad\n%s\n",cudaGetErrorString(cudaGetLastError()));
	cudaMemcpy(bd,b,size,cudaMemcpyHostToDevice);
    //printf("\nAfter HostToDevice Memcpy for bd\n%s\n",cudaGetErrorString(cudaGetLastError()));

	//calculate execution configuration
    dim3 blocksize (N, N);		//each block contains N * N threads, each thread calculates 1 data element
    
    //GPU timer code
    float time;
    cudaEvent_t start,stop;			
	cudaEventCreate(&start);
    cudaEventCreate(&stop);
	cudaEventRecord(start,0);
	
    add_matrices<<< 1, blocksize, 2 * size >>>(ad, bd, cd);
	
    cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time,start,stop);			//time taken in kernel call calculated
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaMemcpy(c,cd,size,cudaMemcpyDeviceToHost);
	//printf("\nAfter DeviceToHost Memcpy for c \n%s\n",cudaGetErrorString(cudaGetLastError()));
	
	printf("Matrix A was---\n");
	for(i=0;i<N;i++)
	{
		for(j=0;j<N;j++)
			printf("%f ",a[i*N+j]);
		printf("\n");
	}
	printf("\nMatrix B was---\n");
	for(i=0;i<N;i++)
	{
		for(j=0;j<N;j++)
			printf("%f ",b[i*N+j]);
		printf("\n");
	}
	printf("\nAddition of A and B gives C----\n");
	for(i=0;i<N;i++)
	{
		for(j=0;j<N;j++)
			printf("%f ",c[i*N+j]);              //if correctly evaluated, all values will be 0
		printf("\n");
	}

	printf("\n\nTime taken is %f (ms)\n",time);
     
    //deallocate host and device memories
    cudaFree(ad); cudaFree(bd); cudaFree (cd);
	free(a);free(b);free(c);

	_getch();
    return 1;
}