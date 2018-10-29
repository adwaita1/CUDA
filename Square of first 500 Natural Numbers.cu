*****************
Problem Statement
*****************
Program to calculate square of first 500 natural numbers on GPU.

****************************************************************************************/

#include<stdio.h>
#include<conio.h>

__global__ void square_array(float *ad, int N)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N)
        ad[index] *= ad[index];                             //adding values in GPU memory
}
int main()
{
	float *a;
	float *ad;
    int N = 500;                                  //500 natural numbers 
    unsigned int i, No_of_blocks, No_of_threads;
	size_t size = sizeof(float) * N;

    //allocate memory on host
	a=(float*)malloc(size);
		
    //allocate memory on device
	cudaMalloc(&ad,size);
	//printf("\nAfter cudaMalloc for ad\n%s\n",cudaGetErrorString(cudaGetLastError()));
	
    //initialize host memory with its own indices
    for(i=0; i<N; i++)
    {
        a[i]=(float)i;
    }

	//copy data from host memory to device memory
	cudaMemcpy(ad,a,size,cudaMemcpyHostToDevice);
    //printf("\nAfter HostToDevice Memcpy for ad\n%s\n",cudaGetErrorString(cudaGetLastError()));
	
    //calculate execution configuration
    if (N > 512)
    {
        No_of_threads = 512;
        No_of_blocks = (N / 512) + (((N % 512) == 0) ? 0 : 1);
       
    }
    else 
    {
        No_of_threads = N;
        No_of_blocks = 1;
    }
    dim3 block (No_of_threads, 1, 1);
    dim3 grid (No_of_blocks, 1, 1);
    
    //GPU timer code
    float time;
    cudaEvent_t start,stop;			
	cudaEventCreate(&start);
    cudaEventCreate(&stop);
	cudaEventRecord(start,0);
	
    //launch kernel with optimum threads
    square_array<<< grid, block >>>(ad,N);
	
    cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time,start,stop);			//time taken in kernel call calculated
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

    //copy back the results
	cudaMemcpy(a,ad,size,cudaMemcpyDeviceToHost);
	//printf("\nAfter DeviceToHost Memcpy for a\n%s\n",cudaGetErrorString(cudaGetLastError()));
	
	//print the results
	printf("\nAddition of above two VECTORS on GPU evaluates to = \n");
    for (i = 0; i < N; i++)
        printf("%f\n", a[i]);                       //if correctly evaluated, all values will be 0
    printf("\n\nTime taken is %f (ms)\n",time);
    
    //deallocate host and device memories
    cudaFree(ad); 
	free(a);

	_getch();
    return 1;
}