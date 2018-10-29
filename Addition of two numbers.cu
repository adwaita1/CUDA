
*****************
Problem Statement
*****************
Program to Add two numbers on GPU.

****************************************************************************************/

#include<stdio.h>
#include<conio.h>

__global__ void add_number(float *ad,float *bd)
{

        *ad += *bd;                             //adding values in GPU memory
}
int main()
{
	float *a,*b;
	float *ad,*bd;
	size_t size = sizeof(float);

    //allocate memory on host
	a=(float*)malloc(size);
	b=(float*)malloc(size);
	
    //allocate memory on device
	cudaMalloc(&ad,size);
	//printf("\nAfter cudaMalloc for ad\n%s\n",cudaGetErrorString(cudaGetLastError()));
	cudaMalloc(&bd,size);
	//printf("\nAfter cudaMalloc for bd\n%s\n",cudaGetErrorString(cudaGetLastError()));
	
    printf("\nEnter two numbers\n");
    scanf("%f%f",a,b);

	//copy data from host memory to device memory
	cudaMemcpy(ad,a,size,cudaMemcpyHostToDevice);
    //printf("\nAfter HostToDevice Memcpy for ad\n%s\n",cudaGetErrorString(cudaGetLastError()));
	cudaMemcpy(bd,b,size,cudaMemcpyHostToDevice);
    //printf("\nAfter HostToDevice Memcpy for bd\n%s\n",cudaGetErrorString(cudaGetLastError()));
	
    //GPU timer code
    float time;
    cudaEvent_t start,stop;			
	cudaEventCreate(&start);
    cudaEventCreate(&stop);
	cudaEventRecord(start,0);
	
    //launch kernel with only one thread
    add_number<<< 1,1 >>>(ad,bd);
	
    cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time,start,stop);			//time taken in kernel call calculated
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

    //copy back the results
	cudaMemcpy(a,ad,size,cudaMemcpyDeviceToHost);
	//printf("\nAfter DeviceToHost Memcpy for a\n%s\n",cudaGetErrorString(cudaGetLastError()));
	
	//print the results
	printf("\nAddition of above two numbers on GPU evaluates to = %f",*a);
    printf("\n\nTime taken is %f (ms)\n",time);
    
    //deallocate host and device memories
    cudaFree(ad); cudaFree(bd);
	free(a);free(b);

	_getch();
    return 1;
}