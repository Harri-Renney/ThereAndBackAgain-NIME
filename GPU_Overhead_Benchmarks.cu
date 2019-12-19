#include <stdio.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "CUDA_Wrapper.hpp"

__global__ void null_kernel(void) {
}

__global__ void copy_buffer(float* srcBuffer, float* dstBuffer)
{
	uint32_t idx = threadIdx.x;

	dstBuffer[idx] = srcBuffer[idx];
}

namespace CUDA_Kernels
{

	void nullKernelExecute()
	{
		null_kernel << <1, 1>> > ();
	}
	void copyBufferExecute(size_t aN, float* srcBuffer, float* dstBuffer)
	{
		float* d_srcBuffer;
		float* d_dstBuffer;
		cudaMalloc((void**)&d_srcBuffer, sizeof(float) * aN);
		cudaMalloc((void**)&d_dstBuffer, sizeof(float) * aN);

		cudaMemcpy(d_srcBuffer, srcBuffer, sizeof(float) * aN, cudaMemcpyHostToDevice);
		cudaMemcpy(d_dstBuffer, dstBuffer, sizeof(float) * aN, cudaMemcpyHostToDevice);

		dim3 threadsPerBlock(aN);
		copy_buffer << <1, aN >> > (d_srcBuffer, d_dstBuffer);

		cudaMemcpy(dstBuffer, d_dstBuffer, sizeof(float) * aN, cudaMemcpyDeviceToHost);

		cudaFree(d_srcBuffer);
		cudaFree(d_dstBuffer);
	}

}

//void wrapper(void)
//{
//	null_kernel<<<1,1>>>();
//	uint32_t N = 1;
//	float* d_out;
//	cudaMalloc((void**)&d_out, sizeof(float) * N);
//
//	
//
//	float* checkWorking = (float*)malloc(sizeof(float) * N);
//	cudaMemcpy(d_out, checkWorking, sizeof(float) * N, cudaMemcpyHostToDevice);
//	//*checkWorking = 0.0;
//	test_kernel << <1, 1 >> > (d_out);
//
//	// Transfer data back to host memory
//	cudaMemcpy(checkWorking, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);
//	printf("Hello, world! %f", checkWorking[0]);
//}