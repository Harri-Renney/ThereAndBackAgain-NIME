#include <stdio.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "CUDA_Wrapper.hpp"

__global__ void null_kernel(void) {
}

__global__ void copy_buffer(float* srcBuffer, float* dstBuffer)
{
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	dstBuffer[idx] = srcBuffer[idx];
}

__global__ void singlesampleprocessing(float sample)
{
	
}

__global__
void ftdtCompute(float* gridOne, float* gridTwo, float* gridThree, float* boundaryGain, int samplesIndex, float* samples, float* excitation, int listenerPosition, int excitationPosition, float propagationFactor, float dampingFactor, int rotationIndex)
{
	int blockId = blockIdx.y * gridDim.x + blockIdx.x;

	//Get index for current and neighbouring nodes//
	//@ToDo - Need to check this, different indexing to openCL!//
	int ixy = blockId * blockDim.x + threadIdx.x;
	int ixMy = (blockId - 1) * blockDim.x + threadIdx.x;
	int ixPy = (blockId + 1) * blockDim.x + threadIdx.x;
	int ixyM = (blockId) * blockDim.x + threadIdx.x - 1;
	int ixyP = (blockId)* blockDim.x + threadIdx.x + 1;

	//Determine each buffer in relation to time from a rotation index//
	float* nMOne;
	float* n;
	float* nPOne;
	if (rotationIndex == 0)
	{
		nMOne = gridOne;
		n = gridTwo;
		nPOne = gridThree;
	}
	else if (rotationIndex == 1)
	{
		nMOne = gridTwo;
		n = gridThree;
		nPOne = gridOne;
	}
	else if (rotationIndex == 2)
	{
		nMOne = gridThree;
		n = gridOne;
		nPOne = gridTwo;
	}

	//Initalise pressure values//
	float centrePressureNMO = nMOne[ixy];
	float centrePressureN = n[ixy];
	float leftPressure;
	float rightPressure;
	float upPressure;
	float downPressure;

	//Predicate method//
	//leftPressure = boundaryGain[ixy] > 0.0 ? n[ixy] * boundaryGain[ixy] : n[ixMy];
	//rightPressure = boundaryGain[ixy] > 0.0 ? n[ixy] * boundaryGain[ixy] : n[ixPy];
	//upPressure = boundaryGain[ixy] > 0.0 ? n[ixy] * boundaryGain[ixy] : n[ixyM];
	//downPressure = boundaryGain[ixy] > 0.0 ? n[ixy] * boundaryGain[ixy] : n[ixyP];

	if (boundaryGain[ixy] > 0.01)
	{
		leftPressure = n[ixy] * boundaryGain[ixy];
		rightPressure = n[ixy] * boundaryGain[ixy];
		upPressure = n[ixy] * boundaryGain[ixy];
		downPressure = n[ixy] * boundaryGain[ixy];
	}
	else
	{
		leftPressure = n[ixMy];
		rightPressure = n[ixPy];
		upPressure = n[ixyM];
		downPressure = n[ixyP];
	}

	//Calculate the nex pressure value//
	float newPressure = 2 * centrePressureN;
	newPressure += (dampingFactor - 1.0) * centrePressureNMO;
	newPressure += propagationFactor * (leftPressure + rightPressure + upPressure + downPressure - (4 * centrePressureN));
	newPressure *= 1.0 / (dampingFactor + 1.0);


	//If the cell is the listener position, sets the next sound sample in buffer to value contained here//
	if (ixy == listenerPosition)
	{
		samples[samplesIndex] = n[ixy];
	}

	if (ixy == excitationPosition)	//If the position is an excitation...
	{
		newPressure += excitation[samplesIndex];	//Input excitation value into point. Then increment to next excitation in next iteration.
	}

	nPOne[ixy] = newPressure;
}

namespace CUDA_Kernels
{

	void nullKernelExecute()
	{
		null_kernel << <1, 1024>> > ();
	}
	void copyBufferExecute(size_t aN, float* srcBuffer, float* dstBuffer, float* cudaSrcBuffer, float* cudaDstBuffer)
	{
		size_t numBlocks = aN / 1024;
		dim3 threadsPerBlock(1024);
		copy_buffer << <numBlocks, threadsPerBlock >> > (cudaSrcBuffer, cudaDstBuffer);
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