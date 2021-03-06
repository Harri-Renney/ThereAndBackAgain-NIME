#include <stdio.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "GPU_Overhead_Benchmarks_CUDA.hpp"
#define M_PI	3.14159265358979323846
#define M_E		2.718281828459

__global__ void null_kernel(void) {
}

__global__ void copy_buffer(float* srcBuffer, float* dstBuffer)
{
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	dstBuffer[idx] = srcBuffer[idx];
}

__global__ void single_sample(float* singleSample)
{
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	const float coefficient = 0.5;

	singleSample[0] = singleSample[0] * coefficient;
}

__global__ void simple_buffer_processing(float* inputBuffer, float* outputBuffer)
{
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	float attenuatedSample = inputBuffer[idx] * 0.5;
	//float attenuatedSample = inputBuffer[idx] * pow(M_E, -idx);
	outputBuffer[idx] = attenuatedSample;
}

__global__ void complex_buffer_processing(float* inputBuffer, float* outputBuffer)
{
	int32_t globalSize = gridDim.x * blockDim.x;
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t limUpper = globalSize - 2;
	int32_t limLower = 2;

	//float attenuationCoefficient = pow(M_E, -idx);

	float smoothedSample = inputBuffer[idx];
	if (idx > limLower & idx < limUpper)
	{
		smoothedSample = ((inputBuffer[idx - 2] + 2.0 * inputBuffer[idx - 1] + 3.0 * inputBuffer[idx] + 2.0 * inputBuffer[idx + 1] + inputBuffer[idx + 2]) / 9.0);
	}

	//float smoothedSample = idx > limLower & idx < limUpper ? ((inputBuffer[idx-2] + 2.0 * inputBuffer[idx-1] + 3.0 * inputBuffer[idx] + 2.0 * inputBuffer[idx+1] + inputBuffer[idx+2]) / 9.0) : inputBuffer[idx];
	outputBuffer[idx] = smoothedSample;
	//outputBuffer[idx] = smoothedSample * attenuationCoefficient;
}

__global__
void simple_buffer_synthesis(int sampleRate, float frequency, float* outputBuffer)
{
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	float amplitude = 0.5;
	float relativeFrequency = frequency / (float)sampleRate;
	int time = idx;
	float currentSample = amplitude * sin(2.0 * M_PI * relativeFrequency * time);
	outputBuffer[idx] = currentSample;
}

//@ToDo - Tri using loop for to fill buffer and __syncthreads(). Won't work, this is for thread synchonization only!!
__global__
void complex_buffer_synthesis(float* gridOne, float* gridTwo, float* gridThree, float* boundaryGain, int samplesIndex, float* samples, float* excitation, int listenerPosition, int excitationPosition, float propagationFactor, float dampingFactor, int rotationIndex)
{
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int ixy = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	int ixMy = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + (threadIdx.x-1);
	int ixPy = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + (threadIdx.x+1);
	int ixyM = blockId * (blockDim.x * blockDim.y) + ((threadIdx.y-1) * blockDim.x) + threadIdx.x;
	int ixyP = blockId * (blockDim.x * blockDim.y) + ((threadIdx.y+1) * blockDim.x) + threadIdx.x;

	//@ToDo - 
	//Alternative way found at: https://stackoverflow.com/questions/20205941/cuda-mandelbrot-set
	//Sounds more clear?? need check//
	//int row = blockIdx.y * blockDim.y + threadIdx.y;  // WIDTH
	//int col = blockIdx.x * blockDim.x + threadIdx.x;  // HEIGHT
	//int ixy = row * 64 + col;
	//int ixMy = (row-1) * 64 + col;
	//int ixPy = (row+1) * 64 + col;
	//int ixyM = row * 64 + (col-1);
	//int ixyP = row * 64 + (col+1);

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

	if (boundaryGain[ixy] > 0.00)
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

//__global__
//void complex_buffer_synthesis(float* grid, int* gridSize, float* boundaryGain, int samplesIndex, float* samples, float* excitation, int listenerPosition, int excitationPosition, float propagationFactor, float dampingFactor, int rotationIndex)
//{
//	//int blockId = blockIdx.y * gridDim.x + blockIdx.x;
//	//
//	////Get index for current and neighbouring nodes//
//	////@ToDo - Need to check this, different indexing to openCL!//
//	//int ixy = blockId * blockDim.x + threadIdx.x;
//	//int ixMy = (blockId - 1) * blockDim.x + threadIdx.x;
//	//int ixPy = (blockId + 1) * blockDim.x + threadIdx.x;
//	//int ixyM = (blockId) * blockDim.x + threadIdx.x - 1;
//	//int ixyP = (blockId)* blockDim.x + threadIdx.x + 1;
//
//	int theGridSize = *gridSize;
//
//	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
//	//int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
//	int ixy = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
//	int ixMy = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + (threadIdx.x - 1);
//	int ixPy = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + (threadIdx.x + 1);
//	int ixyM = blockId * (blockDim.x * blockDim.y) + ((threadIdx.y - 1) * blockDim.x) + threadIdx.x;
//	int ixyP = blockId * (blockDim.x * blockDim.y) + ((threadIdx.y + 1) * blockDim.x) + threadIdx.x;
//
//	//Determine each buffer in relation to time from a rotation index//
//	int idxMOne = (rotationIndex - 1) % 3;
//	int idxN = (rotationIndex);
//	int idxPOne = (rotationIndex + 1) % 3;
//
//	//Initalise pressure values//
//	float centrePressureNMO = grid[(idxMOne * theGridSize) + ixy];
//	float centrePressureN = grid[(idxN * theGridSize) + ixy];
//	float leftPressure;
//	float rightPressure;
//	float upPressure;
//	float downPressure;
//
//	//Predicate method//
//	//leftPressure = boundaryGain[ixy] > 0.0 ? n[ixy] * boundaryGain[ixy] : n[ixMy];
//	//rightPressure = boundaryGain[ixy] > 0.0 ? n[ixy] * boundaryGain[ixy] : n[ixPy];
//	//upPressure = boundaryGain[ixy] > 0.0 ? n[ixy] * boundaryGain[ixy] : n[ixyM];
//	//downPressure = boundaryGain[ixy] > 0.0 ? n[ixy] * boundaryGain[ixy] : n[ixyP];
//
//	if (boundaryGain[ixy] > 0.01)
//	{
//		leftPressure = centrePressureN * boundaryGain[ixy];
//		rightPressure = centrePressureN * boundaryGain[ixy];
//		upPressure = centrePressureN * boundaryGain[ixy];
//		downPressure = centrePressureN * boundaryGain[ixy];
//	}
//	else
//	{
//		leftPressure = grid[(idxN * theGridSize) + ixMy];
//		rightPressure = grid[(idxN * theGridSize) + ixPy];
//		upPressure = grid[(idxN * theGridSize) + ixyM];
//		downPressure = grid[(idxN * theGridSize) + ixyP];
//	}
//
//	//Calculate the nex pressure value//
//	float newPressure = 2 * centrePressureN;
//	newPressure += (dampingFactor - 1.0) * centrePressureNMO;
//	newPressure += propagationFactor * (leftPressure + rightPressure + upPressure + downPressure - (4 * centrePressureN));
//	newPressure *= 1.0 / (dampingFactor + 1.0);
//
//
//	//If the cell is the listener position, sets the next sound sample in buffer to value contained here//
//	if (ixy == listenerPosition)
//	{
//		samples[samplesIndex] = centrePressureN;
//		//printf("sample: %f \n", centrePressureN);
//	}
//
//	if (ixy == excitationPosition)	//If the position is an excitation...
//	{
//		newPressure += excitation[samplesIndex];	//Input excitation value into point. Then increment to next excitation in next iteration.
//	}
//	grid[(idxPOne * theGridSize) + ixy] = newPressure;
//}

__global__
void interrupted_buffer_processing(float* inputBuffer, float* outputBuffer)
{
	int32_t globalSize = gridDim.x * blockDim.x;
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	float attenuatedSample = inputBuffer[idx] * 0.5;
	//float attenuatedSample = inputBuffer[idx] * pow(M_E, -idx);
	outputBuffer[idx] = attenuatedSample;
}

namespace CUDA_Kernels
{

	void nullKernelExecute()
	{
		null_kernel << <1, 1>> > ();
	}
	void singleSampleExecute(float* singleSample)
	{
		size_t numBlocks = 1;
		dim3 threadsPerBlock(1);
		single_sample << <numBlocks, threadsPerBlock >> > (singleSample);
	}
	void copyBufferExecute(dim3 aGlobalSize, dim3 aLocalSize, float* cudaSrcBuffer, float* cudaDstBuffer)
	{
		dim3 numBlocks(aGlobalSize.x / aLocalSize.x, aGlobalSize.y / aLocalSize.y);
		dim3 threadsPerBlock(aLocalSize.x, aLocalSize.y);
		copy_buffer << <numBlocks, threadsPerBlock >> > (cudaSrcBuffer, cudaDstBuffer);
	}
	void simpleBufferProcessing(dim3 aGlobalSize, dim3 aLocalSize, float* cudaInputBuffer, float* cudaOutputBuffer)
	{
		dim3 numBlocks(aGlobalSize.x / aLocalSize.x, aGlobalSize.y / aLocalSize.y);
		dim3 threadsPerBlock(aLocalSize.x, aLocalSize.y);
		simple_buffer_processing << <numBlocks, threadsPerBlock >> > (cudaInputBuffer, cudaOutputBuffer);
	}
	void complexBufferProcessing(dim3 aGlobalSize, dim3 aLocalSize, float* cudaInputBuffer, float* cudaOutputBuffer)
	{
		dim3 numBlocks(aGlobalSize.x / aLocalSize.x, aGlobalSize.y / aLocalSize.y);
		dim3 threadsPerBlock(aLocalSize.x, aLocalSize.y);
		complex_buffer_processing << <numBlocks, threadsPerBlock >> > (cudaInputBuffer, cudaOutputBuffer);
	}

	void simpleBufferSynthesis(dim3 aGlobalSize, dim3 aLocalSize, int cudaSampleRate, float cudaFrequency, float* cudaOutputBuffer)
	{
		dim3 numBlocks(aGlobalSize.x / aLocalSize.x, aGlobalSize.y / aLocalSize.y);
		dim3 threadsPerBlock(aLocalSize.x, aLocalSize.y);
		simple_buffer_synthesis << <numBlocks, threadsPerBlock >> > (cudaSampleRate, cudaFrequency, cudaOutputBuffer);
	}
	void complexBufferSynthesis(dim3 aGlobalSize, dim3 aLocalSize, size_t aBufferSize, uint32_t aGridSize, int* gridLength, float* gridOne, float* gridTwo, float* gridThree, float* boundaryGain, int samplesIndex, float* samples, float* excitation, int listenerPosition, int excitationPosition, float propagationFactor, float dampingFactor, int rotationIndex)
	{
		//Grid size is the number of workitems to allocate. So implicit to the kernel//
		dim3 numBlocks(aGlobalSize.x / aLocalSize.x, aGlobalSize.y / aLocalSize.y);
		dim3 threadsPerBlock(aLocalSize.x, aLocalSize.y);
		complex_buffer_synthesis << <numBlocks, threadsPerBlock >> > (gridOne, gridTwo, gridThree, boundaryGain, samplesIndex, samples, excitation, listenerPosition, excitationPosition, propagationFactor, dampingFactor, rotationIndex);
		//complex_buffer_synthesis << <numBlocks, threadsPerBlock >> > (gridOne, gridLength, boundaryGain, samplesIndex, samples, excitation, listenerPosition, excitationPosition, propagationFactor, dampingFactor, rotationIndex);
	}
	void interruptedBufferProcessing(dim3 aGlobalSize, dim3 aLocalSize, float* sampleBuffer, float* outputBuffer)
	{
		dim3 numBlocks(aGlobalSize.x / aLocalSize.x, aGlobalSize.y / aLocalSize.y);
		dim3 threadsPerBlock(aLocalSize.x, aLocalSize.y);
		interrupted_buffer_processing << <numBlocks, threadsPerBlock >> > (sampleBuffer, outputBuffer);
	}
}