#ifndef CUDA_WRAPPER_HPP
#define CUDA_WRAPPER_HPP

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cuda.h>

#include "benchmarker.hpp"
#include "AudioFile.h"

//TIMING USING NVIDIA EVENTS//
//cudaEvent_t start, stop;
//float time;
//
//cudaEventCreate(&start);
//cudaEventCreate(&stop);
//
//cudaEventRecord(start, 0);
//kernel << <grid, threads >> > (d_odata, d_idata, size_x, size_y,
//	NUM_REPS);
//cudaEventRecord(stop, 0);
//cudaEventSynchronize(stop);
//
//cudaEventElapsedTime(&time, start, stop);
//cudaEventDestroy(start);
//cudaEventDestroy(stop);

namespace CUDA_Kernels
{
	void nullKernelExecute();
	void copyBufferExecute(size_t aN, float* srcBuffer, float* dstBuffer, float* cudaSrcBuffer, float* cudaDstBuffer);
	void singleSampleExecute(float* singleSample);
	void simpleBufferProcessing(size_t aN, float* inputBuffer, float* outputBuffer, float* cudaInputBuffer, float* cudaOutputBuffer);
	void complexBufferProcessing(size_t aN, float* inputBuffer, float* outputBuffer, float* cudaInputBuffer, float* cudaOutputBuffer);
	void simpleBufferSynthesis(size_t aN, int* cudaSampleRate, float* cudaFrequency, float* cudaOutputBuffer);
	void complexBufferSynthesis(size_t aBufferSize, size_t aGridSize, float* gridOne, float* gridTwo, float* gridThree, float* boundaryGain, int* samplesIndex, float* samples, float* excitation, int* listenerPosition, int* excitationPosition, float* propagationFactor, float* dampingFactor, int* rotationIndex);
	void interruptedBufferProcessing(size_t aBufferSize, float* sampleBuffer, float* outputBuffer);
}

class CUDA_Wrapper
{
private:
	static const uint32_t GIGA_BYTE = 1024 * 1024 * 1024;
	static const uint32_t MEGA_BYTE = 1024 * 1024;
	static const uint32_t KILO_BYTE = 1024;
	size_t standardBufferlengths[10];
	uint32_t bufferSize_ = GIGA_BYTE;
	uint32_t bufferLength_ = bufferSize_ / sizeof(float);
	uint32_t sampleRate_ = 44100;

	Benchmarker benchmarker_;
	
public:
	CUDA_Wrapper() : benchmarker_("cudalog.csv", { "Test_Name", "Total_Time", "Average_Time", "Max_Time", "Min_Time", "Max_Difference", "Average_Difference" })
	{
		for (size_t i = 0; i != 10; ++i)
			standardBufferlengths[i] = 2^i;
	}
	void cuda_000_nullkernel(size_t aN, bool isWarmup)
	{
		//Execute and average//
		std::cout << "Executing test: cuda_000_nullkernel" << std::endl;
		if (isWarmup)
			CUDA_Kernels::nullKernelExecute();
		for (uint32_t i = 0; i != aN; ++i)
		{
			benchmarker_.startTimer("cuda_000_nullkernel");
			CUDA_Kernels::nullKernelExecute();
			cudaDeviceSynchronize();
			//cudaEventSynchronize()
			//cudaEventQuery(0);
			benchmarker_.pauseTimer("cuda_000_nullkernel");
		}
		benchmarker_.elapsedTimer("cuda_000_nullkernel");

		bool isSuccessful = true;
		std::cout << "cuda_001_CPUtoGPU successful: " << isSuccessful << std::endl << std::endl;
	}

	void cuda_001_CPUtoGPU(size_t aN, bool isWarmup)
	{
		//Test preperation//
		float* hostBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		float* deviceBuffer;
		cudaMalloc((void**)&deviceBuffer, bufferSize_);

		//Execute and average//
		std::cout << "Executing test: cuda_001_CPUtoGPU" << std::endl;
		if (isWarmup)
			cudaMemcpy(deviceBuffer, hostBuffer, bufferSize_, cudaMemcpyHostToDevice);
		for (uint32_t i = 0; i != aN; ++i)
		{
			benchmarker_.startTimer("cuda_001_CPUtoGPU");
			cudaMemcpy(deviceBuffer, hostBuffer, bufferSize_, cudaMemcpyHostToDevice);
			//cudaMemcpyAsync(deviceBuffer, hostBuffer, bufferSize_, cudaMemcpyHostToDevice);
			benchmarker_.pauseTimer("cuda_001_CPUtoGPU");
		}
		benchmarker_.elapsedTimer("cuda_001_CPUtoGPU");

		//Check contents//
		bool isSuccessful = true;
		float* checkBuffer = new float[bufferLength_];
		cudaMemcpy(checkBuffer, deviceBuffer, bufferSize_, cudaMemcpyDeviceToHost);
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			if (checkBuffer[i] != hostBuffer[i])
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cuda_001_CPUtoGPU successful: " << isSuccessful << std::endl << std::endl;

		//Cleanup//
		cudaFree(deviceBuffer);

		delete(checkBuffer);
		delete(hostBuffer);
	}
	void cuda_002_GPUtoCPU(size_t aN, bool isWarmup)
	{
		//Test preperation//
		float* hostBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		float* deviceBuffer;
		cudaMalloc((void**)&deviceBuffer, bufferSize_);
		cudaMemcpy(deviceBuffer, hostBuffer, bufferSize_, cudaMemcpyHostToDevice);

		float* checkBuffer = new float[bufferLength_];

		//Execute and average//
		std::cout << "Executing test: cuda_002_GPUtoCPU" << std::endl;
		if (isWarmup)
			cudaMemcpy(checkBuffer, deviceBuffer, bufferSize_, cudaMemcpyDeviceToHost);
		for (uint32_t i = 0; i != aN; ++i)
		{
			benchmarker_.startTimer("cuda_002_GPUtoCPU");
			cudaMemcpy(checkBuffer, deviceBuffer, bufferSize_, cudaMemcpyDeviceToHost);
			//cudaMemcpyAsync(deviceBuffer, hostBuffer, bufferSize_, cudaMemcpyHostToDevice);
			benchmarker_.pauseTimer("cuda_002_GPUtoCPU");
		}
		benchmarker_.elapsedTimer("cuda_002_GPUtoCPU");

		//Check contents//
		bool isSuccessful = true;
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			if (checkBuffer[i] != hostBuffer[i])
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cuda_002_GPUtoCPU successful: " << isSuccessful << std::endl << std::endl;

		//Cleanup//
		cudaFree(deviceBuffer);

		delete(checkBuffer);
		delete(hostBuffer);
	}
	void cuda_003_CPUtoGPUtoCPU(size_t aN, bool isWarmup)
	{
		//Test preperation//
		float* hostBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		float* deviceBuffer;
		cudaMalloc((void**)&deviceBuffer, bufferSize_);

		float* checkBuffer = new float[bufferLength_];

		//Execute and average//
		std::cout << "Executing test: cuda_003_CPUtoGPUtoCPU" << std::endl;
		if (isWarmup)
			cudaMemcpy(deviceBuffer, hostBuffer, bufferSize_, cudaMemcpyHostToDevice);
		for (uint32_t i = 0; i != aN; ++i)
		{
			benchmarker_.startTimer("cuda_003_CPUtoGPUtoCPU");
			cudaMemcpy(deviceBuffer, hostBuffer, bufferSize_, cudaMemcpyHostToDevice);
			cudaMemcpy(checkBuffer, deviceBuffer, bufferSize_, cudaMemcpyDeviceToHost);
			//cudaMemcpyAsync(deviceBuffer, hostBuffer, bufferSize_, cudaMemcpyHostToDevice);
			benchmarker_.pauseTimer("cuda_003_CPUtoGPUtoCPU");
		}
		benchmarker_.elapsedTimer("cuda_003_CPUtoGPUtoCPU");

		//Check contents//
		bool isSuccessful = true;
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			if (checkBuffer[i] != hostBuffer[i])
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cuda_003_CPUtoGPUtoCPU successful: " << isSuccessful << std::endl << std::endl;

		//Cleanup//
		cudaFree(deviceBuffer);

		delete(checkBuffer);
		delete(hostBuffer);
	}
	////Essentially CPUtoGPUtoCPU but with pinned memory//
	//void cuda_004_pinnedmemory(size_t aN)
	//{
	//	std::cout << "Write to buffer using cudaMemcpy()" << std::endl;
	//	//Write to buffer using OpenCL cudaMemcpy()//
	//	float* d_buffer;
	//	cudaMalloc((void**)&d_buffer, sizeof(float) * bufferSize_);

	//	char* memoryBuffer = new char[bufferSize_];
	//	memset(memoryBuffer, '1', bufferSize_);

	//	char* memoryBufferCheck = new char[bufferSize_];
	//	memset(memoryBufferCheck, '1', bufferSize_);


	//	auto mappedMemory = openCL.pinMappedMemory("writeDstBuffer", bufferSize_);

	//	std::cout << "Write to buffer using enqueueMapBuffer()" << std::endl;
	//	for (uint32_t i = 0; i != aN; ++i)
	//	{
	//		benchmarker_.startTimer("writeToGPUMapped");
	//		memcpy(memoryBuffer, mappedMemory, bufferSize_);
	//		benchmarker_.pauseTimer("writeToGPUMapped");
	//	}
	//	benchmarker_.elapsedTimer("writeToGPUMapped");

	//	openCL.deleteBuffer("writeDstBuffer");

	//	delete memoryBuffer;
	//	delete memoryBufferCheck;

	//	//memset(memoryBufferCheck, '1', bufferSize_);
	//}

	//cudaMemcpy()
	//cudaMemcpyAsync()
	//cudaDeviceSynchronize()
	void cuda_005_cpymemory(size_t aN, bool isWarmup)
	{
		//Test preperation//
		float* srcMemoryBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			srcMemoryBuffer[i] = 42.0;
		float* dstMemoryBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			dstMemoryBuffer[i] = 0.0;

		float* d_srcBuffer;
		float* d_dstBuffer;
		cudaMalloc((void**)&d_srcBuffer, bufferSize_);
		cudaMalloc((void**)&d_dstBuffer, bufferSize_);

		cudaMemcpy(d_srcBuffer, srcMemoryBuffer, bufferSize_, cudaMemcpyHostToDevice);
		cudaMemcpy(d_dstBuffer, dstMemoryBuffer, bufferSize_, cudaMemcpyHostToDevice);

		//Execute and average//
		std::cout << "Executing test: cuda_005_cpymemory" << std::endl;
		if(isWarmup)
			cudaMemcpy(d_dstBuffer, d_srcBuffer, bufferSize_, cudaMemcpyDeviceToDevice);
		for (uint32_t i = 0; i != aN; ++i)
		{
			benchmarker_.startTimer("cuda_005_cpymemory");
			//cudaMemcpyAsync(d_srcBuffer, d_dstBuffer, bufferSize_, cudaMemcpyDeviceToDevice);
			cudaMemcpy(d_dstBuffer, d_srcBuffer, bufferSize_, cudaMemcpyDeviceToDevice);
			//cudaDeviceSynchronize();
			benchmarker_.pauseTimer("cuda_005_cpymemory");
		}
		benchmarker_.elapsedTimer("cuda_005_cpymemory");

		//Check contents//
		bool isSuccessful = true;
		cudaMemcpy(dstMemoryBuffer, d_dstBuffer, bufferSize_, cudaMemcpyDeviceToHost);
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			if (dstMemoryBuffer[i] != srcMemoryBuffer[i])
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cuda_005_cpymemory successful: " << isSuccessful << std::endl << std::endl;

		cudaFree(d_srcBuffer);
		cudaFree(d_dstBuffer);
		delete(srcMemoryBuffer);
		delete(dstMemoryBuffer);
	}
	void cuda_006_cpymemorykernel(size_t aN, bool isWarmup)
	{
		float* srcMemoryBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			srcMemoryBuffer[i] = 42.0;
		float* dstMemoryBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			dstMemoryBuffer[i] = 0.0;

		float* d_srcBuffer;
		float* d_dstBuffer;
		cudaMalloc((void**)&d_srcBuffer, bufferSize_);
		cudaMalloc((void**)&d_dstBuffer, bufferSize_);

		cudaMemcpy(d_srcBuffer, srcMemoryBuffer, bufferSize_, cudaMemcpyHostToDevice);
		cudaMemcpy(d_dstBuffer, dstMemoryBuffer, bufferSize_, cudaMemcpyHostToDevice);

		//Execute and average//
		std::cout << "Executing test: cuda_006_cpymemorykernel" << std::endl;
		if(isWarmup)
			CUDA_Kernels::copyBufferExecute(bufferLength_, srcMemoryBuffer, dstMemoryBuffer, d_srcBuffer, d_dstBuffer);
		for (uint32_t i = 0; i != aN; ++i)
		{
			benchmarker_.startTimer("cuda_006_cpymemorykernel");
			CUDA_Kernels::copyBufferExecute(bufferLength_, srcMemoryBuffer, dstMemoryBuffer, d_srcBuffer, d_dstBuffer);
			//cudaDeviceSynchronize();
			benchmarker_.pauseTimer("cuda_006_cpymemorykernel");
		}
		benchmarker_.elapsedTimer("cuda_006_cpymemorykernel");

		//Check contents//
		bool isSuccessful = true;
		cudaMemcpy(dstMemoryBuffer, d_dstBuffer, bufferSize_, cudaMemcpyDeviceToHost);
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			if (dstMemoryBuffer[i] != srcMemoryBuffer[i])
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cuda_006_cpymemorykernel successful: " << isSuccessful << std::endl << std::endl;

		cudaFree(d_srcBuffer);
		cudaFree(d_dstBuffer);
		delete(srcMemoryBuffer);
		delete(dstMemoryBuffer);
	}
	void cuda_007_singlesample(size_t aN, bool isWarmup)
	{
		float* hostSingleSample = new float;
		float* checkSingleSample = new float;
		*hostSingleSample = 42.0;
		*checkSingleSample = 0.0;

		float* deviceSingleBuffer;
		cudaMalloc((void**)&deviceSingleBuffer, sizeof(float));

		cudaMemcpy(deviceSingleBuffer, hostSingleSample, bufferSize_, cudaMemcpyHostToDevice);

		//Execute and average//
		std::cout << "Executing test: cuda_007_singlesample" << std::endl;
		if (isWarmup)
			CUDA_Kernels::singleSampleExecute(hostSingleSample);
		for (uint32_t i = 0; i != aN; ++i)
		{
			benchmarker_.startTimer("cuda_007_singlesample");
			CUDA_Kernels::singleSampleExecute(hostSingleSample);
			//cudaDeviceSynchronize();
			benchmarker_.pauseTimer("cuda_007_singlesample");
		}
		benchmarker_.elapsedTimer("cuda_007_singlesample");

		//Check contents//
		bool isSuccessful = true;
		cudaMemcpy(checkSingleSample, deviceSingleBuffer, sizeof(float), cudaMemcpyDeviceToHost);
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			if (checkSingleSample[i] != hostSingleSample[i] * 0.5)
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cuda_007_singlesample successful: " << isSuccessful << std::endl << std::endl;

		cudaFree(deviceSingleBuffer);
		delete(hostSingleSample);
		delete(checkSingleSample);
	}
	void cuda_008_simplebufferprocessing(size_t aN, bool isWarmup)
	{
		float* inputBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			inputBuffer[i] = 42.0;
		float* outputBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			outputBuffer[i] = 0.0;

		float* d_inputBuffer;
		float* d_outputBuffer;
		cudaMalloc((void**)&d_inputBuffer, bufferSize_);
		cudaMalloc((void**)&d_outputBuffer, bufferSize_);

		cudaMemcpy(d_inputBuffer, inputBuffer, bufferSize_, cudaMemcpyHostToDevice);
		cudaMemcpy(d_outputBuffer, outputBuffer, bufferSize_, cudaMemcpyHostToDevice);

		//Execute and average//
		std::cout << "Executing test: cuda_008_simplebufferprocessing" << std::endl;
		if (isWarmup)
		{
			CUDA_Kernels::simpleBufferProcessing(bufferLength_, inputBuffer, outputBuffer, d_inputBuffer, d_outputBuffer);
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			benchmarker_.startTimer("cuda_008_simplebufferprocessing");
			CUDA_Kernels::simpleBufferProcessing(bufferLength_, inputBuffer, outputBuffer, d_inputBuffer, d_outputBuffer);
			//cudaDeviceSynchronize();
			benchmarker_.pauseTimer("cuda_008_simplebufferprocessing");
		}
		benchmarker_.elapsedTimer("cuda_008_simplebufferprocessing");

		//Check contents//
		bool isSuccessful = true;
		cudaMemcpy(outputBuffer, d_outputBuffer, bufferSize_, cudaMemcpyDeviceToHost);
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			if (outputBuffer[i] != inputBuffer[i])
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cuda_008_simplebufferprocessing successful: " << isSuccessful << std::endl << std::endl;

		cudaFree(d_inputBuffer);
		cudaFree(d_outputBuffer);
		delete(inputBuffer);
		delete(outputBuffer);
	}
	void cuda_009_complexbufferprocessing(size_t aN, bool isWarmup)
	{
		float* inputBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			inputBuffer[i] = 42.0 * (i % 4);
		float* outputBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			outputBuffer[i] = 0.0;

		float* d_inputBuffer;
		float* d_outputBuffer;
		cudaMalloc((void**)&d_inputBuffer, bufferSize_);
		cudaMalloc((void**)&d_outputBuffer, bufferSize_);

		cudaMemcpy(d_inputBuffer, inputBuffer, bufferSize_, cudaMemcpyHostToDevice);
		cudaMemcpy(d_outputBuffer, outputBuffer, bufferSize_, cudaMemcpyHostToDevice);

		//Execute and average//
		std::cout << "Executing test: cuda_008_simplebufferprocessing" << std::endl;
		if (isWarmup)
		{
			CUDA_Kernels::complexBufferProcessing(bufferLength_, inputBuffer, outputBuffer, d_inputBuffer, d_outputBuffer);
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			benchmarker_.startTimer("cuda_008_simplebufferprocessing");
			CUDA_Kernels::complexBufferProcessing(bufferLength_, inputBuffer, outputBuffer, d_inputBuffer, d_outputBuffer);
			//cudaDeviceSynchronize();
			benchmarker_.pauseTimer("cuda_008_simplebufferprocessing");
		}
		benchmarker_.elapsedTimer("cuda_008_simplebufferprocessing");

		//Check contents//
		bool isSuccessful = true;
		cudaMemcpy(outputBuffer, d_outputBuffer, bufferSize_, cudaMemcpyDeviceToHost);
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			if (outputBuffer[i] != inputBuffer[i])
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cuda_008_simplebufferprocessing successful: " << isSuccessful << std::endl << std::endl;

		cudaFree(d_inputBuffer);
		cudaFree(d_outputBuffer);
		delete(inputBuffer);
		delete(outputBuffer);
	}
	void cuda_010_simplebuffersynthesis(size_t aN, bool isWarmup)
	{
		int* sampleRate = new int;
		*sampleRate = 44100;
		float* frequency = new float;
		*frequency = 1400.0;
		float* outputBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			outputBuffer[i] = 0.0;

		int* d_sampleRate;
		float* d_frequency;
		float* d_outputBuffer;
		cudaMalloc((void**)&d_sampleRate, sizeof(int));
		cudaMalloc((void**)&d_frequency, sizeof(float));
		cudaMalloc((void**)&d_outputBuffer, bufferSize_);

		cudaMemcpy(d_sampleRate, sampleRate, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_frequency, frequency, sizeof(float), cudaMemcpyHostToDevice);

		//Execute and average//
		std::cout << "Executing test: cuda_010_simplebuffersynthesis" << std::endl;
		if (isWarmup)
		{
			CUDA_Kernels::simpleBufferSynthesis(bufferLength_, d_sampleRate, d_frequency, d_outputBuffer);
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			benchmarker_.startTimer("cuda_010_simplebuffersynthesis");
			CUDA_Kernels::simpleBufferSynthesis(bufferLength_, d_sampleRate, d_frequency, d_outputBuffer);
			//cudaDeviceSynchronize();
			benchmarker_.pauseTimer("cuda_010_simplebuffersynthesis");
		}
		benchmarker_.elapsedTimer("cuda_010_simplebuffersynthesis");

		//Save audio to file for inspection//
		outputAudioFile("cl_010_simplebuffersynthesis.wav", outputBuffer, bufferLength_);
		std::cout << "cl_010_simplebuffersynthesis successful: Inspect audio log \"cl_010_simplebuffersynthesis.wav\"" << std::endl << std::endl;

		cudaFree(d_sampleRate); 
		cudaFree(d_frequency);
		cudaFree(d_outputBuffer);
		
		delete sampleRate;
		delete frequency;
		delete outputBuffer;
	}
	void cuda_011_complexbuffersynthesis(size_t aN, bool isWarmup)
	{
		//FDTD CUDA allocation//
		size_t bufferLength;
		size_t gridSize;
		float* d_gridOne;
		float* d_gridTwo;
		float* d_gridThree;
		float* d_boundaryGain;
		int* d_samplesIndex;
		float* d_samples;
		float* d_excitation;
		int* d_listenerPosition;
		int* d_excitationPosition;
		float* d_propagationFactor;
		float* d_dampingFactor;
		int* d_rotationIndex;
		cudaMalloc((void**)&d_gridOne, gridSize);
		cudaMalloc((void**)&d_gridTwo, gridSize);
		cudaMalloc((void**)&d_gridThree, gridSize);
		cudaMalloc((void**)&d_boundaryGain, gridSize);
		cudaMalloc((void**)&d_samplesIndex, sizeof(int));
		cudaMalloc((void**)&d_samples, bufferLength * sizeof(float));
		cudaMalloc((void**)&d_excitation, bufferLength * sizeof(float));
		cudaMalloc((void**)&d_listenerPosition, sizeof(int));
		cudaMalloc((void**)&d_excitationPosition, sizeof(int));
		cudaMalloc((void**)&d_propagationFactor, sizeof(float));
		cudaMalloc((void**)&d_dampingFactor, sizeof(float));
		cudaMalloc((void**)&d_rotationIndex, sizeof(int));

		//FDTD static variables//
		float* boundaryGain = new float[bufferLength];
		int* listenerPosition = new int;
		int* excitationPosition = new int;
		float* propagationFactor = new float;
		float* dampingFactor = new float;

		cudaMemcpy(d_boundaryGain, boundaryGain, gridSize, cudaMemcpyHostToDevice);
		cudaMemcpy(d_listenerPosition, listenerPosition, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_excitationPosition, excitationPosition, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_propagationFactor, propagationFactor, sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_dampingFactor, dampingFactor, sizeof(float), cudaMemcpyHostToDevice);

		//FDTD dynamic variables//
		int* samplesIndex = new int;
		float* samples = new float[bufferLength];
		float* excitation = new float[bufferLength];
		int* rotationIndex = new int;

		cudaMemcpy(d_samplesIndex, samplesIndex, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_samples, samples, bufferLength * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_excitation, excitation, bufferLength * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_rotationIndex, rotationIndex, sizeof(int), cudaMemcpyHostToDevice);

		//Initialise host variables to copy to GPU//
		(*rotationIndex) = 0;
		(*samplesIndex) = 0;
		for (size_t i = 0; i != bufferLength; ++i)
			excitation[i] = 0.5;

		//Execute and average//
		std::cout << "Executing test: cuda_011_complexbuffersynthesis" << std::endl;
		if (isWarmup)
		{
			CUDA_Kernels::complexBufferSynthesis(bufferLength, gridSize, d_gridOne, d_gridTwo, d_gridThree, d_boundaryGain, d_samplesIndex, d_samples, d_excitation, d_listenerPosition, d_excitationPosition, d_propagationFactor, d_dampingFactor, d_rotationIndex);
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			for (uint32_t j = 0; j != bufferLength; ++j)
			{
				//Update variables//
				cudaMemcpy(d_rotationIndex, rotationIndex, sizeof(int), cudaMemcpyHostToDevice);
				cudaMemcpy(d_samplesIndex, samplesIndex, sizeof(int), cudaMemcpyHostToDevice);
				cudaMemcpy(d_samples, samples, bufferLength * sizeof(float), cudaMemcpyHostToDevice);

				benchmarker_.startTimer("cuda_011_complexbuffersynthesis");
				CUDA_Kernels::complexBufferSynthesis(bufferLength, gridSize, d_gridOne, d_gridTwo, d_gridThree, d_boundaryGain, d_samplesIndex, d_samples, d_excitation, d_listenerPosition, d_excitationPosition, d_propagationFactor, d_dampingFactor, d_rotationIndex);
				//cudaDeviceSynchronize();
				benchmarker_.pauseTimer("cuda_011_complexbuffersynthesis");

				//Get synthesizer samples//
				cudaMemcpy(samples, d_samples, bufferLength * sizeof(float), cudaMemcpyDeviceToHost);

				(*samplesIndex)++;
				(*rotationIndex) = (*rotationIndex + 1) % 3;
			}
			(*samplesIndex) = 0;
		}
		benchmarker_.elapsedTimer("cuda_011_complexbuffersynthesis");

		cudaFree(d_gridOne);
		cudaFree(d_gridTwo);
		cudaFree(d_gridThree);
		cudaFree(d_boundaryGain);
		cudaFree(d_samplesIndex);
		cudaFree(d_samples);
		cudaFree(d_excitation);
		cudaFree(d_listenerPosition);
		cudaFree(d_excitationPosition);
		cudaFree(d_propagationFactor);
		cudaFree(d_dampingFactor);
		cudaFree(d_rotationIndex);
	}
	void cuda_012_interruptedbufferprocessing(size_t aN, bool isWarmup)
	{
		float* inputBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			inputBuffer[i] = 42.0;
		float* outputBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			outputBuffer[i] = 0.0;

		float* d_inputBuffer;
		float* d_outputBuffer;
		cudaMalloc((void**)&d_inputBuffer, bufferSize_);
		cudaMalloc((void**)&d_outputBuffer, bufferSize_);

		cudaMemcpy(d_inputBuffer, inputBuffer, bufferSize_, cudaMemcpyHostToDevice);
		cudaMemcpy(d_outputBuffer, outputBuffer, bufferSize_, cudaMemcpyHostToDevice);

		//Execute and average//
		std::cout << "Executing test: cuda_008_simplebufferprocessing" << std::endl;
		if (isWarmup)
		{
			CUDA_Kernels::simpleBufferProcessing(bufferLength_, inputBuffer, outputBuffer, d_inputBuffer, d_outputBuffer);
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			benchmarker_.startTimer("cuda_008_simplebufferprocessing");
			CUDA_Kernels::interruptedBufferProcessing(bufferLength_, d_inputBuffer, d_outputBuffer);
			//cudaDeviceSynchronize();
			benchmarker_.pauseTimer("cuda_008_simplebufferprocessing");
		}
		benchmarker_.elapsedTimer("cuda_008_simplebufferprocessing");

		//Check contents//
		bool isSuccessful = true;
		cudaMemcpy(outputBuffer, d_outputBuffer, bufferSize_, cudaMemcpyDeviceToHost);
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			if (outputBuffer[i] != inputBuffer[i])
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cuda_008_simplebufferprocessing successful: " << isSuccessful << std::endl << std::endl;

		cudaFree(d_inputBuffer);
		cudaFree(d_outputBuffer);
		delete(inputBuffer);
		delete(outputBuffer);
	}

	//This test finds buffer size where a unidirectional communication setup would loose real-time reliability//
	void cuda_013_unidirectionaltest()
	{
		//Test preperation//
		

		//Execute real-time test//
		uint32_t calculatedSamples = 0;
		while (calculatedSamples < sampleRate_)
		{
			//Process bufferlength_ of samples//
			benchmarker_.startTimer("cuda_013_unidirectionaltest");
			//...//
			benchmarker_.pauseTimer("cuda_013_unidirectionaltest");

			calculatedSamples += bufferLength_;
		}
		benchmarker_.elapsedTimer("cuda_013_unidirectionaltest");
	}
	//This test finds buffer size where a bidirectional communication setup would loose real-time reliability//
	void cuda_014_bidirectionaltest()
	{
		//Test preperation//
		float* hostBuffer = new float[bufferSize_];
		for (size_t i = 0; i != bufferSize_; ++i)
			hostBuffer[i] = 0.0;

		float* deviceBuffer;
		cudaMalloc((void**)&deviceBuffer, bufferSize_);

		//Execute real-time test//
		uint32_t calculatedSamples = 0;
		while (calculatedSamples < sampleRate_)
		{
			//Process bufferlength_ of samples//
			benchmarker_.startTimer("cuda_014_unidirectionaltest");
			cudaMemcpy(deviceBuffer, hostBuffer, bufferSize_, cudaMemcpyHostToDevice);
			//..///
			cudaMemcpy(hostBuffer, deviceBuffer, bufferSize_, cudaMemcpyDeviceToHost);
			benchmarker_.pauseTimer("cuda_014_unidirectionaltest");

			calculatedSamples += bufferLength_;
		}
		benchmarker_.elapsedTimer("cuda_014_unidirectionaltest");
	}

	static void outputAudioFile(const char* aPath, float* aAudioBuffer, uint32_t aAudioLength)
	{
		//SF_INFO sfinfo;
		//sfinfo.channels = 1;
		//sfinfo.samplerate = 44100;
		//sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;
		//
		////printf("writing: %s\n", file_path);
		//SNDFILE *outfile = sf_open(file_path, SFM_WRITE, &sfinfo);
		//if (sf_error(outfile) != SF_ERR_NO_ERROR) {
		//	printf("error: %s\n", sf_strerror(outfile));
		//}
		//sf_write_float(outfile, &audio_buffer[0], audio_length);
		//sf_write_sync(outfile);
		//sf_close(outfile);

		AudioFile<float> audioFile;
		AudioFile<float>::AudioBuffer buffer;

		buffer.resize(1);
		buffer[0].resize(aAudioLength);
		audioFile.setBitDepth(24);
		audioFile.setSampleRate(44100);

		for (int k = 0; k != aAudioLength; ++k)
			buffer[0][k] = (float)aAudioBuffer[k];

		audioFile.setAudioBuffer(buffer);
		audioFile.save(aPath);
	}

	static bool isCudaAvailable()
	{
		int deviceCount = 0;
		cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

		return (deviceCount > 0 ? true : false);
	}

	static void printAvailableDevices()
	{
		printf(
			" CUDA Device Query (Runtime API) version (CUDART static linking)\n\n");

		int deviceCount = 0;
		cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

		if (error_id != cudaSuccess) {
			printf("cudaGetDeviceCount returned %d\n-> %s\n",
				static_cast<int>(error_id), cudaGetErrorString(error_id));
			printf("Result = FAIL\n");
			//exit(EXIT_FAILURE);
		}

		// This function call returns 0 if there are no CUDA capable devices.
		if (deviceCount == 0) {
			printf("There are no available device(s) that support CUDA\n");
		}
		else {
			printf("Detected %d CUDA Capable device(s)\n", deviceCount);
		}

		int dev, driverVersion = 0, runtimeVersion = 0;

		for (dev = 0; dev < deviceCount; ++dev) {
			cudaSetDevice(dev);
			cudaDeviceProp deviceProp;
			cudaGetDeviceProperties(&deviceProp, dev);

			printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

			// Console log
			cudaDriverGetVersion(&driverVersion);
			cudaRuntimeGetVersion(&runtimeVersion);
			printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
				driverVersion / 1000, (driverVersion % 100) / 10,
				runtimeVersion / 1000, (runtimeVersion % 100) / 10);
			printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
				deviceProp.major, deviceProp.minor);

			char msg[256];
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
			sprintf_s(msg, sizeof(msg),
				"  Total amount of global memory:                 %.0f MBytes "
				"(%llu bytes)\n",
				static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
				(unsigned long long)deviceProp.totalGlobalMem);
#else
			snprintf(msg, sizeof(msg),
				"  Total amount of global memory:                 %.0f MBytes "
				"(%llu bytes)\n",
				static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
				(unsigned long long)deviceProp.totalGlobalMem);
#endif
			printf("%s", msg);

			printf("  (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores\n",
				deviceProp.multiProcessorCount,
				_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
				_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) *
				deviceProp.multiProcessorCount);
			printf(
				"  GPU Max Clock rate:                            %.0f MHz (%0.2f "
				"GHz)\n",
				deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);

#if CUDART_VERSION >= 5000
			// This is supported in CUDA 5.0 (runtime API device properties)
			printf("  Memory Clock rate:                             %.0f Mhz\n",
				deviceProp.memoryClockRate * 1e-3f);
			printf("  Memory Bus Width:                              %d-bit\n",
				deviceProp.memoryBusWidth);

			if (deviceProp.l2CacheSize) {
				printf("  L2 Cache Size:                                 %d bytes\n",
					deviceProp.l2CacheSize);
			}

#else
			// This only available in CUDA 4.0-4.2 (but these were only exposed in the
			// CUDA Driver API)
			int memoryClock;
			getCudaAttribute<int>(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
				dev);
			printf("  Memory Clock rate:                             %.0f Mhz\n",
				memoryClock * 1e-3f);
			int memBusWidth;
			getCudaAttribute<int>(&memBusWidth,
				CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
			printf("  Memory Bus Width:                              %d-bit\n",
				memBusWidth);
			int L2CacheSize;
			getCudaAttribute<int>(&L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);

			if (L2CacheSize) {
				printf("  L2 Cache Size:                                 %d bytes\n",
					L2CacheSize);
			}

#endif

			printf(
				"  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, "
				"%d), 3D=(%d, %d, %d)\n",
				deviceProp.maxTexture1D, deviceProp.maxTexture2D[0],
				deviceProp.maxTexture2D[1], deviceProp.maxTexture3D[0],
				deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
			printf(
				"  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
				deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
			printf(
				"  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d "
				"layers\n",
				deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
				deviceProp.maxTexture2DLayered[2]);

			printf("  Total amount of constant memory:               %zu bytes\n",
				deviceProp.totalConstMem);
			printf("  Total amount of shared memory per block:       %zu bytes\n",
				deviceProp.sharedMemPerBlock);
			printf("  Total number of registers available per block: %d\n",
				deviceProp.regsPerBlock);
			printf("  Warp size:                                     %d\n",
				deviceProp.warpSize);
			printf("  Maximum number of threads per multiprocessor:  %d\n",
				deviceProp.maxThreadsPerMultiProcessor);
			printf("  Maximum number of threads per block:           %d\n",
				deviceProp.maxThreadsPerBlock);
			printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
				deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
				deviceProp.maxThreadsDim[2]);
			printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
				deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
				deviceProp.maxGridSize[2]);
			printf("  Maximum memory pitch:                          %zu bytes\n",
				deviceProp.memPitch);
			printf("  Texture alignment:                             %zu bytes\n",
				deviceProp.textureAlignment);
			printf(
				"  Concurrent copy and kernel execution:          %s with %d copy "
				"engine(s)\n",
				(deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
			printf("  Run time limit on kernels:                     %s\n",
				deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
			printf("  Integrated GPU sharing Host Memory:            %s\n",
				deviceProp.integrated ? "Yes" : "No");
			printf("  Support host page-locked memory mapping:       %s\n",
				deviceProp.canMapHostMemory ? "Yes" : "No");
			printf("  Alignment requirement for Surfaces:            %s\n",
				deviceProp.surfaceAlignment ? "Yes" : "No");
			printf("  Device has ECC support:                        %s\n",
				deviceProp.ECCEnabled ? "Enabled" : "Disabled");
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
			printf("  CUDA Device Driver Mode (TCC or WDDM):         %s\n",
				deviceProp.tccDriver ? "TCC (Tesla Compute Cluster Driver)"
				: "WDDM (Windows Display Driver Model)");
#endif
			printf("  Device supports Unified Addressing (UVA):      %s\n",
				deviceProp.unifiedAddressing ? "Yes" : "No");
			printf("  Device supports Compute Preemption:            %s\n",
				deviceProp.computePreemptionSupported ? "Yes" : "No");
			printf("  Supports Cooperative Kernel Launch:            %s\n",
				deviceProp.cooperativeLaunch ? "Yes" : "No");
			printf("  Supports MultiDevice Co-op Kernel Launch:      %s\n",
				deviceProp.cooperativeMultiDeviceLaunch ? "Yes" : "No");
			printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n",
				deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);

			const char *sComputeMode[] = {
				"Default (multiple host threads can use ::cudaSetDevice() with device "
				"simultaneously)",
				"Exclusive (only one host thread in one process is able to use "
				"::cudaSetDevice() with this device)",
				"Prohibited (no host thread can use ::cudaSetDevice() with this "
				"device)",
				"Exclusive Process (many threads in one process is able to use "
				"::cudaSetDevice() with this device)",
				"Unknown",
				NULL };
			printf("  Compute Mode:\n");
			printf("     < %s >\n", sComputeMode[deviceProp.computeMode]);
		}

		// If there are 2 or more GPUs, query to determine whether RDMA is supported
		if (deviceCount >= 2) {
			cudaDeviceProp prop[64];
			int gpuid[64];  // we want to find the first two GPUs that can support P2P
			int gpu_p2p_count = 0;

			for (int i = 0; i < deviceCount; i++) {
				checkCudaErrors(cudaGetDeviceProperties(&prop[i], i));

				// Only boards based on Fermi or later can support P2P
				if ((prop[i].major >= 2)
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
					// on Windows (64-bit), the Tesla Compute Cluster driver for windows
					// must be enabled to support this
					&& prop[i].tccDriver
#endif
					) {
					// This is an array of P2P capable GPUs
					gpuid[gpu_p2p_count++] = i;
				}
			}

			// Show all the combinations of support P2P GPUs
			int can_access_peer;

			if (gpu_p2p_count >= 2) {
				for (int i = 0; i < gpu_p2p_count; i++) {
					for (int j = 0; j < gpu_p2p_count; j++) {
						if (gpuid[i] == gpuid[j]) {
							continue;
						}
						checkCudaErrors(
							cudaDeviceCanAccessPeer(&can_access_peer, gpuid[i], gpuid[j]));
						printf("> Peer access from %s (GPU%d) -> %s (GPU%d) : %s\n",
							prop[gpuid[i]].name, gpuid[i], prop[gpuid[j]].name, gpuid[j],
							can_access_peer ? "Yes" : "No");
					}
				}
			}
		}

		// csv masterlog info
		// *****************************
		// exe and CUDA driver name
		printf("\n");
		std::string sProfileString = "deviceQuery, CUDA Driver = CUDART";
		char cTemp[16];

		// driver version
		sProfileString += ", CUDA Driver Version = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
		sprintf_s(cTemp, 10, "%d.%d", driverVersion / 1000, (driverVersion % 100) / 10);
#else
		snprintf(cTemp, sizeof(cTemp), "%d.%d", driverVersion / 1000,
			(driverVersion % 100) / 10);
#endif
		sProfileString += cTemp;

		// Runtime version
		sProfileString += ", CUDA Runtime Version = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
		sprintf_s(cTemp, 10, "%d.%d", runtimeVersion / 1000, (runtimeVersion % 100) / 10);
#else
		snprintf(cTemp, sizeof(cTemp), "%d.%d", runtimeVersion / 1000,
			(runtimeVersion % 100) / 10);
#endif
		sProfileString += cTemp;

		// Device count
		sProfileString += ", NumDevs = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
		sprintf_s(cTemp, 10, "%d", deviceCount);
#else
		snprintf(cTemp, sizeof(cTemp), "%d", deviceCount);
#endif
		sProfileString += cTemp;
		sProfileString += "\n";
		printf("%s", sProfileString.c_str());

		printf("Result = PASS\n");

	}
};

#endif
