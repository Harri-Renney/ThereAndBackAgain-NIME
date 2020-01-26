#ifndef GPU_OVERHEAD_BENCHMARKS_CUDA_HPP
#define GPU_OVERHEAD_BENCHMARKS_CUDA_HPP

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cuda.h>
#include <random>

#include "GPU_Overhead_Benchmarks.hpp"
#include "benchmarker.hpp"
#include "AudioFile.h"
#include "FDTD_Grid.hpp"

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
	void copyBufferExecute(dim3 aGlobalSize, dim3 aLocalSize, float* cudaSrcBuffer, float* cudaDstBuffer);
	void singleSampleExecute(float* singleSample);
	void simpleBufferProcessing(dim3 aGlobalSize, dim3 aLocalSize, float* cudaInputBuffer, float* cudaOutputBuffer);
	void complexBufferProcessing(dim3 aGlobalSize, dim3 aLocalSize, float* cudaInputBuffer, float* cudaOutputBuffer);
	void simpleBufferSynthesis(dim3 aGlobalSize, dim3 aLocalSize, int* cudaSampleRate, float* cudaFrequency, float* cudaOutputBuffer);
	void complexBufferSynthesis(dim3 aGlobalSize, dim3 aLocalSize, size_t aBufferSize, size_t aGridSize, float* gridOne, float* gridTwo, float* gridThree, float* boundaryGain, int* samplesIndex, float* samples, float* excitation, int* listenerPosition, int* excitationPosition, float* propagationFactor, float* dampingFactor, int* rotationIndex);
	void interruptedBufferProcessing(dim3 aGlobalSize, dim3 aLocalSize, float* sampleBuffer, float* outputBuffer);
}

class GPU_Overhead_Benchmarks_CUDA : GPU_Overhead_Benchmarks
{
private:
	Benchmarker cudaBenchmarker_;

	dim3 globalWorkspace_;
	dim3 localWorkspace_;

	//CUDA Profiling//
	cudaEvent_t cudaEventStart;
	cudaEvent_t cudaEventEnd;
	float cudaTimeElapsed = 0.0f;
	
public:
	GPU_Overhead_Benchmarks_CUDA() : cudaBenchmarker_("cudalog.csv", { "Test_Name", "Total_Time", "Average_Time", "Max_Time", "Min_Time", "Max_Difference", "Average_Difference" })
	{
		bufferSizes[0] = 1;
		for (size_t i = 1; i != bufferSizesLength; ++i)
		{
			bufferSizes[i] = bufferSizes[i - 1] * 2;
		}
	}

	void runGeneralBenchmarks(uint64_t aNumRepetitions) override
	{
		for (uint32_t i = 0; i != bufferSizesLength; ++i)
		{
			uint64_t currentBufferSize = bufferSizes[i];
			std::string benchmarkFileName = "cuda_";
			std::string strBufferSize = std::to_string(currentBufferSize);
			benchmarkFileName.append("buffersize");
			benchmarkFileName.append(strBufferSize);
			benchmarkFileName.append(".csv");
			cudaBenchmarker_ = Benchmarker(benchmarkFileName, { "Test_Name", "Total_Time", "Average_Time", "Max_Time", "Min_Time", "Max_Difference", "Average_Difference" });
			setBufferLength(currentBufferSize);

			//Run tests with setup//
			cuda_nullkernel(aNumRepetitions, true);
			cuda_cputogpu_standard(aNumRepetitions, true);
			cuda_cputogpu_unified(aNumRepetitions, true);
			cuda_gputocpu_standard(aNumRepetitions, true);
			cuda_gputocpu_unified(aNumRepetitions, true);
			cuda_cputogputocpu_standard(aNumRepetitions, true);
			cuda_cputogputocpu_unified(aNumRepetitions, true);
			cuda_devicetransfer_standard(aNumRepetitions, true);
			cuda_devicetransfer_unified(aNumRepetitions, true);
			cuda_devicetransferkernel_standard(aNumRepetitions, true);
			cuda_devicetransferkernel_unified(aNumRepetitions, true);
			cuda_simplebufferprocessing_standard(aNumRepetitions, true);
			cuda_simplebufferprocessing_unified(aNumRepetitions, true);
			cuda_complexbufferprocessing_standard(aNumRepetitions, true);
			cuda_complexbufferprocessing_unified(aNumRepetitions, true);
			cuda_simplebuffersynthesis_standard(aNumRepetitions, true);
			cuda_simplebuffersynthesis_unified(aNumRepetitions, true);
			cuda_complexbuffersynthesis_standard(aNumRepetitions, true);
			cuda_complexbuffersynthesis_unified(aNumRepetitions, true);
			cuda_interruptedbufferprocessing_standard(aNumRepetitions, true);
			cuda_interruptedbufferprocessing_unified(aNumRepetitions, true);
		}
	}
	void runRealTimeBenchmarks(uint64_t aFrameRate) override
	{
		cuda_unidirectional_baseline(aFrameRate, true);
		cuda_unidirectional_processing(aFrameRate, true);
		cuda_bidirectional_baseline(aFrameRate, true);
		cuda_bidirectional_processing(aFrameRate, true);
	}

	void cuda_nullkernel(size_t aN, bool isWarmup)
	{
		//Execute & Profile//
		std::cout << "Executing test: cuda_nullkernel" << std::endl;
		if (isWarmup)
		{
			CUDA_Kernels::nullKernelExecute();
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			cudaBenchmarker_.startTimer("cuda_nullkernel");
			CUDA_Kernels::nullKernelExecute();
			cudaDeviceSynchronize();
			cudaBenchmarker_.pauseTimer("cuda_nullkernel");
		}
		cudaBenchmarker_.elapsedTimer("cuda_nullkernel");

		bool isSuccessful = true;
		std::cout << "cuda_nullkernel successful: " << isSuccessful << std::endl << std::endl;
	}
	void cuda_cputogpu_standard(size_t aN, bool isWarmup)
	{
		//Test preperation//
		float* checkBuffer = new float[bufferLength_];
		float* hostBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		float* deviceBuffer;
		cudaMalloc((void**)&deviceBuffer, bufferSize_);

		//Execute & Profile//
		std::cout << "Executing test: cuda_cputogpu_standard" << std::endl;
		if (isWarmup)
		{
			cudaMemcpy(deviceBuffer, hostBuffer, bufferSize_, cudaMemcpyHostToDevice);
			cudaDeviceSynchronize();
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			cudaBenchmarker_.startTimer("cuda_cputogpu_standard");
			cudaMemcpy(deviceBuffer, hostBuffer, bufferSize_, cudaMemcpyHostToDevice);
			cudaDeviceSynchronize();
			cudaBenchmarker_.pauseTimer("cuda_cputogpu_standard");
		}
		cudaBenchmarker_.elapsedTimer("cuda_cputogpu_standard");

		//Check contents//
		bool isSuccessful = true;
		cudaMemcpy(checkBuffer, deviceBuffer, bufferSize_, cudaMemcpyDeviceToHost);
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			if (checkBuffer[i] != hostBuffer[i])
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cuda_cputogpu_standard successful: " << isSuccessful << std::endl << std::endl;

		//Cleanup//
		cudaFree(deviceBuffer);

		delete(checkBuffer);
		delete(hostBuffer);
	}
	void cuda_cputogpu_unified(size_t aN, bool isWarmup)
	{
		//Test preperation//
		float* checkBuffer = new float[bufferLength_];
		float* hostBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		float* hostUnifiedBuffer;
		cudaMallocManaged(&hostUnifiedBuffer, bufferSize_);

		//Execute & Profile//
		std::cout << "Executing test: cuda_cputogpu_unified" << std::endl;
		if (isWarmup)
		{
			cudaMemcpy(hostUnifiedBuffer, hostBuffer, bufferSize_, cudaMemcpyHostToHost);
			cudaDeviceSynchronize();
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			cudaBenchmarker_.startTimer("cuda_cputogpu_unified");
			cudaMemcpy(hostUnifiedBuffer, hostBuffer, bufferSize_, cudaMemcpyHostToHost);
			cudaDeviceSynchronize();
			cudaBenchmarker_.pauseTimer("cuda_cputogpu_unified");
		}
		cudaBenchmarker_.elapsedTimer("cuda_cputogpu_unified");

		//Check contents//
		bool isSuccessful = true;
		cudaMemcpy(checkBuffer, hostUnifiedBuffer, bufferSize_, cudaMemcpyHostToHost);
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			if (checkBuffer[i] != hostBuffer[i])
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cuda_cputogpu_unified successful: " << isSuccessful << std::endl << std::endl;

		//Cleanup//
		cudaFree(hostUnifiedBuffer);

		delete(checkBuffer);
		delete(hostBuffer);
	}
	void cuda_cputogpu_pinned(size_t aN, bool isWarmup)
	{
		//Test preperation//
		float* checkBuffer = new float[bufferLength_];
		float* hostBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		float* hostPinnedBuffer;
		cudaMallocHost(&hostPinnedBuffer, bufferSize_);

		//Execute & Profile//
		std::cout << "Executing test: cuda_cputogpu_pinned" << std::endl;
		if (isWarmup)
		{
			cudaMemcpy(hostPinnedBuffer, hostBuffer, bufferSize_, cudaMemcpyHostToHost);
			cudaDeviceSynchronize();
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			cudaBenchmarker_.startTimer("cuda_cputogpu_pinned");
			cudaMemcpy(hostPinnedBuffer, hostBuffer, bufferSize_, cudaMemcpyHostToHost);
			cudaDeviceSynchronize();
			cudaBenchmarker_.pauseTimer("cuda_cputogpu_pinned");
		}
		cudaBenchmarker_.elapsedTimer("cuda_cputogpu_pinned");

		//Check contents//
		bool isSuccessful = true;
		cudaMemcpy(checkBuffer, hostBuffer, bufferSize_, cudaMemcpyHostToHost);
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			if (checkBuffer[i] != hostBuffer[i])
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cuda_cputogpu_pinned successful: " << isSuccessful << std::endl << std::endl;

		//Cleanup//
		cudaFree(hostPinnedBuffer);

		delete(checkBuffer);
		delete(hostBuffer);
	}
	void cuda_gputocpu_standard(size_t aN, bool isWarmup)
	{
		//Test preperation//
		float* checkBuffer = new float[bufferLength_];
		float* hostBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		float* deviceBuffer;
		cudaMalloc((void**)&deviceBuffer, bufferSize_);

		cudaMemcpy(deviceBuffer, hostBuffer, bufferSize_, cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();

		//Execute & Profile//
		std::cout << "Executing test: cuda_gputocpu_standard" << std::endl;
		if (isWarmup)
		{
			cudaMemcpy(checkBuffer, deviceBuffer, bufferSize_, cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			cudaBenchmarker_.startTimer("cuda_gputocpu_standard");
			cudaMemcpy(checkBuffer, deviceBuffer, bufferSize_, cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
			cudaBenchmarker_.pauseTimer("cuda_gputocpu_standard");
		}
		cudaBenchmarker_.elapsedTimer("cuda_gputocpu_standard");

		//Check contents//
		bool isSuccessful = true;
		cudaMemcpy(checkBuffer, deviceBuffer, bufferSize_, cudaMemcpyDeviceToHost);
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			if (checkBuffer[i] != hostBuffer[i])
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cuda_gputocpu_standard successful: " << isSuccessful << std::endl << std::endl;

		//Cleanup//
		cudaFree(deviceBuffer);

		delete(checkBuffer);
		delete(hostBuffer);
	}
	void cuda_gputocpu_unified(size_t aN, bool isWarmup)
	{
		//Test preperation//
		float* checkBuffer = new float[bufferLength_];
		float* hostBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		float* hostUnifiedBuffer;
		cudaMallocManaged(&hostUnifiedBuffer, bufferSize_);

		cudaMemcpy(hostUnifiedBuffer, hostBuffer, bufferSize_, cudaMemcpyHostToHost);
		cudaDeviceSynchronize();

		//Execute & Profile//
		std::cout << "Executing test: cuda_gputocpu_unified" << std::endl;
		if (isWarmup)
		{
			cudaMemcpy(checkBuffer, hostUnifiedBuffer, bufferSize_, cudaMemcpyHostToHost);
			cudaDeviceSynchronize();
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			cudaBenchmarker_.startTimer("cuda_gputocpu_unified");
			cudaMemcpy(checkBuffer, hostUnifiedBuffer, bufferSize_, cudaMemcpyHostToHost);
			cudaDeviceSynchronize();
			cudaBenchmarker_.pauseTimer("cuda_gputocpu_unified");
		}
		cudaBenchmarker_.elapsedTimer("cuda_gputocpu_unified");

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
		std::cout << "cuda_gputocpu_unified successful: " << isSuccessful << std::endl << std::endl;

		//Cleanup//
		cudaFree(hostUnifiedBuffer);

		delete(checkBuffer);
		delete(hostBuffer);
	}
	void cuda_gputocpu_pinned(size_t aN, bool isWarmup)
	{
		//Test preperation//
		float* hostBuffer = new float[bufferLength_];
		float* checkBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		float* hostPinnedBuffer;
		cudaMallocHost(&hostPinnedBuffer, bufferSize_);

		cudaMemcpy(hostPinnedBuffer, hostBuffer, bufferSize_, cudaMemcpyHostToHost);
		cudaDeviceSynchronize();

		//Execute & Profile//
		std::cout << "Executing test: cuda_gputocpu_pinned" << std::endl;
		if (isWarmup)
		{
			cudaMemcpy(checkBuffer, hostPinnedBuffer, bufferSize_, cudaMemcpyHostToHost);
			cudaDeviceSynchronize();
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			cudaBenchmarker_.startTimer("cuda_gputocpu_pinned");
			cudaMemcpy(checkBuffer, hostPinnedBuffer, bufferSize_, cudaMemcpyHostToHost);
			cudaDeviceSynchronize();
			cudaBenchmarker_.pauseTimer("cuda_gputocpu_pinned");
		}
		cudaBenchmarker_.elapsedTimer("cuda_gputocpu_pinned");

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
		std::cout << "cuda_gputocpu_pinned successful: " << isSuccessful << std::endl << std::endl;

		//Cleanup//
		cudaFree(hostPinnedBuffer);

		delete(checkBuffer);
		delete(hostBuffer);
	}
	void cuda_cputogputocpu_standard(size_t aN, bool isWarmup)
	{
		//Test preperation//
		float* checkBuffer = new float[bufferLength_];
		float* hostBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		float* deviceBuffer;
		cudaMalloc((void**)&deviceBuffer, bufferSize_);

		//Execute & Profile//
		std::cout << "Executing test: cuda_cputogputocpu_standard" << std::endl;
		if (isWarmup)
		{
			cudaMemcpy(deviceBuffer, hostBuffer, bufferSize_, cudaMemcpyHostToDevice);
			cudaDeviceSynchronize();
			cudaMemcpy(hostBuffer, deviceBuffer, bufferSize_, cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			cudaBenchmarker_.startTimer("cuda_cputogputocpu_standard");
			cudaMemcpy(deviceBuffer, hostBuffer, bufferSize_, cudaMemcpyHostToDevice);
			cudaDeviceSynchronize();
			cudaMemcpy(hostBuffer, deviceBuffer, bufferSize_, cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
			cudaBenchmarker_.pauseTimer("cuda_cputogputocpu_standard");
		}
		cudaBenchmarker_.elapsedTimer("cuda_cputogputocpu_standard");

		//Check contents//
		bool isSuccessful = true;
		cudaMemcpy(checkBuffer, deviceBuffer, bufferSize_, cudaMemcpyDeviceToHost);
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			if (checkBuffer[i] != hostBuffer[i])
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cuda_cputogputocpu_standard successful: " << isSuccessful << std::endl << std::endl;

		//Cleanup//
		cudaFree(deviceBuffer);

		delete(checkBuffer);
		delete(hostBuffer);
	}
	void cuda_cputogputocpu_unified(size_t aN, bool isWarmup)
	{
		//Test preperation//
		float* checkBuffer = new float[bufferLength_];
		float* hostBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		float* hostUnifiedBuffer;
		cudaMallocManaged(&hostUnifiedBuffer, bufferSize_);
		//Execute & Profile//
		std::cout << "Executing test: cuda_cputogputocpu_unified" << std::endl;
		if (isWarmup)
		{
			cudaMemcpy(hostUnifiedBuffer, hostBuffer, bufferSize_, cudaMemcpyHostToHost);
			cudaDeviceSynchronize();
			cudaMemcpy(checkBuffer, hostUnifiedBuffer, bufferSize_, cudaMemcpyHostToHost);
			cudaDeviceSynchronize();
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			cudaBenchmarker_.startTimer("cuda_cputogputocpu_unified");
			cudaMemcpy(hostUnifiedBuffer, hostBuffer, bufferSize_, cudaMemcpyHostToHost);
			cudaDeviceSynchronize();
			cudaMemcpy(checkBuffer, hostUnifiedBuffer, bufferSize_, cudaMemcpyHostToHost);
			cudaDeviceSynchronize();
			cudaBenchmarker_.pauseTimer("cuda_cputogputocpu_unified");
		}
		cudaBenchmarker_.elapsedTimer("cuda_cputogputocpu_unified");

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
		std::cout << "cuda_cputogputocpu_unified successful: " << isSuccessful << std::endl << std::endl;

		//Cleanup//
		cudaFree(hostUnifiedBuffer);

		delete(checkBuffer);
		delete(hostBuffer);
	}
	void cuda_cputogputocpu_pinnedmemory(size_t aN, bool isWarmup)
	{
		//Test preperation//
		float* checkBuffer = new float[bufferLength_];
		float* hostBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		float* hostPinnedBuffer;
		cudaMallocHost(&hostPinnedBuffer, bufferSize_);

		//Execute & Profile//
		std::cout << "Executing test: cuda_cputogputocpu_pinnedmemory" << std::endl;
		if (isWarmup)
		{
			cudaMemcpy(hostPinnedBuffer, hostBuffer, bufferSize_, cudaMemcpyHostToHost);
			cudaDeviceSynchronize();
			cudaMemcpy(checkBuffer, hostPinnedBuffer, bufferSize_, cudaMemcpyHostToHost);
			cudaDeviceSynchronize();
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			cudaBenchmarker_.startTimer("cuda_cputogputocpu_pinnedmemory");
			cudaMemcpy(hostPinnedBuffer, hostBuffer, bufferSize_, cudaMemcpyHostToHost);
			cudaDeviceSynchronize();
			cudaMemcpy(checkBuffer, hostPinnedBuffer, bufferSize_, cudaMemcpyHostToHost);
			cudaDeviceSynchronize();
			cudaBenchmarker_.pauseTimer("cuda_cputogputocpu_pinnedmemory");
		}
		cudaBenchmarker_.elapsedTimer("cuda_cputogputocpu_pinnedmemory");

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
		std::cout << "cuda_cputogputocpu_pinnedmemory successful: " << isSuccessful << std::endl << std::endl;

		//Cleanup//
		cudaFree(hostPinnedBuffer);

		delete(checkBuffer);
		delete(hostBuffer);
	}
	void cuda_devicetransfer_standard(size_t aN, bool isWarmup)
	{
		//Test preperation//
		float* checkBuffer = new float[bufferLength_];
		float* hostBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		float* deviceBufferSrc;
		float* deviceBufferDst;
		cudaMalloc((void**)&deviceBufferSrc, bufferSize_);
		cudaMalloc((void**)&deviceBufferDst, bufferSize_);

		cudaMemcpy(deviceBufferSrc, hostBuffer, bufferSize_, cudaMemcpyHostToDevice);

		//Execute & Profile//
		std::cout << "Executing test: cuda_devicetransfer_standard" << std::endl;
		if (isWarmup)
		{
			cudaMemcpy(deviceBufferDst, deviceBufferSrc, bufferSize_, cudaMemcpyDeviceToDevice);
			cudaDeviceSynchronize();
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			cudaBenchmarker_.startTimer("cuda_devicetransfer_standard");
			cudaMemcpy(deviceBufferDst, deviceBufferSrc, bufferSize_, cudaMemcpyDeviceToDevice);
			cudaDeviceSynchronize();
			cudaBenchmarker_.pauseTimer("cuda_devicetransfer_standard");
		}
		cudaBenchmarker_.elapsedTimer("cuda_devicetransfer_standard");

		//Check contents//
		bool isSuccessful = true;
		cudaMemcpy(checkBuffer, deviceBufferDst, bufferSize_, cudaMemcpyDeviceToHost);
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			if (checkBuffer[i] != hostBuffer[i])
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cuda_devicetransfer_standard successful: " << isSuccessful << std::endl << std::endl;

		//Cleanup//
		cudaFree(deviceBufferSrc);
		cudaFree(deviceBufferDst);

		delete(checkBuffer);
		delete(hostBuffer);
	}
	void cuda_devicetransfer_unified(size_t aN, bool isWarmup)
	{
		//Test preperation//
		float* checkBuffer = new float[bufferLength_];
		float* hostBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		float* hostUnifiedBufferSrc;
		float* hostUnifiedBufferDst;
		cudaMallocManaged((void**)&hostUnifiedBufferSrc, bufferSize_);
		cudaMallocManaged((void**)&hostUnifiedBufferDst, bufferSize_);

		for (size_t i = 0; i != bufferLength_; ++i)
			hostUnifiedBufferSrc[i] = 42.0;

		//Execute & Profile//
		std::cout << "Executing test: cuda_devicetransfer_unified" << std::endl;
		if (isWarmup)
		{
			cudaMemcpy(hostUnifiedBufferDst, hostUnifiedBufferSrc, bufferSize_, cudaMemcpyHostToHost);	//@ToDo - These are host unified memory, so does cudaMemcpyDeviceToDevice work?
			cudaDeviceSynchronize();
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			cudaBenchmarker_.startTimer("cuda_devicetransfer_unified");
			cudaMemcpy(hostUnifiedBufferDst, hostUnifiedBufferSrc, bufferSize_, cudaMemcpyHostToHost);	//@ToDo - These are host unified memory, so does cudaMemcpyDeviceToDevice work?
			cudaDeviceSynchronize();
			cudaBenchmarker_.pauseTimer("cuda_devicetransfer_unified");
		}
		cudaBenchmarker_.elapsedTimer("cuda_devicetransfer_unified");

		//Check contents//
		bool isSuccessful = true;
		cudaMemcpy(checkBuffer, hostUnifiedBufferDst, bufferSize_, cudaMemcpyHostToHost);
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			if (checkBuffer[i] != hostBuffer[i])
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cuda_devicetransfer_unified successful: " << isSuccessful << std::endl << std::endl;

		//Cleanup//
		cudaFree(hostUnifiedBufferSrc);
		cudaFree(hostUnifiedBufferDst);

		delete(checkBuffer);
		delete(hostBuffer);
	}
	void cuda_devicetransfer_pinned(size_t aN, bool isWarmup)
	{
		//Test preperation//
		float* checkBuffer = new float[bufferLength_];
		float* hostBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		float* hostPinnedBufferSrc;
		float* hostPinnedBufferDst;
		cudaMallocHost((void**)&hostPinnedBufferSrc, bufferSize_);
		cudaMallocHost((void**)&hostPinnedBufferDst, bufferSize_);

		for (size_t i = 0; i != bufferLength_; ++i)
			hostPinnedBufferSrc[i] = 42.0;

		//Execute & Profile//
		std::cout << "Executing test: cuda_devicetransfer_pinned" << std::endl;
		if (isWarmup)
		{
			cudaMemcpy(hostPinnedBufferDst, hostPinnedBufferSrc, bufferSize_, cudaMemcpyHostToHost);	//@ToDo - These are host unified memory, so does cudaMemcpyDeviceToDevice work?
			cudaDeviceSynchronize();
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			cudaBenchmarker_.startTimer("cuda_devicetransfer_pinned");
			cudaMemcpy(hostPinnedBufferDst, hostPinnedBufferSrc, bufferSize_, cudaMemcpyHostToHost);	//@ToDo - These are host unified memory, so does cudaMemcpyDeviceToDevice work?
			cudaDeviceSynchronize();
			cudaBenchmarker_.pauseTimer("cuda_devicetransfer_pinned");
		}
		cudaBenchmarker_.elapsedTimer("cuda_devicetransfer_pinned");

		//Check contents//
		bool isSuccessful = true;
		cudaMemcpy(checkBuffer, hostPinnedBufferDst, bufferSize_, cudaMemcpyHostToHost);
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			if (checkBuffer[i] != hostBuffer[i])
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cuda_devicetransfer_pinned successful: " << isSuccessful << std::endl << std::endl;

		//Cleanup//
		cudaFree(hostPinnedBufferSrc);
		cudaFree(hostPinnedBufferDst);

		delete(checkBuffer);
		delete(hostBuffer);
	}
	void cuda_devicetransferkernel_standard(size_t aN, bool isWarmup)
	{
		//Test preperation//
		float* checkBuffer = new float[bufferLength_];
		float* hostBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		float* deviceBufferSrc;
		float* deviceBufferDst;
		cudaMalloc((void**)&deviceBufferSrc, bufferSize_);
		cudaMalloc((void**)&deviceBufferDst, bufferSize_);

		cudaMemcpy(deviceBufferSrc, hostBuffer, bufferSize_, cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();

		//Execute & Profile//
		std::cout << "Executing test: cuda_devicetransferkernel_standard" << std::endl;
		if (isWarmup)
		{
			CUDA_Kernels::copyBufferExecute(globalWorkspace_, localWorkspace_, deviceBufferSrc, deviceBufferDst);
			cudaDeviceSynchronize();
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			cudaBenchmarker_.startTimer("cuda_devicetransferkernel_standard");
			CUDA_Kernels::copyBufferExecute(globalWorkspace_, localWorkspace_, deviceBufferSrc, deviceBufferDst);
			cudaDeviceSynchronize();
			cudaBenchmarker_.pauseTimer("cuda_devicetransferkernel_standard");
		}
		cudaBenchmarker_.elapsedTimer("cuda_devicetransferkernel_standard");

		//Check contents//
		bool isSuccessful = true;
		cudaMemcpy(checkBuffer, deviceBufferDst, bufferSize_, cudaMemcpyDeviceToHost);
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			if (checkBuffer[i] != hostBuffer[i])
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cuda_devicetransferkernel_standard successful: " << isSuccessful << std::endl << std::endl;

		//Cleanup//
		cudaFree(deviceBufferSrc);
		cudaFree(deviceBufferDst);

		delete(checkBuffer);
		delete(hostBuffer);
	}
	void cuda_devicetransferkernel_unified(size_t aN, bool isWarmup)
	{
		//Test preperation//
		float* checkBuffer = new float[bufferLength_];
		float* hostBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		float* hostUnifiedBufferSrc;
		float* hostUnifiedBufferDst;
		cudaMallocManaged((void**)&hostUnifiedBufferSrc, bufferSize_);
		cudaMallocManaged((void**)&hostUnifiedBufferDst, bufferSize_);

		for (size_t i = 0; i != bufferLength_; ++i)
			hostUnifiedBufferSrc[i] = 42.0;

		//Execute & Profile//
		std::cout << "Executing test: cuda_devicetransferkernel_unified" << std::endl;
		if (isWarmup)
		{
			CUDA_Kernels::copyBufferExecute(globalWorkspace_, localWorkspace_, hostUnifiedBufferSrc, hostUnifiedBufferDst);
			cudaDeviceSynchronize();
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			cudaBenchmarker_.startTimer("cuda_devicetransferkernel_unified");
			CUDA_Kernels::copyBufferExecute(globalWorkspace_, localWorkspace_, hostUnifiedBufferSrc, hostUnifiedBufferDst);
			cudaDeviceSynchronize();
			cudaBenchmarker_.pauseTimer("cuda_devicetransferkernel_unified");
		}
		cudaBenchmarker_.elapsedTimer("cuda_devicetransferkernel_unified");

		//Check contents//
		bool isSuccessful = true;
		cudaMemcpy(checkBuffer, hostUnifiedBufferDst, bufferSize_, cudaMemcpyHostToHost);
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			if (checkBuffer[i] != hostBuffer[i])
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cuda_devicetransferkernel_unified successful: " << isSuccessful << std::endl << std::endl;

		//Cleanup//
		cudaFree(hostUnifiedBufferSrc);
		cudaFree(hostUnifiedBufferDst);

		delete(checkBuffer);
		delete(hostBuffer);
	}
	void cuda_devicetransferkernel_pinned(size_t aN, bool isWarmup)
	{
		//Test preperation//
		float* checkBuffer = new float[bufferLength_];
		float* hostBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		float* hostPinnedBufferSrc;
		float* hostPinnedBufferDst;
		cudaMallocHost((void**)&hostPinnedBufferSrc, bufferSize_);
		cudaMallocHost((void**)&hostPinnedBufferDst, bufferSize_);

		for (size_t i = 0; i != bufferLength_; ++i)
			hostPinnedBufferSrc[i] = 42.0;

		//Execute & Profile//
		std::cout << "Executing test: cuda_devicetransferkernel_pinned" << std::endl;
		if (isWarmup)
		{
			CUDA_Kernels::copyBufferExecute(globalWorkspace_, localWorkspace_, hostPinnedBufferSrc, hostPinnedBufferDst);
			cudaDeviceSynchronize();
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			cudaBenchmarker_.startTimer("cuda_devicetransferkernel_pinned");
			CUDA_Kernels::copyBufferExecute(globalWorkspace_, localWorkspace_, hostPinnedBufferSrc, hostPinnedBufferDst);
			cudaDeviceSynchronize();
			cudaBenchmarker_.pauseTimer("cuda_devicetransferkernel_pinned");
		}
		cudaBenchmarker_.elapsedTimer("cuda_devicetransferkernel_pinned");

		//Check contents//
		bool isSuccessful = true;
		cudaMemcpy(checkBuffer, hostPinnedBufferDst, bufferSize_, cudaMemcpyHostToHost);
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			if (checkBuffer[i] != hostBuffer[i])
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cuda_devicetransferkernel_pinned successful: " << isSuccessful << std::endl << std::endl;

		//Cleanup//
		cudaFree(hostPinnedBufferSrc);
		cudaFree(hostPinnedBufferDst);

		delete(checkBuffer);
		delete(hostBuffer);
	}
	void cuda_simplebufferprocessing_standard(size_t aN, bool isWarmup)
	{
		//Test preperation//
		float* checkBuffer = new float[bufferLength_];
		float* hostBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		float* deviceBufferInput;
		float* deviceBufferOutput;
		cudaMalloc((void**)&deviceBufferInput, bufferSize_);
		cudaMalloc((void**)&deviceBufferOutput, bufferSize_);

		//Execute & Profile//
		std::cout << "Executing test: cuda_simplebufferprocessing_standard" << std::endl;
		if (isWarmup)
		{
			cudaMemcpy(deviceBufferInput, hostBuffer, bufferSize_, cudaMemcpyHostToDevice);
			cudaDeviceSynchronize();
			CUDA_Kernels::simpleBufferProcessing(globalWorkspace_, localWorkspace_, deviceBufferInput, deviceBufferOutput);
			cudaDeviceSynchronize();
			cudaMemcpy(checkBuffer, deviceBufferOutput, bufferSize_, cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			cudaBenchmarker_.startTimer("cuda_simplebufferprocessing_standard");
			cudaMemcpy(deviceBufferInput, hostBuffer, bufferSize_, cudaMemcpyHostToDevice);
			cudaDeviceSynchronize();
			CUDA_Kernels::simpleBufferProcessing(globalWorkspace_, localWorkspace_, deviceBufferInput, deviceBufferOutput);
			cudaDeviceSynchronize();
			cudaMemcpy(checkBuffer, deviceBufferOutput, bufferSize_, cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
			cudaBenchmarker_.pauseTimer("cuda_simplebufferprocessing_standard");
		}
		cudaBenchmarker_.elapsedTimer("cuda_simplebufferprocessing_standard");

		//Check contents//
		bool isSuccessful = true;
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			float attenuatedSample = hostBuffer[i] * 0.5;
			if (attenuatedSample != checkBuffer[i])
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cuda_simplebufferprocessing_standard successful: " << isSuccessful << std::endl << std::endl;

		//Cleanup//
		cudaFree(deviceBufferInput);
		cudaFree(deviceBufferOutput);

		delete(checkBuffer);
		delete(hostBuffer);
	}
	void cuda_simplebufferprocessing_unified(size_t aN, bool isWarmup)
	{
		//Test preperation//
		float* checkBuffer = new float[bufferLength_];
		float* hostBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		float* hostUnifiedBufferInput;
		float* hostUnifiedBufferOutput;
		cudaMallocManaged((void**)&hostUnifiedBufferInput, bufferSize_);
		cudaMallocManaged((void**)&hostUnifiedBufferOutput, bufferSize_);

		//Execute & Profile//
		std::cout << "Executing test: cuda_simplebufferprocessing_unified" << std::endl;
		if (isWarmup)
		{
			cudaMemcpy(hostUnifiedBufferInput, hostBuffer, bufferSize_, cudaMemcpyHostToHost);
			cudaDeviceSynchronize();
			CUDA_Kernels::simpleBufferProcessing(globalWorkspace_, localWorkspace_, hostUnifiedBufferInput, hostUnifiedBufferOutput);
			cudaDeviceSynchronize();
			cudaMemcpy(checkBuffer, hostUnifiedBufferOutput, bufferSize_, cudaMemcpyHostToHost);
			cudaDeviceSynchronize();
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			cudaBenchmarker_.startTimer("cuda_simplebufferprocessing_unified");
			cudaMemcpy(hostUnifiedBufferInput, hostBuffer, bufferSize_, cudaMemcpyHostToHost);
			cudaDeviceSynchronize();
			CUDA_Kernels::simpleBufferProcessing(globalWorkspace_, localWorkspace_, hostUnifiedBufferInput, hostUnifiedBufferOutput);
			cudaDeviceSynchronize();
			cudaMemcpy(checkBuffer, hostUnifiedBufferOutput, bufferSize_, cudaMemcpyHostToHost);
			cudaDeviceSynchronize();
			cudaBenchmarker_.pauseTimer("cuda_simplebufferprocessing_unified");
		}
		cudaBenchmarker_.elapsedTimer("cuda_simplebufferprocessing_unified");

		//Check contents//
		bool isSuccessful = true;
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			float attenuatedSample = hostBuffer[i] * 0.5;
			if (attenuatedSample != checkBuffer[i])
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cuda_simplebufferprocessing_unified successful: " << isSuccessful << std::endl << std::endl;

		//Cleanup//
		cudaFree(hostUnifiedBufferInput);
		cudaFree(hostUnifiedBufferOutput);

		delete(checkBuffer);
		delete(hostBuffer);
	}
	void cuda_simplebufferprocessing_pinned(size_t aN, bool isWarmup)
	{
		//Test preperation//
		float* checkBuffer = new float[bufferLength_];
		float* hostBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		float* hostPinnedBufferInput;
		float* hostPinnedBufferOutput;
		cudaMallocHost((void**)&hostPinnedBufferInput, bufferSize_);
		cudaMallocHost((void**)&hostPinnedBufferOutput, bufferSize_);

		for (size_t i = 0; i != bufferLength_; ++i)
			hostPinnedBufferInput[i] = 42.0;

		//Execute & Profile//
		std::cout << "Executing test: cuda_simplebufferprocessing_pinned" << std::endl;
		if (isWarmup)
		{
			cudaMemcpy(hostPinnedBufferInput, hostBuffer, bufferSize_, cudaMemcpyHostToHost);
			cudaDeviceSynchronize();
			CUDA_Kernels::simpleBufferProcessing(globalWorkspace_, localWorkspace_, hostPinnedBufferInput, hostPinnedBufferOutput);
			cudaDeviceSynchronize();
			cudaMemcpy(checkBuffer, hostPinnedBufferOutput, bufferSize_, cudaMemcpyHostToHost);
			cudaDeviceSynchronize();
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			cudaBenchmarker_.startTimer("cuda_simplebufferprocessing_pinned");
			cudaMemcpy(hostPinnedBufferInput, hostBuffer, bufferSize_, cudaMemcpyHostToHost);
			cudaDeviceSynchronize();
			CUDA_Kernels::simpleBufferProcessing(globalWorkspace_, localWorkspace_, hostPinnedBufferInput, hostPinnedBufferOutput);
			cudaDeviceSynchronize();
			cudaMemcpy(checkBuffer, hostPinnedBufferOutput, bufferSize_, cudaMemcpyHostToHost);
			cudaDeviceSynchronize();
			cudaBenchmarker_.pauseTimer("cuda_simplebufferprocessing_pinned");
		}
		cudaBenchmarker_.elapsedTimer("cuda_simplebufferprocessing_pinned");

		//Check contents//
		bool isSuccessful = true;
		cudaMemcpy(checkBuffer, hostPinnedBufferOutput, bufferSize_, cudaMemcpyHostToHost);
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			float attenuatedSample = hostBuffer[i] * 0.5;
			if (attenuatedSample != checkBuffer[i])
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cuda_simplebufferprocessing_pinned successful: " << isSuccessful << std::endl << std::endl;

		//Cleanup//
		cudaFree(hostPinnedBufferInput);
		cudaFree(hostPinnedBufferOutput);

		delete(checkBuffer);
		delete(hostBuffer);
	}
	void cuda_complexbufferprocessing_standard(size_t aN, bool isWarmup)
	{
		//Test preperation//
		float* checkBuffer = new float[bufferLength_];
		float* hostBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		float* deviceBufferInput;
		float* deviceBufferOutput;
		cudaMalloc((void**)&deviceBufferInput, bufferSize_);
		cudaMalloc((void**)&deviceBufferOutput, bufferSize_);

		cudaMemcpy(deviceBufferInput, hostBuffer, bufferSize_, cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();

		//Execute & Profile//
		std::cout << "Executing test: cuda_complexbufferprocessing_standard" << std::endl;
		if (isWarmup)
		{
			cudaMemcpy(deviceBufferInput, hostBuffer, bufferSize_, cudaMemcpyHostToDevice);
			cudaDeviceSynchronize();
			CUDA_Kernels::complexBufferProcessing(globalWorkspace_, localWorkspace_, deviceBufferInput, deviceBufferOutput);
			cudaDeviceSynchronize();
			cudaMemcpy(checkBuffer, deviceBufferOutput, bufferSize_, cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			cudaBenchmarker_.startTimer("cuda_complexbufferprocessing_standard");
			cudaMemcpy(deviceBufferInput, hostBuffer, bufferSize_, cudaMemcpyHostToDevice);
			cudaDeviceSynchronize();
			CUDA_Kernels::complexBufferProcessing(globalWorkspace_, localWorkspace_, deviceBufferInput, deviceBufferOutput);
			cudaDeviceSynchronize();
			cudaMemcpy(checkBuffer, deviceBufferOutput, bufferSize_, cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
			cudaBenchmarker_.pauseTimer("cuda_complexbufferprocessing_standard");
		}
		cudaBenchmarker_.elapsedTimer("cuda_complexbufferprocessing_standard");

		//Check contents//
		bool isSuccessful = true;
		cudaMemcpy(checkBuffer, deviceBufferOutput, bufferSize_, cudaMemcpyDeviceToHost);
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			//Calculate triangular smoothing on CPU to compare//
			float smoothedSample = hostBuffer[i];
			if (i > 2 & i < bufferLength_ - 2)
			{
				smoothedSample = ((hostBuffer[i - 2] + 2.0 * hostBuffer[i - 1] + 3.0 * hostBuffer[i] + 2.0 * hostBuffer[i + 1] + hostBuffer[i + 2]) / 9.0);
			}
			if (checkBuffer[i] != smoothedSample)
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cuda_complexbufferprocessing_standard successful: " << isSuccessful << std::endl << std::endl;

		//Cleanup//
		cudaFree(deviceBufferInput);
		cudaFree(deviceBufferOutput);

		delete(checkBuffer);
		delete(hostBuffer);
	}
	void cuda_complexbufferprocessing_unified(size_t aN, bool isWarmup)
	{
		//Test preperation//
		float* checkBuffer = new float[bufferLength_];
		float* hostBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		float* hostUnifiedBufferInput;
		float* hostUnifiedBufferOutput;
		cudaMallocManaged((void**)&hostUnifiedBufferInput, bufferSize_);
		cudaMallocManaged((void**)&hostUnifiedBufferOutput, bufferSize_);

		for (size_t i = 0; i != bufferLength_; ++i)
			hostUnifiedBufferInput[i] = 42.0;

		//Execute & Profile//
		std::cout << "Executing test: cuda_complexbufferprocessing_unified" << std::endl;
		if (isWarmup)
		{
			cudaMemcpy(hostUnifiedBufferInput, hostBuffer, bufferSize_, cudaMemcpyHostToHost);
			cudaDeviceSynchronize();
			CUDA_Kernels::complexBufferProcessing(globalWorkspace_, localWorkspace_, hostUnifiedBufferInput, hostUnifiedBufferOutput);
			cudaDeviceSynchronize();
			cudaMemcpy(checkBuffer, hostUnifiedBufferOutput, bufferSize_, cudaMemcpyHostToHost);	//@ToDo - Need this? The data is already in a host accessible pointer! Remove this.
			cudaDeviceSynchronize();
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			cudaBenchmarker_.startTimer("cuda_complexbufferprocessing_unified");
			cudaMemcpy(hostUnifiedBufferInput, hostBuffer, bufferSize_, cudaMemcpyHostToHost);
			cudaDeviceSynchronize();
			CUDA_Kernels::complexBufferProcessing(globalWorkspace_, localWorkspace_, hostUnifiedBufferInput, hostUnifiedBufferOutput);
			cudaDeviceSynchronize();
			cudaMemcpy(checkBuffer, hostUnifiedBufferOutput, bufferSize_, cudaMemcpyHostToHost);	//@ToDo - Need this? The data is already in a host accessible pointer! Remove this.
			cudaDeviceSynchronize();
			cudaBenchmarker_.pauseTimer("cuda_complexbufferprocessing_unified");
		}
		cudaBenchmarker_.elapsedTimer("cuda_complexbufferprocessing_unified");

		//Check contents//
		bool isSuccessful = true;
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			//Calculate triangular smoothing on CPU to compare//
			float smoothedSample = hostBuffer[i];
			if (i > 2 & i < bufferLength_ - 2)
			{
				smoothedSample = ((hostBuffer[i - 2] + 2.0 * hostBuffer[i - 1] + 3.0 * hostBuffer[i] + 2.0 * hostBuffer[i + 1] + hostBuffer[i + 2]) / 9.0);
			}
			if (checkBuffer[i] != smoothedSample)
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cuda_complexbufferprocessing_unified successful: " << isSuccessful << std::endl << std::endl;

		//Cleanup//
		cudaFree(hostUnifiedBufferInput);
		cudaFree(hostUnifiedBufferOutput);

		delete(checkBuffer);
		delete(hostBuffer);
	}
	void cuda_complexbufferprocessing_pinned(size_t aN, bool isWarmup)
	{
		//Test preperation//
		float* checkBuffer = new float[bufferLength_];
		float* hostBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		float* hostPinnedBufferInput;
		float* hostPinnedBufferOutput;
		cudaMallocHost((void**)&hostPinnedBufferInput, bufferSize_);
		cudaMallocHost((void**)&hostPinnedBufferOutput, bufferSize_);

		for (size_t i = 0; i != bufferLength_; ++i)
			hostPinnedBufferInput[i] = 42.0;

		//Execute & Profile//
		std::cout << "Executing test: cuda_complexbufferprocessing_unified" << std::endl;
		if (isWarmup)
		{
			cudaMemcpy(hostPinnedBufferInput, hostBuffer, bufferSize_, cudaMemcpyHostToHost);
			cudaDeviceSynchronize();
			CUDA_Kernels::complexBufferProcessing(globalWorkspace_, localWorkspace_, hostPinnedBufferInput, hostPinnedBufferOutput);
			cudaDeviceSynchronize();
			cudaMemcpy(checkBuffer, hostPinnedBufferOutput, bufferSize_, cudaMemcpyHostToHost);
			cudaDeviceSynchronize();
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			cudaBenchmarker_.startTimer("cuda_complexbufferprocessing_unified");
			cudaMemcpy(hostPinnedBufferInput, hostBuffer, bufferSize_, cudaMemcpyHostToHost);
			cudaDeviceSynchronize();
			CUDA_Kernels::complexBufferProcessing(globalWorkspace_, localWorkspace_, hostPinnedBufferInput, hostPinnedBufferOutput);
			cudaDeviceSynchronize();
			cudaMemcpy(checkBuffer, hostPinnedBufferOutput, bufferSize_, cudaMemcpyHostToHost);
			cudaDeviceSynchronize();
			cudaBenchmarker_.pauseTimer("cuda_complexbufferprocessing_unified");
		}
		cudaBenchmarker_.elapsedTimer("cuda_complexbufferprocessing_unified");

		//Check contents//
		bool isSuccessful = true;
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			//Calculate triangular smoothing on CPU to compare//
			float smoothedSample = hostBuffer[i];
			if (i > 2 & i < bufferLength_ - 2)
			{
				smoothedSample = ((hostBuffer[i - 2] + 2.0 * hostBuffer[i - 1] + 3.0 * hostBuffer[i] + 2.0 * hostBuffer[i + 1] + hostBuffer[i + 2]) / 9.0);
			}
			if (checkBuffer[i] != smoothedSample)
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cuda_complexbufferprocessing_unified successful: " << isSuccessful << std::endl << std::endl;

		//Cleanup//
		cudaFree(hostPinnedBufferInput);
		cudaFree(hostPinnedBufferOutput);

		delete(checkBuffer);
		delete(hostBuffer);
	}
	void cuda_simplebuffersynthesis_standard(size_t aN, bool isWarmup)
	{
		int sampleRate = 44100;
		float frequency = 1400.0;
		float* outputBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			outputBuffer[i] = 0.0;

		int* deviceSampleRate;
		float* deviceFrequency;
		float* deviceBufferOutput;
		cudaMalloc((void**)&deviceSampleRate, sizeof(int));
		cudaMalloc((void**)&deviceFrequency, sizeof(float));
		cudaMalloc((void**)&deviceBufferOutput, bufferSize_);

		cudaMemcpy(deviceFrequency, &sampleRate, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(deviceFrequency, &frequency, sizeof(float), cudaMemcpyHostToDevice);

		//Execute and average//
		std::cout << "Executing test: cuda_010_simplebuffersynthesis" << std::endl;
		if (isWarmup)
		{
			CUDA_Kernels::simpleBufferSynthesis(globalWorkspace_, localWorkspace_, deviceSampleRate, deviceFrequency, deviceBufferOutput);
			cudaDeviceSynchronize();
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			cudaBenchmarker_.startTimer("cuda_010_simplebuffersynthesis");
			CUDA_Kernels::simpleBufferSynthesis(globalWorkspace_, localWorkspace_, deviceSampleRate, deviceFrequency, deviceBufferOutput);
			cudaDeviceSynchronize();
			cudaMemcpy(outputBuffer, deviceBufferOutput, bufferSize_, cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
			cudaBenchmarker_.pauseTimer("cuda_010_simplebuffersynthesis");
		}
		cudaBenchmarker_.elapsedTimer("cuda_010_simplebuffersynthesis");

		//Save audio to file for inspection//
		outputAudioFile("cl_010_simplebuffersynthesis.wav", outputBuffer, bufferLength_);
		std::cout << "cl_010_simplebuffersynthesis successful: Inspect audio log \"cl_010_simplebuffersynthesis.wav\"" << std::endl << std::endl;

		cudaFree(deviceSampleRate);
		cudaFree(deviceFrequency);
		cudaFree(deviceBufferOutput);

		delete outputBuffer;
	}
	void cuda_simplebuffersynthesis_unified(size_t aN, bool isWarmup)
	{
		int sampleRate = 44100;
		float frequency = 1400.0;
		float* outputBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			outputBuffer[i] = 0.0;

		int* hostUnifiedSampleRate;
		float* hostUnifiedFrequency;
		float* hostUnifiedBufferOutput;
		cudaMallocManaged((void**)&hostUnifiedSampleRate, sizeof(int));
		cudaMallocManaged((void**)&hostUnifiedFrequency, sizeof(float));
		cudaMallocManaged((void**)&hostUnifiedBufferOutput, bufferSize_);

		*hostUnifiedSampleRate = sampleRate;
		*hostUnifiedFrequency = frequency;

		//Execute and average//
		std::cout << "Executing test: cuda_simplebuffersynthesis_unified" << std::endl;
		if (isWarmup)
		{
			CUDA_Kernels::simpleBufferSynthesis(globalWorkspace_, localWorkspace_, hostUnifiedSampleRate, hostUnifiedFrequency, hostUnifiedBufferOutput);
			cudaDeviceSynchronize();
			cudaMemcpy(outputBuffer, hostUnifiedBufferOutput, bufferSize_, cudaMemcpyHostToHost);
			cudaDeviceSynchronize();
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			cudaBenchmarker_.startTimer("cuda_simplebuffersynthesis_unified");
			CUDA_Kernels::simpleBufferSynthesis(globalWorkspace_, localWorkspace_, hostUnifiedSampleRate, hostUnifiedFrequency, hostUnifiedBufferOutput);
			cudaDeviceSynchronize();
			cudaMemcpy(outputBuffer, hostUnifiedBufferOutput, bufferSize_, cudaMemcpyHostToHost);
			cudaDeviceSynchronize();
			cudaBenchmarker_.pauseTimer("cuda_simplebuffersynthesis_unified");
		}
		cudaBenchmarker_.elapsedTimer("cuda_simplebuffersynthesis_unified");

		//Save audio to file for inspection//
		outputAudioFile("cuda_simplebuffersynthesis_unified.wav", outputBuffer, bufferLength_);
		std::cout << "cuda_simplebuffersynthesis_unified successful: Inspect audio log \"cuda_simplebuffersynthesis_unified.wav\"" << std::endl << std::endl;

		cudaFree(hostUnifiedSampleRate);
		cudaFree(hostUnifiedFrequency);
		cudaFree(hostUnifiedBufferOutput);

		delete outputBuffer;
	}
	void cuda_simplebuffersynthesis_pinned(size_t aN, bool isWarmup)
	{
		int sampleRate = 44100;
		float frequency = 1400.0;
		float* outputBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			outputBuffer[i] = 0.0;

		int* hostPinnedSampleRate;
		float* hostPinnedFrequency;
		float* hostPinnedBufferOutput;
		cudaMallocHost((void**)&hostPinnedSampleRate, sizeof(int));
		cudaMallocHost((void**)&hostPinnedFrequency, sizeof(float));
		cudaMallocHost((void**)&hostPinnedBufferOutput, bufferSize_);

		*hostPinnedSampleRate = sampleRate;
		*hostPinnedFrequency = frequency;

		//Execute and average//
		std::cout << "Executing test: cuda_simplebuffersynthesis_pinned" << std::endl;
		if (isWarmup)
		{
			CUDA_Kernels::simpleBufferSynthesis(globalWorkspace_, localWorkspace_, hostPinnedSampleRate, hostPinnedFrequency, hostPinnedBufferOutput);
			cudaDeviceSynchronize();
			cudaMemcpy(outputBuffer, hostPinnedBufferOutput, bufferSize_, cudaMemcpyHostToHost);
			cudaDeviceSynchronize();
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			cudaBenchmarker_.startTimer("cuda_simplebuffersynthesis_pinned");
			CUDA_Kernels::simpleBufferSynthesis(globalWorkspace_, localWorkspace_, hostPinnedSampleRate, hostPinnedFrequency, hostPinnedBufferOutput);
			cudaDeviceSynchronize();
			cudaMemcpy(outputBuffer, hostPinnedBufferOutput, bufferSize_, cudaMemcpyHostToHost);
			cudaDeviceSynchronize();
			cudaBenchmarker_.pauseTimer("cuda_simplebuffersynthesis_pinned");
		}
		cudaBenchmarker_.elapsedTimer("cuda_simplebuffersynthesis_pinned");

		//Save audio to file for inspection//
		outputAudioFile("cuda_simplebuffersynthesis_pinned.wav", outputBuffer, bufferLength_);
		std::cout << "cuda_simplebuffersynthesis_pinned successful: Inspect audio log \"cuda_simplebuffersynthesis_pinned.wav\"" << std::endl << std::endl;

		cudaFree(hostPinnedSampleRate);
		cudaFree(hostPinnedFrequency);
		cudaFree(hostPinnedBufferOutput);

		delete outputBuffer;
	}
	void cuda_complexbuffersynthesis_standard(size_t aN, bool isWarmup)
	{
		//FDTD CUDA allocation//
		size_t bufferLength;
		uint32_t gridWidth = 64;
		uint32_t gridHeight = 64;
		size_t gridSize = gridWidth * gridHeight * sizeof(float);
		float boundaryGain = 0.5;
		float propagationFactor = 0.06;
		float dampingFactor = 0.0005;
		Model model_ = Model(gridWidth, gridHeight, boundaryGain);
		float* boundaryGainGrid = new float[bufferLength_];
		boundaryGainGrid = model_.getBoundaryGridBuffer();

		float* samples = new float[bufferLength_];
		float* excitation = new float[bufferLength_];

		//Initialise host variables to copy to GPU//
		for (size_t i = 0; i != bufferLength_; ++i)
		{
			//Create initial impulse as excitation//
			if (i < bufferLength_ / 1000)
				excitation[i] = 0.5;
			else
				excitation[i] = 0.0;
		}

		dim3 tempGlobalSize = { gridWidth, gridHeight };
		dim3 tempLocalSize = { 8, 8 };
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
		cudaMallocHost((void**)&d_samplesIndex, sizeof(int));
		cudaMallocHost((void**)&d_samples, bufferLength_ * sizeof(float));
		cudaMallocHost((void**)&d_excitation, bufferLength_ * sizeof(float));
		cudaMalloc((void**)&d_listenerPosition, sizeof(int));
		cudaMalloc((void**)&d_excitationPosition, sizeof(int));
		cudaMalloc((void**)&d_propagationFactor, sizeof(float));
		cudaMalloc((void**)&d_dampingFactor, sizeof(float));
		cudaMallocHost((void**)&d_rotationIndex, sizeof(int));

		for (uint32_t i = 0; i != bufferLength_; ++i)
			d_samples[i] = 0.0;
		for (uint32_t i = 0; i != bufferLength_; ++i)
			d_excitation[i] = excitation[i];

		//FDTD static variables//
		int listenerPosition;
		model_.setListenerPosition(8, 8);
		listenerPosition = model_.getListenerPosition();
		int excitationPosition;
		model_.setExcitationPosition(32, 32);
		excitationPosition = model_.getExcitationPosition();

		cudaMemcpy(d_boundaryGain, boundaryGainGrid, gridSize, cudaMemcpyHostToDevice);
		cudaMemcpy(d_listenerPosition, &listenerPosition, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_excitationPosition, &excitationPosition, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_propagationFactor, &propagationFactor, sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_dampingFactor, &dampingFactor, sizeof(float), cudaMemcpyHostToDevice);

		//@ToDo - Write to constant coefficients. Propagation and damping//
		//cudaMemcpyToSymbol(c_ax, &ax, sizeof(float), 0, cudaMemcpyHostToDevice);
		//cudaMemcpyToSymbol(c_ax, &ax, sizeof(float), 0, cudaMemcpyHostToDevice);

		//FDTD dynamic variables//
		int samplesIndex = 0;
		int rotationIndex = 0;
		cudaMemcpy(d_samplesIndex, &samplesIndex, sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_rotationIndex, &rotationIndex, sizeof(float), cudaMemcpyHostToDevice);

		float* soundBuffer = new float[bufferLength_ * 2];

		//Execute and average//
		std::cout << "Executing test: cuda_complexbuffersynthesis_standard" << std::endl;
		if (isWarmup)
		{
			//CUDA_Kernels::complexBufferSynthesis(tempGlobalSize, tempLocalSize, bufferLength_, gridSize, d_gridOne, d_gridTwo, d_gridThree, d_boundaryGain, d_samplesIndex, d_samples, d_excitation, d_listenerPosition, d_excitationPosition, d_propagationFactor, d_dampingFactor, d_rotationIndex);
			//cudaDeviceSynchronize();
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			cudaBenchmarker_.startTimer("cuda_complexbuffersynthesis_standard");
			for (uint32_t j = 0; j != bufferLength_; ++j)
			{
				//Update variables//
				//cudaMemcpy(d_rotationIndex, rotationIndex, sizeof(int), cudaMemcpyHostToDevice);
				//cudaMemcpy(d_samplesIndex, samplesIndex, sizeof(int), cudaMemcpyHostToDevice);

				//cudaDeviceSynchronize();
				CUDA_Kernels::complexBufferSynthesis(tempGlobalSize, tempLocalSize, bufferLength_, gridSize, d_gridOne, d_gridTwo, d_gridThree, d_boundaryGain, d_samplesIndex, d_samples, d_excitation, d_listenerPosition, d_excitationPosition, d_propagationFactor, d_dampingFactor, d_rotationIndex);
				cudaDeviceSynchronize();

				samplesIndex++;
				rotationIndex = (rotationIndex + 1) % 3;
				cudaMemcpy(d_rotationIndex, &rotationIndex, sizeof(int), cudaMemcpyHostToDevice);
				cudaMemcpy(d_samplesIndex, &samplesIndex, sizeof(int), cudaMemcpyHostToDevice);
			}
			cudaBenchmarker_.pauseTimer("cuda_complexbuffersynthesis_standard");
			samplesIndex = 0;
			cudaMemcpy(d_samplesIndex, &samplesIndex, sizeof(int), cudaMemcpyHostToDevice);

			//Get synthesizer samples//
			cudaMemcpy(samples, d_samples, bufferLength_ * sizeof(float), cudaMemcpyDeviceToHost);
		}
		cudaBenchmarker_.elapsedTimer("cuda_complexbuffersynthesis_standard");

		//Save audio to file for inspection//
		for (int j = 0; j != bufferLength_; ++j)
			soundBuffer[j] = d_samples[j];
		outputAudioFile("cuda_complexbuffersynthesis_standard.wav", soundBuffer, bufferLength_);
		std::cout << "cuda_complexbuffersynthesis_standard successful: Inspect audio log \"cuda_complexbuffersynthesis_standard.wav\"" << std::endl << std::endl;

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
	void cuda_complexbuffersynthesis_unified(size_t aN, bool isWarmup)
	{
		//FDTD CUDA allocation//
		size_t bufferLength;
		uint32_t gridWidth = 64;
		uint32_t gridHeight = 64;
		size_t gridSize = gridWidth * gridHeight * sizeof(float);
		float boundaryGain = 0.5;
		float propagationFactor = 0.06;
		float dampingFactor = 0.0005;
		Model model_ = Model(gridWidth, gridHeight, boundaryGain);
		float* boundaryGainGrid = new float[bufferLength_];
		boundaryGainGrid = model_.getBoundaryGridBuffer();

		float* samples = new float[bufferLength_];
		float* excitation = new float[bufferLength_];

		//Initialise host variables to copy to GPU//
		for (size_t i = 0; i != bufferLength_; ++i)
		{
			//Create initial impulse as excitation//
			if (i < bufferLength_ / 1000)
				excitation[i] = 0.5;
			else
				excitation[i] = 0.0;
		}

		dim3 tempGlobalSize = { gridWidth, gridHeight };
		dim3 tempLocalSize = { 8, 8 };
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
		cudaMallocManaged((void**)&d_samplesIndex, sizeof(int));
		cudaMallocManaged((void**)&d_samples, bufferLength_ * sizeof(float));
		cudaMallocManaged((void**)&d_excitation, bufferLength_ * sizeof(float));
		cudaMalloc((void**)&d_listenerPosition, sizeof(int));
		cudaMalloc((void**)&d_excitationPosition, sizeof(int));
		cudaMalloc((void**)&d_propagationFactor, sizeof(float));
		cudaMalloc((void**)&d_dampingFactor, sizeof(float));
		cudaMallocManaged((void**)&d_rotationIndex, sizeof(int));

		for (uint32_t i = 0; i != bufferLength_; ++i)
			d_samples[i] = 0.0;
		for (uint32_t i = 0; i != bufferLength_; ++i)
			d_excitation[i] = excitation[i];

		//FDTD static variables//
		int listenerPosition;
		model_.setListenerPosition(8, 8);
		listenerPosition = model_.getListenerPosition();
		int excitationPosition;
		model_.setExcitationPosition(32, 32);
		excitationPosition = model_.getExcitationPosition();

		cudaMemcpy(d_boundaryGain, boundaryGainGrid, gridSize, cudaMemcpyHostToDevice);
		cudaMemcpy(d_listenerPosition, &listenerPosition, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_excitationPosition, &excitationPosition, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_propagationFactor, &propagationFactor, sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_dampingFactor, &dampingFactor, sizeof(float), cudaMemcpyHostToDevice);

		//FDTD dynamic variables//
		int samplesIndex = 0;
		int rotationIndex = 0;

		*d_samplesIndex = samplesIndex;
		*d_rotationIndex = rotationIndex;

		float* soundBuffer = new float[bufferLength_ * 2];

		//Execute and average//
		std::cout << "Executing test: cuda_complexbuffersynthesis_unified" << std::endl;
		if (isWarmup)
		{
			//CUDA_Kernels::complexBufferSynthesis(tempGlobalSize, tempLocalSize, bufferLength_, gridSize, d_gridOne, d_gridTwo, d_gridThree, d_boundaryGain, d_samplesIndex, d_samples, d_excitation, d_listenerPosition, d_excitationPosition, d_propagationFactor, d_dampingFactor, d_rotationIndex);
			//cudaDeviceSynchronize();
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			cudaBenchmarker_.startTimer("cuda_complexbuffersynthesis_unified");
			for (uint32_t j = 0; j != bufferLength_; ++j)
			{
				//Update variables//
				//cudaMemcpy(d_rotationIndex, rotationIndex, sizeof(int), cudaMemcpyHostToDevice);
				//cudaMemcpy(d_samplesIndex, samplesIndex, sizeof(int), cudaMemcpyHostToDevice);

				//cudaDeviceSynchronize();
				CUDA_Kernels::complexBufferSynthesis(tempGlobalSize, tempLocalSize, bufferLength_, gridSize, d_gridOne, d_gridTwo, d_gridThree, d_boundaryGain, d_samplesIndex, d_samples, d_excitation, d_listenerPosition, d_excitationPosition, d_propagationFactor, d_dampingFactor, d_rotationIndex);
				cudaDeviceSynchronize();

				samplesIndex++;
				rotationIndex = (rotationIndex + 1) % 3;
				//cudaMemcpy(d_rotationIndex, &rotationIndex, sizeof(int), cudaMemcpyHostToHost);
				//cudaMemcpy(d_samplesIndex, &samplesIndex, sizeof(int), cudaMemcpyHostToHost);
				*d_samplesIndex = samplesIndex;
				*d_rotationIndex = rotationIndex;
			}
			cudaBenchmarker_.pauseTimer("cuda_complexbuffersynthesis_unified");
			samplesIndex = 0;
			*d_samplesIndex = samplesIndex;

			//Get synthesizer samples//
			cudaMemcpy(samples, d_samples, bufferLength_ * sizeof(float), cudaMemcpyHostToHost);
		}
		cudaBenchmarker_.elapsedTimer("cuda_complexbuffersynthesis_unified");

		//Save audio to file for inspection//
		for (int j = 0; j != bufferLength_; ++j)
			soundBuffer[j] = d_samples[j];
		outputAudioFile("cuda_complexbuffersynthesis_unified.wav", soundBuffer, bufferLength_);
		std::cout << "cuda_complexbuffersynthesis_unified successful: Inspect audio log \"cuda_complexbuffersynthesis_unified.wav\"" << std::endl << std::endl;

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
	void cuda_complexbuffersynthesis_pinned(size_t aN, bool isWarmup)
	{
		//FDTD CUDA allocation//
		size_t bufferLength;
		uint32_t gridWidth = 64;
		uint32_t gridHeight = 64;
		size_t gridSize = gridWidth * gridHeight * sizeof(float);
		float boundaryGain = 0.5;
		float propagationFactor = 0.06;
		float dampingFactor = 0.0005;
		Model model_ = Model(gridWidth, gridHeight, boundaryGain);
		float* boundaryGainGrid = new float[bufferLength_];
		boundaryGainGrid = model_.getBoundaryGridBuffer();

		float* samples = new float[bufferLength_];
		float* excitation = new float[bufferLength_];

		//Initialise host variables to copy to GPU//
		for (size_t i = 0; i != bufferLength_; ++i)
		{
			//Create initial impulse as excitation//
			if (i < bufferLength_ / 1000)
				excitation[i] = 0.5;
			else
				excitation[i] = 0.0;
		}

		dim3 tempGlobalSize = { gridWidth, gridHeight };
		dim3 tempLocalSize = { 8, 8 };
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
		cudaMallocHost((void**)&d_samplesIndex, sizeof(int));
		cudaMallocHost((void**)&d_samples, bufferLength_ * sizeof(float));
		cudaMallocHost((void**)&d_excitation, bufferLength_ * sizeof(float));
		cudaMalloc((void**)&d_listenerPosition, sizeof(int));
		cudaMalloc((void**)&d_excitationPosition, sizeof(int));
		cudaMalloc((void**)&d_propagationFactor, sizeof(float));
		cudaMalloc((void**)&d_dampingFactor, sizeof(float));
		cudaMallocHost((void**)&d_rotationIndex, sizeof(int));

		for (uint32_t i = 0; i != bufferLength_; ++i)
			d_samples[i] = 0.0;
		for (uint32_t i = 0; i != bufferLength_; ++i)
			d_excitation[i] = excitation[i];

		//FDTD static variables//
		int listenerPosition;
		model_.setListenerPosition(8, 8);
		listenerPosition = model_.getListenerPosition();
		int excitationPosition;
		model_.setExcitationPosition(32, 32);
		excitationPosition = model_.getExcitationPosition();
		
		cudaMemcpy(d_boundaryGain, boundaryGainGrid, gridSize, cudaMemcpyHostToDevice);
		cudaMemcpy(d_listenerPosition, &listenerPosition, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_excitationPosition, &excitationPosition, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_propagationFactor, &propagationFactor, sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_dampingFactor, &dampingFactor, sizeof(float), cudaMemcpyHostToDevice);

		//FDTD dynamic variables//
		int samplesIndex = 0;
		int rotationIndex = 0;

		*d_samplesIndex = samplesIndex;
		*d_rotationIndex = rotationIndex;

		float* soundBuffer = new float[bufferLength_ * 2];

		//Execute and average//
		std::cout << "Executing test: cuda_complexbuffersynthesis_pinned" << std::endl;
		if (isWarmup)
		{
			//CUDA_Kernels::complexBufferSynthesis(tempGlobalSize, tempLocalSize, bufferLength_, gridSize, d_gridOne, d_gridTwo, d_gridThree, d_boundaryGain, d_samplesIndex, d_samples, d_excitation, d_listenerPosition, d_excitationPosition, d_propagationFactor, d_dampingFactor, d_rotationIndex);
			//cudaDeviceSynchronize();
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			cudaBenchmarker_.startTimer("cuda_complexbuffersynthesis_pinned");
			for (uint32_t j = 0; j != bufferLength_; ++j)
			{
				//Update variables//
				//cudaMemcpy(d_rotationIndex, rotationIndex, sizeof(int), cudaMemcpyHostToDevice);
				//cudaMemcpy(d_samplesIndex, samplesIndex, sizeof(int), cudaMemcpyHostToDevice);

				//cudaDeviceSynchronize();
				CUDA_Kernels::complexBufferSynthesis(tempGlobalSize, tempLocalSize, bufferLength_, gridSize, d_gridOne, d_gridTwo, d_gridThree, d_boundaryGain, d_samplesIndex, d_samples, d_excitation, d_listenerPosition, d_excitationPosition, d_propagationFactor, d_dampingFactor, d_rotationIndex);
				cudaDeviceSynchronize();

				samplesIndex++;
				rotationIndex = (rotationIndex + 1) % 3;
				//cudaMemcpy(d_rotationIndex, &rotationIndex, sizeof(int), cudaMemcpyHostToHost);
				//cudaMemcpy(d_samplesIndex, &samplesIndex, sizeof(int), cudaMemcpyHostToHost);
				*d_samplesIndex = samplesIndex;
				*d_rotationIndex = rotationIndex;
			}
			cudaBenchmarker_.pauseTimer("cuda_complexbuffersynthesis_pinned");
			samplesIndex = 0;
			*d_samplesIndex = samplesIndex;

			//Get synthesizer samples//
			cudaMemcpy(samples, d_samples, bufferLength_ * sizeof(float), cudaMemcpyHostToHost);
		}
		cudaBenchmarker_.elapsedTimer("cuda_complexbuffersynthesis_pinned");

		//Save audio to file for inspection//
		for (int j = 0; j != bufferLength_; ++j)
			soundBuffer[j] = d_samples[j];
		outputAudioFile("cuda_complexbuffersynthesis_pinned.wav", soundBuffer, bufferLength_);
		std::cout << "cuda_complexbuffersynthesis_pinned successful: Inspect audio log \"cuda_complexbuffersynthesis_pinned.wav\"" << std::endl << std::endl;

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
	void cuda_interruptedbufferprocessing_standard(size_t aN, bool isWarmup)
	{
		//Test preperation//
		datatype* checkBuffer = new datatype[bufferLength_];
		datatype* hostBuffer = new datatype[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		float* deviceBufferInput;
		float* deviceBufferOutput;
		cudaMalloc((void**)&deviceBufferInput, bufferSize_);
		cudaMalloc((void**)&deviceBufferOutput, bufferSize_);

		std::default_random_engine generator;
		std::uniform_int_distribution<int> distribution(1, 10);

		//Execute and average//
		std::cout << "Executing test: cuda_interruptedbufferprocessing_standard" << std::endl;
		if (isWarmup)
		{
			cudaMemcpy(deviceBufferInput, hostBuffer, bufferSize_, cudaMemcpyHostToDevice);
			cudaDeviceSynchronize();
			CUDA_Kernels::simpleBufferProcessing(globalWorkspace_, localWorkspace_, deviceBufferInput, deviceBufferOutput);
			cudaDeviceSynchronize();
			cudaMemcpy(checkBuffer, deviceBufferOutput, bufferSize_, cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
		}
		for (int32_t i = 0; i != aN; ++i)
		{
			cudaBenchmarker_.startTimer("cuda_interruptedbufferprocessing_standard");
			cudaMemcpy(deviceBufferInput, hostBuffer, bufferSize_, cudaMemcpyHostToDevice);
			cudaDeviceSynchronize();
			CUDA_Kernels::simpleBufferProcessing(globalWorkspace_, localWorkspace_, deviceBufferInput, deviceBufferOutput);
			cudaDeviceSynchronize();
			cudaMemcpy(checkBuffer, deviceBufferOutput, bufferSize_, cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
			cudaBenchmarker_.pauseTimer("cuda_interruptedbufferprocessing_standard");

			int randomChance = distribution(generator);
			if (randomChance > 5)
				--i;

		}
		cudaBenchmarker_.elapsedTimer("cuda_interruptedbufferprocessing_standard");

		//Check contents//
		bool isSuccessful = true;
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			float attenuatedSample = hostBuffer[i] * 0.5;
			if (attenuatedSample != checkBuffer[i])
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cuda_interruptedbufferprocessing_standard successful: " << isSuccessful << std::endl << std::endl;

		//Cleanup//
		cudaFree(deviceBufferInput);
		cudaFree(deviceBufferOutput);

		delete hostBuffer;
		delete checkBuffer;
	}
	void cuda_interruptedbufferprocessing_unified(size_t aN, bool isWarmup)
	{
		//Test preperation//
		datatype* checkBuffer = new datatype[bufferLength_];
		datatype* hostBuffer = new datatype[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		float* deviceBufferUnifiedInput;
		float* deviceBufferUnifiedOutput;
		cudaMallocManaged((void**)&deviceBufferUnifiedInput, bufferSize_);
		cudaMallocManaged((void**)&deviceBufferUnifiedOutput, bufferSize_);

		std::default_random_engine generator;
		std::uniform_int_distribution<int> distribution(1, 10);

		//Execute and average//
		std::cout << "Executing test: cuda_interruptedbufferprocessing_standard" << std::endl;
		if (isWarmup)
		{
			cudaMemcpy(deviceBufferUnifiedInput, hostBuffer, bufferSize_, cudaMemcpyHostToHost);
			cudaDeviceSynchronize();
			CUDA_Kernels::simpleBufferProcessing(globalWorkspace_, localWorkspace_, deviceBufferUnifiedInput, deviceBufferUnifiedOutput);
			cudaDeviceSynchronize();
			cudaMemcpy(checkBuffer, deviceBufferUnifiedOutput, bufferSize_, cudaMemcpyHostToHost);
			cudaDeviceSynchronize();
		}
		for (int32_t i = 0; i != aN; ++i)
		{
			cudaBenchmarker_.startTimer("cuda_interruptedbufferprocessing_standard");
			cudaMemcpy(deviceBufferUnifiedInput, hostBuffer, bufferSize_, cudaMemcpyHostToHost);
			cudaDeviceSynchronize();
			CUDA_Kernels::simpleBufferProcessing(globalWorkspace_, localWorkspace_, deviceBufferUnifiedInput, deviceBufferUnifiedOutput);
			cudaDeviceSynchronize();
			cudaMemcpy(checkBuffer, deviceBufferUnifiedOutput, bufferSize_, cudaMemcpyHostToHost);
			cudaDeviceSynchronize();
			cudaBenchmarker_.pauseTimer("cuda_interruptedbufferprocessing_standard");

			int randomChance = distribution(generator);
			if (randomChance > 5)
				--i;

		}
		cudaBenchmarker_.elapsedTimer("cuda_interruptedbufferprocessing_standard");

		//Check contents//
		bool isSuccessful = true;
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			float attenuatedSample = hostBuffer[i] * 0.5;
			if (attenuatedSample != checkBuffer[i])
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cuda_interruptedbufferprocessing_standard successful: " << isSuccessful << std::endl << std::endl;

		//Cleanup//
		cudaFree(deviceBufferUnifiedInput);
		cudaFree(deviceBufferUnifiedOutput);

		delete hostBuffer;
		delete checkBuffer;
	}
	void cuda_interruptedbufferprocessing_pinned(size_t aN, bool isWarmup)
	{
		//Test preperation//
		datatype* checkBuffer = new datatype[bufferLength_];
		datatype* hostBuffer = new datatype[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		float* deviceBufferPinnedInput;
		float* deviceBufferPinnedOutput;
		cudaMallocHost((void**)&deviceBufferPinnedInput, bufferSize_);
		cudaMallocHost((void**)&deviceBufferPinnedOutput, bufferSize_);

		std::default_random_engine generator;
		std::uniform_int_distribution<int> distribution(1, 10);

		//Execute and average//
		std::cout << "Executing test: cuda_interruptedbufferprocessing_standard" << std::endl;
		if (isWarmup)
		{
			cudaMemcpy(deviceBufferPinnedInput, hostBuffer, bufferSize_, cudaMemcpyHostToHost);
			cudaDeviceSynchronize();
			CUDA_Kernels::simpleBufferProcessing(globalWorkspace_, localWorkspace_, deviceBufferPinnedInput, deviceBufferPinnedOutput);
			cudaDeviceSynchronize();
			cudaMemcpy(checkBuffer, deviceBufferPinnedOutput, bufferSize_, cudaMemcpyHostToHost);
			cudaDeviceSynchronize();
		}
		for (int32_t i = 0; i != aN; ++i)
		{
			cudaBenchmarker_.startTimer("cuda_interruptedbufferprocessing_standard");
			cudaMemcpy(deviceBufferPinnedInput, hostBuffer, bufferSize_, cudaMemcpyHostToHost);
			cudaDeviceSynchronize();
			CUDA_Kernels::simpleBufferProcessing(globalWorkspace_, localWorkspace_, deviceBufferPinnedInput, deviceBufferPinnedOutput);
			cudaDeviceSynchronize();
			cudaMemcpy(checkBuffer, deviceBufferPinnedOutput, bufferSize_, cudaMemcpyHostToHost);
			cudaDeviceSynchronize();
			cudaBenchmarker_.pauseTimer("cuda_interruptedbufferprocessing_standard");

			int randomChance = distribution(generator);
			if (randomChance > 5)
				--i;

		}
		cudaBenchmarker_.elapsedTimer("cuda_interruptedbufferprocessing_standard");

		//Check contents//
		bool isSuccessful = true;
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			float attenuatedSample = hostBuffer[i] * 0.5;
			if (attenuatedSample != checkBuffer[i])
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cuda_interruptedbufferprocessing_standard successful: " << isSuccessful << std::endl << std::endl;

		//Cleanup//
		cudaFree(deviceBufferPinnedInput);
		cudaFree(deviceBufferPinnedOutput);

		delete hostBuffer;
		delete checkBuffer;
	}

	void cuda_unidirectional_baseline(size_t aFrameRate, bool isWarmup)
	{
		//Prepare new file for cuda_unidirectional_baseline//
		std::string strBenchmarkFileName = "cuda_unidirectional_baseline_framerate";
		std::string strFrameRate = std::to_string(aFrameRate);
		strBenchmarkFileName.append(strFrameRate);
		strBenchmarkFileName.append(".csv");
		cudaBenchmarker_ = Benchmarker(strBenchmarkFileName, { "Buffer_Size", "Total_Time", "Average_Time", "Max_Time", "Min_Time", "Max_Difference", "Average_Difference" });

		uint64_t numSamplesComputed = 0;
		for (size_t i = 0; i != bufferSizesLength; ++i)
		{
			uint64_t currentBufferLength = bufferSizes[i];
			uint64_t currentBufferSize = currentBufferLength * sizeof(float);
			setBufferLength(currentBufferLength);
			if (currentBufferLength > aFrameRate)
				break;

			std::string strBenchmarkName = "";
			std::string strBufferSize = std::to_string(currentBufferLength);
			strBenchmarkName.append(strBufferSize);

			uint64_t numSamplesComputed = 0;
			float* inBuf = new float[currentBufferLength];
			float* outBuf = new float[currentBufferLength];
			for (size_t i = 0; i != currentBufferLength; ++i)
				inBuf[i] = i;

			void* deviceBufferOutput;
			cudaMalloc((void**)&deviceBufferOutput, currentBufferSize);

			while (numSamplesComputed < aFrameRate)
			{
				cudaBenchmarker_.startTimer(strBenchmarkName);
				CUDA_Kernels::nullKernelExecute();
				cudaDeviceSynchronize();
				cudaMemcpy(outBuf, deviceBufferOutput, bufferSize_, cudaMemcpyDeviceToHost);
				cudaDeviceSynchronize();
				cudaBenchmarker_.pauseTimer(strBenchmarkName);

				numSamplesComputed += currentBufferLength;
			}
			cudaBenchmarker_.elapsedTimer(strBenchmarkName);

			numSamplesComputed = 0;

			delete inBuf;
			delete outBuf;
		}
	}
	void cuda_unidirectional_processing(size_t aFrameRate, bool isWarmup)
	{
		//Prepare new file for cuda_unidirectional_processing//
		std::string strBenchmarkFileName = "cuda_unidirectional_processing_framerate";
		std::string strFrameRate = std::to_string(aFrameRate);
		strBenchmarkFileName.append(strFrameRate);
		strBenchmarkFileName.append(".csv");
		cudaBenchmarker_ = Benchmarker(strBenchmarkFileName, { "Buffer_Size", "Total_Time", "Average_Time", "Max_Time", "Min_Time", "Max_Difference", "Average_Difference" });

		uint64_t numSamplesComputed = 0;
		for (size_t i = 0; i != bufferSizesLength; ++i)
		{
			uint64_t currentBufferLength = bufferSizes[i];
			uint64_t currentBufferSize = currentBufferLength * sizeof(float);
			setBufferLength(currentBufferLength);
			if (currentBufferLength > aFrameRate)
				break;

			std::string strBenchmarkName = "";
			std::string strBufferSize = std::to_string(currentBufferLength);
			strBenchmarkName.append(strBufferSize);

			uint64_t numSamplesComputed = 0;

			const int sampleRate = 44100;
			const float frequency = 1400.0;
			float* inBuf = new float[currentBufferLength];
			float* outBuf = new float[currentBufferLength];
			for (size_t i = 0; i != currentBufferLength; ++i)
				inBuf[i] = i;

			int* deviceSampleRate;
			float* deviceFrequency;
			float* deviceBufferOutput;
			cudaMalloc((void**)&deviceSampleRate, sizeof(int));
			cudaMalloc((void**)&deviceFrequency, sizeof(float));
			cudaMalloc((void**)&deviceBufferOutput, bufferSize_);

			cudaMemcpy(deviceFrequency, &sampleRate, sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(deviceFrequency, &frequency, sizeof(float), cudaMemcpyHostToDevice);

			float* soundBuffer = new float[currentBufferLength > aFrameRate ? currentBufferLength : aFrameRate * 2];

			while (numSamplesComputed < aFrameRate)
			{
				cudaBenchmarker_.startTimer(strBenchmarkName);
				CUDA_Kernels::simpleBufferSynthesis(globalWorkspace_, localWorkspace_, deviceSampleRate, deviceFrequency, deviceBufferOutput);
				cudaDeviceSynchronize();
				cudaMemcpy(outBuf, deviceBufferOutput, bufferSize_, cudaMemcpyDeviceToHost);
				cudaDeviceSynchronize();
				cudaBenchmarker_.pauseTimer(strBenchmarkName);

				//Log audio for inspection if necessary//
				for (int j = 0; j != currentBufferLength; ++j)
					soundBuffer[numSamplesComputed + j] = outBuf[j];

				numSamplesComputed += currentBufferLength;
			}
			cudaBenchmarker_.elapsedTimer(strBenchmarkName);

			//Save audio to file for inspection//
			outputAudioFile("cuda_unidirectional_processing.wav", soundBuffer, aFrameRate);
			std::cout << "cuda_unidirectional_processing successful: Inspect audio log \"cuda_unidirectional_processing.wav\"" << std::endl << std::endl;

			numSamplesComputed = 0;

			delete inBuf;
			delete outBuf;
		}
	}
	void cuda_bidirectional_baseline(size_t aFrameRate, bool isWarmup)
	{
		//Prepare new file for cuda_bidirectional_baseline//
		std::string strBenchmarkFileName = "cuda_bidirectional_baseline_framerate";
		std::string strFrameRate = std::to_string(aFrameRate);
		strBenchmarkFileName.append(strFrameRate);
		strBenchmarkFileName.append(".csv");
		cudaBenchmarker_ = Benchmarker(strBenchmarkFileName, { "Buffer_Size", "Total_Time", "Average_Time", "Max_Time", "Min_Time", "Max_Difference", "Average_Difference" });

		uint64_t numSamplesComputed = 0;
		for (size_t i = 0; i != bufferSizesLength; ++i)
		{
			uint64_t currentBufferLength = bufferSizes[i];
			uint64_t currentBufferSize = currentBufferLength * sizeof(float);
			setBufferLength(currentBufferLength);
			if (currentBufferLength > aFrameRate)
				break;

			std::string strBenchmarkName = "";
			std::string strBufferSize = std::to_string(currentBufferLength);
			strBenchmarkName.append(strBufferSize);

			uint64_t numSamplesComputed = 0;
			float* inBuf = new float[currentBufferLength];
			float* outBuf = new float[currentBufferLength];
			for (size_t i = 0; i != currentBufferLength; ++i)
				inBuf[i] = i;

			void* deviceBufferInput;
			void* deviceBufferOutput;
			cudaMalloc((void**)&deviceBufferInput, currentBufferSize);
			cudaMalloc((void**)&deviceBufferOutput, currentBufferSize);

			while (numSamplesComputed < aFrameRate)
			{
				cudaBenchmarker_.startTimer(strBenchmarkName);
				cudaMemcpy(deviceBufferInput, inBuf, bufferSize_, cudaMemcpyHostToDevice);
				cudaDeviceSynchronize();
				CUDA_Kernels::nullKernelExecute();
				cudaDeviceSynchronize();
				cudaMemcpy(outBuf, deviceBufferOutput, bufferSize_, cudaMemcpyDeviceToHost);
				cudaDeviceSynchronize();
				cudaBenchmarker_.pauseTimer(strBenchmarkName);

				numSamplesComputed += currentBufferLength;
			}
			cudaBenchmarker_.elapsedTimer(strBenchmarkName);

			numSamplesComputed = 0;

			delete inBuf;
			delete outBuf;
		}
	}
	void cuda_bidirectional_processing(size_t aFrameRate, bool isWarmup)
	{
		//Prepare new file for cuda_bidirectional_processing//
		std::string strBenchmarkFileName = "cuda_bidirectional_processing_framerate";
		std::string strFrameRate = std::to_string(aFrameRate);
		strBenchmarkFileName.append(strFrameRate);
		strBenchmarkFileName.append(".csv");
		cudaBenchmarker_ = Benchmarker(strBenchmarkFileName, { "Buffer_Size", "Total_Time", "Average_Time", "Max_Time", "Min_Time", "Max_Difference", "Average_Difference" });

		uint64_t numSamplesComputed = 0;
		for (size_t i = 0; i != bufferSizesLength; ++i)
		{
			uint64_t currentBufferLength = bufferSizes[i];
			uint64_t currentBufferSize = currentBufferLength * sizeof(float);
			setBufferLength(currentBufferLength);
			if (currentBufferLength > aFrameRate)
				break;

			std::string strBenchmarkName = "";
			std::string strBufferSize = std::to_string(currentBufferLength);
			strBenchmarkName.append(strBufferSize);

			//FDTD CUDA allocation//
			size_t bufferLength;
			uint32_t gridWidth = 64;
			uint32_t gridHeight = 64;
			size_t gridSize = gridWidth * gridHeight * sizeof(float);
			float boundaryGain = 0.5;
			float propagationFactor = 0.06;
			float dampingFactor = 0.0005;
			Model model_ = Model(gridWidth, gridHeight, boundaryGain);
			float* boundaryGainGrid = new float[bufferLength_];
			boundaryGainGrid = model_.getBoundaryGridBuffer();

			float* samples = new float[bufferLength_];
			float* excitation = new float[bufferLength_];

			//Initialise host variables to copy to GPU//
			for (size_t i = 0; i != currentBufferLength; ++i)
			{
				excitation[i] = 0.5;
			}

			dim3 tempGlobalSize = { gridWidth, gridHeight };
			dim3 tempLocalSize = { 16, 16 };
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
			cudaMallocHost((void**)&d_samplesIndex, sizeof(int));
			cudaMallocHost((void**)&d_samples, currentBufferLength * sizeof(float));
			cudaMallocHost((void**)&d_excitation, currentBufferLength * sizeof(float));
			cudaMalloc((void**)&d_listenerPosition, sizeof(int));
			cudaMalloc((void**)&d_excitationPosition, sizeof(int));
			cudaMalloc((void**)&d_propagationFactor, sizeof(float));
			cudaMalloc((void**)&d_dampingFactor, sizeof(float));
			cudaMallocHost((void**)&d_rotationIndex, sizeof(int));

			for (uint32_t i = 0; i != currentBufferLength; ++i)
				d_samples[i] = 0.0;
			for (uint32_t i = 0; i != currentBufferLength; ++i)
				d_excitation[i] = excitation[i];

			//FDTD static variables//
			int listenerPosition;
			model_.setListenerPosition(8, 8);
			listenerPosition = model_.getListenerPosition();
			int excitationPosition;
			model_.setExcitationPosition(32, 32);
			excitationPosition = model_.getExcitationPosition();

			cudaMemcpy(d_boundaryGain, boundaryGainGrid, gridSize, cudaMemcpyHostToDevice);
			cudaMemcpy(d_listenerPosition, &listenerPosition, sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(d_excitationPosition, &excitationPosition, sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(d_propagationFactor, &propagationFactor, sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_dampingFactor, &dampingFactor, sizeof(float), cudaMemcpyHostToDevice);

			//@ToDo - Write to constant coefficients. Propagation and damping//
			//cudaMemcpyToSymbol(c_ax, &ax, sizeof(float), 0, cudaMemcpyHostToDevice);
			//cudaMemcpyToSymbol(c_ax, &ax, sizeof(float), 0, cudaMemcpyHostToDevice);

			//FDTD dynamic variables//
			int samplesIndex = 0;
			int rotationIndex = 0;

			*d_samplesIndex = samplesIndex;
			*d_rotationIndex = rotationIndex;

			float* soundBuffer = new float[currentBufferLength > aFrameRate ? currentBufferLength : aFrameRate * 2];

			while (numSamplesComputed < aFrameRate)
			{
				cudaBenchmarker_.startTimer(strBenchmarkName);
				for (uint32_t i = 0; i != currentBufferLength; ++i)
				{
					CUDA_Kernels::complexBufferSynthesis(tempGlobalSize, tempLocalSize, currentBufferLength, gridSize, d_gridOne, d_gridTwo, d_gridThree, d_boundaryGain, d_samplesIndex, d_samples, d_excitation, d_listenerPosition, d_excitationPosition, d_propagationFactor, d_dampingFactor, d_rotationIndex);
					cudaDeviceSynchronize();
					samplesIndex++;
					rotationIndex = (rotationIndex + 1) % 3;
					*d_samplesIndex = samplesIndex;
					*d_rotationIndex = rotationIndex;
				}
				//Get synthesizer samples//
				cudaMemcpy(samples, d_samples, currentBufferLength * sizeof(float), cudaMemcpyDeviceToHost);
				samplesIndex = 0;
				*d_samplesIndex = samplesIndex;
				cudaBenchmarker_.pauseTimer(strBenchmarkName);

				//Log audio for inspection if necessary//
				for (int j = 0; j != currentBufferLength; ++j)
					soundBuffer[numSamplesComputed + j] = samples[j];

				numSamplesComputed += currentBufferLength;
			}
			cudaBenchmarker_.elapsedTimer(strBenchmarkName);

			//Save audio to file for inspection//
			outputAudioFile("cuda_bidirectional_processing.wav", soundBuffer, aFrameRate);
			std::cout << "cuda_bidirectional_processing successful: Inspect audio log \"cuda_bidirectional_processing.wav\"" << std::endl << std::endl;

			numSamplesComputed = 0;

			delete samples;
			delete excitation;
		}
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

	void setWorkspaceSize(uint32_t aGlobalSize, uint32_t aLocalSize)
	{
		globalWorkspace_ = dim3(aGlobalSize, 1, 1);
		localWorkspace_ = dim3(aLocalSize, 1, 1);
	}
	void setWorkspaceSize(dim3 aGlobalSize, dim3 aLocalSize)
	{
		globalWorkspace_ = aGlobalSize;
		localWorkspace_ = aLocalSize;
	}

	void setLocalWorkspace(uint64_t aGlobalSize)
	{
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, 0);
		uint64_t maxLocalWorkspace = deviceProp.maxThreadsPerBlock;
		uint64_t localWorkspace = aGlobalSize > maxLocalWorkspace ? maxLocalWorkspace : aGlobalSize;

		dim3 newGlobalSize = aGlobalSize;
		dim3 newLocalSize = localWorkspace;
		setWorkspaceSize(newGlobalSize, newLocalSize);
	}
	void setBufferSize(uint64_t aBufferSize)
	{
		bufferSize_ = aBufferSize;
		bufferLength_ = bufferSize_ / sizeof(datatype);

		setLocalWorkspace(bufferLength_);
	}
	void setBufferLength(uint64_t aBufferLength)
	{
		bufferLength_ = aBufferLength;
		bufferSize_ = bufferLength_ * sizeof(datatype);

		setLocalWorkspace(bufferLength_);
	}

	static bool isCudaAvailable()
	{
		int deviceCount = 0;
		cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

		return (deviceCount > 0 ? true : false);
	}

	static bool cudaCompatible()
	{
		int cudaRuntimeVersion = 0;
		int cudaDriverVersion = 0;
		cudaRuntimeGetVersion(&cudaRuntimeVersion);
		cudaDriverGetVersion(&cudaDriverVersion);
		std::cout << "CUDA runtime version: " << cudaRuntimeVersion << std::endl;
		std::cout << "CUDA driver version: " << cudaDriverVersion << std::endl;
		if (cudaRuntimeGetVersion == 0 || cudaDriverVersion == 0)
		{
			std::cout << "CUDA runtime or driver version missing" << std::endl;
			return false;
		}

		int numCudaDevices = GPU_Overhead_Benchmarks_CUDA::isCudaAvailable();
		std::cout << "Number of available CUDA devices: " << numCudaDevices << std::endl;
		if (numCudaDevices == 0)
		{
			std::cout << "No CUDA device is detected" << std::endl;
			return false;
		}
		return true;
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
