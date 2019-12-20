#ifndef GPUOVERHEAD_BENCHMARKS_HPP
#define GPUOVERHEAD_BENCHMARKS_HPP

//#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include <CL/cl2.hpp>
//#include <CL/cl.hpp>
#include <CL/cl_gl.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>

//OpenGL API Dependencies//
//#define GLFW_EXPOSE_NATIVE_WIN32
//#define GLFW_EXPOSE_NATIVE_WGL
#include <windows.h>
//#include <glad\glad.h> 
//#include <GLFW\glfw3.h>

#include "OpenCL_Wrapper.h"
#include "CUDA_Wrapper.hpp"
#include "Benchmarker.hpp"

class GPUOverhead_Benchmarks
{
private:
	
	typedef float datatype;
	uint32_t sampleRate_ = 44100;
	uint32_t bufferSize_ = 1024;
	uint32_t bufferLength_ = bufferSize_ / sizeof(datatype);

	OpenCL_Wrapper openCL;
	CUDA_Wrapper cuda_;
	cl::NDRange globalWorkspace_;
	cl::NDRange localWorkspace_;

	Benchmarker benchmarker_;
public:
	static const uint32_t GIGA_BYTE = 1024 * 1024 * 1024;
	static const uint32_t MEGA_BYTE = 1024 * 1024;
	static const uint32_t KILO_BYTE = 1024;
	GPUOverhead_Benchmarks(uint32_t aPlatform, uint32_t aDevice) : openCL(aPlatform, aDevice), cuda_()
	{
	}

	void cl_000_nullKernel(size_t aN, bool isWarmup)
	{
		//Execute and average//
		std::cout << "Executing test: cl_000_nullKernel" << std::endl;
		openCL.setWorkspaceSize(1024, 32);
		if (isWarmup)
		{
			openCL.enqueueKernel("cl_000_nullKernel");
			openCL.waitKernel();
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			benchmarker_.startTimer("cl_000_nullKernel");
			openCL.enqueueKernel("cl_000_nullKernel");
			openCL.waitKernel();
			benchmarker_.pauseTimer("cl_000_nullKernel");
		}
		benchmarker_.elapsedTimer("cl_000_nullKernel");

		bool isSuccessful = true;
		std::cout << "Null kernel successful? " << isSuccessful << std::endl << std::endl;
	}
	void cl_001_CPUtoGPU(size_t aN, bool isWarmup)
	{	
		//Test preperation//
		datatype* checkBuffer = new datatype[bufferLength_];
		datatype* hostBuffer = new datatype[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		openCL.createBuffer("deviceBuffer", CL_MEM_WRITE_ONLY, bufferSize_);

		//Execute and average//
		std::cout << "Executing test: cl_001_CPUtoGPU" << std::endl;
		if (isWarmup)
		{
			openCL.writeBuffer("deviceBuffer", bufferSize_, hostBuffer);
			openCL.waitKernel();
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			//std::cout << "Write to buffer using enqueueWriteBuffer()" << std::endl;
			benchmarker_.startTimer("cl_001_CPUtoGPU");
			openCL.writeBuffer("deviceBuffer", bufferSize_, hostBuffer);
			openCL.waitKernel();

			benchmarker_.pauseTimer("cl_001_CPUtoGPU");
		}
		benchmarker_.elapsedTimer("cl_001_CPUtoGPU");

		bool isSuccessful = true;
		openCL.readBuffer("deviceBuffer", bufferSize_, checkBuffer);
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			if (checkBuffer[i] != hostBuffer[i])
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cl_001_CPUtoGPU successful: " << isSuccessful << std::endl << std::endl;

		openCL.deleteBuffer("deviceBuffer");
		delete hostBuffer;
		delete checkBuffer;
	}

	void cl_002_GPUtoCPU(size_t aN, bool isWarmup)
	{
		//Test preperation//
		datatype* hostBuffer = new datatype[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		openCL.createBuffer("deviceBuffer", CL_MEM_READ_WRITE, bufferSize_);
		openCL.writeBuffer("deviceBuffer", bufferSize_, hostBuffer);

		datatype* checkBuffer = new datatype[bufferLength_];

		//Execute and average//
		std::cout << "Executing test: cl_002_GPUtoCPU" << std::endl;
		if (isWarmup)
			openCL.readBuffer("deviceBuffer", bufferSize_, checkBuffer);
		for (uint32_t i = 0; i != aN; ++i)
		{
			benchmarker_.startTimer("cl_002_GPUtoCPU");
			openCL.readBuffer("deviceBuffer", bufferSize_, checkBuffer);
			//cudaMemcpyAsync(deviceBuffer, hostBuffer, bufferSize_, cudaMemcpyHostToDevice);
			benchmarker_.pauseTimer("cl_002_GPUtoCPU");
		}
		benchmarker_.elapsedTimer("cl_002_GPUtoCPU");

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
		std::cout << "cl_002_GPUtoCPU successful: " << isSuccessful << std::endl << std::endl;

		//Cleanup//
		openCL.deleteBuffer("deviceBuffer");

		delete(checkBuffer);
		delete(hostBuffer);
	}
	void cl_003_CPUtoGPUtoCPU(size_t aN, bool isWarmup)
	{
		//Test preperation//
		datatype* hostBuffer = new datatype[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		datatype* checkBuffer = new datatype[bufferLength_];

		openCL.createBuffer("deviceBuffer", CL_MEM_READ_WRITE, bufferSize_);

		//Execute and average//
		std::cout << "Executing test: cl_003_CPUtoGPUtoCPU" << std::endl;
		if (isWarmup)
		{
			openCL.writeBuffer("deviceBuffer", bufferSize_, hostBuffer);
			openCL.readBuffer("deviceBuffer", bufferSize_, checkBuffer);
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			benchmarker_.startTimer("cl_003_CPUtoGPUtoCPU");
			openCL.writeBuffer("deviceBuffer", bufferSize_, hostBuffer);
			openCL.readBuffer("deviceBuffer", bufferSize_, checkBuffer);
			//cudaMemcpyAsync(deviceBuffer, hostBuffer, bufferSize_, cudaMemcpyHostToDevice);
			benchmarker_.pauseTimer("cl_003_CPUtoGPUtoCPU");
		}
		benchmarker_.elapsedTimer("cl_003_CPUtoGPUtoCPU");

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
		std::cout << "cl_003_CPUtoGPUtoCPU successful: " << isSuccessful << std::endl << std::endl;

		//Cleanup//
		openCL.deleteBuffer("deviceBuffer");

		delete(checkBuffer);
		delete(hostBuffer);
	}

	void cl_004_pinnedmemory(size_t aN, bool isWarmup)
	{
		//Test preperation//
		datatype* hostBuffer = new datatype[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		datatype* checkBuffer = new datatype[bufferLength_];

		//Write to buffer using OpenCL enqueueMapBuffer()//
		openCL.createBuffer("deviceBuffer", CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, bufferSize_);

		auto mappedMemory = openCL.pinMappedMemory("deviceBuffer", bufferSize_);
		memcpy(mappedMemory, hostBuffer, bufferSize_);

		//Execute and average//
		std::cout << "Executing test: cl_004_pinnedmemory" << std::endl;
		if (isWarmup)
		{
			//openCL.enqueueCopyBuffer("srcBuffer", "dstBuffer", bufferSize_);
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			benchmarker_.startTimer("cl_004_pinnedmemory");
			memcpy(checkBuffer, mappedMemory, bufferSize_);
			benchmarker_.pauseTimer("cl_004_pinnedmemory");
		}
		benchmarker_.elapsedTimer("cl_004_pinnedmemory");

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
		std::cout << "cl_004_pinnedmemory successful: " << isSuccessful << std::endl << std::endl;

		openCL.unpinMappedMemory("deviceBuffer", mappedMemory, bufferSize_);
		openCL.deleteBuffer("deviceBuffer");

		delete hostBuffer;
		delete checkBuffer;
	}

	void cl_005_cpymemory(size_t aN, bool isWarmup)
	{
		//Test preperation//
		float* srcMemoryBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			srcMemoryBuffer[i] = 4.0;
		float* dstMemoryBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			dstMemoryBuffer[i] = 0.0;

		openCL.createBuffer("srcBuffer", CL_MEM_READ_ONLY, bufferSize_);
		openCL.createBuffer("dstBuffer", CL_MEM_WRITE_ONLY, bufferSize_);

		openCL.writeBuffer("srcBuffer", bufferSize_, srcMemoryBuffer);
		openCL.writeBuffer("dstBuffer", bufferSize_, dstMemoryBuffer);

		//Execute and average//
		std::cout << "Executing test: cl_005_cpymemory" << std::endl;
		if (isWarmup)
		{
			openCL.enqueueCopyBuffer("srcBuffer", "dstBuffer", bufferSize_);
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			benchmarker_.startTimer("cl_005_cpymemory");
			openCL.enqueueCopyBuffer("srcBuffer", "dstBuffer", bufferSize_);
			//openCL.waitKernel();
			benchmarker_.pauseTimer("cl_005_cpymemory");
		}
		benchmarker_.elapsedTimer("cl_005_cpymemory");

		//Check contents//
		bool isSuccessful = true;
		openCL.readBuffer("dstBuffer", bufferSize_, dstMemoryBuffer);
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			if (dstMemoryBuffer[i] != srcMemoryBuffer[i])
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cl_005_cpymemory successful: " << isSuccessful << std::endl << std::endl;

		openCL.deleteBuffer("srcBuffer");
		openCL.deleteBuffer("dstBuffer");

		delete srcMemoryBuffer;
		delete dstMemoryBuffer;
	}
	void cl_006_cpymemorykernel(size_t aN, bool isWarmup)
	{
		//Copy between buffer using OpenCL kernel//
		openCL.createKernelProgram("resources/kernels/GPU_Overhead_Benchmarks.cl", "");
		openCL.createKernel("cl_006_cpymemorykernel");
		openCL.setWorkspaceSize(bufferLength_, 1024);

		//Test preperation//
		float* srcMemoryBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			srcMemoryBuffer[i] = 42.0;
		float* dstMemoryBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			dstMemoryBuffer[i] = 0.0;

		openCL.createBuffer("srcBuffer", CL_MEM_READ_WRITE, bufferSize_);
		openCL.createBuffer("dstBuffer", CL_MEM_READ_WRITE, bufferSize_);

		openCL.writeBuffer("srcBuffer", bufferSize_, srcMemoryBuffer);
		openCL.writeBuffer("dstBuffer", bufferSize_, dstMemoryBuffer);

		openCL.setKernelArgument("cl_006_cpymemorykernel", "srcBuffer", 0, sizeof(cl_mem));
		openCL.setKernelArgument("cl_006_cpymemorykernel", "dstBuffer", 1, sizeof(cl_mem));

		//Execute and average//
		std::cout << "Executing test: cl_006_cpymemorykernel" << std::endl;
		if (isWarmup)
		{
			openCL.enqueueKernel("cl_006_cpymemorykernel");
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			benchmarker_.startTimer("cl_006_cpymemorykernel");

			openCL.enqueueKernel("cl_006_cpymemorykernel");
			openCL.waitKernel();

			benchmarker_.pauseTimer("cl_006_cpymemorykernel");
		}
		benchmarker_.elapsedTimer("cl_006_cpymemorykernel");

		//Check contents//
		bool isSuccessful = true;
		openCL.readBuffer("dstBuffer", bufferSize_, dstMemoryBuffer);
		openCL.waitKernel();
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			if (srcMemoryBuffer[i] != dstMemoryBuffer[i])
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cl_006_cpymemorykernel successful: " << isSuccessful << std::endl << std::endl;

		openCL.deleteBuffer("srcBuffer");
		openCL.deleteBuffer("dstBuffer");

		delete srcMemoryBuffer;
		delete dstMemoryBuffer;
	}
	void runUnidirectionalBenchmarks(size_t aN)
	{
		//averageBenchmark(writeToGPU, 10);
		//writeToGPU(aN);
		//copyBuffer(aN);
		//copyBufferKernel(aN);
		
	}
	void writeReadToGPU(size_t aN)
	{
		openCL.createBuffer("readWriteBuffer", CL_MEM_READ_WRITE, bufferSize_);

		char* writeMemoryBuffer = new char[bufferSize_];
		memset(writeMemoryBuffer, '1', bufferSize_);
		char* readMemoryBuffer = new char[bufferSize_];
		memset(readMemoryBuffer, '0', bufferSize_);

		std::cout << "Write then read from same buffer" << std::endl << std::endl;

		for (uint32_t i = 0; i != aN; ++i)
		{
			benchmarker_.startTimer("writeReadToGPU");

			openCL.writeBuffer("readWriteBuffer", bufferSize_, writeMemoryBuffer);
			openCL.readBuffer("readWriteBuffer", bufferSize_, readMemoryBuffer);
			openCL.waitKernel();

			benchmarker_.pauseTimer("writeReadToGPU");
		}
		benchmarker_.elapsedTimer("writeReadToGPU");

		//Check contents//
		char* memoryBufferCheck = new char[bufferSize_];
		memset(memoryBufferCheck, '1', bufferSize_);
		bool isSuccessful = true;
		for (uint32_t i = 0; i != bufferSize_; ++i)
		{
			if (writeMemoryBuffer[i] != readMemoryBuffer[i])
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "Write & Read buffer successful? " << isSuccessful << std::endl << std::endl;

		openCL.deleteBuffer("readWriteBuffer");
	}
	void runBidirectionalBenchmarks(size_t aN)
	{
		writeReadToGPU(aN);
	}

	void setBufferSize(uint32_t aBufferSize)
	{
		bufferSize_ = aBufferSize;
		bufferLength_ = bufferSize_ / sizeof(datatype);
	}

	static void printAvailableDevices()
	{
		printAvailableDevices();
	}
};

#endif