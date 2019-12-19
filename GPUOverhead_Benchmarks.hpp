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
	uint32_t sampleRate_ = 44100;
	uint32_t bufferSize_ = 1024;

	OpenCL_Wrapper openCL;
	CUDA_Wrapper cuda_;
	cl::NDRange globalWorkspace_;
	cl::NDRange localWorkspace_;

	Benchmarker benchmarker_;
public:
	GPUOverhead_Benchmarks(uint32_t aPlatform, uint32_t aDevice) : openCL(aPlatform, aDevice), cuda_()
	{
	}

	void nullKernel(size_t aN)
	{
		std::cout << "Null kernel" << std::endl;

		openCL.setWorkspaceSize(1024, 32);
		for (uint32_t i = 0; i != aN; ++i)
		{
			benchmarker_.startTimer("nullKernel");
			openCL.enqueueKernel("nullKernel");
			openCL.waitKernel();
			benchmarker_.pauseTimer("nullKernel");
		}
		benchmarker_.elapsedTimer("nullKernel");

		std::cout << "Null kernel successful? " << true << std::endl << std::endl;
	}

	void writeToGPUMapped(size_t aN)
	{
		//Write to buffer using OpenCL enqueueMapBuffer()//
		openCL.createBuffer("writeDstBuffer", CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, bufferSize_);

		char* memoryBuffer = new char[bufferSize_];
		memset(memoryBuffer, '1', bufferSize_);

		char* memoryBufferCheck = new char[bufferSize_];
		memset(memoryBufferCheck, '1', bufferSize_);


		auto mappedMemory = openCL.pinMappedMemory("writeDstBuffer", bufferSize_);

		std::cout << "Write to buffer using enqueueMapBuffer()" << std::endl;
		for (uint32_t i = 0; i != aN; ++i)
		{
			benchmarker_.startTimer("writeToGPUMapped");
			memcpy(memoryBuffer, mappedMemory, bufferSize_);
			benchmarker_.pauseTimer("writeToGPUMapped");
		}
		benchmarker_.elapsedTimer("writeToGPUMapped");

		openCL.deleteBuffer("writeDstBuffer");

		delete memoryBuffer;
		delete memoryBufferCheck;

		//memset(memoryBufferCheck, '1', bufferSize_);
	}
	void writeToGPU(size_t aN)
	{	
		//Write to buffer using OpenCL enqueueWriteBuffer()//
		openCL.createBuffer("writeDstBuffer", CL_MEM_WRITE_ONLY, bufferSize_);

		char* memoryBuffer = new char[bufferSize_];
		memset(memoryBuffer, '1', bufferSize_);

		char* memoryBufferCheck = new char[bufferSize_];
		memset(memoryBufferCheck, '1', bufferSize_);

		std::cout << "Write to buffer using enqueueWriteBuffer()" << std::endl;
		for (uint32_t i = 0; i != aN; ++i)
		{
			//std::cout << "Write to buffer using enqueueWriteBuffer()" << std::endl;
			benchmarker_.startTimer("writeToGPU");
			openCL.writeBuffer("writeDstBuffer", bufferSize_, memoryBuffer);
			openCL.waitKernel();

			benchmarker_.pauseTimer("writeToGPU");
			//benchmarker_.endTimer();
			//benchmarker_.elapsedTimer();

			bool isSuccessful = true;
			openCL.readBuffer("writeDstBuffer", bufferSize_, memoryBufferCheck);
			for (uint32_t i = 0; i != bufferSize_; ++i)
			{
				if (memoryBuffer[i] != memoryBufferCheck[i])
				{
					isSuccessful = false;
					break;
				}
			}
			//std::cout << "Write buffer successful? " << isSuccessful << std::endl << std::endl;
		}
		benchmarker_.elapsedTimer("writeToGPU");

		openCL.deleteBuffer("writeDstBuffer");

		delete memoryBuffer;
		delete memoryBufferCheck;
	}

	void copyBuffer(size_t aN)
	{
		//init(1,0);

		//Copy between buffers using OpenCL clEnqueueCopyBuffer()//
		openCL.createBuffer("writeSrcBuffer", CL_MEM_READ_ONLY, bufferSize_);
		openCL.createBuffer("writeDstBuffer", CL_MEM_WRITE_ONLY, bufferSize_);

		char* srcMemoryBuffer = new char[bufferSize_];
		memset(srcMemoryBuffer, '1', bufferSize_);
		char* dstMemoryBuffer = new char[bufferSize_];
		memset(dstMemoryBuffer, '0', bufferSize_);
		openCL.writeBuffer("writeSrcBuffer", bufferSize_, srcMemoryBuffer);
		openCL.writeBuffer("writeDstBuffer", bufferSize_, dstMemoryBuffer);
		//setKernelArgument("copyBuffer", "writeSrcBuffer", 0, sizeof(cl_mem));
		//setKernelArgument("copyBuffer", "writeDstBuffer", 1, sizeof(cl_mem));

		std::cout << "Copying between two buffers using clEnqueueCopyBuffer" << std::endl << std::endl;

		for (uint32_t i = 0; i != aN; ++i)
		{
			benchmarker_.startTimer("copyBuffer");
			openCL.enqueueCopyBuffer("writeSrcBuffer", "writeDstBuffer", bufferSize_);
			openCL.waitKernel();
			benchmarker_.pauseTimer("copyBuffer");
		}
		benchmarker_.elapsedTimer("copyBuffer");

		//Check contents//
		bool isSuccessful = true;
		openCL.readBuffer("writeDstBuffer", bufferSize_, dstMemoryBuffer);
		for (uint32_t i = 0; i != bufferSize_; ++i)
		{
			if (srcMemoryBuffer[i] != dstMemoryBuffer[i])
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "Copy buffer kernel successful? " << isSuccessful << std::endl << std::endl;

		openCL.deleteBuffer("writeSrcBuffer");
		openCL.deleteBuffer("writeDstBuffer");

		delete srcMemoryBuffer;
		delete dstMemoryBuffer;
	}
	void copyBufferKernel(size_t aN)
	{
		//init(1, 0);

		//Copy between buffer using OpenCL kernel//
		openCL.createKernelProgram("resources/kernels/GPU_Overhead_Benchmarks.cl", "");
		openCL.createKernel("copyBuffer");
		openCL.setWorkspaceSize(bufferSize_, 8);

		char* srcMemoryBuffer = new char[bufferSize_];
		memset(srcMemoryBuffer, '1', bufferSize_);
		char* dstMemoryBuffer = new char[bufferSize_];
		memset(dstMemoryBuffer, '0', bufferSize_);
		openCL.createBuffer("writeSrcBuffer", CL_MEM_READ_WRITE, bufferSize_);
		openCL.createBuffer("writeDstBuffer", CL_MEM_READ_WRITE, bufferSize_);
		openCL.writeBuffer("writeSrcBuffer", bufferSize_, srcMemoryBuffer);
		openCL.writeBuffer("writeDstBuffer", bufferSize_, dstMemoryBuffer);
		openCL.setKernelArgument("copyBuffer", "writeSrcBuffer", 0, sizeof(cl_mem));
		openCL.setKernelArgument("copyBuffer", "writeDstBuffer", 1, sizeof(cl_mem));

		std::cout << "Copying between two buffers using kernel" << std::endl << std::endl;

		for (uint32_t i = 0; i != aN; ++i)
		{
			benchmarker_.startTimer("copyBufferKernel");

			openCL.enqueueKernel("copyBuffer");
			//openCL.waitKernel();

			benchmarker_.pauseTimer("copyBufferKernel");
		}
		benchmarker_.elapsedTimer("copyBufferKernel");

		//Check contents//
		bool isSuccessful = true;
		openCL.readBuffer("writeDstBuffer", bufferSize_, dstMemoryBuffer);
		openCL.waitKernel();
		for (uint32_t i = 0; i != bufferSize_; ++i)
		{
			if (srcMemoryBuffer[i] != dstMemoryBuffer[i])
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "Copy buffer kernel successful? " << isSuccessful << std::endl << std::endl;

		openCL.deleteBuffer("writeSrcBuffer");
		openCL.deleteBuffer("writeDstBuffer");

		delete srcMemoryBuffer;
		delete dstMemoryBuffer;
	}
	void runUnidirectionalBenchmarks(size_t aN)
	{
		//averageBenchmark(writeToGPU, 10);
		//writeToGPU();
		copyBuffer(aN);
		copyBufferKernel(aN);
		
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
	}

	static void printAvailableDevices()
	{
		printAvailableDevices();
	}
};

#endif