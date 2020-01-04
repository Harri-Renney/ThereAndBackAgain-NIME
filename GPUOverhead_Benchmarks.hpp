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
#include <random>

//OpenGL API Dependencies//
//#define GLFW_EXPOSE_NATIVE_WIN32
//#define GLFW_EXPOSE_NATIVE_WGL
#include <windows.h>
//#include <glad\glad.h> 
//#include <GLFW\glfw3.h>

#include "OpenCL_Wrapper.h"
#include "CUDA_Wrapper.hpp"
#include "Benchmarker.hpp"
#include "OpenCL_FDTD.hpp"
#include "AudioFile.h"

class GPUOverhead_Benchmarks
{
private:
	static const size_t bufferSizesLength = 13;
	uint32_t bufferSizes[bufferSizesLength];
	
	typedef float datatype;
	uint32_t sampleRate_ = 44100;
	uint32_t bufferSize_ = 1024;
	uint32_t bufferLength_ = bufferSize_ / sizeof(datatype);

	OpenCL_Wrapper openCL;
	CUDA_Wrapper cuda_;
	cl::NDRange globalWorkspace_;
	cl::NDRange localWorkspace_;

	Benchmarker benchmarker_;

	OpenCL_FDTD fdtdSynth;
public:
	static const uint32_t GIGA_BYTE = 1024 * 1024 * 1024;
	static const uint32_t MEGA_BYTE = 1024 * 1024;
	static const uint32_t KILO_BYTE = 1024;
	GPUOverhead_Benchmarks(uint32_t aPlatform, uint32_t aDevice) : openCL(aPlatform, aDevice), cuda_(), fdtdSynth()
	{
		bufferSizes[0] = 1;
		for (size_t i = 1; i != 11; ++i)
		{
			bufferSizes[i] = bufferSizes[i - 1] * 2;
		}
		bufferSizes[11] = MEGA_BYTE;
		bufferSizes[12] = GIGA_BYTE;

		OpenCL_FDTD_Arguments args;
		args.isDebug = false;
		args.isBenchmark = false;
		args.modelWidth = 64;
		args.modelHeight = 64;
		args.propagationFactor = 0.06;
		args.dampingCoefficient = 0.0005;
		args.boundaryGain = 0.5;
		args.listenerPosition[0] = 8;
		args.listenerPosition[1] = 8;
		args.excitationPosition[0] = 32;
		args.excitationPosition[1] = 32;
		args.workGroupDimensions[0] = 16;
		args.workGroupDimensions[1] = 16;
		args.bufferSize = 128;
		args.kernelSource = "resources/kernels/fdtdGlobal.cl";
		new(&fdtdSynth) OpenCL_FDTD(args);
		//fdtdSynth = OpenCL_FDTD();
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

	void cl_004_mappedmemory(size_t aN, bool isWarmup)
	{
		//Test preperation//
		datatype* hostBuffer = new datatype[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		datatype* checkBuffer = new datatype[bufferLength_];

		//Write to buffer using OpenCL enqueueMapBuffer()//
		openCL.createBuffer("deviceBuffer", CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, bufferSize_);

		auto mappedMemory = openCL.mapMemory("deviceBuffer", bufferSize_);
		memcpy(mappedMemory, hostBuffer, bufferSize_);

		//Execute and average//
		std::cout << "Executing test: cl_004_mappedmemory" << std::endl;
		if (isWarmup)
		{
			memcpy(mappedMemory, hostBuffer, bufferSize_);
			memcpy(checkBuffer, mappedMemory, bufferSize_);
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			benchmarker_.startTimer("cl_004_mappedmemory");
			memcpy(mappedMemory, hostBuffer, bufferSize_);
			memcpy(checkBuffer, mappedMemory, bufferSize_);
			benchmarker_.pauseTimer("cl_004_mappedmemory");
		}
		benchmarker_.elapsedTimer("cl_004_mappedmemory");

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
		std::cout << "cl_004_mappedmemory successful: " << isSuccessful << std::endl << std::endl;

		openCL.unmapMemory("deviceBuffer", mappedMemory, bufferSize_);
		openCL.deleteBuffer("deviceBuffer");

		delete hostBuffer;
		delete checkBuffer;
	}

	void cl_005_cpymemory(size_t aN, bool isWarmup)
	{
		//Test preperation//
		float* srcMemoryBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			srcMemoryBuffer[i] = 42.0;
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
		//Test preperation//
		openCL.createKernelProgram("resources/kernels/GPU_Overhead_Benchmarks.cl", "");
		openCL.createKernel("cl_006_cpymemorykernel");
		openCL.setWorkspaceSize(bufferLength_, 256);

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
			//openCL.waitKernel();

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
	void cl_007_singlesample(size_t aN, bool isWarmup)
	{
		//Test preperation//
		openCL.createKernelProgram("resources/kernels/GPU_Overhead_Benchmarks.cl", "");
		openCL.createKernel("cl_007_singlesample");
		openCL.setWorkspaceSize(1, 1);

		float* hostSingleSample = new float;
		float* checkSingleSample = new float;
		*hostSingleSample = 42.0;
		*checkSingleSample = 0.0;

		openCL.createBuffer("singleBuffer", CL_MEM_READ_WRITE, sizeof(float));
		openCL.setKernelArgument("cl_007_singlesample", "singleBuffer", 0, sizeof(cl_mem));

		//Execute and average//
		std::cout << "Executing test: cl_007_singlesample" << std::endl;
		if (isWarmup)
		{
			openCL.writeBuffer("singleBuffer", sizeof(float), hostSingleSample);
			openCL.enqueueKernel("cl_007_singlesample");
			openCL.readBuffer("singleBuffer", sizeof(float), checkSingleSample);
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			benchmarker_.startTimer("cl_007_singlesample");

			openCL.writeBuffer("singleBuffer", sizeof(float), hostSingleSample);
			openCL.enqueueKernel("cl_007_singlesample");
			openCL.readBuffer("singleBuffer", sizeof(float), checkSingleSample);

			benchmarker_.pauseTimer("cl_007_singlesample");
		}
		benchmarker_.elapsedTimer("cl_007_singlesample");

		//Check contents//
		bool isSuccessful = true;
		if (checkSingleSample[0] != hostSingleSample[0] * 0.5)
		{
			isSuccessful = false;
		}
		std::cout << "cl_007_singlesample successful: " << isSuccessful << std::endl << std::endl;

		openCL.deleteBuffer("singleBuffer");

		delete checkSingleSample;
		delete hostSingleSample;
	}
	void cl_007_singlesamplemapping(size_t aN, bool isWarmup)
	{
		//Test preperation//
		openCL.createKernelProgram("resources/kernels/GPU_Overhead_Benchmarks.cl", "");
		openCL.createKernel("cl_007_singlesample");
		openCL.setWorkspaceSize(1, 1);

		float* hostSingleSample = new float;
		float* checkSingleSample = new float;
		*hostSingleSample = 42.0;
		*checkSingleSample = 0.0;

		openCL.createBuffer("singleBuffer", CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(float));
		openCL.setKernelArgument("cl_007_singlesample", "singleBuffer", 0, sizeof(cl_mem));

		//Execute and average//
		std::cout << "Executing test: cl_007_singlesample" << std::endl;
		if (isWarmup)
		{
			auto mappedMemoryOne = openCL.mapMemory("singleBuffer", sizeof(float));
			memcpy(mappedMemoryOne, hostSingleSample, sizeof(float));
			openCL.unmapMemory("singleBuffer", mappedMemoryOne, sizeof(float));

			openCL.enqueueKernel("cl_007_singlesample");
			
			auto mappedMemoryTwo = openCL.mapMemory("singleBuffer", sizeof(float));
			memcpy(checkSingleSample, mappedMemoryTwo, sizeof(float));
			openCL.unmapMemory("singleBuffer", mappedMemoryTwo, sizeof(float));
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			benchmarker_.startTimer("cl_007_singlesample");

			auto mappedMemoryOne = openCL.mapMemory("singleBuffer", sizeof(float));
			memcpy(mappedMemoryOne, hostSingleSample, sizeof(float));
			openCL.unmapMemory("singleBuffer", mappedMemoryOne, sizeof(float));

			openCL.enqueueKernel("cl_007_singlesample");

			auto mappedMemoryTwo = openCL.mapMemory("singleBuffer", sizeof(float));
			memcpy(checkSingleSample, mappedMemoryTwo, sizeof(float));
			openCL.unmapMemory("singleBuffer", mappedMemoryTwo, sizeof(float));

			benchmarker_.pauseTimer("cl_007_singlesample");
		}
		benchmarker_.elapsedTimer("cl_007_singlesample");

		//Check contents//
		bool isSuccessful = true;
		if (checkSingleSample[0] != hostSingleSample[0] * 0.5)
		{
			isSuccessful = false;
		}
		std::cout << "cl_007_singlesample successful: " << isSuccessful << std::endl << std::endl;

		openCL.deleteBuffer("singleBuffer");

		delete checkSingleSample;
		delete hostSingleSample;
	}

	void cl_008_simplebufferprocessing(size_t aN, bool isWarmup)
	{
		//Test preperation//
		openCL.createKernelProgram("resources/kernels/GPU_Overhead_Benchmarks.cl", "");
		openCL.createKernel("cl_008_simplebufferprocessing");
		openCL.setWorkspaceSize(bufferLength_, 256);

		float* srcMemoryBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			srcMemoryBuffer[i] = 42.0;
		float* dstMemoryBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			dstMemoryBuffer[i] = 0.0;

		openCL.createBuffer("inputBuffer", CL_MEM_READ_ONLY, bufferSize_);
		openCL.createBuffer("outputBuffer", CL_MEM_WRITE_ONLY, bufferSize_);

		openCL.setKernelArgument("cl_008_simplebufferprocessing", "inputBuffer", 0, sizeof(cl_mem));
		openCL.setKernelArgument("cl_008_simplebufferprocessing", "outputBuffer", 1, sizeof(cl_mem));

		//Execute and average//
		std::cout << "Executing test: cl_008_simplebufferprocessing" << std::endl;
		if (isWarmup)
		{
			openCL.writeBuffer("inputBuffer", bufferSize_, srcMemoryBuffer);
			openCL.enqueueKernel("cl_008_simplebufferprocessing");
			openCL.readBuffer("outputBuffer", bufferSize_, dstMemoryBuffer);
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			benchmarker_.startTimer("cl_008_simplebufferprocessing");

			openCL.writeBuffer("inputBuffer", bufferSize_, srcMemoryBuffer);
			openCL.enqueueKernel("cl_008_simplebufferprocessing");
			openCL.readBuffer("outputBuffer", bufferSize_, dstMemoryBuffer);

			benchmarker_.pauseTimer("cl_008_simplebufferprocessing");
		}
		benchmarker_.elapsedTimer("cl_008_simplebufferprocessing");

		//Check contents//
		bool isSuccessful = true;
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			//Calculate simple attenuation on CPU to compare//
			srcMemoryBuffer[i] = srcMemoryBuffer[i] * 0.5;
			if (srcMemoryBuffer[i] != dstMemoryBuffer[i])
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cl_008_simplebufferprocessing successful: " << isSuccessful << std::endl << std::endl;

		openCL.deleteBuffer("inputBuffer");
		openCL.deleteBuffer("outputBuffer");

		delete srcMemoryBuffer;
		delete dstMemoryBuffer;
	}
	void cl_008_simplebufferprocessingmapping(size_t aN, bool isWarmup)
	{
		//Test preperation//
		openCL.createKernelProgram("resources/kernels/GPU_Overhead_Benchmarks.cl", "");
		openCL.createKernel("cl_008_simplebufferprocessing");
		openCL.setWorkspaceSize(bufferLength_, 256);

		void* mappedMemoryOne;
		void* mappedMemoryTwo;

		float* srcMemoryBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			srcMemoryBuffer[i] = 42.0;
		float* dstMemoryBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			dstMemoryBuffer[i] = 0.0;

		openCL.createBuffer("inputBuffer", CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, bufferSize_);
		openCL.createBuffer("outputBuffer", CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, bufferSize_);

		openCL.setKernelArgument("cl_008_simplebufferprocessing", "inputBuffer", 0, sizeof(cl_mem));
		openCL.setKernelArgument("cl_008_simplebufferprocessing", "outputBuffer", 1, sizeof(cl_mem));

		//Execute and average//
		std::cout << "Executing test: cl_008_simplebufferprocessing" << std::endl;
		if (isWarmup)
		{
			mappedMemoryOne = openCL.mapMemory("inputBuffer", bufferSize_);
			memcpy(mappedMemoryOne, srcMemoryBuffer, bufferSize_);
			openCL.unmapMemory("inputBuffer", mappedMemoryOne, bufferSize_);

			openCL.enqueueKernel("cl_008_simplebufferprocessing");

			mappedMemoryTwo = openCL.mapMemory("outputBuffer", bufferSize_);
			memcpy(dstMemoryBuffer, mappedMemoryTwo, bufferSize_);
			openCL.unmapMemory("outputBuffer", mappedMemoryTwo, bufferSize_);
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			benchmarker_.startTimer("cl_008_simplebufferprocessing");

			mappedMemoryOne = openCL.mapMemory("inputBuffer", bufferSize_);
			memcpy(mappedMemoryOne, srcMemoryBuffer, bufferSize_);
			openCL.unmapMemory("inputBuffer", mappedMemoryOne, bufferSize_);

			openCL.enqueueKernel("cl_008_simplebufferprocessing");

			mappedMemoryTwo = openCL.mapMemory("outputBuffer", bufferSize_);
			memcpy(dstMemoryBuffer, mappedMemoryTwo, bufferSize_);
			openCL.unmapMemory("outputBuffer", mappedMemoryTwo, bufferSize_);

			benchmarker_.pauseTimer("cl_008_simplebufferprocessing");
		}
		benchmarker_.elapsedTimer("cl_008_simplebufferprocessing");

		//Check contents//
		bool isSuccessful = true;
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			//Calculate simple attenuation on CPU to compare//
			srcMemoryBuffer[i] = srcMemoryBuffer[i] * 0.5;
			if (srcMemoryBuffer[i] != dstMemoryBuffer[i])
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cl_008_simplebufferprocessing successful: " << isSuccessful << std::endl << std::endl;

		openCL.deleteBuffer("inputBuffer");
		openCL.deleteBuffer("outputBuffer");

		/*delete mappedMemoryOne;
		delete mappedMemoryTwo;*/
		delete srcMemoryBuffer;
		delete dstMemoryBuffer;
	}
	void cl_009_complexbufferprocessing(size_t aN, bool isWarmup)
	{
		//Test preperation//
		openCL.createKernelProgram("resources/kernels/GPU_Overhead_Benchmarks.cl", "");
		openCL.createKernel("cl_009_complexbufferprocessing");
		openCL.setWorkspaceSize(bufferLength_, 256);

		float* srcMemoryBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			srcMemoryBuffer[i] = 42.0 * (i % 4);
		float* dstMemoryBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			dstMemoryBuffer[i] = 0.0;

		openCL.createBuffer("inputBuffer", CL_MEM_READ_ONLY, bufferSize_);
		openCL.createBuffer("outputBuffer", CL_MEM_WRITE_ONLY, bufferSize_);

		openCL.setKernelArgument("cl_009_complexbufferprocessing", "inputBuffer", 0, sizeof(cl_mem));
		openCL.setKernelArgument("cl_009_complexbufferprocessing", "outputBuffer", 1, sizeof(cl_mem));

		//Execute and average//
		std::cout << "Executing test: cl_009_complexbufferprocessing" << std::endl;
		if (isWarmup)
		{
			openCL.writeBuffer("inputBuffer", bufferSize_, srcMemoryBuffer);
			openCL.enqueueKernel("cl_009_complexbufferprocessing");
			openCL.readBuffer("outputBuffer", bufferSize_, dstMemoryBuffer);
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			benchmarker_.startTimer("cl_009_complexbufferprocessing");

			openCL.writeBuffer("inputBuffer", bufferSize_, srcMemoryBuffer);
			openCL.enqueueKernel("cl_009_complexbufferprocessing");
			openCL.waitKernel();
			openCL.readBuffer("outputBuffer", bufferSize_, dstMemoryBuffer);

			benchmarker_.pauseTimer("cl_009_complexbufferprocessing");
		}
		benchmarker_.elapsedTimer("cl_009_complexbufferprocessing");

		//Check contents//
		bool isSuccessful = true;
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			//Calculate triangular smoothing on CPU to compare//
			float smoothedSample = srcMemoryBuffer[i];
			if (i > 2 & i < bufferLength_-2)
			{
				smoothedSample = ((srcMemoryBuffer[i - 2] + 2.0 * srcMemoryBuffer[i - 1] + 3.0 * srcMemoryBuffer[i] + 2.0 * srcMemoryBuffer[i + 1] + srcMemoryBuffer[i + 2]) / 9.0);
			}
			if (smoothedSample != dstMemoryBuffer[i])
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cl_009_complexbufferprocessing successful: " << isSuccessful << std::endl << std::endl;

		openCL.deleteBuffer("inputBuffer");
		openCL.deleteBuffer("outputBuffer");

		delete srcMemoryBuffer;
		delete dstMemoryBuffer;
	}
	void cl_009_complexbufferprocessingmapping(size_t aN, bool isWarmup)
	{
		//Test preperation//
		openCL.createKernelProgram("resources/kernels/GPU_Overhead_Benchmarks.cl", "");
		openCL.createKernel("cl_009_complexbufferprocessing");
		openCL.setWorkspaceSize(bufferLength_, 256);

		float* srcMemoryBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			srcMemoryBuffer[i] = 42.0 * (i % 4);
		float* dstMemoryBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			dstMemoryBuffer[i] = 0.0;

		openCL.createBuffer("inputBuffer", CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bufferSize_);
		openCL.createBuffer("outputBuffer", CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bufferSize_);

		openCL.setKernelArgument("cl_009_complexbufferprocessing", "inputBuffer", 0, sizeof(cl_mem));
		openCL.setKernelArgument("cl_009_complexbufferprocessing", "outputBuffer", 1, sizeof(cl_mem));

		//Execute and average//
		std::cout << "Executing test: cl_009_complexbufferprocessing" << std::endl;
		if (isWarmup)
		{
			auto mappedMemoryOne = openCL.mapMemory("inputBuffer", bufferSize_);
			memcpy(mappedMemoryOne, srcMemoryBuffer, bufferSize_);
			openCL.unmapMemory("inputBuffer", mappedMemoryOne, bufferSize_);

			openCL.enqueueKernel("cl_009_complexbufferprocessing");

			auto mappedMemoryTwo = openCL.mapMemory("outputBuffer", bufferSize_);
			memcpy(dstMemoryBuffer, mappedMemoryTwo, bufferSize_);
			openCL.unmapMemory("outputBuffer", mappedMemoryTwo, bufferSize_);
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			benchmarker_.startTimer("cl_009_complexbufferprocessing");

			auto mappedMemoryOne = openCL.mapMemory("inputBuffer", bufferSize_);
			memcpy(mappedMemoryOne, srcMemoryBuffer, bufferSize_);
			openCL.unmapMemory("inputBuffer", mappedMemoryOne, bufferSize_);

			openCL.enqueueKernel("cl_009_complexbufferprocessing");

			auto mappedMemoryTwo = openCL.mapMemory("outputBuffer", bufferSize_);
			memcpy(dstMemoryBuffer, mappedMemoryTwo, bufferSize_);
			openCL.unmapMemory("outputBuffer", mappedMemoryTwo, bufferSize_);

			benchmarker_.pauseTimer("cl_009_complexbufferprocessing");
		}
		benchmarker_.elapsedTimer("cl_009_complexbufferprocessing");

		//Check contents//
		bool isSuccessful = true;
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			//Calculate triangular smoothing on CPU to compare//
			float smoothedSample = srcMemoryBuffer[i];
			if (i > 2 & i < bufferLength_ - 2)
			{
				smoothedSample = ((srcMemoryBuffer[i - 2] + 2.0 * srcMemoryBuffer[i - 1] + 3.0 * srcMemoryBuffer[i] + 2.0 * srcMemoryBuffer[i + 1] + srcMemoryBuffer[i + 2]) / 9.0);
			}
			if (smoothedSample != dstMemoryBuffer[i])
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cl_009_complexbufferprocessing successful: " << isSuccessful << std::endl << std::endl;

		openCL.deleteBuffer("inputBuffer");
		openCL.deleteBuffer("outputBuffer");

		delete srcMemoryBuffer;
		delete dstMemoryBuffer;
	}

	void cl_010_simplebuffersynthesis(size_t aN, bool isWarmup)
	{
		//Test preperation//
		float* srcMemoryBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			srcMemoryBuffer[i] = 42.0;
		float* dstMemoryBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			dstMemoryBuffer[i] = 0.0;

		//Execute and average//
		std::cout << "Executing test: cl_010_simplebuffersynthesis" << std::endl;
		if (isWarmup)
		{

		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			benchmarker_.startTimer("cl_010_simplebuffersynthesis");



			benchmarker_.pauseTimer("cl_010_simplebuffersynthesis");
		}
		benchmarker_.elapsedTimer("cl_010_simplebuffersynthesis");

		//Check contents//
		bool isSuccessful = true;
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			if (srcMemoryBuffer[i] != dstMemoryBuffer[i])
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cl_010_simplebuffersynthesis successful: " << isSuccessful << std::endl << std::endl;

		openCL.deleteBuffer("srcBuffer");
		openCL.deleteBuffer("dstBuffer");

		delete srcMemoryBuffer;
		delete dstMemoryBuffer;
	}
	void cl_011_complexbuffersynthesis(size_t aN, bool isWarmup)
	{
		//Test preperation//
		float* inBuf = new float[bufferSize_];
		float* outBuf = new float[bufferSize_];
		for (size_t i = 0; i != bufferSize_; ++i)
			inBuf[i] = 0.5;

		OpenCL_FDTD_Arguments args;
		args.isDebug = false;
		args.isBenchmark = false;
		args.modelWidth = 64;
		args.modelHeight = 64;
		args.propagationFactor = 0.06;
		args.dampingCoefficient = 0.0005;
		args.boundaryGain = 0.5;
		args.listenerPosition[0] = 8;
		args.listenerPosition[1] = 8;
		args.excitationPosition[0] = 32;
		args.excitationPosition[1] = 32;
		args.workGroupDimensions[0] = 16;
		args.workGroupDimensions[1] = 16;
		args.bufferSize = bufferSize_;
		args.kernelSource = "resources/kernels/fdtdGlobal.cl";
		OpenCL_FDTD fdtdSynth = OpenCL_FDTD(args);

		//Execute and average//
		std::cout << "Executing test: cl_011_complexbuffersynthesis" << std::endl;
		if (isWarmup)
		{
			fdtdSynth.fillBuffer(bufferSize_, inBuf, outBuf);
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			benchmarker_.startTimer("cl_011_complexbuffersynthesis");

			fdtdSynth.fillBuffer(bufferSize_, inBuf, outBuf);

			benchmarker_.pauseTimer("cl_011_complexbuffersynthesis");
		}
		benchmarker_.elapsedTimer("cl_011_complexbuffersynthesis");

		//Check contents//
		bool isSuccessful = true;
		std::cout << "cl_011_complexbuffersynthesis successful: " << isSuccessful << std::endl << std::endl;

		//Cleanup//
		delete inBuf;
		delete outBuf;
	}

	void cl_012_interruptedbufferprocessing(size_t aN, bool isWarmup)
	{
		//Test preperation//
		openCL.createKernelProgram("resources/kernels/GPU_Overhead_Benchmarks.cl", "");
		openCL.createKernel("cl_012_interruptedbufferprocessing");
		openCL.setWorkspaceSize(bufferLength_, 256);

		void* mappedMemoryOne;
		void* mappedMemoryTwo;

		bool isBufferCorrect = true;		//Variable indicating if buffer requires recalculating.
		std::default_random_engine generator;
		std::uniform_int_distribution<int> distribution(1, 10);
		float* srcMemoryBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			srcMemoryBuffer[i] = 42.0;
		float* dstMemoryBuffer = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			dstMemoryBuffer[i] = 0.0;

		openCL.createBuffer("inputBuffer", CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, bufferSize_);
		openCL.createBuffer("outputBuffer", CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, bufferSize_);

		openCL.setKernelArgument("cl_012_interruptedbufferprocessing", "inputBuffer", 0, sizeof(cl_mem));
		openCL.setKernelArgument("cl_012_interruptedbufferprocessing", "outputBuffer", 1, sizeof(cl_mem));

		//Execute and average//
		std::cout << "Executing test: cl_012_interruptedbufferprocessing" << std::endl;
		if (isWarmup)
		{
			mappedMemoryOne = openCL.mapMemory("inputBuffer", bufferSize_);
			memcpy(mappedMemoryOne, srcMemoryBuffer, bufferSize_);
			openCL.unmapMemory("inputBuffer", mappedMemoryOne, bufferSize_);

			openCL.enqueueKernel("cl_012_interruptedbufferprocessing");

			mappedMemoryTwo = openCL.mapMemory("outputBuffer", bufferSize_);
			memcpy(dstMemoryBuffer, mappedMemoryTwo, bufferSize_);
			openCL.unmapMemory("outputBuffer", mappedMemoryTwo, bufferSize_);
		}
		for (int32_t i = 0; i != aN; ++i)
		{
			benchmarker_.startTimer("cl_012_interruptedbufferprocessing");

			mappedMemoryOne = openCL.mapMemory("inputBuffer", bufferSize_);
			memcpy(mappedMemoryOne, srcMemoryBuffer, bufferSize_);
			openCL.unmapMemory("inputBuffer", mappedMemoryOne, bufferSize_);

			openCL.enqueueKernel("cl_012_interruptedbufferprocessing");
			openCL.waitKernel();

			mappedMemoryTwo = openCL.mapMemory("outputBuffer", bufferSize_);
			memcpy(dstMemoryBuffer, mappedMemoryTwo, bufferSize_);
			openCL.unmapMemory("outputBuffer", mappedMemoryTwo, bufferSize_);
			openCL.waitKernel();

			benchmarker_.pauseTimer("cl_012_interruptedbufferprocessing");

			int randomChance = distribution(generator);
			if (randomChance > 5)
				--i;
				
		}
		benchmarker_.elapsedTimer("cl_012_interruptedbufferprocessing");

		//Check contents//
		bool isSuccessful = true;
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			//Calculate simple attenuation on CPU to compare//
			srcMemoryBuffer[i] = srcMemoryBuffer[i] * 0.5;
			if (srcMemoryBuffer[i] != dstMemoryBuffer[i])
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cl_012_interruptedbufferprocessing successful: " << isSuccessful << std::endl << std::endl;

		//Cleanup//
		openCL.deleteBuffer("inputBuffer");
		openCL.deleteBuffer("outputBuffer");
		delete srcMemoryBuffer;
		delete dstMemoryBuffer;
	}

	void unidirectionalBaseline(size_t aN, size_t sampleRate)
	{
		//Test preperation//
		datatype* hostBuffer = new datatype[MEGA_BYTE / sizeof(float)];
		for (size_t i = 0; i != MEGA_BYTE / sizeof(float); ++i)
			hostBuffer[i] = 42.0;
		datatype* checkBuffer = new datatype[MEGA_BYTE / sizeof(float)];

		openCL.createBuffer("deviceBuffer", CL_MEM_READ_WRITE, MEGA_BYTE);

		uint32_t numSamplesComputed = 0;
		for (size_t i = 0; i != bufferSizesLength - 2; ++i)
		{
			uint32_t currentBufferSize = bufferSizes[i];

			openCL.enqueueKernel("cl_000_nullKernel");
			openCL.readBuffer("deviceBuffer", currentBufferSize * sizeof(float), checkBuffer);
			benchmarker_.startTimer("totalTime");
			while (numSamplesComputed < sampleRate)
			{

				benchmarker_.startTimer("bufferTime");
				openCL.enqueueKernel("cl_000_nullKernel");
				openCL.readBuffer("deviceBuffer", currentBufferSize * sizeof(float), checkBuffer);
				benchmarker_.pauseTimer("bufferTime");

				numSamplesComputed += currentBufferSize;
			}
			benchmarker_.pauseTimer("totalTime");
			benchmarker_.elapsedTimer("totalTime");
			benchmarker_.elapsedTimer("bufferTime");

			numSamplesComputed = 0;
		}

		openCL.deleteBuffer("deviceBuffer");
		delete hostBuffer;
		delete checkBuffer;
	}
	void runUnidirectionalBenchmarks(size_t aN, size_t sampleRate)
	{
		unidirectionalBaseline(aN, sampleRate);
	}
	void bidirectionalComplexSynthesis(size_t aN, size_t sampleRate)
	{
		uint32_t numSamplesComputed = 0;
		for (size_t i = 0; i != bufferSizesLength-2; ++i)
		{
			uint32_t currentBufferSize = bufferSizes[i];

			OpenCL_FDTD_Arguments args;
			args.isDebug = false;
			args.isBenchmark = false;
			args.modelWidth = 64;
			args.modelHeight = 64;
			args.propagationFactor = 0.06;
			args.dampingCoefficient = 0.0005;
			args.boundaryGain = 0.5;
			args.listenerPosition[0] = 8;
			args.listenerPosition[1] = 8;
			args.excitationPosition[0] = 32;
			args.excitationPosition[1] = 32;
			args.workGroupDimensions[0] = 16;
			args.workGroupDimensions[1] = 16;
			args.bufferSize = currentBufferSize;
			args.kernelSource = "resources/kernels/fdtdGlobal.cl";
			OpenCL_FDTD fdtdSynth = OpenCL_FDTD(args);
			//new(&fdtdSynth) OpenCL_FDTD(args);

			//fdtdSynth.setBufferSize(currentBufferSize);
			//fdtdSynth.init();

			uint32_t numSamplesComputed = 0;
			float* inBuf = new float[currentBufferSize];
			float* outBuf = new float[currentBufferSize];
			for (size_t i = 0; i != currentBufferSize; ++i)
				inBuf[i] = 0.5;

			float* soundBuffer = new float[sampleRate*2];

			benchmarker_.startTimer("totalTime");
			while (numSamplesComputed < sampleRate)
			{

				benchmarker_.startTimer("bufferTime");
				fdtdSynth.fillBuffer(currentBufferSize, inBuf, outBuf);
				benchmarker_.pauseTimer("bufferTime");

				//Log audio for inspection if necessary//
				for (int j = 0; j != currentBufferSize; ++j)
					soundBuffer[numSamplesComputed + j] = outBuf[j];

				numSamplesComputed += currentBufferSize;
			}
			benchmarker_.pauseTimer("totalTime");
			benchmarker_.elapsedTimer("totalTime");
			benchmarker_.elapsedTimer("bufferTime");

			outputAudioFile("test.wav", soundBuffer, sampleRate);	//Save audio to file for inspection.

			numSamplesComputed = 0;

			delete inBuf;
			delete outBuf;
		}
	}
	void runBidirectionalBenchmarks(size_t aN, size_t sampleRate)
	{
		//Test preperation//
		datatype* hostBuffer = new datatype[MEGA_BYTE / sizeof(float)];
		for (size_t i = 0; i != MEGA_BYTE / sizeof(float); ++i)
			hostBuffer[i] = 42.0;
		datatype* checkBuffer = new datatype[MEGA_BYTE / sizeof(float)];

		openCL.createBuffer("deviceBuffer", CL_MEM_READ_WRITE, MEGA_BYTE);

		uint32_t numSamplesComputed = 0;
		for (size_t i = 0; i != bufferSizesLength - 2; ++i)
		{
			uint32_t currentBufferSize = bufferSizes[i];

			openCL.writeBuffer("deviceBuffer", currentBufferSize * sizeof(float), hostBuffer);
			openCL.enqueueKernel("cl_000_nullKernel");
			openCL.readBuffer("deviceBuffer", currentBufferSize * sizeof(float), checkBuffer);
			benchmarker_.startTimer("totalTime");
			while (numSamplesComputed < sampleRate)
			{

				benchmarker_.startTimer("bufferTime");
				openCL.writeBuffer("deviceBuffer", currentBufferSize * sizeof(float), hostBuffer);
				openCL.enqueueKernel("cl_000_nullKernel");
				openCL.readBuffer("deviceBuffer", currentBufferSize * sizeof(float), checkBuffer);
				benchmarker_.pauseTimer("bufferTime");

				numSamplesComputed += currentBufferSize;
			}
			benchmarker_.pauseTimer("totalTime");
			benchmarker_.elapsedTimer("totalTime");
			benchmarker_.elapsedTimer("bufferTime");

			numSamplesComputed = 0;
		}

		openCL.deleteBuffer("deviceBuffer");
		delete hostBuffer;
		delete checkBuffer;
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
};

#endif