#ifndef GPU_OVERHEAD_BENCHMARKS_OPENCL_HPP
#define GPU_OVERHEAD_BENCHMARKS_OPENCL_HPP

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

#include "GPU_Overhead_Benchmarks.hpp"
#include "OpenCL_Wrapper.h"
#include "Benchmarker.hpp"
#include "OpenCL_FDTD.hpp"
#include "AudioFile.h"

class GPU_Overhead_Benchmarks_OpenCL : public GPU_Overhead_Benchmarks
{
private:
	OpenCL_Wrapper openCL;
	cl::NDRange globalWorkspace_;
	cl::NDRange localWorkspace_;

	Benchmarker clBenchmarker_;

	OpenCL_FDTD fdtdSynth;

	//OpenCL objects//
	cl::Context context_;
	cl::Device device_;
	cl::CommandQueue commandQueue_;
	cl::Program kernelProgram_;
	cl::Event kernelBenchmark_;

	cl::Event clEvent;
public:
	GPU_Overhead_Benchmarks_OpenCL(uint32_t aPlatform, uint32_t aDevice) : fdtdSynth(), clBenchmarker_("openclog.csv", { "Test_Name", "Total_Time", "Average_Time", "Max_Time", "Min_Time", "Max_Difference", "Average_Difference" })
	{
		openCL = OpenCL_Wrapper(aPlatform, aDevice, context_, device_, commandQueue_);
		bufferSizes[0] = 1;
		for (size_t i = 1; i != bufferSizesLength; ++i)
		{
			bufferSizes[i] = bufferSizes[i - 1] * 2;
		}

		//Build the program - Define kernel constants//
		char options[1024];
		snprintf(options, sizeof(options),
			" -cl-fast-relaxed-math"
			//" -cl-single-precision-constant"
			//""
		);

		//SeqMemoryTest0//
		openCL.createKernelProgram(context_, kernelProgram_, "resources/kernels/GPU_Overhead_Benchmarks.cl", options);

		//Initialise workgroup dimensions//
		globalWorkspace_ = cl::NDRange(1024);
		localWorkspace_ = cl::NDRange(256);
	}

	void runGeneralBenchmarks(uint64_t aNumRepetitions, bool isWarmup) override
	{
		for (uint32_t i = 0; i != bufferSizesLength; ++i)
		{
			uint64_t currentBufferSize = bufferSizes[i];
			std::string benchmarkFileName = "cl_";
			std::string strBufferSize = std::to_string(currentBufferSize);
			benchmarkFileName.append("buffersize");
			benchmarkFileName.append(strBufferSize);
			benchmarkFileName.append(".csv");
			clBenchmarker_ = Benchmarker(benchmarkFileName, { "Test_Name", "Total_Time", "Average_Time", "Max_Time", "Min_Time", "Max_Difference", "Average_Difference" });
			setBufferLength(currentBufferSize);

			//Run tests with setup//
			cl_nullkernel(aNumRepetitions, isWarmup);
			cl_cputogpu_standard(aNumRepetitions, isWarmup);
			cl_cputogpu_pinned(aNumRepetitions, isWarmup);
			cl_gputocpu_standard(aNumRepetitions, isWarmup);
			cl_gputocpu_pinned(aNumRepetitions, isWarmup);
			cl_cputogputocpu_standard(aNumRepetitions, isWarmup);
			cl_cputogputocpu_pinned(aNumRepetitions, isWarmup);
			cl_devicetransfer_standard(aNumRepetitions, isWarmup);
			cl_devicetransfer_pinned(aNumRepetitions, isWarmup);
			cl_devicetransferkernel_standard(aNumRepetitions, isWarmup);
			cl_devicetransferkernel_pinned(aNumRepetitions, isWarmup);
			cl_simplebufferprocessing_standard(aNumRepetitions, isWarmup);
			cl_simplebufferprocessing_pinned(aNumRepetitions, isWarmup);
			cl_complexbufferprocessing_standard(aNumRepetitions, isWarmup);
			cl_complexbufferprocessing_pinned(aNumRepetitions, isWarmup);
			cl_simplebuffersynthesis_standard(aNumRepetitions, isWarmup);
			cl_simplebuffersynthesis_pinned(aNumRepetitions, isWarmup);
			cl_complexbuffersynthesis_standard(aNumRepetitions, isWarmup);
			cl_complexbuffersynthesis_pinned(aNumRepetitions, isWarmup);
			cl_interruptedbufferprocessing_standard(aNumRepetitions, isWarmup);
			cl_interruptedbufferprocessing_pinned(aNumRepetitions, isWarmup);
		}
	}
	void runRealTimeBenchmarks(uint64_t aFrameRate, bool isWarmup) override
	{
		cl_unidirectional_baseline(aFrameRate, isWarmup);
		cl_unidirectional_processing(aFrameRate, isWarmup);
		cl_bidirectional_baseline(aFrameRate, isWarmup);
		cl_bidirectional_processing(aFrameRate, isWarmup);
	}

	void cl_nullkernel(size_t aN, bool isWarmup)
	{
		//Test Preperation//
		cl::Kernel nullKernel;
		openCL.createKernel(context_, kernelProgram_, nullKernel, "cl_000_nullKernel");

		//Execute & Profile//
		std::cout << "Executing test: cl_nullkernel" << std::endl;
		setLocalWorkspace(bufferLength_);
		if (isWarmup)
		{
			openCL.enqueueKernel(commandQueue_, nullKernel, globalWorkspace_, localWorkspace_, clEvent);
			openCL.waitCommandQueue(commandQueue_);
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			clBenchmarker_.startTimer("cl_nullkernel");
			openCL.enqueueKernel(commandQueue_, nullKernel, globalWorkspace_, localWorkspace_);
			//openCL.enqueueKernel(commandQueue_, nullKernel, globalWorkspace_, localWorkspace_, clEvent);
			//openCL.waitEvent(clEvent);
			openCL.waitCommandQueue(commandQueue_);
			clBenchmarker_.pauseTimer("cl_nullkernel");

			//openCL.profileEvent(clEvent);
		}
		clBenchmarker_.elapsedTimer("cl_nullkernel");

		//Check results//
		bool isSuccessful = true;
		std::cout << "cl_nullkernel successful:  " << isSuccessful << std::endl << std::endl;
	}
	void cl_cputogpu_standard(size_t aN, bool isWarmup)
	{
		//Test preperation//
		datatype* checkBuffer = new datatype[bufferLength_];
		datatype* hostBuffer = new datatype[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		cl::Buffer deviceBuffer;
		openCL.createBuffer(context_, deviceBuffer, CL_MEM_READ_WRITE, bufferSize_);

		cl::Kernel testKernel;
		openCL.createKernel(context_, kernelProgram_, testKernel, "cl_testkernel");
		openCL.setKernelArgument(testKernel, deviceBuffer, 0, sizeof(cl_mem));

		//Execute & Profile//
		std::cout << "Executing test: cl_cputogpu_standard" << std::endl;
		if (isWarmup)
		{
			openCL.writeBuffer(commandQueue_,deviceBuffer, bufferSize_, hostBuffer);
			openCL.waitCommandQueue(commandQueue_);
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			clBenchmarker_.startTimer("cl_cputogpu_standard");
			openCL.writeBuffer(commandQueue_, deviceBuffer, bufferSize_, hostBuffer);
			openCL.waitCommandQueue(commandQueue_);
			clBenchmarker_.pauseTimer("cl_cputogpu_standard");

			//Run kernel outside timer for results//
			openCL.enqueueKernel(commandQueue_, testKernel, globalWorkspace_, localWorkspace_);
			openCL.waitCommandQueue(commandQueue_);
		}
		clBenchmarker_.elapsedTimer("cl_cputogpu_standard");

		//Check results//
		bool isSuccessful = true;
		openCL.readBuffer(commandQueue_, deviceBuffer, bufferSize_, checkBuffer);
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			float attenuatedSample = hostBuffer[i] * 0.5;
			if (attenuatedSample != checkBuffer[i])
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cl_cputogpu_standard successful: " << isSuccessful << std::endl << std::endl;

		//Cleanup//

		delete hostBuffer;
		delete checkBuffer;
	}
	void cl_cputogpu_pinned(size_t aN, bool isWarmup)
	{
		//Test preperation//
		datatype* checkBuffer = new datatype[bufferLength_];
		datatype* hostBuffer = new datatype[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		cl::Buffer deviceBuffer;
		openCL.createBuffer(context_, deviceBuffer, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bufferSize_);

		void* pinned = openCL.mapMemory(commandQueue_, deviceBuffer, CL_MAP_WRITE, bufferSize_);
		openCL.unmapMemory(commandQueue_, deviceBuffer, pinned, bufferSize_);

		cl::Kernel testKernel;
		openCL.createKernel(context_, kernelProgram_, testKernel, "cl_testkernel");
		openCL.setKernelArgument(testKernel, deviceBuffer, 0, sizeof(cl_mem));

		//Execute and average//
		std::cout << "Executing test: cl_cputogpu_pinned" << std::endl;
		if (isWarmup)
		{
			pinned = openCL.mapMemory(commandQueue_, deviceBuffer, CL_MAP_WRITE, bufferSize_);
			std::memcpy(pinned, hostBuffer, bufferSize_);
			openCL.unmapMemory(commandQueue_, deviceBuffer, pinned, bufferSize_);
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			clBenchmarker_.startTimer("cl_cputogpu_mappedbuffer");
			pinned = openCL.mapMemory(commandQueue_, deviceBuffer, CL_MAP_WRITE, bufferSize_);
			std::memcpy(pinned, hostBuffer, bufferSize_);
			openCL.unmapMemory(commandQueue_, deviceBuffer, pinned, bufferSize_);
			clBenchmarker_.pauseTimer("cl_cputogpu_mappedbuffer");

			//Run kernel outside timer for results//
			openCL.enqueueKernel(commandQueue_, testKernel, globalWorkspace_, localWorkspace_);
			openCL.waitCommandQueue(commandQueue_);
		}
		clBenchmarker_.elapsedTimer("cl_cputogpu_mappedbuffer");

		//Check results//
		bool isSuccessful = true;
		openCL.readBuffer(commandQueue_, deviceBuffer, bufferSize_, checkBuffer);
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			float attenuatedSample = hostBuffer[i] * 0.5;
			if (attenuatedSample != checkBuffer[i])
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cl_cputogpu_mappedbuffer successful: " << isSuccessful << std::endl << std::endl;

		//Cleanup//
		//openCL.deleteBuffer(deviceBuffer);

		delete hostBuffer;
		delete checkBuffer;
	}
	void cl_gputocpu_standard(size_t aN, bool isWarmup)
	{
		//Test preperation//
		datatype* checkBuffer = new datatype[bufferLength_];
		datatype* hostBuffer = new datatype[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		cl::Buffer deviceBuffer;
		openCL.createBuffer(context_, deviceBuffer, CL_MEM_READ_WRITE, bufferSize_);		//@ToDo - Can we do CL_MEM_READ_ONLY for possible optimization?

		openCL.writeBuffer(commandQueue_, deviceBuffer, bufferSize_, hostBuffer);

		cl::Kernel testKernel;
		openCL.createKernel(context_, kernelProgram_, testKernel, "cl_testkernel");
		openCL.setKernelArgument(testKernel, deviceBuffer, 0, sizeof(cl_mem));

		//Execute & Profile//
		std::cout << "Executing test: cl_gputocpu_standard" << std::endl;
		if (isWarmup)
		{
			openCL.readBuffer(commandQueue_, deviceBuffer, bufferSize_, checkBuffer);
			openCL.waitCommandQueue(commandQueue_);
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			openCL.writeBuffer(commandQueue_, deviceBuffer, bufferSize_, hostBuffer);

			clBenchmarker_.startTimer("cl_gputocpu_standard");
			openCL.readBuffer(commandQueue_, deviceBuffer, bufferSize_, checkBuffer);
			openCL.waitCommandQueue(commandQueue_);
			clBenchmarker_.pauseTimer("cl_gputocpu_standard");

			//Run kernel outside timer for results//
			openCL.enqueueKernel(commandQueue_, testKernel, globalWorkspace_, localWorkspace_);
			openCL.waitCommandQueue(commandQueue_);
		}
		clBenchmarker_.elapsedTimer("cl_gputocpu_standard");

		//Check results//
		bool isSuccessful = true;
		openCL.readBuffer(commandQueue_, deviceBuffer, bufferSize_, checkBuffer);
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			float attenuatedSample = hostBuffer[i] * 0.5;
			if (attenuatedSample != checkBuffer[i])
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cl_gputocpu_standard successful: " << isSuccessful << std::endl << std::endl;

		//Cleanup//
		//openCL.deleteBuffer(deviceBuffer);

		delete hostBuffer;
		delete checkBuffer;
	}
	void cl_gputocpu_pinned(size_t aN, bool isWarmup)
	{
		//Test preperation//
		datatype* checkBuffer = new datatype[bufferLength_];
		datatype* hostBuffer = new datatype[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		cl::Buffer deviceBuffer;
		openCL.createBuffer(context_, deviceBuffer, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bufferSize_);

		void* pinned = openCL.mapMemory(commandQueue_, deviceBuffer, CL_MAP_WRITE, bufferSize_);
		std::memcpy(pinned, hostBuffer, bufferSize_);
		openCL.unmapMemory(commandQueue_, deviceBuffer, pinned, bufferSize_);

		cl::Kernel testKernel;
		openCL.createKernel(context_, kernelProgram_, testKernel, "cl_testkernel");
		openCL.setKernelArgument(testKernel, deviceBuffer, 0, sizeof(cl_mem));

		//Execute and average//
		std::cout << "Executing test: cl_gputocpu_pinned" << std::endl;
		if (isWarmup)
		{
			pinned = openCL.mapMemory(commandQueue_, deviceBuffer, CL_MAP_READ, bufferSize_);
			std::memcpy(checkBuffer, pinned, bufferSize_);
			openCL.unmapMemory(commandQueue_, deviceBuffer, pinned, bufferSize_);
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			openCL.writeBuffer(commandQueue_, deviceBuffer, bufferSize_, hostBuffer);

			clBenchmarker_.startTimer("cl_gputocpu_pinned");
			pinned = openCL.mapMemory(commandQueue_, deviceBuffer, CL_MAP_READ, bufferSize_);
			std::memcpy(checkBuffer, pinned, bufferSize_);
			openCL.unmapMemory(commandQueue_, deviceBuffer, pinned, bufferSize_);
			clBenchmarker_.pauseTimer("cl_gputocpu_pinned");

			//Run kernel outside timer for results//
			openCL.enqueueKernel(commandQueue_, testKernel, globalWorkspace_, localWorkspace_);
			openCL.waitCommandQueue(commandQueue_);
		}
		clBenchmarker_.elapsedTimer("cl_gputocpu_pinned");

		//Check results//
		bool isSuccessful = true;
		openCL.readBuffer(commandQueue_, deviceBuffer, bufferSize_, checkBuffer);
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			float attenuatedSample = hostBuffer[i] * 0.5;
			if (attenuatedSample != checkBuffer[i])
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cl_gputocpu_pinned successful: " << isSuccessful << std::endl << std::endl;

		//Cleanup//
		//openCL.deleteBuffer(deviceBuffer);

		delete hostBuffer;
		delete checkBuffer;
	}
	void cl_cputogputocpu_standard(size_t aN, bool isWarmup)
	{
		//Test preperation//
		datatype* checkBuffer = new datatype[bufferLength_];
		datatype* hostBuffer = new datatype[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		cl::Buffer deviceBuffer;
		openCL.createBuffer(context_, deviceBuffer, CL_MEM_WRITE_ONLY, bufferSize_);

		cl::Kernel testKernel;
		openCL.createKernel(context_, kernelProgram_, testKernel, "cl_testkernel");
		openCL.setKernelArgument(testKernel, deviceBuffer, 0, sizeof(cl_mem));

		//Execute & Profile//
		std::cout << "Executing test: cl_cputogputocpu_standard" << std::endl;
		if (isWarmup)
		{
			openCL.writeBuffer(commandQueue_, deviceBuffer, bufferSize_, hostBuffer);
			openCL.waitCommandQueue(commandQueue_);
			openCL.readBuffer(commandQueue_, deviceBuffer, bufferSize_, checkBuffer);
			openCL.waitCommandQueue(commandQueue_);
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			clBenchmarker_.startTimer("cl_cputogputocpu_standard");
			openCL.writeBuffer(commandQueue_, deviceBuffer, bufferSize_, hostBuffer);
			openCL.waitCommandQueue(commandQueue_);
			clBenchmarker_.waitTimer("cl_cputogputocpu_standard");

			//Run kernel outside timer for results//
			openCL.enqueueKernel(commandQueue_, testKernel, globalWorkspace_, localWorkspace_);
			openCL.waitCommandQueue(commandQueue_);

			clBenchmarker_.resumeTimer("cl_cputogputocpu_standard");
			openCL.readBuffer(commandQueue_, deviceBuffer, bufferSize_, checkBuffer);
			openCL.waitCommandQueue(commandQueue_);
			clBenchmarker_.pauseTimer("cl_cputogputocpu_standard");
		}
		clBenchmarker_.elapsedTimer("cl_cputogputocpu_standard");

		//Check results//
		bool isSuccessful = true;
		openCL.readBuffer(commandQueue_, deviceBuffer, bufferSize_, checkBuffer);
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			float attenuatedSample = hostBuffer[i] * 0.5;
			if (attenuatedSample != checkBuffer[i])
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cl_cputogputocpu_standard successful: " << isSuccessful << std::endl << std::endl;

		//Cleanup//
		//openCL.deleteBuffer(deviceBuffer);

		delete hostBuffer;
		delete checkBuffer;
	}
	void cl_cputogputocpu_pinned(size_t aN, bool isWarmup)
	{
		//Test preperation//
		datatype* checkBuffer = new datatype[bufferLength_];
		datatype* hostBuffer = new datatype[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		cl::Buffer deviceBuffer;
		openCL.createBuffer(context_, deviceBuffer, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bufferSize_);

		void* pinned = openCL.mapMemory(commandQueue_, deviceBuffer, CL_MAP_WRITE, bufferSize_);
		openCL.unmapMemory(commandQueue_, deviceBuffer, pinned, bufferSize_);

		cl::Kernel testKernel;
		openCL.createKernel(context_, kernelProgram_, testKernel, "cl_testkernel");
		openCL.setKernelArgument(testKernel, deviceBuffer, 0, sizeof(cl_mem));

		//Execute and average//
		std::cout << "Executing test: cl_cputogputocpu_pinned" << std::endl;
		if (isWarmup)
		{
			pinned = openCL.mapMemory(commandQueue_, deviceBuffer, CL_MAP_WRITE, bufferSize_);
			std::memcpy(pinned, hostBuffer, bufferSize_);
			openCL.unmapMemory(commandQueue_, deviceBuffer, pinned, bufferSize_);
			pinned = openCL.mapMemory(commandQueue_, deviceBuffer, CL_MAP_READ, bufferSize_);
			std::memcpy(checkBuffer, pinned, bufferSize_);
			openCL.unmapMemory(commandQueue_, deviceBuffer, pinned, bufferSize_);
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			clBenchmarker_.startTimer("cl_cputogputocpu_pinned");
			pinned = openCL.mapMemory(commandQueue_, deviceBuffer, CL_MAP_WRITE, bufferSize_);
			std::memcpy(pinned, hostBuffer, bufferSize_);
			openCL.unmapMemory(commandQueue_, deviceBuffer, pinned, bufferSize_);
			clBenchmarker_.waitTimer("cl_cputogputocpu_pinned");

			//Run kernel outside timer for results//
			openCL.enqueueKernel(commandQueue_, testKernel, globalWorkspace_, localWorkspace_);
			openCL.waitCommandQueue(commandQueue_);

			clBenchmarker_.resumeTimer("cl_cputogputocpu_pinned");
			pinned = openCL.mapMemory(commandQueue_, deviceBuffer, CL_MAP_READ, bufferSize_);
			std::memcpy(checkBuffer, pinned, bufferSize_);
			openCL.unmapMemory(commandQueue_, deviceBuffer, pinned, bufferSize_);
			clBenchmarker_.pauseTimer("cl_cputogputocpu_pinned");
		}
		clBenchmarker_.elapsedTimer("cl_cputogputocpu_pinned");

		//Check results//
		bool isSuccessful = true;
		openCL.readBuffer(commandQueue_, deviceBuffer, bufferSize_, checkBuffer);
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			float attenuatedSample = hostBuffer[i] * 0.5;
			if (attenuatedSample != checkBuffer[i])
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cl_cputogputocpu_pinned successful: " << isSuccessful << std::endl << std::endl;

		//Cleanup//
		//openCL.deleteBuffer(deviceBuffer);

		delete hostBuffer;
		delete checkBuffer;
	}
	void cl_devicetransfer_standard(size_t aN, bool isWarmup)
	{
		//Test preperation//
		datatype* checkBuffer = new datatype[bufferLength_];
		datatype* hostBuffer = new datatype[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		cl::Buffer deviceBufferSrc;
		cl::Buffer deviceBufferDst;
		openCL.createBuffer(context_, deviceBufferSrc, CL_MEM_READ_WRITE, bufferSize_);
		openCL.createBuffer(context_, deviceBufferDst, CL_MEM_READ_WRITE, bufferSize_);

		openCL.writeBuffer(commandQueue_, deviceBufferSrc, bufferSize_, hostBuffer);

		//Execute and average//
		std::cout << "Executing test: cl_devicetransfer_standard" << std::endl;
		if (isWarmup)
		{
			openCL.enqueueCopyBuffer(commandQueue_, deviceBufferSrc, deviceBufferDst, bufferSize_);
			openCL.waitCommandQueue(commandQueue_);
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			clBenchmarker_.startTimer("cl_devicetransfer_standard");
			openCL.enqueueCopyBuffer(commandQueue_, deviceBufferSrc, deviceBufferDst, bufferSize_);
			openCL.waitCommandQueue(commandQueue_);
			clBenchmarker_.pauseTimer("cl_devicetransfer_standard");
		}
		clBenchmarker_.elapsedTimer("cl_devicetransfer_standard");

		//Check contents//
		bool isSuccessful = true;
		openCL.readBuffer(commandQueue_, deviceBufferSrc, bufferSize_, checkBuffer);
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			if (checkBuffer[i] != hostBuffer[i])
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cl_devicetransfer_standard successful: " << isSuccessful << std::endl << std::endl;

		//Cleanup//
		//openCL.deleteBuffer(deviceBufferSrc);
		//openCL.deleteBuffer(deviceBufferDst);

		delete hostBuffer;
		delete checkBuffer;
	}
	void cl_devicetransfer_pinned(size_t aN, bool isWarmup)
	{
		//Test preperation//
		datatype* checkBuffer = new datatype[bufferLength_];
		datatype* hostBuffer = new datatype[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		cl::Buffer deviceBufferSrc;
		cl::Buffer deviceBufferDst;
		openCL.createBuffer(context_, deviceBufferSrc, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bufferSize_);
		openCL.createBuffer(context_, deviceBufferDst, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bufferSize_);

		openCL.writeBuffer(commandQueue_, deviceBufferSrc, bufferSize_, hostBuffer);

		//Execute and average//
		std::cout << "Executing test: cl_devicetransfer_pinned" << std::endl;
		if (isWarmup)
		{
			openCL.enqueueCopyBuffer(commandQueue_, deviceBufferSrc, deviceBufferDst, bufferSize_);
			openCL.waitCommandQueue(commandQueue_);
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			clBenchmarker_.startTimer("cl_devicetransfer_pinned");
			openCL.enqueueCopyBuffer(commandQueue_, deviceBufferSrc, deviceBufferDst, bufferSize_);
			openCL.waitCommandQueue(commandQueue_);
			clBenchmarker_.pauseTimer("cl_devicetransfer_pinned");
		}
		clBenchmarker_.elapsedTimer("cl_devicetransfer_pinned");

		//Check contents//
		bool isSuccessful = true;
		openCL.readBuffer(commandQueue_, deviceBufferSrc, bufferSize_, checkBuffer);
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			if (checkBuffer[i] != hostBuffer[i])
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cl_devicetransfer_pinned successful: " << isSuccessful << std::endl << std::endl;

		//Cleanup//
		//openCL.deleteBuffer(deviceBufferSrc);
		//openCL.deleteBuffer(deviceBufferDst);

		delete hostBuffer;
		delete checkBuffer;
	}
	void cl_devicetransferkernel_standard(size_t aN, bool isWarmup)
	{
		//Test preperation//
		datatype* checkBuffer = new datatype[bufferLength_];
		datatype* hostBuffer = new datatype[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		cl::Buffer deviceBufferSrc;
		cl::Buffer deviceBufferDst;
		openCL.createBuffer(context_, deviceBufferSrc, CL_MEM_READ_WRITE, bufferSize_);
		openCL.createBuffer(context_, deviceBufferDst, CL_MEM_READ_WRITE, bufferSize_);

		openCL.writeBuffer(commandQueue_, deviceBufferSrc, bufferSize_, hostBuffer);

		cl::Kernel transferKernel;
		openCL.createKernel(context_, kernelProgram_, transferKernel, "cl_006_cpymemorykernel");
		openCL.setKernelArgument(transferKernel, deviceBufferSrc, 0, sizeof(cl_mem));
		openCL.setKernelArgument(transferKernel, deviceBufferDst, 1, sizeof(cl_mem));

		//Execute and average//
		std::cout << "Executing test: cl_devicetransferkernel_standard" << std::endl;
		if (isWarmup)
		{
			openCL.enqueueKernel(commandQueue_, transferKernel, globalWorkspace_, localWorkspace_);
			openCL.waitCommandQueue(commandQueue_);
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			clBenchmarker_.startTimer("cl_devicetransferkernel_standard");
			openCL.enqueueKernel(commandQueue_, transferKernel, globalWorkspace_, localWorkspace_);
			openCL.waitCommandQueue(commandQueue_);
			clBenchmarker_.pauseTimer("cl_devicetransferkernel_standard");
		}
		clBenchmarker_.elapsedTimer("cl_devicetransferkernel_standard");

		//Check contents//
		bool isSuccessful = true;
		openCL.readBuffer(commandQueue_, deviceBufferSrc, bufferSize_, checkBuffer);
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			if (checkBuffer[i] != hostBuffer[i])
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cl_devicetransferkernel_standard successful: " << isSuccessful << std::endl << std::endl;

		//Cleanup//
		//openCL.deleteBuffer(deviceBufferSrc);
		//openCL.deleteBuffer(deviceBufferDst);

		delete hostBuffer;
		delete checkBuffer;
	}
	void cl_devicetransferkernel_pinned(size_t aN, bool isWarmup)
	{
		//Test preperation//
		datatype* checkBuffer = new datatype[bufferLength_];
		datatype* hostBuffer = new datatype[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		cl::Buffer deviceBufferSrc;
		cl::Buffer deviceBufferDst;
		openCL.createBuffer(context_, deviceBufferSrc, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bufferSize_);
		openCL.createBuffer(context_, deviceBufferDst, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bufferSize_);

		openCL.writeBuffer(commandQueue_, deviceBufferSrc, bufferSize_, hostBuffer);

		cl::Kernel transferKernel;
		openCL.createKernel(context_, kernelProgram_, transferKernel, "cl_006_cpymemorykernel");
		openCL.setKernelArgument(transferKernel, deviceBufferSrc, 0, sizeof(cl_mem));
		openCL.setKernelArgument(transferKernel, deviceBufferDst, 1, sizeof(cl_mem));

		//Execute and average//
		std::cout << "Executing test: cl_devicetransferkernel_pinned" << std::endl;
		if (isWarmup)
		{
			openCL.enqueueKernel(commandQueue_, transferKernel, globalWorkspace_, localWorkspace_);
			openCL.waitCommandQueue(commandQueue_);
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			clBenchmarker_.startTimer("cl_devicetransferkernel_pinned");
			openCL.enqueueKernel(commandQueue_, transferKernel, globalWorkspace_, localWorkspace_);
			openCL.waitCommandQueue(commandQueue_);
			clBenchmarker_.pauseTimer("cl_devicetransferkernel_pinned");
		}
		clBenchmarker_.elapsedTimer("cl_devicetransferkernel_pinned");

		//Check contents//
		bool isSuccessful = true;
		openCL.readBuffer(commandQueue_, deviceBufferSrc, bufferSize_, checkBuffer);
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			if (checkBuffer[i] != hostBuffer[i])
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cl_devicetransferkernel_pinned successful: " << isSuccessful << std::endl << std::endl;

		//Cleanup//
		//openCL.deleteBuffer(deviceBufferSrc);
		//openCL.deleteBuffer(deviceBufferDst);

		delete hostBuffer;
		delete checkBuffer;
	}
	void cl_simplebufferprocessing_standard(size_t aN, bool isWarmup)
	{
		//Test preperation//
		datatype* checkBuffer = new datatype[bufferLength_];
		datatype* hostBuffer = new datatype[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		cl::Buffer deviceBufferSrc;
		cl::Buffer deviceBufferDst;
		openCL.createBuffer(context_, deviceBufferSrc, CL_MEM_READ_WRITE, bufferSize_);
		openCL.createBuffer(context_, deviceBufferDst, CL_MEM_READ_WRITE, bufferSize_);

		openCL.writeBuffer(commandQueue_, deviceBufferSrc, bufferSize_, hostBuffer);

		cl::Kernel simpleBufferProcessingKernel;
		openCL.createKernel(context_, kernelProgram_, simpleBufferProcessingKernel, "cl_008_simplebufferprocessing");
		openCL.setKernelArgument(simpleBufferProcessingKernel, deviceBufferSrc, 0, sizeof(cl_mem));
		openCL.setKernelArgument(simpleBufferProcessingKernel, deviceBufferDst, 1, sizeof(cl_mem));

		//Execute and average//
		std::cout << "Executing test: cl_simplebufferprocessing_standard" << std::endl;
		if (isWarmup)
		{
			openCL.writeBuffer(commandQueue_, deviceBufferSrc, bufferSize_, hostBuffer);
			openCL.waitCommandQueue(commandQueue_);
			openCL.enqueueKernel(commandQueue_, simpleBufferProcessingKernel, globalWorkspace_, localWorkspace_);
			openCL.waitCommandQueue(commandQueue_);
			openCL.readBuffer(commandQueue_, deviceBufferDst, bufferSize_, checkBuffer);
			openCL.waitCommandQueue(commandQueue_);
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			clBenchmarker_.startTimer("cl_simplebufferprocessing_standard");
			openCL.writeBuffer(commandQueue_, deviceBufferSrc, bufferSize_, hostBuffer);
			openCL.waitCommandQueue(commandQueue_);
			openCL.enqueueKernel(commandQueue_, simpleBufferProcessingKernel, globalWorkspace_, localWorkspace_);
			openCL.waitCommandQueue(commandQueue_);
			openCL.readBuffer(commandQueue_, deviceBufferDst, bufferSize_, checkBuffer);
			openCL.waitCommandQueue(commandQueue_);
			clBenchmarker_.pauseTimer("cl_simplebufferprocessing_standard");
		}
		clBenchmarker_.elapsedTimer("cl_simplebufferprocessing_standard");

		//Check contents//
		bool isSuccessful = true;
		openCL.readBuffer(commandQueue_, deviceBufferDst, bufferSize_, checkBuffer);
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			float attenuatedSample = hostBuffer[i] * 0.5;
			if (attenuatedSample != checkBuffer[i])
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cl_simplebufferprocessing_standard successful: " << isSuccessful << std::endl << std::endl;

		//Cleanup//
		//openCL.deleteBuffer(deviceBufferSrc);
		//openCL.deleteBuffer(deviceBufferDst);

		delete hostBuffer;
		delete checkBuffer;
	}
	void cl_simplebufferprocessing_pinned(size_t aN, bool isWarmup)
	{
		//Test preperation//
		datatype* checkBuffer = new datatype[bufferLength_];
		datatype* hostBuffer = new datatype[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		cl::Buffer deviceBufferSrc;
		cl::Buffer deviceBufferDst;
		openCL.createBuffer(context_, deviceBufferSrc, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bufferSize_);
		openCL.createBuffer(context_, deviceBufferDst, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bufferSize_);

		openCL.writeBuffer(commandQueue_, deviceBufferSrc, bufferSize_, hostBuffer);

		cl::Kernel simpleBufferProcessingKernel;
		openCL.createKernel(context_, kernelProgram_, simpleBufferProcessingKernel, "cl_008_simplebufferprocessing");
		openCL.setKernelArgument(simpleBufferProcessingKernel, deviceBufferSrc, 0, sizeof(cl_mem));
		openCL.setKernelArgument(simpleBufferProcessingKernel, deviceBufferDst, 1, sizeof(cl_mem));

		void* pinned = openCL.mapMemory(commandQueue_, deviceBufferSrc, CL_MAP_WRITE, bufferSize_);
		openCL.unmapMemory(commandQueue_, deviceBufferSrc, pinned, bufferSize_);

		//Execute and average//
		std::cout << "Executing test: cl_simplebufferprocessing_pinned" << std::endl;
		if (isWarmup)
		{
			pinned = openCL.mapMemory(commandQueue_, deviceBufferSrc, CL_MAP_WRITE, bufferSize_);
			std::memcpy(pinned, hostBuffer, bufferSize_);
			openCL.unmapMemory(commandQueue_, deviceBufferSrc, pinned, bufferSize_);

			openCL.enqueueKernel(commandQueue_, simpleBufferProcessingKernel, globalWorkspace_, localWorkspace_);
			openCL.waitCommandQueue(commandQueue_);

			pinned = openCL.mapMemory(commandQueue_, deviceBufferDst, CL_MAP_READ, bufferSize_);
			std::memcpy(checkBuffer, pinned, bufferSize_);
			openCL.unmapMemory(commandQueue_, deviceBufferDst, pinned, bufferSize_);
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			clBenchmarker_.startTimer("cl_simplebufferprocessing_pinned");
			pinned = openCL.mapMemory(commandQueue_, deviceBufferSrc, CL_MAP_WRITE, bufferSize_);
			std::memcpy(pinned, hostBuffer, bufferSize_);
			openCL.unmapMemory(commandQueue_, deviceBufferSrc, pinned, bufferSize_);

			openCL.enqueueKernel(commandQueue_, simpleBufferProcessingKernel, globalWorkspace_, localWorkspace_);
			openCL.waitCommandQueue(commandQueue_);

			pinned = openCL.mapMemory(commandQueue_, deviceBufferDst, CL_MAP_READ, bufferSize_);
			std::memcpy(checkBuffer, pinned, bufferSize_);
			openCL.unmapMemory(commandQueue_, deviceBufferDst, pinned, bufferSize_);
			clBenchmarker_.pauseTimer("cl_simplebufferprocessing_pinned");
		}
		clBenchmarker_.elapsedTimer("cl_simplebufferprocessing_pinned");

		//Check contents//
		bool isSuccessful = true;
		openCL.readBuffer(commandQueue_, deviceBufferDst, bufferSize_, checkBuffer);
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			float attenuatedSample = hostBuffer[i] * 0.5;
			if (attenuatedSample != checkBuffer[i])
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cl_simplebufferprocessing_pinned successful: " << isSuccessful << std::endl << std::endl;

		//Cleanup//
		//openCL.deleteBuffer(deviceBufferSrc);
		//openCL.deleteBuffer(deviceBufferDst);

		delete hostBuffer;
		delete checkBuffer;
	}
	void cl_complexbufferprocessing_standard(size_t aN, bool isWarmup)
	{
		//Test preperation//
		datatype* checkBuffer = new datatype[bufferLength_];
		datatype* hostBuffer = new datatype[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		cl::Buffer deviceBufferSrc;
		cl::Buffer deviceBufferDst;
		openCL.createBuffer(context_, deviceBufferSrc, CL_MEM_READ_WRITE, bufferSize_);
		openCL.createBuffer(context_, deviceBufferDst, CL_MEM_READ_WRITE, bufferSize_);

		openCL.writeBuffer(commandQueue_, deviceBufferSrc, bufferSize_, hostBuffer);

		cl::Kernel complexBufferProcessingKernel;
		openCL.createKernel(context_, kernelProgram_, complexBufferProcessingKernel, "cl_009_complexbufferprocessing");
		openCL.setKernelArgument(complexBufferProcessingKernel, deviceBufferSrc, 0, sizeof(cl_mem));
		openCL.setKernelArgument(complexBufferProcessingKernel, deviceBufferDst, 1, sizeof(cl_mem));

		//Execute and average//
		std::cout << "Executing test: cl_complexbufferprocessing_standard" << std::endl;
		if (isWarmup)
		{
			openCL.writeBuffer(commandQueue_, deviceBufferSrc, bufferSize_, hostBuffer);
			openCL.waitCommandQueue(commandQueue_);
			openCL.enqueueKernel(commandQueue_, complexBufferProcessingKernel, globalWorkspace_, localWorkspace_);
			openCL.waitCommandQueue(commandQueue_);
			openCL.readBuffer(commandQueue_, deviceBufferSrc, bufferSize_, checkBuffer);
			openCL.waitCommandQueue(commandQueue_);
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			clBenchmarker_.startTimer("cl_complexbufferprocessing_standard");
			openCL.writeBuffer(commandQueue_, deviceBufferSrc, bufferSize_, hostBuffer);
			openCL.waitCommandQueue(commandQueue_);
			openCL.enqueueKernel(commandQueue_, complexBufferProcessingKernel, globalWorkspace_, localWorkspace_);
			openCL.waitCommandQueue(commandQueue_);
			openCL.readBuffer(commandQueue_, deviceBufferSrc, bufferSize_, checkBuffer);
			openCL.waitCommandQueue(commandQueue_);
			clBenchmarker_.pauseTimer("cl_complexbufferprocessing_standard");
		}
		clBenchmarker_.elapsedTimer("cl_complexbufferprocessing_standard");

		//Check contents//
		bool isSuccessful = true;
		openCL.readBuffer(commandQueue_, deviceBufferSrc, bufferSize_, checkBuffer);
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
		std::cout << "cl_complexbufferprocessing_standard successful: " << isSuccessful << std::endl << std::endl;

		//Cleanup//
		//openCL.deleteBuffer(deviceBufferSrc);
		//openCL.deleteBuffer(deviceBufferDst);

		delete hostBuffer;
		delete checkBuffer;
	}
	void cl_complexbufferprocessing_pinned(size_t aN, bool isWarmup)
	{
		//Test preperation//
		datatype* checkBuffer = new datatype[bufferLength_];
		datatype* hostBuffer = new datatype[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		cl::Buffer deviceBufferSrc;
		cl::Buffer deviceBufferDst;
		openCL.createBuffer(context_, deviceBufferSrc, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bufferSize_);
		openCL.createBuffer(context_, deviceBufferDst, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bufferSize_);

		openCL.writeBuffer(commandQueue_, deviceBufferSrc, bufferSize_, hostBuffer);

		cl::Kernel complexBufferProcessingKernel;
		openCL.createKernel(context_, kernelProgram_, complexBufferProcessingKernel, "cl_009_complexbufferprocessing");
		openCL.setKernelArgument(complexBufferProcessingKernel, deviceBufferSrc, 0, sizeof(cl_mem));
		openCL.setKernelArgument(complexBufferProcessingKernel, deviceBufferDst, 1, sizeof(cl_mem));

		void* pinned = openCL.mapMemory(commandQueue_, deviceBufferSrc, CL_MAP_WRITE, bufferSize_);
		openCL.unmapMemory(commandQueue_, deviceBufferSrc, pinned, bufferSize_);

		//Execute and average//
		std::cout << "Executing test: cl_complexbufferprocessing_pinned" << std::endl;
		if (isWarmup)
		{
			pinned = openCL.mapMemory(commandQueue_, deviceBufferSrc, CL_MAP_WRITE, bufferSize_);
			std::memcpy(pinned, hostBuffer, bufferSize_);
			openCL.unmapMemory(commandQueue_, deviceBufferSrc, pinned, bufferSize_);

			openCL.enqueueKernel(commandQueue_, complexBufferProcessingKernel, globalWorkspace_, localWorkspace_);
			openCL.waitCommandQueue(commandQueue_);

			pinned = openCL.mapMemory(commandQueue_, deviceBufferDst, CL_MAP_READ, bufferSize_);
			std::memcpy(checkBuffer, pinned, bufferSize_);
			openCL.unmapMemory(commandQueue_, deviceBufferDst, pinned, bufferSize_);
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			clBenchmarker_.startTimer("cl_complexbufferprocessing_pinned");
			pinned = openCL.mapMemory(commandQueue_, deviceBufferSrc, CL_MAP_WRITE, bufferSize_);
			std::memcpy(pinned, hostBuffer, bufferSize_);
			openCL.unmapMemory(commandQueue_, deviceBufferSrc, pinned, bufferSize_);

			openCL.enqueueKernel(commandQueue_, complexBufferProcessingKernel, globalWorkspace_, localWorkspace_);
			openCL.waitCommandQueue(commandQueue_);

			pinned = openCL.mapMemory(commandQueue_, deviceBufferDst, CL_MAP_READ, bufferSize_);
			std::memcpy(checkBuffer, pinned, bufferSize_);
			openCL.unmapMemory(commandQueue_, deviceBufferDst, pinned, bufferSize_);
			clBenchmarker_.pauseTimer("cl_complexbufferprocessing_pinned");
		}
		clBenchmarker_.elapsedTimer("cl_complexbufferprocessing_pinned");

		//Check contents//
		bool isSuccessful = true;
		openCL.readBuffer(commandQueue_, deviceBufferSrc, bufferSize_, checkBuffer);
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
		std::cout << "cl_complexbufferprocessing_pinned successful: " << isSuccessful << std::endl << std::endl;

		//Cleanup//
		//openCL.deleteBuffer(deviceBufferSrc);
		//openCL.deleteBuffer(deviceBufferDst);

		delete hostBuffer;
		delete checkBuffer;
	}
	void cl_simplebuffersynthesis_standard(size_t aN, bool isWarmup)
	{
		//Test preperation//
		const int sampleRate = 44100;
		const float frequency = 1400.0;
		datatype* checkBuffer = new datatype[bufferLength_];
		datatype* hostBuffer = new datatype[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		cl::Buffer deviceBufferoutput;
		openCL.createBuffer(context_, deviceBufferoutput, CL_MEM_READ_WRITE, bufferSize_);

		cl::Kernel simpleBufferSynthesisKernel;
		openCL.createKernel(context_, kernelProgram_, simpleBufferSynthesisKernel, "cl_010_simplebuffersynthesis");
		openCL.setKernelArgument(simpleBufferSynthesisKernel, (void*)&sampleRate, 0, sizeof(int));
		openCL.setKernelArgument(simpleBufferSynthesisKernel, (void*)&frequency, 1, sizeof(float));
		openCL.setKernelArgument(simpleBufferSynthesisKernel, deviceBufferoutput, 2, sizeof(cl_mem));

		//Execute and average//
		std::cout << "Executing test: cl_simplebuffersynthesis_standard" << std::endl;
		if (isWarmup)
		{
			openCL.enqueueKernel(commandQueue_, simpleBufferSynthesisKernel, globalWorkspace_, localWorkspace_);
			openCL.waitCommandQueue(commandQueue_);
			openCL.readBuffer(commandQueue_, deviceBufferoutput, bufferSize_, checkBuffer);
			openCL.waitCommandQueue(commandQueue_);
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			clBenchmarker_.startTimer("cl_simplebuffersynthesis_standard");
			openCL.enqueueKernel(commandQueue_, simpleBufferSynthesisKernel, globalWorkspace_, localWorkspace_);
			openCL.waitCommandQueue(commandQueue_);
			openCL.readBuffer(commandQueue_, deviceBufferoutput, bufferSize_, checkBuffer);
			openCL.waitCommandQueue(commandQueue_);
			clBenchmarker_.pauseTimer("cl_simplebuffersynthesis_standard");
		}
		clBenchmarker_.elapsedTimer("cl_simplebuffersynthesis_standard");

		//Save audio to file for inspection//
		outputAudioFile("cl_simplebuffersynthesis_standard.wav", checkBuffer, bufferLength_);
		std::cout << "cl_simplebuffersynthesis_standard successful: Inspect audio log \"cl_simplebuffersynthesis_standard.wav\"" << std::endl << std::endl;

		//Cleanup//
		//openCL.deleteBuffer(deviceBufferoutput);

		delete hostBuffer;
		delete checkBuffer;
	}
	void cl_simplebuffersynthesis_pinned(size_t aN, bool isWarmup)
	{
		//Test preperation//
		const int sampleRate = 44100;
		const float frequency = 1400.0;
		datatype* checkBuffer = new datatype[bufferLength_];
		datatype* hostBuffer = new datatype[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		cl::Buffer deviceBufferOutput;
		openCL.createBuffer(context_, deviceBufferOutput, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bufferSize_);

		cl::Kernel simpleBufferSynthesisKernel;
		openCL.createKernel(context_, kernelProgram_, simpleBufferSynthesisKernel, "cl_010_simplebuffersynthesis");
		openCL.setKernelArgument(simpleBufferSynthesisKernel, (void*)&sampleRate, 0, sizeof(int));
		openCL.setKernelArgument(simpleBufferSynthesisKernel, (void*)&frequency, 1, sizeof(float));
		openCL.setKernelArgument(simpleBufferSynthesisKernel, deviceBufferOutput, 2, sizeof(cl_mem));

		void* pinned = openCL.mapMemory(commandQueue_, deviceBufferOutput, CL_MAP_READ, bufferSize_);
		openCL.unmapMemory(commandQueue_, deviceBufferOutput, pinned, bufferSize_);

		//Execute and average//
		std::cout << "Executing test: cl_simplebuffersynthesis_pinned" << std::endl;
		if (isWarmup)
		{
			openCL.enqueueKernel(commandQueue_, simpleBufferSynthesisKernel, globalWorkspace_, localWorkspace_);
			openCL.waitCommandQueue(commandQueue_);
			pinned = openCL.mapMemory(commandQueue_, deviceBufferOutput, CL_MAP_READ, bufferSize_);
			std::memcpy(checkBuffer, pinned, bufferSize_);
			openCL.unmapMemory(commandQueue_, deviceBufferOutput, pinned, bufferSize_);
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			clBenchmarker_.startTimer("cl_simplebuffersynthesis_pinned");
			openCL.enqueueKernel(commandQueue_, simpleBufferSynthesisKernel, globalWorkspace_, localWorkspace_);
			openCL.waitCommandQueue(commandQueue_);
			pinned = openCL.mapMemory(commandQueue_, deviceBufferOutput, CL_MAP_READ, bufferSize_);
			std::memcpy(checkBuffer, pinned, bufferSize_);
			openCL.unmapMemory(commandQueue_, deviceBufferOutput, pinned, bufferSize_);
			clBenchmarker_.pauseTimer("cl_simplebuffersynthesis_pinned");
		}
		clBenchmarker_.elapsedTimer("cl_simplebuffersynthesis_pinned");

		//Save audio to file for inspection//
		outputAudioFile("cl_simplebuffersynthesis_pinned.wav", checkBuffer, bufferLength_);
		std::cout << "cl_simplebuffersynthesis_pinned successful: Inspect audio log \"cl_simplebuffersynthesis_pinned.wav\"" << std::endl << std::endl;

		//Cleanup//
		//openCL.deleteBuffer(deviceBufferOutput);

		delete hostBuffer;
		delete checkBuffer;
	}
	void cl_complexbuffersynthesis_standard(size_t aN, bool isWarmup)
	{
		//Test preperation//
		uint32_t numSamplesComputed = 0;
		float* inBuf = new float[bufferLength_];
		float* outBuf = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
		{
			//Create initial impulse as excitation//
			if (i < bufferLength_ / 1000)
				inBuf[i] = 0.5;
			else
				inBuf[i] = 0.0;
		}
		float* soundBuffer = new float[bufferLength_ * 2];

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
		args.bufferSize = bufferLength_;
		args.kernelSource = "resources/kernels/fdtdGlobal.cl";
		OpenCL_FDTD fdtdSynth = OpenCL_FDTD(args, false);	//@ToDo - Need to have selected device control!

		//Execute and average//
		std::cout << "Executing test: cl_complexbuffersynthesis_standard" << std::endl;
		if (isWarmup)
		{
			fdtdSynth.fillBuffer(1, inBuf, outBuf);
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			clBenchmarker_.startTimer("cl_complexbuffersynthesis_standard");

			fdtdSynth.fillBuffer(bufferLength_, inBuf, outBuf);

			clBenchmarker_.pauseTimer("cl_complexbuffersynthesis_standard");
		}
		clBenchmarker_.elapsedTimer("cl_complexbuffersynthesis_standard");

		//Save audio to file for inspection//
		for (int j = 0; j != bufferLength_; ++j)
			soundBuffer[j] = outBuf[j];
		outputAudioFile("cl_complexbuffersynthesis_standard.wav", soundBuffer, bufferLength_);
		std::cout << "cl_complexbuffersynthesis_standard successful: Inspect audio log \"cl_complexbuffersynthesis_standard.wav\"" << std::endl << std::endl;

		//Cleanup//
		delete inBuf;
		delete outBuf;
	}
	//@ToDo - Invetsigate if can just map once with OpenCL, or need to  map/unmap between each read and write to buffers. Currently works with just one map!? But online resources say this is undefined?//
	void cl_complexbuffersynthesis_pinned(size_t aN, bool isWarmup)
	{
		//Test preperation//
		uint32_t numSamplesComputed = 0;
		float* inBuf = new float[bufferLength_];
		float* outBuf = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
		{
			//Create initial impulse as excitation//
			if (i < bufferLength_ / 1000)
				inBuf[i] = 0.5;
			else
				inBuf[i] = 0.0;
		}
		float* soundBuffer = new float[bufferLength_ * 2];

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
		args.bufferSize = bufferLength_;
		args.kernelSource = "resources/kernels/fdtdGlobal.cl";
		OpenCL_FDTD fdtdSynth = OpenCL_FDTD(args, true);	//@ToDo - Need to have selected device control!

		//Execute and average//
		std::cout << "Executing test: cl_complexbuffersynthesis_pinned" << std::endl;
		if (isWarmup)
		{
			fdtdSynth.fillBuffer(1, inBuf, outBuf);
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			clBenchmarker_.startTimer("cl_complexbuffersynthesis_pinned");

			fdtdSynth.fillBuffer(bufferLength_, inBuf, outBuf);

			clBenchmarker_.pauseTimer("cl_complexbuffersynthesis_pinned");
		}
		clBenchmarker_.elapsedTimer("cl_complexbuffersynthesis_pinned");

		//Save audio to file for inspection//
		for (int j = 0; j != bufferLength_; ++j)
			soundBuffer[j] = outBuf[j];
		outputAudioFile("cl_complexbuffersynthesis_pinned.wav", soundBuffer, bufferLength_);
		std::cout << "cl_complexbuffersynthesis_pinned successful: Inspect audio log \"cl_complexbuffersynthesis_pinned.wav\"" << std::endl << std::endl;

		//Cleanup//
		delete inBuf;
		delete outBuf;
	}
	void cl_interruptedbufferprocessing_standard(size_t aN, bool isWarmup)
	{
		//Test preperation//
		datatype* checkBuffer = new datatype[bufferLength_];
		datatype* hostBuffer = new datatype[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		cl::Buffer deviceBufferSrc;
		cl::Buffer deviceBufferDst;
		openCL.createBuffer(context_, deviceBufferSrc, CL_MEM_READ_WRITE, bufferSize_);
		openCL.createBuffer(context_, deviceBufferDst, CL_MEM_READ_WRITE, bufferSize_);

		cl::Kernel simpleBufferProcessingKernel;
		openCL.createKernel(context_, kernelProgram_, simpleBufferProcessingKernel, "cl_008_simplebufferprocessing");
		openCL.setKernelArgument(simpleBufferProcessingKernel, deviceBufferSrc, 0, sizeof(cl_mem));
		openCL.setKernelArgument(simpleBufferProcessingKernel, deviceBufferDst, 1, sizeof(cl_mem));

		std::default_random_engine generator;
		std::uniform_int_distribution<int> distribution(1, 10);

		//Execute and average//
		std::cout << "Executing test: cl_interruptedbufferprocessing_standard" << std::endl;
		if (isWarmup)
		{
			openCL.writeBuffer(commandQueue_, deviceBufferSrc, bufferSize_, hostBuffer);
			openCL.waitCommandQueue(commandQueue_);
			openCL.enqueueKernel(commandQueue_, simpleBufferProcessingKernel, globalWorkspace_, localWorkspace_);
			openCL.waitCommandQueue(commandQueue_);
			openCL.readBuffer(commandQueue_, deviceBufferDst, bufferSize_, checkBuffer);
			openCL.waitCommandQueue(commandQueue_);
		}
		for (int32_t i = 0; i != aN; ++i)
		{
			clBenchmarker_.startTimer("cl_interruptedbufferprocessing_standard");

			openCL.writeBuffer(commandQueue_, deviceBufferSrc, bufferSize_, hostBuffer);
			openCL.waitCommandQueue(commandQueue_);
			openCL.enqueueKernel(commandQueue_, simpleBufferProcessingKernel, globalWorkspace_, localWorkspace_);
			openCL.waitCommandQueue(commandQueue_);
			openCL.readBuffer(commandQueue_, deviceBufferDst, bufferSize_, checkBuffer);
			openCL.waitCommandQueue(commandQueue_);

			clBenchmarker_.pauseTimer("cl_interruptedbufferprocessing_standard");

			int randomChance = distribution(generator);
			if (randomChance > 5)
				--i;

		}
		clBenchmarker_.elapsedTimer("cl_interruptedbufferprocessing_standard");

		//Check contents//
		bool isSuccessful = true;
		openCL.readBuffer(commandQueue_, deviceBufferDst, bufferSize_, checkBuffer);
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			float attenuatedSample = hostBuffer[i] * 0.5;
			if (attenuatedSample != checkBuffer[i])
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cl_interruptedbufferprocessing_standard successful: " << isSuccessful << std::endl << std::endl;

		//Cleanup//
		delete hostBuffer;
		delete checkBuffer;
	}
	void cl_interruptedbufferprocessing_pinned(size_t aN, bool isWarmup)
	{
		//Test preperation//
		datatype* checkBuffer = new datatype[bufferLength_];
		datatype* hostBuffer = new datatype[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		void* pinned;
		cl::Buffer deviceBufferSrc;
		cl::Buffer deviceBufferDst;
		openCL.createBuffer(context_, deviceBufferSrc, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bufferSize_);
		openCL.createBuffer(context_, deviceBufferDst, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bufferSize_);

		cl::Kernel simpleBufferProcessingKernel;
		openCL.createKernel(context_, kernelProgram_, simpleBufferProcessingKernel, "cl_008_simplebufferprocessing");
		openCL.setKernelArgument(simpleBufferProcessingKernel, deviceBufferSrc, 0, sizeof(cl_mem));
		openCL.setKernelArgument(simpleBufferProcessingKernel, deviceBufferDst, 1, sizeof(cl_mem));

		std::default_random_engine generator;
		std::uniform_int_distribution<int> distribution(1, 10);

		//Execute and average//
		std::cout << "Executing test: cl_interruptedbufferprocessing_pinned" << std::endl;
		if (isWarmup)
		{
			pinned = openCL.mapMemory(commandQueue_, deviceBufferSrc, CL_MAP_WRITE, bufferSize_);
			std::memcpy(pinned, hostBuffer, bufferSize_);
			openCL.unmapMemory(commandQueue_, deviceBufferSrc, pinned, bufferSize_);

			openCL.enqueueKernel(commandQueue_, simpleBufferProcessingKernel, globalWorkspace_, localWorkspace_);
			openCL.waitCommandQueue(commandQueue_);

			pinned = openCL.mapMemory(commandQueue_, deviceBufferDst, CL_MAP_READ, bufferSize_);
			std::memcpy(checkBuffer, pinned, bufferSize_);
			openCL.unmapMemory(commandQueue_, deviceBufferDst, pinned, bufferSize_);
		}
		for (int32_t i = 0; i != aN; ++i)
		{
			clBenchmarker_.startTimer("cl_interruptedbufferprocessing_pinned");

			pinned = openCL.mapMemory(commandQueue_, deviceBufferSrc, CL_MAP_WRITE, bufferSize_);
			std::memcpy(pinned, hostBuffer, bufferSize_);
			openCL.unmapMemory(commandQueue_, deviceBufferSrc, pinned, bufferSize_);

			openCL.enqueueKernel(commandQueue_, simpleBufferProcessingKernel, globalWorkspace_, localWorkspace_);
			openCL.waitCommandQueue(commandQueue_);

			pinned = openCL.mapMemory(commandQueue_, deviceBufferDst, CL_MAP_READ, bufferSize_);
			std::memcpy(checkBuffer, pinned, bufferSize_);
			openCL.unmapMemory(commandQueue_, deviceBufferDst, pinned, bufferSize_);

			clBenchmarker_.pauseTimer("cl_interruptedbufferprocessing_pinned");

			int randomChance = distribution(generator);
			if (randomChance > 5)
				--i;

		}
		clBenchmarker_.elapsedTimer("cl_interruptedbufferprocessing_pinned");

		//Check contents//
		bool isSuccessful = true;
		openCL.readBuffer(commandQueue_, deviceBufferDst, bufferSize_, checkBuffer);
		for (uint32_t i = 0; i != bufferLength_; ++i)
		{
			float attenuatedSample = hostBuffer[i] * 0.5;
			if (attenuatedSample != checkBuffer[i])
			{
				isSuccessful = false;
				break;
			}
		}
		std::cout << "cl_interruptedbufferprocessing_pinned successful: " << isSuccessful << std::endl << std::endl;

		//Cleanup//
		delete hostBuffer;
		delete checkBuffer;
	}

	void cl_unidirectional_baseline(size_t aFrameRate, bool isWarmup)
	{
		//Prepare new file for cl_bidirectional_processing//
		std::string strBenchmarkFileName = "cl_unidirectional_baseline_framerate";
		std::string strFrameRate = std::to_string(aFrameRate);
		strBenchmarkFileName.append(strFrameRate);
		strBenchmarkFileName.append(".csv");
		clBenchmarker_ = Benchmarker(strBenchmarkFileName, { "Buffer_Size", "Total_Time", "Average_Time", "Max_Time", "Min_Time", "Max_Difference", "Average_Difference" });

		cl::Kernel nullKernel;
		openCL.createKernel(context_, kernelProgram_, nullKernel, "cl_000_nullKernel");

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

			cl::Buffer deviceBufferOutput;
			openCL.createBuffer(context_, deviceBufferOutput, CL_MEM_READ_WRITE, bufferSize_);

			if (isWarmup)
			{
				openCL.enqueueKernel(commandQueue_, nullKernel, 1, 1);
				openCL.waitCommandQueue(commandQueue_);
				openCL.readBuffer(commandQueue_, deviceBufferOutput, currentBufferSize, outBuf);
				openCL.waitCommandQueue(commandQueue_);
			}
			while (numSamplesComputed < aFrameRate)
			{
				clBenchmarker_.startTimer(strBenchmarkName);
				openCL.enqueueKernel(commandQueue_, nullKernel, 1, 1);
				openCL.waitCommandQueue(commandQueue_);
				openCL.readBuffer(commandQueue_, deviceBufferOutput, currentBufferSize, outBuf);
				openCL.waitCommandQueue(commandQueue_);
				clBenchmarker_.pauseTimer(strBenchmarkName);

				numSamplesComputed += currentBufferLength;
			}
			clBenchmarker_.elapsedTimer(strBenchmarkName);

			numSamplesComputed = 0;

			delete inBuf;
			delete outBuf;
		}
	}
	void cl_unidirectional_processing(size_t aFrameRate, bool isWarmup)
	{
		//Prepare new file for cl_bidirectional_processing//
		std::string strBenchmarkFileName = "cl_unidirectional_processing_framerate";
		std::string strFrameRate = std::to_string(aFrameRate);
		strBenchmarkFileName.append(strFrameRate);
		strBenchmarkFileName.append(".csv");
		clBenchmarker_ = Benchmarker(strBenchmarkFileName, { "Buffer_Size", "Total_Time", "Average_Time", "Max_Time", "Min_Time", "Max_Difference", "Average_Difference" });

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

			const int sampleRate = 44100;
			const float frequency = 1400.0;
			cl::Buffer deviceBufferOutput;
			openCL.createBuffer(context_, deviceBufferOutput, CL_MEM_READ_WRITE, bufferSize_);

			cl::Kernel simpleBufferSynthesisKernel;
			openCL.createKernel(context_, kernelProgram_, simpleBufferSynthesisKernel, "cl_010_simplebuffersynthesis");
			openCL.setKernelArgument(simpleBufferSynthesisKernel, (void*)&sampleRate, 0, sizeof(int));
			openCL.setKernelArgument(simpleBufferSynthesisKernel, (void*)&frequency, 1, sizeof(float));
			openCL.setKernelArgument(simpleBufferSynthesisKernel, deviceBufferOutput, 2, sizeof(cl_mem));

			float* soundBuffer = new float[currentBufferLength > aFrameRate ? currentBufferLength : aFrameRate * 2];

			if (isWarmup)
			{
				openCL.enqueueKernel(commandQueue_, simpleBufferSynthesisKernel, globalWorkspace_, localWorkspace_);
				openCL.waitCommandQueue(commandQueue_);
				openCL.readBuffer(commandQueue_, deviceBufferOutput, currentBufferSize, outBuf);
				openCL.waitCommandQueue(commandQueue_);
			}
			while (numSamplesComputed < aFrameRate)
			{
				clBenchmarker_.startTimer(strBenchmarkName);
				openCL.enqueueKernel(commandQueue_, simpleBufferSynthesisKernel, globalWorkspace_, localWorkspace_);
				openCL.waitCommandQueue(commandQueue_);
				openCL.readBuffer(commandQueue_, deviceBufferOutput, currentBufferSize, outBuf);
				openCL.waitCommandQueue(commandQueue_);
				clBenchmarker_.pauseTimer(strBenchmarkName);

				//Log audio for inspection if necessary//
				for (int j = 0; j != currentBufferLength; ++j)
					soundBuffer[numSamplesComputed + j] = outBuf[j];

				numSamplesComputed += currentBufferLength;
			}
			clBenchmarker_.elapsedTimer(strBenchmarkName);

			//Save audio to file for inspection//
			outputAudioFile("cl_unidirectional_processing.wav", soundBuffer, aFrameRate);
			std::cout << "cl_unidirectional_processing successful: Inspect audio log \"cl_unidirectional_processing.wav\"" << std::endl << std::endl;

			numSamplesComputed = 0;

			delete inBuf;
			delete outBuf;
		}
	}
	void cl_bidirectional_baseline(size_t aFrameRate, bool isWarmup)
	{
		//Prepare new file for cl_bidirectional_processing//
		std::string strBenchmarkFileName = "cl_bidirectional_baseline_framerate";
		std::string strFrameRate = std::to_string(aFrameRate);
		strBenchmarkFileName.append(strFrameRate);
		strBenchmarkFileName.append(".csv");
		clBenchmarker_ = Benchmarker(strBenchmarkFileName, { "Buffer_Size", "Total_Time", "Average_Time", "Max_Time", "Min_Time", "Max_Difference", "Average_Difference" });

		cl::Kernel nullKernel;
		openCL.createKernel(context_, kernelProgram_, nullKernel, "cl_000_nullKernel");

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

			cl::Buffer deviceBufferInput;
			cl::Buffer deviceBufferOutput;
			openCL.createBuffer(context_, deviceBufferInput, CL_MEM_WRITE_ONLY, bufferSize_);
			openCL.createBuffer(context_, deviceBufferOutput, CL_MEM_READ_ONLY, bufferSize_);

			float* soundBuffer = new float[currentBufferLength > aFrameRate ? currentBufferLength : aFrameRate * 2];

			if (isWarmup)
			{
				openCL.writeBuffer(commandQueue_, deviceBufferInput, currentBufferSize, inBuf);
				openCL.waitCommandQueue(commandQueue_);
				openCL.enqueueKernel(commandQueue_, nullKernel, 1, 1);
				openCL.waitCommandQueue(commandQueue_);
				openCL.readBuffer(commandQueue_, deviceBufferOutput, currentBufferSize, outBuf);
				openCL.waitCommandQueue(commandQueue_);
			}
			while (numSamplesComputed < aFrameRate)
			{
				clBenchmarker_.startTimer(strBenchmarkName);
				openCL.writeBuffer(commandQueue_, deviceBufferInput, currentBufferSize, inBuf);
				openCL.waitCommandQueue(commandQueue_);
				openCL.enqueueKernel(commandQueue_, nullKernel, 1, 1);
				openCL.waitCommandQueue(commandQueue_);
				openCL.readBuffer(commandQueue_, deviceBufferOutput, currentBufferSize, outBuf);
				openCL.waitCommandQueue(commandQueue_);
				clBenchmarker_.pauseTimer(strBenchmarkName);

				numSamplesComputed += currentBufferLength;
			}
			clBenchmarker_.elapsedTimer(strBenchmarkName);

			numSamplesComputed = 0;

			delete inBuf;
			delete outBuf;
		}
	}
	void cl_bidirectional_processing(size_t aFrameRate, bool isWarmup)
	{
		//Prepare new file for cl_bidirectional_processing//
		std::string strBenchmarkFileName = "cl_bidirectional_processing_framerate";
		std::string strFrameRate = std::to_string(aFrameRate);
		strBenchmarkFileName.append(strFrameRate);
		strBenchmarkFileName.append(".csv");
		clBenchmarker_ = Benchmarker(strBenchmarkFileName, { "Buffer_Size", "Total_Time", "Average_Time", "Max_Time", "Min_Time", "Max_Difference", "Average_Difference" });

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
			args.bufferSize = currentBufferLength;
			args.kernelSource = "resources/kernels/fdtdGlobal.cl";
			OpenCL_FDTD fdtdSynth = OpenCL_FDTD(args, false);	//@ToDo - Need to have selected device control!

			uint64_t numSamplesComputed = 0;
			float* inBuf = new float[currentBufferLength];
			float* outBuf = new float[currentBufferLength];
			for (size_t i = 0; i != currentBufferLength; ++i)
				inBuf[i] = 0.5;

			float* soundBuffer = new float[currentBufferLength > aFrameRate ? currentBufferLength : aFrameRate*2];

			if (isWarmup)
			{
				fdtdSynth.fillBuffer(1, inBuf, outBuf);
			}
			while (numSamplesComputed < aFrameRate)
			{
				clBenchmarker_.startTimer(strBenchmarkName);
				fdtdSynth.fillBuffer(currentBufferLength, inBuf, outBuf);
				clBenchmarker_.pauseTimer(strBenchmarkName);

				//Log audio for inspection if necessary//
				for (int j = 0; j != currentBufferLength; ++j)
					soundBuffer[numSamplesComputed + j] = outBuf[j];

				numSamplesComputed += currentBufferLength;
			}
			clBenchmarker_.elapsedTimer(strBenchmarkName);

			//Save audio to file for inspection//
			outputAudioFile("cl_bidirectional_processing.wav", soundBuffer, aFrameRate);
			std::cout << "cl_bidirectional_processing successful: Inspect audio log \"cl_bidirectional_processing.wav\"" << std::endl << std::endl;

			numSamplesComputed = 0;

			delete inBuf;
			delete outBuf;
		}
	}

	void cl_mappingmemory(uint32_t aN)
	{
		//Prepare new file for cl_mappingmemory//
		std::string strBenchmarkFileName = "cl_mappingmemory";
		strBenchmarkFileName.append(".csv");
		clBenchmarker_ = Benchmarker(strBenchmarkFileName, { "Type", "Total_Time", "Average_Time", "Max_Time", "Min_Time", "Max_Difference", "Average_Difference" });

		//Prepare host and check buffer//
		datatype* checkBuffer = new datatype[bufferLength_];
		datatype* hostBuffer = new datatype[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		//Create a host visible buffer//
		cl::Buffer deviceBufferPinned;
		openCL.createBuffer(context_, deviceBufferPinned, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bufferSize_);

		void* pinned;

		//Benchmark mapping//
		for (uint32_t i = 0; i != aN; ++i)
		{
			clBenchmarker_.startTimer("Map");
			pinned = openCL.mapMemory(commandQueue_, deviceBufferPinned, CL_MAP_WRITE, bufferSize_);
			clBenchmarker_.pauseTimer("Map");
			openCL.unmapMemory(commandQueue_, deviceBufferPinned, pinned, bufferSize_);
		}
		clBenchmarker_.elapsedTimer("Map");

		//Benchmark unmapping//
		for (uint32_t i = 0; i != aN; ++i)
		{
			pinned = openCL.mapMemory(commandQueue_, deviceBufferPinned, CL_MAP_WRITE, bufferSize_);
			clBenchmarker_.startTimer("Unmap");
			openCL.unmapMemory(commandQueue_, deviceBufferPinned, pinned, bufferSize_);
			clBenchmarker_.pauseTimer("Unmap");
		}
		clBenchmarker_.elapsedTimer("Unmap");

		//Benchmark write//
		for (uint32_t i = 0; i != aN; ++i)
		{
			pinned = openCL.mapMemory(commandQueue_, deviceBufferPinned, CL_MAP_WRITE, bufferSize_);
			clBenchmarker_.startTimer("write");
			std::memcpy(pinned, hostBuffer, bufferSize_);
			clBenchmarker_.pauseTimer("write");
			openCL.unmapMemory(commandQueue_, deviceBufferPinned, pinned, bufferSize_);
		}
		clBenchmarker_.elapsedTimer("write");

		//Benchmark mapwrite//
		for (uint32_t i = 0; i != aN; ++i)
		{
			clBenchmarker_.startTimer("mapwrite");
			pinned = openCL.mapMemory(commandQueue_, deviceBufferPinned, CL_MAP_WRITE, bufferSize_);
			std::memcpy(pinned, hostBuffer, bufferSize_);
			openCL.unmapMemory(commandQueue_, deviceBufferPinned, pinned, bufferSize_);
			clBenchmarker_.pauseTimer("mapwrite");
		}
		clBenchmarker_.elapsedTimer("mapwrite");

		//Benchmark read//
		for (uint32_t i = 0; i != aN; ++i)
		{
			pinned = openCL.mapMemory(commandQueue_, deviceBufferPinned, CL_MAP_READ, bufferSize_);
			clBenchmarker_.startTimer("read");
			std::memcpy(checkBuffer, pinned, bufferSize_);
			clBenchmarker_.pauseTimer("read");
			openCL.unmapMemory(commandQueue_, deviceBufferPinned, pinned, bufferSize_);
		}
		clBenchmarker_.elapsedTimer("read");

		//Benchmark mapread//
		for (uint32_t i = 0; i != aN; ++i)
		{
			clBenchmarker_.startTimer("mapread");
			pinned = openCL.mapMemory(commandQueue_, deviceBufferPinned, CL_MAP_READ, bufferSize_);
			std::memcpy(checkBuffer, pinned, bufferSize_);
			openCL.unmapMemory(commandQueue_, deviceBufferPinned, pinned, bufferSize_);
			clBenchmarker_.pauseTimer("mapread");
		}
		clBenchmarker_.elapsedTimer("mapread");

		//Benchmark mapwriteread//
		for (uint32_t i = 0; i != aN; ++i)
		{
			clBenchmarker_.startTimer("mapwriteread");
			pinned = openCL.mapMemory(commandQueue_, deviceBufferPinned, CL_MAP_WRITE | CL_MAP_READ, bufferSize_);
			std::memcpy(pinned, hostBuffer, bufferSize_);
			std::memcpy(checkBuffer, pinned, bufferSize_);
			openCL.unmapMemory(commandQueue_, deviceBufferPinned, pinned, bufferSize_);
			clBenchmarker_.pauseTimer("mapwriteread");
		}
		clBenchmarker_.elapsedTimer("mapwriteread");
	}
	void setWorkspaceSize(uint32_t aGlobalSize, uint32_t aLocalSize)
	{
		globalWorkspace_ = cl::NDRange(aGlobalSize, 1, 1);
		localWorkspace_ = cl::NDRange(aLocalSize, 1, 1);
	}
	void setWorkspaceSize(cl::NDRange aGlobalSize, cl::NDRange aLocalSize)
	{
		globalWorkspace_ = aGlobalSize;
		localWorkspace_ = aLocalSize;
	} 

	void setLocalWorkspace(uint64_t aGlobalSize)
	{
		uint64_t maxLocalWorkspace = openCL.getMaxLocalWorkspace(device_);
		uint64_t localWorkspace = aGlobalSize > maxLocalWorkspace ? maxLocalWorkspace : aGlobalSize;

		cl::NDRange newGlobalSize = aGlobalSize;
		cl::NDRange newLocalSize = localWorkspace;
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

	static bool openclCompatible()
	{
		cl::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);

		if (platforms.size() != 0)
		{
			std::cout << "OpenCL Platform Version: " << platforms[0].getInfo<CL_PLATFORM_VERSION>() << std::endl << std::endl;
			return true;
		}

		return false;
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
		audioFile.setSampleRate(96000);

		for (int k = 0; k != aAudioLength; ++k)
			buffer[0][k] = (float)aAudioBuffer[k];

		audioFile.setAudioBuffer(buffer);
		audioFile.save(aPath);
	}
};

#endif