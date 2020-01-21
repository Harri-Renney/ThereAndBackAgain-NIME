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
	static const size_t bufferSizesLength = 13;
	uint64_t bufferSizes[bufferSizesLength];

	typedef float datatype;
	uint32_t sampleRate_ = 44100;
	uint64_t bufferSize_ = 1024;
	uint64_t bufferLength_ = bufferSize_ / sizeof(datatype);
public:
	static const uint32_t GIGA_BYTE = 1024 * 1024 * 1024;
	static const uint32_t MEGA_BYTE = 1024 * 1024;
	static const uint32_t KILO_BYTE = 1024;
	GPU_Overhead_Benchmarks_OpenCL(uint32_t aPlatform, uint32_t aDevice) : fdtdSynth(), clBenchmarker_("openclog.csv", { "Test_Name", "Total_Time", "Average_Time", "Max_Time", "Min_Time", "Max_Difference", "Average_Difference" })
	{
		openCL = OpenCL_Wrapper(aPlatform, aDevice, context_, device_, commandQueue_);
		bufferSizes[0] = 1;
		for (size_t i = 1; i != 11; ++i)
		{
			bufferSizes[i] = bufferSizes[i - 1] * 2;
		}
		bufferSizes[11] = MEGA_BYTE;
		bufferSizes[12] = GIGA_BYTE;

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

	void runGeneralBenchmarks(uint64_t aNumRepetitions) override
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
			if (i > 9)
				setBufferSize(currentBufferSize);

			//Run tests with setup//
			cl_nullkernel(aNumRepetitions, true);
			cl_cputogpu_standard(aNumRepetitions, true);
			cl_cputogpu_mappedmemory(aNumRepetitions, true);
			cl_gputocpu_standard(aNumRepetitions, true);
			cl_gputocpu_mappedmemory(aNumRepetitions, true);
			cl_cputogputocpu_standard(aNumRepetitions, true);
			cl_cputogputocpu_mappedmemory(aNumRepetitions, true);
			cl_devicetransfer_standard(aNumRepetitions, true);
			cl_devicetransfer_mappedmemory(aNumRepetitions, true);
			cl_devicetransferkernel_standard(aNumRepetitions, true);
			cl_devicetransferkernel_mappedmemory(aNumRepetitions, true);
			cl_simplebufferprocessing_standard(aNumRepetitions, true);
			cl_simplebufferprocessing_mappedmemory(aNumRepetitions, true);
			cl_complexbufferprocessing_standard(aNumRepetitions, true);
			cl_complexbufferprocessing_mappedmemory(aNumRepetitions, true);
			cl_simplebuffersynthesis_standard(aNumRepetitions, true);
			cl_simplebuffersynthesis_mappedmemory(aNumRepetitions, true);
			cl_complexbuffersynthesis_standard(aNumRepetitions, true);
			cl_complexbuffersynthesis_mappedmemory(aNumRepetitions, true);
			cl_interruptedbufferprocessing_standard(aNumRepetitions, true);
			cl_interruptedbufferprocessing_mappedmemory(aNumRepetitions, true);
		}
	}
	void runRealTimeBenchmarks(uint64_t aFrameRate) override
	{
		cl_unidirectional_baseline(aFrameRate, true);
		cl_unidirectional_processing(aFrameRate, true);
		cl_bidirectional_baseline(aFrameRate, true);
		cl_bidirectional_processing(aFrameRate, true);
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
			openCL.enqueueKernel(commandQueue_, nullKernel, globalWorkspace_, localWorkspace_);
			openCL.waitCommandQueue(commandQueue_);
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			clBenchmarker_.startTimer("cl_nullkernel");
			openCL.enqueueKernel(commandQueue_, nullKernel, globalWorkspace_, localWorkspace_);
			openCL.waitCommandQueue(commandQueue_);
			clBenchmarker_.pauseTimer("cl_nullkernel");
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
	void cl_cputogpu_mappedmemory(size_t aN, bool isWarmup)
	{
		//Test preperation//
		datatype* checkBuffer = new datatype[bufferLength_];
		datatype* hostBuffer = new datatype[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		cl::Buffer deviceBuffer;
		openCL.createBuffer(context_, deviceBuffer, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bufferSize_);

		void* mappedMemory = openCL.mapMemory(commandQueue_, deviceBuffer, CL_MAP_WRITE, bufferSize_);
		openCL.unmapMemory(commandQueue_, deviceBuffer, mappedMemory, bufferSize_);

		cl::Kernel testKernel;
		openCL.createKernel(context_, kernelProgram_, testKernel, "cl_testkernel");
		openCL.setKernelArgument(testKernel, deviceBuffer, 0, sizeof(cl_mem));

		//Execute and average//
		std::cout << "Executing test: cl_cputogpu_mappedmemory" << std::endl;
		if (isWarmup)
		{
			mappedMemory = openCL.mapMemory(commandQueue_, deviceBuffer, CL_MAP_WRITE, bufferSize_);
			std::memcpy(mappedMemory, hostBuffer, bufferSize_);
			openCL.unmapMemory(commandQueue_, deviceBuffer, mappedMemory, bufferSize_);
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			clBenchmarker_.startTimer("cl_cputogpu_mappedbuffer");
			mappedMemory = openCL.mapMemory(commandQueue_, deviceBuffer, CL_MAP_WRITE, bufferSize_);
			std::memcpy(mappedMemory, hostBuffer, bufferSize_);
			openCL.unmapMemory(commandQueue_, deviceBuffer, mappedMemory, bufferSize_);
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
	void cl_gputocpu_mappedmemory(size_t aN, bool isWarmup)
	{
		//Test preperation//
		datatype* checkBuffer = new datatype[bufferLength_];
		datatype* hostBuffer = new datatype[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		cl::Buffer deviceBuffer;
		openCL.createBuffer(context_, deviceBuffer, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bufferSize_);

		void* mappedMemory = openCL.mapMemory(commandQueue_, deviceBuffer, CL_MAP_WRITE, bufferSize_);
		std::memcpy(mappedMemory, hostBuffer, bufferSize_);
		openCL.unmapMemory(commandQueue_, deviceBuffer, mappedMemory, bufferSize_);

		cl::Kernel testKernel;
		openCL.createKernel(context_, kernelProgram_, testKernel, "cl_testkernel");
		openCL.setKernelArgument(testKernel, deviceBuffer, 0, sizeof(cl_mem));

		//Execute and average//
		std::cout << "Executing test: cl_gputocpu_mappedmemory" << std::endl;
		if (isWarmup)
		{
			mappedMemory = openCL.mapMemory(commandQueue_, deviceBuffer, CL_MAP_READ, bufferSize_);
			std::memcpy(checkBuffer, mappedMemory, bufferSize_);
			openCL.unmapMemory(commandQueue_, deviceBuffer, mappedMemory, bufferSize_);
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			clBenchmarker_.startTimer("cl_gputocpu_mappedmemory");
			mappedMemory = openCL.mapMemory(commandQueue_, deviceBuffer, CL_MAP_READ, bufferSize_);
			std::memcpy(checkBuffer, mappedMemory, bufferSize_);
			openCL.unmapMemory(commandQueue_, deviceBuffer, mappedMemory, bufferSize_);
			clBenchmarker_.pauseTimer("cl_gputocpu_mappedmemory");

			//Run kernel outside timer for results//
			openCL.enqueueKernel(commandQueue_, testKernel, globalWorkspace_, localWorkspace_);
			openCL.waitCommandQueue(commandQueue_);
		}
		clBenchmarker_.elapsedTimer("cl_gputocpu_mappedmemory");

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
		std::cout << "cl_gputocpu_mappedmemory successful: " << isSuccessful << std::endl << std::endl;

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
	void cl_cputogputocpu_mappedmemory(size_t aN, bool isWarmup)
	{
		//Test preperation//
		datatype* checkBuffer = new datatype[bufferLength_];
		datatype* hostBuffer = new datatype[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
			hostBuffer[i] = 42.0;

		cl::Buffer deviceBuffer;
		openCL.createBuffer(context_, deviceBuffer, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bufferSize_);

		void* mappedMemory = openCL.mapMemory(commandQueue_, deviceBuffer, CL_MAP_WRITE, bufferSize_);
		openCL.unmapMemory(commandQueue_, deviceBuffer, mappedMemory, bufferSize_);

		cl::Kernel testKernel;
		openCL.createKernel(context_, kernelProgram_, testKernel, "cl_testkernel");
		openCL.setKernelArgument(testKernel, deviceBuffer, 0, sizeof(cl_mem));

		//Execute and average//
		std::cout << "Executing test: cl_cputogputocpu_mappedmemory" << std::endl;
		if (isWarmup)
		{
			mappedMemory = openCL.mapMemory(commandQueue_, deviceBuffer, CL_MAP_WRITE, bufferSize_);
			std::memcpy(mappedMemory, hostBuffer, bufferSize_);
			openCL.unmapMemory(commandQueue_, deviceBuffer, mappedMemory, bufferSize_);
			mappedMemory = openCL.mapMemory(commandQueue_, deviceBuffer, CL_MAP_READ, bufferSize_);
			std::memcpy(checkBuffer, mappedMemory, bufferSize_);
			openCL.unmapMemory(commandQueue_, deviceBuffer, mappedMemory, bufferSize_);
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			clBenchmarker_.startTimer("cl_cputogputocpu_mappedmemory");
			mappedMemory = openCL.mapMemory(commandQueue_, deviceBuffer, CL_MAP_WRITE, bufferSize_);
			std::memcpy(mappedMemory, hostBuffer, bufferSize_);
			openCL.unmapMemory(commandQueue_, deviceBuffer, mappedMemory, bufferSize_);
			clBenchmarker_.waitTimer("cl_cputogputocpu_mappedmemory");

			//Run kernel outside timer for results//
			openCL.enqueueKernel(commandQueue_, testKernel, globalWorkspace_, localWorkspace_);
			openCL.waitCommandQueue(commandQueue_);

			clBenchmarker_.resumeTimer("cl_cputogputocpu_mappedmemory");
			mappedMemory = openCL.mapMemory(commandQueue_, deviceBuffer, CL_MAP_READ, bufferSize_);
			std::memcpy(checkBuffer, mappedMemory, bufferSize_);
			openCL.unmapMemory(commandQueue_, deviceBuffer, mappedMemory, bufferSize_);
			clBenchmarker_.pauseTimer("cl_cputogputocpu_mappedmemory");
		}
		clBenchmarker_.elapsedTimer("cl_cputogputocpu_mappedmemory");

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
		std::cout << "cl_cputogputocpu_mappedmemory successful: " << isSuccessful << std::endl << std::endl;

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
	void cl_devicetransfer_mappedmemory(size_t aN, bool isWarmup)
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
		std::cout << "Executing test: cl_devicetransfer_mappedmemory" << std::endl;
		if (isWarmup)
		{
			openCL.enqueueCopyBuffer(commandQueue_, deviceBufferSrc, deviceBufferDst, bufferSize_);
			openCL.waitCommandQueue(commandQueue_);
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			clBenchmarker_.startTimer("cl_devicetransfer_mappedmemory");
			openCL.enqueueCopyBuffer(commandQueue_, deviceBufferSrc, deviceBufferDst, bufferSize_);
			openCL.waitCommandQueue(commandQueue_);
			clBenchmarker_.pauseTimer("cl_devicetransfer_mappedmemory");
		}
		clBenchmarker_.elapsedTimer("cl_devicetransfer_mappedmemory");

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
		std::cout << "cl_devicetransfer_mappedmemory successful: " << isSuccessful << std::endl << std::endl;

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
	void cl_devicetransferkernel_mappedmemory(size_t aN, bool isWarmup)
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
		std::cout << "Executing test: cl_devicetransferkernel_mappedmemory" << std::endl;
		if (isWarmup)
		{
			openCL.enqueueKernel(commandQueue_, transferKernel, globalWorkspace_, localWorkspace_);
			openCL.waitCommandQueue(commandQueue_);
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			clBenchmarker_.startTimer("cl_devicetransferkernel_mappedmemory");
			openCL.enqueueKernel(commandQueue_, transferKernel, globalWorkspace_, localWorkspace_);
			openCL.waitCommandQueue(commandQueue_);
			clBenchmarker_.pauseTimer("cl_devicetransferkernel_mappedmemory");
		}
		clBenchmarker_.elapsedTimer("cl_devicetransferkernel_mappedmemory");

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
		std::cout << "cl_devicetransferkernel_mappedmemory successful: " << isSuccessful << std::endl << std::endl;

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
	void cl_simplebufferprocessing_mappedmemory(size_t aN, bool isWarmup)
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

		void* mappedMemory = openCL.mapMemory(commandQueue_, deviceBufferSrc, CL_MAP_WRITE, bufferSize_);
		openCL.unmapMemory(commandQueue_, deviceBufferSrc, mappedMemory, bufferSize_);

		//Execute and average//
		std::cout << "Executing test: cl_simplebufferprocessing_mappedmemory" << std::endl;
		if (isWarmup)
		{
			mappedMemory = openCL.mapMemory(commandQueue_, deviceBufferSrc, CL_MAP_WRITE, bufferSize_);
			std::memcpy(mappedMemory, hostBuffer, bufferSize_);
			openCL.unmapMemory(commandQueue_, deviceBufferSrc, mappedMemory, bufferSize_);

			openCL.enqueueKernel(commandQueue_, simpleBufferProcessingKernel, globalWorkspace_, localWorkspace_);
			openCL.waitCommandQueue(commandQueue_);

			mappedMemory = openCL.mapMemory(commandQueue_, deviceBufferDst, CL_MAP_READ, bufferSize_);
			std::memcpy(checkBuffer, mappedMemory, bufferSize_);
			openCL.unmapMemory(commandQueue_, deviceBufferDst, mappedMemory, bufferSize_);
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			clBenchmarker_.startTimer("cl_simplebufferprocessing_mappedmemory");
			mappedMemory = openCL.mapMemory(commandQueue_, deviceBufferSrc, CL_MAP_WRITE, bufferSize_);
			std::memcpy(mappedMemory, hostBuffer, bufferSize_);
			openCL.unmapMemory(commandQueue_, deviceBufferSrc, mappedMemory, bufferSize_);

			openCL.enqueueKernel(commandQueue_, simpleBufferProcessingKernel, globalWorkspace_, localWorkspace_);
			openCL.waitCommandQueue(commandQueue_);

			mappedMemory = openCL.mapMemory(commandQueue_, deviceBufferDst, CL_MAP_READ, bufferSize_);
			std::memcpy(checkBuffer, mappedMemory, bufferSize_);
			openCL.unmapMemory(commandQueue_, deviceBufferDst, mappedMemory, bufferSize_);
			clBenchmarker_.pauseTimer("cl_simplebufferprocessing_mappedmemory");
		}
		clBenchmarker_.elapsedTimer("cl_simplebufferprocessing_mappedmemory");

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
		std::cout << "cl_simplebufferprocessing_mappedmemory successful: " << isSuccessful << std::endl << std::endl;

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
	void cl_complexbufferprocessing_mappedmemory(size_t aN, bool isWarmup)
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

		void* mappedMemory = openCL.mapMemory(commandQueue_, deviceBufferSrc, CL_MAP_WRITE, bufferSize_);
		openCL.unmapMemory(commandQueue_, deviceBufferSrc, mappedMemory, bufferSize_);

		//Execute and average//
		std::cout << "Executing test: cl_complexbufferprocessing_mappedmemory" << std::endl;
		if (isWarmup)
		{
			mappedMemory = openCL.mapMemory(commandQueue_, deviceBufferSrc, CL_MAP_WRITE, bufferSize_);
			std::memcpy(mappedMemory, hostBuffer, bufferSize_);
			openCL.unmapMemory(commandQueue_, deviceBufferSrc, mappedMemory, bufferSize_);

			openCL.enqueueKernel(commandQueue_, complexBufferProcessingKernel, globalWorkspace_, localWorkspace_);
			openCL.waitCommandQueue(commandQueue_);

			mappedMemory = openCL.mapMemory(commandQueue_, deviceBufferDst, CL_MAP_READ, bufferSize_);
			std::memcpy(checkBuffer, mappedMemory, bufferSize_);
			openCL.unmapMemory(commandQueue_, deviceBufferDst, mappedMemory, bufferSize_);
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			clBenchmarker_.startTimer("cl_complexbufferprocessing_mappedmemory");
			mappedMemory = openCL.mapMemory(commandQueue_, deviceBufferSrc, CL_MAP_WRITE, bufferSize_);
			std::memcpy(mappedMemory, hostBuffer, bufferSize_);
			openCL.unmapMemory(commandQueue_, deviceBufferSrc, mappedMemory, bufferSize_);

			openCL.enqueueKernel(commandQueue_, complexBufferProcessingKernel, globalWorkspace_, localWorkspace_);
			openCL.waitCommandQueue(commandQueue_);

			mappedMemory = openCL.mapMemory(commandQueue_, deviceBufferDst, CL_MAP_READ, bufferSize_);
			std::memcpy(checkBuffer, mappedMemory, bufferSize_);
			openCL.unmapMemory(commandQueue_, deviceBufferDst, mappedMemory, bufferSize_);
			clBenchmarker_.pauseTimer("cl_complexbufferprocessing_mappedmemory");
		}
		clBenchmarker_.elapsedTimer("cl_complexbufferprocessing_mappedmemory");

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
		std::cout << "cl_complexbufferprocessing_mappedmemory successful: " << isSuccessful << std::endl << std::endl;

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
	void cl_simplebuffersynthesis_mappedmemory(size_t aN, bool isWarmup)
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

		void* mappedMemory = openCL.mapMemory(commandQueue_, deviceBufferOutput, CL_MAP_READ, bufferSize_);
		openCL.unmapMemory(commandQueue_, deviceBufferOutput, mappedMemory, bufferSize_);

		//Execute and average//
		std::cout << "Executing test: cl_simplebuffersynthesis_mappedmemory" << std::endl;
		if (isWarmup)
		{
			openCL.enqueueKernel(commandQueue_, simpleBufferSynthesisKernel, globalWorkspace_, localWorkspace_);
			openCL.waitCommandQueue(commandQueue_);
			mappedMemory = openCL.mapMemory(commandQueue_, deviceBufferOutput, CL_MAP_READ, bufferSize_);
			std::memcpy(checkBuffer, mappedMemory, bufferSize_);
			openCL.unmapMemory(commandQueue_, deviceBufferOutput, mappedMemory, bufferSize_);
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			clBenchmarker_.startTimer("cl_simplebuffersynthesis_mappedmemory");
			openCL.enqueueKernel(commandQueue_, simpleBufferSynthesisKernel, globalWorkspace_, localWorkspace_);
			openCL.waitCommandQueue(commandQueue_);
			mappedMemory = openCL.mapMemory(commandQueue_, deviceBufferOutput, CL_MAP_READ, bufferSize_);
			std::memcpy(checkBuffer, mappedMemory, bufferSize_);
			openCL.unmapMemory(commandQueue_, deviceBufferOutput, mappedMemory, bufferSize_);
			clBenchmarker_.pauseTimer("cl_simplebuffersynthesis_mappedmemory");
		}
		clBenchmarker_.elapsedTimer("cl_simplebuffersynthesis_mappedmemory");

		//Save audio to file for inspection//
		outputAudioFile("cl_simplebuffersynthesis_mappedmemory.wav", checkBuffer, bufferLength_);
		std::cout << "cl_simplebuffersynthesis_mappedmemory successful: Inspect audio log \"cl_simplebuffersynthesis_mappedmemory.wav\"" << std::endl << std::endl;

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
		OpenCL_FDTD fdtdSynth = OpenCL_FDTD(args, false);

		//Execute and average//
		std::cout << "Executing test: cl_complexbuffersynthesis_standard" << std::endl;
		if (isWarmup)
		{
			fdtdSynth.fillBuffer(bufferLength_, inBuf, outBuf);
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
	void cl_complexbuffersynthesis_mappedmemory(size_t aN, bool isWarmup)
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
		OpenCL_FDTD fdtdSynth = OpenCL_FDTD(args, true);

		//Execute and average//
		std::cout << "Executing test: cl_complexbuffersynthesis_mappedmemory" << std::endl;
		if (isWarmup)
		{
			fdtdSynth.fillBuffer(bufferLength_, inBuf, outBuf);
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			clBenchmarker_.startTimer("cl_complexbuffersynthesis_mappedmemory");

			fdtdSynth.fillBuffer(bufferLength_, inBuf, outBuf);

			clBenchmarker_.pauseTimer("cl_complexbuffersynthesis_mappedmemory");
		}
		clBenchmarker_.elapsedTimer("cl_complexbuffersynthesis_mappedmemory");

		//Save audio to file for inspection//
		for (int j = 0; j != bufferLength_; ++j)
			soundBuffer[j] = outBuf[j];
		outputAudioFile("cl_complexbuffersynthesis_mappedmemory.wav", soundBuffer, bufferLength_);
		std::cout << "cl_complexbuffersynthesis_mappedmemory successful: Inspect audio log \"cl_complexbuffersynthesis_mappedmemory.wav\"" << std::endl << std::endl;

		//Cleanup//
		delete inBuf;
		delete outBuf;
	}
	void cl_interruptedbufferprocessing_standard(size_t aN, bool isWarmup)
	{

	}
	void cl_interruptedbufferprocessing_mappedmemory(size_t aN, bool isWarmup)
	{

	}

	void cl_unidirectional_baseline(size_t aFrameRate, bool isWarmup)
	{

	}
	void cl_unidirectional_processing(size_t aFrameRate, bool isWarmup)
	{

	}
	void cl_bidirectional_baseline(size_t aFrameRate, bool isWarmup)
	{

	}
	void cl_bidirectional_processing(size_t aFrameRate, bool isWarmup)
	{

	}

	
	void cl_011_complexbuffersynthesis(size_t aN, bool isWarmup)
	{
		//Test preperation//
		uint32_t numSamplesComputed = 0;
		float* inBuf = new float[bufferLength_];
		float* outBuf = new float[bufferLength_];
		for (size_t i = 0; i != bufferLength_; ++i)
		{
			//Create initial impulse as excitation//
			if(i < bufferLength_ / 1000)
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
		OpenCL_FDTD fdtdSynth = OpenCL_FDTD(args, true);

		//Execute and average//
		std::cout << "Executing test: cl_011_complexbuffersynthesis" << std::endl;
		if (isWarmup)
		{
			fdtdSynth.fillBuffer(bufferLength_, inBuf, outBuf);
		}
		for (uint32_t i = 0; i != aN; ++i)
		{
			clBenchmarker_.startTimer("cl_011_complexbuffersynthesis");

			fdtdSynth.fillBuffer(bufferLength_, inBuf, outBuf);

			clBenchmarker_.pauseTimer("cl_011_complexbuffersynthesis");
		}
		clBenchmarker_.elapsedTimer("cl_011_complexbuffersynthesis");

		//Save audio to file for inspection//
		for (int j = 0; j != bufferLength_; ++j)
			soundBuffer[j] = outBuf[j];
		outputAudioFile("cl_011_complexbuffersynthesis.wav", soundBuffer, bufferLength_);
		std::cout << "cl_011_complexbuffersynthesis successful: Inspect audio log \"cl_011_complexbuffersynthesis.wav\"" << std::endl << std::endl;

		//Cleanup//
		delete inBuf;
		delete outBuf;
	}

	//void cl_012_interruptedbufferprocessing(size_t aN, bool isWarmup)
	//{
	//	//Test preperation//
	//	//openCL.createKernelProgram("resources/kernels/GPU_Overhead_Benchmarks.cl", "");
	//	//openCL.createKernel("cl_012_interruptedbufferprocessing");
	//	setLocalWorkspace(bufferLength_);

	//	void* mappedMemory;

	//	bool isBufferCorrect = true;		//Variable indicating if buffer requires recalculating.
	//	std::default_random_engine generator;
	//	std::uniform_int_distribution<int> distribution(1, 10);
	//	float* srcMemoryBuffer = new float[bufferLength_];
	//	for (size_t i = 0; i != bufferLength_; ++i)
	//		srcMemoryBuffer[i] = 42.0;
	//	float* dstMemoryBuffer = new float[bufferLength_];
	//	for (size_t i = 0; i != bufferLength_; ++i)
	//		dstMemoryBuffer[i] = 0.0;

	//	openCL.createBuffer("inputBuffer", CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, bufferSize_);
	//	openCL.createBuffer("outputBuffer", CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, bufferSize_);

	//	openCL.setKernelArgument("cl_012_interruptedbufferprocessing", "inputBuffer", 0, sizeof(cl_mem));
	//	openCL.setKernelArgument("cl_012_interruptedbufferprocessing", "outputBuffer", 1, sizeof(cl_mem));

	//	//Execute and average//
	//	std::cout << "Executing test: cl_012_interruptedbufferprocessing" << std::endl;
	//	if (isWarmup)
	//	{
	//		mappedMemory = openCL.mapMemory("inputBuffer", bufferSize_);
	//		memcpy(mappedMemory, srcMemoryBuffer, bufferSize_);
	//		openCL.unmapMemory("inputBuffer", mappedMemory, bufferSize_);

	//		openCL.enqueueKernel("cl_012_interruptedbufferprocessing");

	//		mappedMemory = openCL.mapMemory("outputBuffer", bufferSize_);
	//		memcpy(dstMemoryBuffer, mappedMemory, bufferSize_);
	//		openCL.unmapMemory("outputBuffer", mappedMemory, bufferSize_);
	//	}
	//	for (int32_t i = 0; i != aN; ++i)
	//	{
	//		clBenchmarker_.startTimer("cl_012_interruptedbufferprocessing");

	//		mappedMemory = openCL.mapMemory("inputBuffer", bufferSize_);
	//		memcpy(mappedMemory, srcMemoryBuffer, bufferSize_);
	//		openCL.unmapMemory("inputBuffer", mappedMemory, bufferSize_);

	//		openCL.enqueueKernel("cl_012_interruptedbufferprocessing");
	//		openCL.waitKernel();

	//		mappedMemory = openCL.mapMemory("outputBuffer", bufferSize_);
	//		memcpy(dstMemoryBuffer, mappedMemory, bufferSize_);
	//		openCL.unmapMemory("outputBuffer", mappedMemory, bufferSize_);
	//		openCL.waitKernel();

	//		clBenchmarker_.pauseTimer("cl_012_interruptedbufferprocessing");

	//		int randomChance = distribution(generator);
	//		if (randomChance > 5)
	//			--i;
	//			
	//	}
	//	clBenchmarker_.elapsedTimer("cl_012_interruptedbufferprocessing");

	//	//Check contents//
	//	bool isSuccessful = true;
	//	for (uint32_t i = 0; i != bufferLength_; ++i)
	//	{
	//		//Calculate simple attenuation on CPU to compare//
	//		srcMemoryBuffer[i] = srcMemoryBuffer[i] * 0.5;
	//		if (srcMemoryBuffer[i] != dstMemoryBuffer[i])
	//		{
	//			isSuccessful = false;
	//			break;
	//		}
	//	}
	//	std::cout << "cl_012_interruptedbufferprocessing successful: " << isSuccessful << std::endl << std::endl;

	//	//Cleanup//
	//	openCL.deleteBuffer("inputBuffer");
	//	openCL.deleteBuffer("outputBuffer");
	//	delete srcMemoryBuffer;
	//	delete dstMemoryBuffer;
	//}

	//void unidirectionalBaseline(size_t aN, size_t sampleRate)
	//{
	//	//Test preperation//
	//	datatype* hostBuffer = new datatype[MEGA_BYTE / sizeof(float)];
	//	for (size_t i = 0; i != MEGA_BYTE / sizeof(float); ++i)
	//		hostBuffer[i] = 42.0;
	//	datatype* checkBuffer = new datatype[MEGA_BYTE / sizeof(float)];

	//	openCL.createBuffer("deviceBuffer", CL_MEM_READ_WRITE, MEGA_BYTE);

	//	uint32_t numSamplesComputed = 0;
	//	for (size_t i = 0; i != bufferSizesLength - 2; ++i)
	//	{
	//		uint32_t currentBufferSize = bufferSizes[i];

	//		openCL.enqueueKernel("cl_000_nullKernel");
	//		openCL.readBuffer("deviceBuffer", currentBufferSize * sizeof(float), checkBuffer);
	//		clBenchmarker_.startTimer("totalTime");
	//		while (numSamplesComputed < sampleRate)
	//		{

	//			clBenchmarker_.startTimer("bufferTime");
	//			openCL.enqueueKernel("cl_000_nullKernel");
	//			openCL.readBuffer("deviceBuffer", currentBufferSize * sizeof(float), checkBuffer);
	//			clBenchmarker_.pauseTimer("bufferTime");

	//			numSamplesComputed += currentBufferSize;
	//		}
	//		clBenchmarker_.pauseTimer("totalTime");
	//		clBenchmarker_.elapsedTimer("totalTime");
	//		clBenchmarker_.elapsedTimer("bufferTime");

	//		numSamplesComputed = 0;
	//	}

	//	openCL.deleteBuffer("deviceBuffer");
	//	delete hostBuffer;
	//	delete checkBuffer;
	//}
	//void runUnidirectionalBenchmarks(size_t aN, size_t sampleRate)
	//{
	//	unidirectionalBaseline(aN, sampleRate);
	//}
	//void bidirectionalComplexSynthesis(size_t aN, size_t sampleRate)
	//{
	//	uint32_t numSamplesComputed = 0;
	//	for (size_t i = 0; i != bufferSizesLength-2; ++i)
	//	{
	//		uint32_t currentBufferSize = bufferSizes[i];

	//		OpenCL_FDTD_Arguments args;
	//		args.isDebug = false;
	//		args.isBenchmark = false;
	//		args.modelWidth = 64;
	//		args.modelHeight = 64;
	//		args.propagationFactor = 0.06;
	//		args.dampingCoefficient = 0.0005;
	//		args.boundaryGain = 0.5;
	//		args.listenerPosition[0] = 8;
	//		args.listenerPosition[1] = 8;
	//		args.excitationPosition[0] = 32;
	//		args.excitationPosition[1] = 32;
	//		args.workGroupDimensions[0] = 16;
	//		args.workGroupDimensions[1] = 16;
	//		args.bufferSize = currentBufferSize;
	//		args.kernelSource = "resources/kernels/fdtdGlobal.cl";
	//		OpenCL_FDTD fdtdSynth = OpenCL_FDTD(args);
	//		//new(&fdtdSynth) OpenCL_FDTD(args);

	//		//fdtdSynth.setBufferSize(currentBufferSize);
	//		//fdtdSynth.init();

	//		uint32_t numSamplesComputed = 0;
	//		float* inBuf = new float[currentBufferSize];
	//		float* outBuf = new float[currentBufferSize];
	//		for (size_t i = 0; i != currentBufferSize; ++i)
	//			inBuf[i] = 0.5;

	//		float* soundBuffer = new float[sampleRate*2];

	//		clBenchmarker_.startTimer("totalTime");
	//		while (numSamplesComputed < sampleRate)
	//		{

	//			clBenchmarker_.startTimer("bufferTime");
	//			fdtdSynth.fillBuffer(currentBufferSize, inBuf, outBuf);
	//			clBenchmarker_.pauseTimer("bufferTime");

	//			//Log audio for inspection if necessary//
	//			for (int j = 0; j != currentBufferSize; ++j)
	//				soundBuffer[numSamplesComputed + j] = outBuf[j];

	//			numSamplesComputed += currentBufferSize;
	//		}
	//		clBenchmarker_.pauseTimer("totalTime");
	//		clBenchmarker_.elapsedTimer("totalTime");
	//		clBenchmarker_.elapsedTimer("bufferTime");

	//		//Save audio to file for inspection//
	//		outputAudioFile("bidirectionalComplexSynthesis.wav", soundBuffer, sampleRate);
	//		std::cout << "bidirectionalComplexSynthesis successful: Inspect audio log \"bidirectionalComplexSynthesis.wav\"" << std::endl << std::endl;

	//		numSamplesComputed = 0;

	//		delete inBuf;
	//		delete outBuf;
	//	}
	//}
	//void runBidirectionalBenchmarks(size_t aN, size_t sampleRate)
	//{
	//	//Test preperation//
	//	datatype* hostBuffer = new datatype[MEGA_BYTE / sizeof(float)];
	//	for (size_t i = 0; i != MEGA_BYTE / sizeof(float); ++i)
	//		hostBuffer[i] = 42.0;
	//	datatype* checkBuffer = new datatype[MEGA_BYTE / sizeof(float)];

	//	openCL.createBuffer("deviceBuffer", CL_MEM_READ_WRITE, MEGA_BYTE);

	//	uint32_t numSamplesComputed = 0;
	//	for (size_t i = 0; i != bufferSizesLength - 2; ++i)
	//	{
	//		uint32_t currentBufferSize = bufferSizes[i];

	//		openCL.writeBuffer("deviceBuffer", currentBufferSize * sizeof(float), hostBuffer);
	//		openCL.enqueueKernel("cl_000_nullKernel");
	//		openCL.readBuffer("deviceBuffer", currentBufferSize * sizeof(float), checkBuffer);
	//		clBenchmarker_.startTimer("totalTime");
	//		while (numSamplesComputed < sampleRate)
	//		{

	//			clBenchmarker_.startTimer("bufferTime");
	//			openCL.writeBuffer("deviceBuffer", currentBufferSize * sizeof(float), hostBuffer);
	//			openCL.enqueueKernel("cl_000_nullKernel");
	//			openCL.readBuffer("deviceBuffer", currentBufferSize * sizeof(float), checkBuffer);
	//			clBenchmarker_.pauseTimer("bufferTime");

	//			numSamplesComputed += currentBufferSize;
	//		}
	//		clBenchmarker_.pauseTimer("totalTime");
	//		clBenchmarker_.elapsedTimer("totalTime");
	//		clBenchmarker_.elapsedTimer("bufferTime");

	//		numSamplesComputed = 0;
	//	}

	//	openCL.deleteBuffer("deviceBuffer");
	//	delete hostBuffer;
	//	delete checkBuffer;
	//}

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