#ifndef OPENCL_FDTD_HPP
#define OPENCL_FDTD_HPP

#include <vector>
#include <fstream>

//#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include <CL/cl2.hpp>
//#include <CL/cl.hpp>

#include "FDTD_Grid.hpp"
#include "Buffer.hpp"
#include "DSP.hpp"

struct OpenCL_FDTD_Arguments
{
	bool isDebug;
	bool isBenchmark;
	unsigned int modelWidth;
	unsigned int modelHeight;
	float boundaryGain;
	unsigned int bufferSize;
	float propagationFactor;
	float dampingCoefficient;
	unsigned int listenerPosition[2];
	unsigned int excitationPosition[2];
	unsigned int workGroupDimensions[2];
	std::string kernelSource;
};

class OpenCL_FDTD : public DSP
{
private:
	//Print Debug Information//
	bool isDebug_;

	//Calculate + Print Benchmarking//
	bool isBenchmark_;
	cl::Event kernelBenchmark_;
	cl_ulong kernelComputeStartTime_;
	cl_ulong kernelComputeEndTime_;
	cl_ulong kernelComputeElapsedTime_;
	cl_ulong kernelOverheadStartTime_;
	cl_ulong kernelOverheadEndTime_;
	cl_ulong kernelOverheadElapsedTime_;

	//Model Dimensions//
	typedef float base_type_;
	const int modelWidth_;
	const int modelHeight_;
	const int gridElements_;
	const int gridByteSize_;
	int listenerPosition_[2];
	int excitationPosition_[2];
	Model model_;

	//Global/Uniform material properties - Later will have these as grids for variable material//
	typedef float property_type_;
	const property_type_ propagationFactor_;
	const property_type_ dampingCoefficient_;

	//Output and excitations//
	unsigned int bufferSize_;
	Buffer<base_type_> output_;
	Buffer<base_type_> excitation_;

	//OpenCL Variables//
	unsigned int globalWorkSpaceX_;
	unsigned int globalWorkSpaceY_;
	unsigned int localWorkSpaceX_;
	unsigned int localWorkSpaceY_;

	//OpenCL objects//
	cl::Context context_;
	cl::CommandQueue commandQueue_;
	cl::Kernel ftdtKernel_;
	cl::NDRange globalws_;
	cl::NDRange localws_;

	//GPU Memory Buffers//
	cl::Buffer nMinusOnePressureBuffer_;
	cl::Buffer nPressureBuffer_;
	cl::Buffer nPlusOnePressureBuffer_;
	cl::Buffer boundaryGridBuffer_;
	cl::Buffer outputBuffer_;
	cl::Buffer excitationBuffer_;
	cl::Buffer localBuffer_;

	int bufferRotationIndex_;

	const std::string path_;

	bool isOptimized;

public:
	OpenCL_FDTD() :
		modelWidth_(0),
		modelHeight_(0),
		gridElements_(modelWidth_*modelHeight_),
		gridByteSize_(gridElements_ * sizeof(float)),
		propagationFactor_(0.0),
		dampingCoefficient_(0.0),
		model_(0, 0, 0.0),
		output_(0),
		excitation_(0),
		bufferSize_(0),
		bufferRotationIndex_(0),
		isOptimized(0)
	{}
	OpenCL_FDTD(int aModelWidth, int aModelHeight, float aBoundaryGain, const int aBufferSize, property_type_ aPropagationFactor, property_type_ aDampingCoefficient, unsigned int listenerPosition[2], unsigned int excitationPosition[2], unsigned int workGroupDimensions[2], const char* aKernelSource, bool aIsDebug, bool aIsBenchmark, bool aIsOptimized) :
		modelWidth_(aModelWidth),
		modelHeight_(aModelHeight),
		gridElements_(modelWidth_*modelHeight_),
		gridByteSize_(gridElements_ * sizeof(float)),
		propagationFactor_(aPropagationFactor),
		dampingCoefficient_(aDampingCoefficient),
		model_(aModelWidth, aModelHeight, aBoundaryGain),
		bufferSize_(aBufferSize),
		output_(bufferSize_),
		excitation_(bufferSize_),
		path_(aKernelSource),
		isDebug_(aIsDebug),
		isBenchmark_(aIsBenchmark),
		bufferRotationIndex_(0),
		isOptimized(aIsOptimized)
	{
		listenerPosition_[0] = listenerPosition[0];
		listenerPosition_[1] = listenerPosition[1];
		excitationPosition_[0] = excitationPosition[0];
		excitationPosition_[1] = excitationPosition[1];
		globalWorkSpaceX_ = modelWidth_;
		globalWorkSpaceY_ = modelHeight_;
		localWorkSpaceX_ = workGroupDimensions[0];
		localWorkSpaceY_ = workGroupDimensions[1];
		kernelComputeElapsedTime_ = 0;
		kernelOverheadStartTime_ = 0;
		if (isOptimized)
			initOptimized();
		else
			initStandard();
	}
	OpenCL_FDTD(OpenCL_FDTD_Arguments args, bool aIsOptimized) :
		modelWidth_(args.modelWidth),
		modelHeight_(args.modelHeight),
		gridElements_(modelWidth_*modelHeight_),
		gridByteSize_(gridElements_ * sizeof(float)),
		propagationFactor_(args.propagationFactor),
		dampingCoefficient_(args.dampingCoefficient),
		model_(modelWidth_, modelHeight_, args.boundaryGain),
		output_(args.bufferSize),
		excitation_(args.bufferSize),
		path_(args.kernelSource),
		isDebug_(args.isDebug),
		isBenchmark_(args.isBenchmark),
		bufferRotationIndex_(0),
		isOptimized(aIsOptimized)
	{
		listenerPosition_[0] = args.listenerPosition[0];
		listenerPosition_[1] = args.listenerPosition[1];
		excitationPosition_[0] = args.excitationPosition[0];
		excitationPosition_[1] = args.excitationPosition[1];
		globalWorkSpaceX_ = modelWidth_;
		globalWorkSpaceY_ = modelHeight_;
		localWorkSpaceX_ = args.workGroupDimensions[0];
		localWorkSpaceY_ = args.workGroupDimensions[1];
		//Set size of NDRange and workgroups//
		globalws_ = cl::NDRange(modelWidth_, modelHeight_);
		localws_ = cl::NDRange(localWorkSpaceX_, localWorkSpaceY_);
		kernelComputeElapsedTime_ = 0;
		kernelOverheadStartTime_ = 0;

		if (isDebug_)
			printAvailableDevices();

		if (isOptimized)
			initOptimized();
		else
			initStandard();
	}
	~OpenCL_FDTD()
	{
		//////////////////////////////////////////////////////
		//Cleanup - Typically do this in reverse of creation//
		//////////////////////////////////////////////////////
	}
	void initStandard()
	{
		int errorStatus = 0;

		///////////////////////////////////////////////
		//This code executes on the host device - CPU//
		///////////////////////////////////////////////

		//Initalise grid with boundaries//
		model_.setListenerPosition(listenerPosition_[0], listenerPosition_[1]);
		model_.setExcitationPosition(excitationPosition_[0], excitationPosition_[1]);

		/////////////////////////////////////
		//Step 1: Set up OpenCL environment//
		/////////////////////////////////////

		//Discover platforms//
		std::vector <cl::Platform> platforms;
		cl::Platform::get(&platforms);

		//Create contex properties for first platform//
		cl_context_properties contextProperties[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[1])(), 0 };	//Need to specify platform 3 for dedicated graphics - Harri Laptop.

		//Create context context using platform for GPU device//
		context_ = cl::Context(CL_DEVICE_TYPE_ALL, contextProperties);

		//Get device list from context//
		std::vector<cl::Device> devices = context_.getInfo<CL_CONTEXT_DEVICES>();

		//Create command queue for first device - Profiling enabled//
		commandQueue_ = cl::CommandQueue(context_, devices[0], CL_QUEUE_PROFILING_ENABLE, &errorStatus);	//Need to specify device 1[0] of platform 3[2] for dedicated graphics - Harri Laptop.
		if (errorStatus)
			std::cout << "ERROR creating command queue for device. Status code: " << errorStatus << std::endl;

		////////////////////////////////////////////////////////////////
		//Step 2: Create and populate memory data structures - Buffers//
		////////////////////////////////////////////////////////////////

		//Create input and output buffer for grid points//
		nMinusOnePressureBuffer_ = cl::Buffer(context_, CL_MEM_READ_ONLY, gridByteSize_);
		nPressureBuffer_ = cl::Buffer(context_, CL_MEM_READ_ONLY, gridByteSize_);
		nPlusOnePressureBuffer_ = cl::Buffer(context_, CL_MEM_WRITE_ONLY, gridByteSize_);
		boundaryGridBuffer_ = cl::Buffer(context_, CL_MEM_READ_ONLY, gridByteSize_);
		outputBuffer_ = cl::Buffer(context_, CL_MEM_WRITE_ONLY, output_.bufferSize_);
		excitationBuffer_ = cl::Buffer(context_, CL_MEM_READ_ONLY, excitation_.bufferSize_);

		//Copy data to newly created device's memory//
		commandQueue_.enqueueWriteBuffer(nMinusOnePressureBuffer_, CL_TRUE, 0, gridByteSize_, model_.getNMinusOneGridBuffer());
		commandQueue_.enqueueWriteBuffer(nPressureBuffer_, CL_TRUE, 0, gridByteSize_, model_.getNGridBuffer());
		commandQueue_.enqueueWriteBuffer(nPlusOnePressureBuffer_, CL_TRUE, 0, gridByteSize_, model_.getNPlusOneGridBuffer());
		commandQueue_.enqueueWriteBuffer(boundaryGridBuffer_, CL_TRUE, 0, gridByteSize_, model_.getBoundaryGridBuffer());

		//////////////////////////////////
		//Step 3: Compile Kernel program//
		//////////////////////////////////

		//Read in program source - I'm going to go for reading from compiled object instead//
		std::ifstream sourceFileName(path_.c_str());
		std::string sourceFile(std::istreambuf_iterator<char>(sourceFileName), (std::istreambuf_iterator<char>()));

		//Create program source object from std::string source code//
		std::vector<std::string> programSources;
		programSources.push_back(sourceFile);
		cl::Program::Sources ftdtSource(programSources);	//Apparently this takes a vector of strings as the program source.

		//Create program from source code//
		cl::Program ftdtProgram(context_, ftdtSource, &errorStatus);
		if (errorStatus)
			std::cout << "ERROR creating program from source. Status code: " << errorStatus << std::endl;

		//Build program//
		ftdtProgram.build();

		//Create kernel program on device//
		ftdtKernel_ = cl::Kernel(ftdtProgram, "ftdtCompute", &errorStatus);
		if (errorStatus)
			std::cout << "ERROR creating kernel. Status code: " << errorStatus << std::endl;

		//Set static kernel arguments//
		ftdtKernel_.setArg(0, sizeof(cl_mem), &nMinusOnePressureBuffer_);
		ftdtKernel_.setArg(1, sizeof(cl_mem), &nPressureBuffer_);
		ftdtKernel_.setArg(2, sizeof(cl_mem), &nPlusOnePressureBuffer_);
		ftdtKernel_.setArg(3, sizeof(cl_mem), &boundaryGridBuffer_);
		ftdtKernel_.setArg(5, sizeof(cl_mem), &outputBuffer_);

		//unsigned int localWorkspaceSize = localWorkSpaceX_ * localWorkSpaceY_ * sizeof(float);
		//ftdtKernel_.setArg(12, localWorkspaceSize, NULL);	//To allocate local memory dynamically, must be given a size here.

		std::cout << "\t\tDevice Name: " << devices[0].getInfo<CL_DEVICE_NAME>() << std::endl;
	}
	void initOptimized()
	{
		int errorStatus = 0;

		///////////////////////////////////////////////
		//This code executes on the host device - CPU//
		///////////////////////////////////////////////

		//Initalise grid with boundaries//
		model_.setListenerPosition(listenerPosition_[0], listenerPosition_[1]);
		model_.setExcitationPosition(excitationPosition_[0], excitationPosition_[1]);

		/////////////////////////////////////
		//Step 1: Set up OpenCL environment//
		/////////////////////////////////////

		//Discover platforms//
		std::vector <cl::Platform> platforms;
		cl::Platform::get(&platforms);

		//Create contex properties for first platform//
		cl_context_properties contextProperties[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[1])(), 0 };	//Need to specify platform 3 for dedicated graphics - Harri Laptop.

		//Create context context using platform for GPU device//
		context_ = cl::Context(CL_DEVICE_TYPE_ALL, contextProperties);

		//Get device list from context//
		std::vector<cl::Device> devices = context_.getInfo<CL_CONTEXT_DEVICES>();

		//Create command queue for first device - Profiling enabled//
		commandQueue_ = cl::CommandQueue(context_, devices[0], CL_QUEUE_PROFILING_ENABLE, &errorStatus);	//Need to specify device 1[0] of platform 3[2] for dedicated graphics - Harri Laptop.
		if (errorStatus)
			std::cout << "ERROR creating command queue for device. Status code: " << errorStatus << std::endl;

		////////////////////////////////////////////////////////////////
		//Step 2: Create and populate memory data structures - Buffers//
		////////////////////////////////////////////////////////////////

		//Create input and output buffer for grid points//
		nMinusOnePressureBuffer_ = cl::Buffer(context_, CL_MEM_READ_ONLY, gridByteSize_);
		nPressureBuffer_ = cl::Buffer(context_, CL_MEM_READ_ONLY, gridByteSize_);
		nPlusOnePressureBuffer_ = cl::Buffer(context_, CL_MEM_WRITE_ONLY, gridByteSize_);
		boundaryGridBuffer_ = cl::Buffer(context_, CL_MEM_READ_ONLY, gridByteSize_);
		outputBuffer_ = cl::Buffer(context_, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, output_.bufferSize_);
		excitationBuffer_ = cl::Buffer(context_, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, excitation_.bufferSize_);

		//Copy data to newly created device's memory//
		commandQueue_.enqueueWriteBuffer(nMinusOnePressureBuffer_, CL_TRUE, 0, gridByteSize_, model_.getNMinusOneGridBuffer());
		commandQueue_.enqueueWriteBuffer(nPressureBuffer_, CL_TRUE, 0, gridByteSize_, model_.getNGridBuffer());
		commandQueue_.enqueueWriteBuffer(nPlusOnePressureBuffer_, CL_TRUE, 0, gridByteSize_, model_.getNPlusOneGridBuffer());
		commandQueue_.enqueueWriteBuffer(boundaryGridBuffer_, CL_TRUE, 0, gridByteSize_, model_.getBoundaryGridBuffer());

		//////////////////////////////////
		//Step 3: Compile Kernel program//
		//////////////////////////////////

		//Read in program source - I'm going to go for reading from compiled object instead//
		std::ifstream sourceFileName(path_.c_str());
		std::string sourceFile(std::istreambuf_iterator<char>(sourceFileName), (std::istreambuf_iterator<char>()));

		//Create program source object from std::string source code//
		std::vector<std::string> programSources;
		programSources.push_back(sourceFile);
		cl::Program::Sources ftdtSource(programSources);	//Apparently this takes a vector of strings as the program source.

		//Create program from source code//
		cl::Program ftdtProgram(context_, ftdtSource, &errorStatus);
		if (errorStatus)
			std::cout << "ERROR creating program from source. Status code: " << errorStatus << std::endl;

		//Build program//
		ftdtProgram.build();

		//Create kernel program on device//
		ftdtKernel_ = cl::Kernel(ftdtProgram, "ftdtCompute", &errorStatus);
		if (errorStatus)
			std::cout << "ERROR creating kernel. Status code: " << errorStatus << std::endl;

		//Set static kernel arguments//
		ftdtKernel_.setArg(0, sizeof(cl_mem), &nMinusOnePressureBuffer_);
		ftdtKernel_.setArg(1, sizeof(cl_mem), &nPressureBuffer_);
		ftdtKernel_.setArg(2, sizeof(cl_mem), &nPlusOnePressureBuffer_);
		ftdtKernel_.setArg(3, sizeof(cl_mem), &boundaryGridBuffer_);
		ftdtKernel_.setArg(5, sizeof(cl_mem), &outputBuffer_);

		//unsigned int localWorkspaceSize = localWorkSpaceX_ * localWorkSpaceY_ * sizeof(float);
		//ftdtKernel_.setArg(12, localWorkspaceSize, NULL);	//To allocate local memory dynamically, must be given a size here.

		mapMemoryOne = commandQueue_.enqueueMapBuffer(excitationBuffer_, TRUE, CL_MAP_WRITE, 0, excitation_.bufferSize_, NULL, NULL);
		mapMemoryTwo = commandQueue_.enqueueMapBuffer(outputBuffer_, TRUE, CL_MAP_WRITE, 0, excitation_.bufferSize_, NULL, NULL);
	}
	float step()
	{
		commandQueue_.enqueueNDRangeKernel(ftdtKernel_, cl::NullRange/*globaloffset*/, globalws_, localws_, NULL, &kernelBenchmark_);

		//Record benchmark time//
		if (isBenchmark_)
		{
			kernelBenchmark_.wait();
			kernelComputeElapsedTime_ += kernelBenchmark_.getProfilingInfo<CL_PROFILING_COMMAND_END>() - kernelBenchmark_.getProfilingInfo<CL_PROFILING_COMMAND_START>();
			kernelOverheadElapsedTime_ += kernelBenchmark_.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>() - kernelBenchmark_.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
		}

		output_.bufferIndex_++;
		excitation_.bufferIndex_++;
		bufferRotationIndex_ = (bufferRotationIndex_ + 1) % 3;

		return 0.0;
	}
	void* mapMemoryOne;
	void* mapMemoryTwo;
	bool compute(unsigned long frames, float* inbuf, float* outbuf)
	{
		if (isOptimized)
		{
			//Set dynamic kernel arguments//
			int listenerPositionArg = model_.getListenerPosition();
			int excitationPositionArg = model_.getExcitationPosition();
			updateDynamicVariables(propagationFactor_, dampingCoefficient_, listenerPositionArg, excitationPositionArg);

			//Load excitation samples into GPU//

			//commandQueue_.enqueueWriteBuffer(excitationBuffer_, CL_TRUE, 0, excitation_.bufferSize_, inbuf);
			//mapMemoryOne = commandQueue_.enqueueMapBuffer(excitationBuffer_, TRUE, NULL, 0, excitation_.bufferSize_, NULL, NULL);
			memcpy(mapMemoryOne, inbuf, excitation_.bufferSize_);
			//commandQueue_.enqueueUnmapMemObject(excitationBuffer_, mapMemoryOne, NULL, NULL);

			ftdtKernel_.setArg(6, sizeof(cl_mem), &excitationBuffer_);

			//Calculate buffer size of synthesizer output samples//
			for (unsigned int i = 0; i != frames; ++i)
			{
				//Increments kernel indices//
				ftdtKernel_.setArg(4, sizeof(int), &output_.bufferIndex_);
				ftdtKernel_.setArg(11, sizeof(int), &bufferRotationIndex_);

				step();
			}

			output_.resetIndex();
			excitation_.resetIndex();

			//commandQueue_.enqueueReadBuffer(outputBuffer_, CL_TRUE, 0, output_.bufferSize_, output_.buffer_);
			//mapMemoryTwo = commandQueue_.enqueueMapBuffer(outputBuffer_, TRUE, NULL, 0, excitation_.bufferSize_, NULL, NULL);
			memcpy(output_.buffer_, mapMemoryTwo, output_.bufferSize_);
			//commandQueue_.enqueueUnmapMemObject(outputBuffer_, mapMemoryTwo, NULL, NULL);
			for (int k = 0; k != frames; ++k)
				outbuf[k] = output_[k];

			return true;
		}
		else
		{
			//Set dynamic kernel arguments//
			int listenerPositionArg = model_.getListenerPosition();
			int excitationPositionArg = model_.getExcitationPosition();
			updateDynamicVariables(propagationFactor_, dampingCoefficient_, listenerPositionArg, excitationPositionArg);

			//Load excitation samples into GPU//
			commandQueue_.enqueueWriteBuffer(excitationBuffer_, CL_TRUE, 0, excitation_.bufferSize_, inbuf);
			//commandQueue_.finish();

			ftdtKernel_.setArg(6, sizeof(cl_mem), &excitationBuffer_);

			//Calculate buffer size of synthesizer output samples//
			for (unsigned int i = 0; i != frames; ++i)
			{
				//Increments kernel indices//
				ftdtKernel_.setArg(4, sizeof(int), &output_.bufferIndex_);
				ftdtKernel_.setArg(11, sizeof(int), &bufferRotationIndex_);

				step();
			}

			output_.resetIndex();
			excitation_.resetIndex();

			commandQueue_.enqueueReadBuffer(outputBuffer_, CL_TRUE, 0, output_.bufferSize_, output_.buffer_);
			//commandQueue_.finish();
			for (int k = 0; k != frames; ++k)
				outbuf[k] = output_[k];

			return true;
		}
		return false;
	}
	void updateDynamicVariables(property_type_ aPropagationFactor, property_type_ aDampingFactor, unsigned int aListenerPosition, unsigned int aExcitationPosition)
	{
		//Sets all dynamic varaiables - Should be set with setters//
		ftdtKernel_.setArg(7, sizeof(int), &aListenerPosition);
		ftdtKernel_.setArg(8, sizeof(int), &aExcitationPosition);
		ftdtKernel_.setArg(9, sizeof(float), &aPropagationFactor);
		ftdtKernel_.setArg(10, sizeof(float), &aDampingFactor);
	}
	void setBufferSize(uint32_t aBufferSize)
	{
		bufferSize_ = aBufferSize;
	}

	void printKernelBenchmarking()
	{
		if (isBenchmark_)
		{
			std::cout << "OpenCL kernel total compute execution time: " << (double)kernelComputeElapsedTime_ / 1000000000.0 << "seconds" << std::endl;
			std::cout << "OpenCL kernel total compute execution time: " << (double)kernelComputeElapsedTime_ / 1000000.0 << "milliseconds" << std::endl;
			std::cout << "OpenCL kernel total compute execution time: " << (double)kernelComputeElapsedTime_ / 1000.0 << "microseconds" << std::endl;

			std::cout << std::endl;

			std::cout << "OpenCL kernel total overhead execution time: " << (double)kernelOverheadElapsedTime_ / 1000000000.0 << "seconds" << std::endl;
			std::cout << "OpenCL kernel total overhead execution time: " << (double)kernelOverheadElapsedTime_ / 1000000.0 << "milliseconds" << std::endl;
			std::cout << "OpenCL kernel total overhead execution time: " << (double)kernelOverheadElapsedTime_ / 1000.0 << "microseconds" << std::endl;

			std::cout << std::endl;
		}
	}

	void printAvailableDevices()
	{
		cl::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);

		//Print all available devices//
		int platform_id = 0;
		std::cout << "Number of Platforms: " << platforms.size() << std::endl << std::endl;
		for (cl::vector<cl::Platform>::iterator it = platforms.begin(); it != platforms.end(); ++it)
		{
			cl::Platform platform(*it);

			std::cout << "Platform ID: " << platform_id++ << std::endl;
			std::cout << "Platform Name: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
			std::cout << "Platform Vendor: " << platform.getInfo<CL_PLATFORM_VENDOR>() << std::endl;

			cl::vector<cl::Device> devices;
			platform.getDevices(CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, &devices);

			int device_id = 0;
			for (cl::vector<cl::Device>::iterator it2 = devices.begin(); it2 != devices.end(); ++it2)
			{
				cl::Device device(*it2);

				std::cout << "\tDevice " << device_id++ << ": " << std::endl;
				std::cout << "\t\tDevice Name: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
				std::cout << "\t\tDevice Type: " << device.getInfo<CL_DEVICE_TYPE>();
				std::cout << " (GPU: " << CL_DEVICE_TYPE_GPU << ", CPU: " << CL_DEVICE_TYPE_CPU << ")" << std::endl;
				std::cout << "\t\tDevice Vendor: " << device.getInfo<CL_DEVICE_VENDOR>() << std::endl;
				std::cout << "\t\tDevice Max Compute Units: " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
				std::cout << "\t\tDevice Global Memory: " << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() << std::endl;
				std::cout << "\t\tDevice Max Clock Frequency: " << device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << std::endl;
				std::cout << "\t\tDevice Max Allocateable Memory: " << device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>() << std::endl;
				std::cout << "\t\tDevice Local Memory: " << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << std::endl;
				std::cout << "\t\tDevice Available: " << device.getInfo< CL_DEVICE_AVAILABLE>() << std::endl;

				//If an AMD platform//
				if (strstr(platform.getInfo<CL_PLATFORM_NAME>().c_str(), "AMD"))
				{
					std::cout << "\tAMD Specific:" << std::endl;
					//std::cout << "\t\tAMD Wavefront size: " << device.getInfo<CL_DEVICE_WAVEFRONT_WIDTH_AMD>() << std::endl;
				}
			}
			std::cout << std::endl;
		}
	}

	//Virtual Functions Defined//
	bool fillBuffer(unsigned long frames, void* inbuf, void* outbuf) override
	{
		float* input = ((float*)(inbuf));
		float* output = (static_cast<float*>(outbuf));
		return compute(frames, input, output);
	}

	float sample() const {
		return 0.0;
	}

	float nextSample() override
	{
		//step(mu_, rho_);
		return sample();
	}

	int getNumOutputs() override
	{
		return 1;
	}

	int getNumInputs() override
	{
		return 0;
	}
};

#endif