#pragma once
#ifndef OPENCL_WRAPPER_H
#define OPENCL_WRAPPER_H

class OpenCL_Wrapper
{
private:
	static const uint32_t GIGA_BYTE = 1024 * 1024 * 1024;
	static const uint32_t MEGA_BYTE = 1024 * 1024;
	static const uint32_t KILO_BYTE = 1024;

	//OpenCL objects//
	cl::Context context_;
	cl::Device device_;
	cl::CommandQueue commandQueue_;
	cl::Program kernelProgram_;
	std::map<std::string, cl::Kernel*> kernels_;
	cl::Event kernelBenchmark_;
	cl::NDRange globalws_;
	cl::NDRange localws_;

	std::map<std::string, cl::Buffer*> buffers_;

	std::map<std::string, cl::Image*> images_;
	std::map<std::string, cl::Sampler*> imageSamplers_;

	//OpenGL Inter-operability//
	cl::Context contextOpenGL_;
	uint32_t vertexBufferObject_;
	uint32_t frameBufferObject_;
	uint32_t vertexArrayObject_;

	cl_int errorStatus_ = 0;
public:
	OpenCL_Wrapper(uint32_t aPlatform, uint32_t aDevice)
	{
		init(aPlatform, aDevice);
	}
	void init(uint32_t aPlatform, uint32_t aDevice)
	{
		/////////////////////////////////////
		//Step 1: Set up OpenCL environment//
		/////////////////////////////////////

		//Discover platforms//
		std::vector <cl::Platform> platforms;
		cl::Platform::get(&platforms);

		intptr_t isOpenGL = (intptr_t)wglGetCurrentContext() == 0 ? false : true;

		//Create contex properties for first platform//
		std::vector<cl_context_properties > contextProperties;
		if (isOpenGL)
		{
			std::cout << "Creating OpenCL context with OpenGL Interop." << std::endl;
			contextProperties.push_back(CL_CONTEXT_PLATFORM); contextProperties.push_back((cl_context_properties)(platforms[aPlatform])());
			contextProperties.push_back(CL_GL_CONTEXT_KHR); contextProperties.push_back((intptr_t)wglGetCurrentContext());
			contextProperties.push_back(CL_WGL_HDC_KHR); contextProperties.push_back((intptr_t)wglGetCurrentDC());
			contextProperties.push_back(0);
		}
		else
		{
			std::cout << "Creating OpenCL context without OpenGL Interop." << std::endl;
			contextProperties.push_back(CL_CONTEXT_PLATFORM);
			contextProperties.push_back((cl_context_properties)(platforms[aPlatform])());
			contextProperties.push_back(0);
		}

		//Create context context using platform for GPU device//
		context_ = cl::Context(CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, contextProperties.data());

		//Get device list from context//
		std::vector<cl::Device> devices = context_.getInfo<CL_CONTEXT_DEVICES>();
		device_ = devices[aDevice];

		//Create command queue for first device - Profiling enabled//
		commandQueue_ = cl::CommandQueue(context_, device_, CL_QUEUE_PROFILING_ENABLE, &errorStatus_);	//Need to specify device 1[0] of platform 3[2] for dedicated graphics - Harri Laptop.
		if (errorStatus_)
			std::cout << "ERROR creating command queue for device. Status code: " << errorStatus_ << std::endl;

		//Define benchmark constants//
		uint32_t globalSize = 256;
		uint32_t repetitions = 1;
		uint32_t memorySize = GIGA_BYTE;

		//Build the program - Define kernel constants//
		char options[1024];
		sprintf(options,
			" -cl-fast-relaxed-math"
			//" -cl-single-precision-constant"
			//""
		);

		//Initialise workgroup dimensions//
		globalws_ = cl::NDRange(globalSize);
		localws_ = cl::NDRange(256);

		//SeqMemoryTest0//
		createKernelProgram("resources/kernels/GPU_Overhead_Benchmarks.cl", options);
		createKernel("nullKernel");
		createKernel("copyBuffer");
	}

	void createKernelProgram(const std::string aSourcePath, const char options[])
	{
		//Read in program source - I'm going to go for reading from compiled object instead//
		std::ifstream sourceFileName(aSourcePath.c_str());
		std::string sourceFile(std::istreambuf_iterator<char>(sourceFileName), (std::istreambuf_iterator<char>()));

		//Create program source object from std::string source code//
		std::vector<std::string> programSources;
		programSources.push_back(sourceFile);
		cl::Program::Sources source(programSources);	//Apparently this takes a vector of strings as the program source.

		//Create program from source code//
		kernelProgram_ = cl::Program(context_, source, &errorStatus_);
		if (errorStatus_)
			std::cout << "ERROR creating program from source. Status code: " << errorStatus_ << std::endl;

		//Build program//
		kernelProgram_.build(options);
	}
	void createKernel(const std::string aKernelName)
	{
		//Create kernel program on device//
		cl::Kernel kernel = cl::Kernel(kernelProgram_, aKernelName.c_str(), &errorStatus_);
		kernels_.insert(std::make_pair(aKernelName, new cl::Kernel(kernelProgram_, aKernelName.c_str(), &errorStatus_)));
		if (errorStatus_)
			std::cout << "ERROR creating kernel. Status code: " << errorStatus_ << std::endl;
	}

	// . //
	void createBuffer(std::string aBufferName, int aMemFlags, unsigned int aBufferSize)
	{
		buffers_.insert(std::make_pair(aBufferName, new cl::Buffer(context_, aMemFlags, aBufferSize)));
	}
	void writeBuffer(std::string aBufferName, unsigned int aBufferSize, void* input)
	{
		commandQueue_.enqueueWriteBuffer(*buffers_[aBufferName], CL_TRUE, 0, aBufferSize, input);
	}
	void readBuffer(std::string aBufferName, unsigned int aBufferSize, void* output)
	{
		commandQueue_.enqueueReadBuffer(*buffers_[aBufferName], CL_TRUE, 0, aBufferSize, output);
	}
	void deleteBuffer(std::string aBufferName)
	{
		cl::Buffer* buf = buffers_[aBufferName];
		clReleaseMemObject((*buf)());
		buffers_.erase(aBufferName);
	}

	// . //
	void setKernelArgument(const std::string aKernelName, const std::string aBufferName, int aIndex, int aSize)
	{
		(*kernels_[aKernelName]).setArg(aIndex, aSize, buffers_[aBufferName]);	//@ToDo - Check this works, and not meant to be using kernels_.find(...)
	}

	void enqueueKernel(const std::string aKernelName)
	{
		commandQueue_.enqueueNDRangeKernel(*kernels_[aKernelName], cl::NullRange/*globaloffset*/, globalws_, localws_, NULL, &kernelBenchmark_);
		kernelBenchmark_.wait();
	}

	void enqueueCopyBuffer(const std::string aSrcBuffer, const std::string aDstBuffer, const uint32_t aSize)
	{
		commandQueue_.enqueueCopyBuffer((*buffers_[aSrcBuffer]), (*buffers_["writeDstBuffer"]), 0, 0, aSize, NULL, NULL);
	}

	void* pinMappedMemory(const std::string aBuffer, const uint32_t aSize)
	{
		return commandQueue_.enqueueMapBuffer((*buffers_[aBuffer]), true, NULL, 0, aSize, NULL, NULL, &errorStatus_);
	}

	void waitKernel()
	{
		commandQueue_.finish();
	}

	void setWorkspaceSize(uint32_t aGlobalSize, uint32_t aLocalSize)
	{
		globalws_ = cl::NDRange(aGlobalSize, 1, 1);
		localws_ = cl::NDRange(aLocalSize, 1, 1);
	}
	void setWorkspaceSize(cl::NDRange aGlobalSize, cl::NDRange aLocalSize)
	{
		globalws_ = aGlobalSize;
		localws_ = aLocalSize;
	}


	//Static Functions//
	static void printAvailableDevices()
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
					//If AMD//
					//std::cout << "\t\tAMD Wavefront size: " << device.getInfo<CL_DEVICE_WAVEFRONT_WIDTH_AMD>() << std::endl;
				}
			}
			std::cout << std::endl;
		}
	}
};

#endif