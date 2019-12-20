#include <stdio.h>
#include <stdlib.h>

#include "GPUOverhead_Benchmarks.hpp"
#include "OpenCL_Wrapper.h"
#include "CUDA_Wrapper.hpp"

int main()
{
	OpenCL_Wrapper::printAvailableDevices();

	GPUOverhead_Benchmarks clBenchmark(0,0);
	
	clBenchmark.setBufferSize(GPUOverhead_Benchmarks::GIGA_BYTE);
	//clBenchmark.setBufferSize(1024);

	clBenchmark.cl_001_CPUtoGPU(10, true);
	clBenchmark.cl_002_GPUtoCPU(10, true);
	
	//clBenchmark.cl_000_nullKernel(10);
	//clBenchmark.runUnidirectionalBenchmarks(10);
	//clBenchmark.runBidirectionalBenchmarks(100);
	//
	//clBenchmark.writeToGPUMapped(1000);
	
	//CUDA_Wrapper::printAvailableDevices();

	//Check CUDA support and device availability//
	bool isCUDA = true;
	int cudaVersion = 0;
	cudaRuntimeGetVersion(&cudaVersion);
	if (cudaVersion > 0)
	{
		std::cout << "CUDA version: " << cudaVersion << std::endl;
	}
	else
	{
		std::cout << "No CUDA version detected." << std::endl;
		isCUDA = false;
	}

	int numCudaDevices = CUDA_Wrapper::isCudaAvailable();
	if (numCudaDevices)
	{
		std::cout << "Number of avilable CUDA devices: " << numCudaDevices << std::endl;
	}
	else
	{
		std::cout << "No CUDA devices detected" << std::endl;
		isCUDA = false;
	}

	if (isCUDA)
	{
		std::cout << "CUDA device and support detected." << std::endl;
		std::cout << "Beginning CUDA benchmarking" << std::endl << std::endl;
		CUDA_Wrapper cudaBenchmark = CUDA_Wrapper();
		cudaBenchmark.cuda_000_nullkernel(10, true);
		cudaBenchmark.cuda_001_CPUtoGPU(10, true);
		cudaBenchmark.cuda_002_GPUtoCPU(10, true);
		cudaBenchmark.cuda_003_CPUtoGPUtoCPU(10, true);
		cudaBenchmark.cuda_005_cpymemory(10, true);
		cudaBenchmark.cuda_006_cpymemorykernel(10, true);
	}
	else
		std::cout << "CUDA device or support ";

	char haltc;
	std::cin >> haltc;

	return 0;
}