#include <stdio.h>
#include <stdlib.h>

#include "GPUOverhead_Benchmarks.hpp"
#include "OpenCL_Wrapper.h"
#include "CUDA_Wrapper.hpp"

int main()
{
	OpenCL_Wrapper::printAvailableDevices();

	GPUOverhead_Benchmarks clBenchmark(0,0);
	
	//@ToDo - Sort out way of dynamically working out localworkgroupsize from buffer size//
	//clBenchmark.setBufferSize(GPUOverhead_Benchmarks::MEGA_BYTE);
	clBenchmark.setBufferSize(1024 * sizeof(float));

	//clBenchmark.cl_000_nullKernel(10, true);
	//clBenchmark.cl_001_CPUtoGPU(10, true);
	//clBenchmark.cl_002_GPUtoCPU(10, true);
	//clBenchmark.cl_003_CPUtoGPUtoCPU(1000, true);
	//clBenchmark.cl_004_mappedmemory(1000, true);
	//clBenchmark.cl_005_cpymemory(10, true);
	//clBenchmark.cl_006_cpymemorykernel(10, true);
	//clBenchmark.cl_007_singlesample(1000, true);
	clBenchmark.cl_007_singlesamplemapping(1000, true);
	//clBenchmark.cl_008_simplebufferprocessing(10000, true);
	//clBenchmark.cl_008_simplebufferprocessingmapping(1000, true);
	//clBenchmark.cl_009_complexbufferprocessing(10000, true);
	//clBenchmark.cl_009_complexbufferprocessingmapping(10000, true);
	//clBenchmark.cl_011_complexbuffersynthesis(10, true);
	clBenchmark.cl_012_interruptedbufferprocessing(1000, true);
	
	//clBenchmark.runUnidirectionalBenchmarks(10, 44100);
	//clBenchmark.runBidirectionalBenchmarks(10, 44100);
	//clBenchmark.bidirectionalComplexSynthesis(10, 44100);
	//
	//clBenchmark.writeToGPUMapped(1000);
	
	//CUDA_Wrapper::printAvailableDevices();

	//Check CUDA support and device availability//
	bool isCUDA = true;
	int cudaRuntimeVersion = 0;
	int cudaDriverVersion = 0;
	cudaRuntimeGetVersion(&cudaRuntimeVersion);
	cudaDriverGetVersion(&cudaDriverVersion);
	std::cout << "CUDA runtime version: " << cudaRuntimeVersion << std::endl;
	std::cout << "CUDA driver version: " << cudaDriverVersion << std::endl;
	if(cudaRuntimeGetVersion == 0 || cudaDriverVersion == 0)
	{
		std::cout << "Necessary CUDA runtime or driver version missing" << std::endl;
		isCUDA = false;
	}

	int numCudaDevices = CUDA_Wrapper::isCudaAvailable();
	std::cout << "Number of available CUDA devices: " << numCudaDevices << std::endl;
	if(numCudaDevices == 0)
	{
		std::cout << "A necessary CUDA device is not detected" << std::endl;
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
		std::cout << "CUDA device or support no present to benchmark CUDA" << std::endl;

	char haltc;
	std::cin >> haltc;

	return 0;
}