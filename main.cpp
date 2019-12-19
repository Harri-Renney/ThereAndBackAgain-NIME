#include <stdio.h>
#include <stdlib.h>

#include "GPUOverhead_Benchmarks.hpp"
#include "OpenCL_Wrapper.h"
#include "CUDA_Wrapper.hpp"

int main()
{
	//GPUOverhead_Benchmarks clBenchmark(0,0);
	//
	//OpenCL_Wrapper::printAvailableDevices();
	//
	//clBenchmark.setBufferSize(1024);
	//
	//clBenchmark.writeToGPU(1000);
	//
	//clBenchmark.nullKernel(100);
	//clBenchmark.runUnidirectionalBenchmarks(100);
	//clBenchmark.runBidirectionalBenchmarks(100);
	//
	//clBenchmark.writeToGPUMapped(1000);
	
	CUDA_Wrapper::printAvailableDevices();

	CUDA_Wrapper cW = CUDA_Wrapper();
	cW.nullkernel();
	cW.copyBufferKernel();

	char haltc;
	std::cin >> haltc;

	return 0;
}