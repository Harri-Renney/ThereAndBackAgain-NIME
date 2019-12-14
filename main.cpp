#include <stdio.h>
#include <stdlib.h>

#include "GPUOverhead_Benchmarks.hpp"
#include "OpenCL_Wrapper.h"

int main()
{
	GPUOverhead_Benchmarks clBenchmark(1,0);

	OpenCL_Wrapper::printAvailableDevices();

	clBenchmark.setBufferSize(1024);

	clBenchmark.writeToGPU(1000);

	clBenchmark.nullKernel(100);
	clBenchmark.runUnidirectionalBenchmarks(100);
	clBenchmark.runBidirectionalBenchmarks(100);

	clBenchmark.writeToGPUMapped(1000);
	
	CUDA_Wrapper::printAvailableDevices();

	char haltc;
	std::cin >> haltc;

	return 0;
}