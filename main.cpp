#include <stdio.h>
#include <stdlib.h>

#include "GPU_Overhead_Benchmarks_OpenCL.hpp"
#include "GPU_Overhead_Benchmarks_CUDA.hpp"

int main()
{
	//////////
	//OpenCL//
	//////////

	OpenCL_Wrapper::printAvailableDevices();

	//Check OpenCL support and device availability//
	bool isOpenCl = GPU_Overhead_Benchmarks_OpenCL::openclCompatible();

	if (isOpenCl)
	{
		std::cout << "OpenCL device and support detected." << std::endl;
		std::cout << "Beginning OpenCL benchmarking" << std::endl << std::endl;
		GPU_Overhead_Benchmarks_OpenCL clBenchmark(0, 0);
	
		clBenchmark.setBufferLength(44100);

		clBenchmark.cl_mappingmemory(10000);
	
		clBenchmark.runGeneralBenchmarks(10000, true);
		//clBenchmark.runRealTimeBenchmarks(44100, true);
	}
	else
		std::cout << "OpenCL device or support not present to benchmark OpenCL." << std::endl;
	
	////////
	//CUDA//
	////////

	//GPU_Overhead_Benchmarks_CUDA::printAvailableDevices();

	////Check CUDA support and device availability//
	//bool isCuda = GPU_Overhead_Benchmarks_CUDA::cudaCompatible();
	//
	//if (isCuda)
	//{
	//	std::cout << "CUDA device and support detected." << std::endl;
	//	std::cout << "Beginning CUDA benchmarking" << std::endl << std::endl;
	//	GPU_Overhead_Benchmarks_CUDA cudaBenchmark = GPU_Overhead_Benchmarks_CUDA();
	//
	//	cudaBenchmark.setBufferLength(44100);
	//
	//	cudaBenchmark.runGeneralBenchmarks(10000, true);
	//	cudaBenchmark.runRealTimeBenchmarks(44100, true);
	//}
	//else
	//	std::cout << "CUDA device or support not present to benchmark CUDA" << std::endl;

	char haltc;
	std::cin >> haltc;

	return 0;
}