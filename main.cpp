#include <stdio.h>
#include <stdlib.h>

#include "GPU_Overhead_Benchmarks_OpenCL.hpp"
#include "GPU_Overhead_Benchmarks_CUDA.hpp"

int main()
{
	//////////
	//OpenCL//
	//////////

	//OpenCL_Wrapper::printAvailableDevices();

	//Check OpenCL support and device availability//
	bool isOpenCl = GPU_Overhead_Benchmarks_OpenCL::openclCompatible();
	
	if (isOpenCl)
	{
		std::vector<OpenCL_Device> clDevices = OpenCL_Wrapper::getOpenclDevices();

		std::cout << "OpenCL device and support detected." << std::endl;
		std::cout << "Beginning OpenCL benchmarking" << std::endl << std::endl;

		for (uint32_t i = 0; i != clDevices.size(); ++i)
		{
			std::cout << "Runnning tests for platform " << clDevices[i].platform_name << " device " << clDevices[i].device_name << std::endl;
			GPU_Overhead_Benchmarks_OpenCL clBenchmark(clDevices[i].platform_id, clDevices[i].device_id);

			//clBenchmark.setBufferLength(44100);

			//clBenchmark.cl_mappingmemory(100);

			//clBenchmark.runGeneralBenchmarks(1000, true);
			clBenchmark.runRealTimeBenchmarks(44100, true);
		}
	}
	else
		std::cout << "OpenCL device or support not present to benchmark OpenCL." << std::endl;
	
	////////
	//CUDA//
	////////

	//GPU_Overhead_Benchmarks_CUDA::printAvailableDevices();

	//Check CUDA support and device availability//
	bool isCuda = GPU_Overhead_Benchmarks_CUDA::cudaCompatible();
	
	if (isCuda)
	{
		std::vector<CUDA_Device> cudaDevics = GPU_Overhead_Benchmarks_CUDA::getCudaDevices();

		std::cout << "CUDA device and support detected." << std::endl;
		std::cout << "Beginning CUDA benchmarking" << std::endl << std::endl;

		for (uint32_t i = 0; i != cudaDevics.size(); ++i)
		{
			std::cout << "Runnning tests for device " << cudaDevics[i].device_name << std::endl;
			GPU_Overhead_Benchmarks_CUDA cudaBenchmark = GPU_Overhead_Benchmarks_CUDA(cudaDevics[i].device_id);

			//cudaBenchmark.setBufferLength(44100);

			//cudaBenchmark.runGeneralBenchmarks(100, true);
			cudaBenchmark.runRealTimeBenchmarks(44100, true);
		}
	}
	else
		std::cout << "CUDA device or support not present to benchmark CUDA" << std::endl;

	char haltc;
	std::cin >> haltc;

	return 0;
}