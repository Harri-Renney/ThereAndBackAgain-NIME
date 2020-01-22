#include <stdio.h>
#include <stdlib.h>

#include "GPU_Overhead_Benchmarks_OpenCL.hpp"
#include "GPU_Overhead_Benchmarks_CUDA.hpp"

int main()
{
	OpenCL_Wrapper::printAvailableDevices();

	// 0,0 = CPU
	// 0,1 = Intel GPU
	// 1,0 = AMD GPU
	GPU_Overhead_Benchmarks_OpenCL clBenchmark(0,0);
	
	//@ToDo - Sort out way of dynamically working out localworkgroupsize from buffer size//
	clBenchmark.setBufferSize(GPU_Overhead_Benchmarks_OpenCL::MEGA_BYTE);
	//clBenchmark.setBufferLength(44100);
	//clBenchmark.setBufferSize(1024 * sizeof(float));
	//clBenchmark.setBufferSize(88200 * sizeof(float));

	clBenchmark.cl_cputogpu_standard(1000, true);
	clBenchmark.cl_cputogpu_mappedmemory(1000, true);
	
	//clBenchmark.runUnidirectionalBenchmarks(10, 44100);
	//clBenchmark.runBidirectionalBenchmarks(1, 44100);
	//clBenchmark.bidirectionalComplexSynthesis(1, 44100);
	//
	//clBenchmark.writeToGPUMapped(1000);

	//clBenchmark.cl_complexbuffersynthesis_standard(1, true);
	//clBenchmark.cl_complexbuffersynthesis_mappedmemory(1, true);

	clBenchmark.cl_bidirectional_baseline(44100, true);
	clBenchmark.cl_bidirectional_processing(44100, true);

	//clBenchmark.runGeneralBenchmarks(10);
	
	GPU_Overhead_Benchmarks_CUDA::printAvailableDevices();

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

	int numCudaDevices = GPU_Overhead_Benchmarks_CUDA::isCudaAvailable();
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
		GPU_Overhead_Benchmarks_CUDA cudaBenchmark = GPU_Overhead_Benchmarks_CUDA();
		//cudaBenchmark.setBufferSize(GPU_Overhead_Benchmarks_OpenCL::GIGA_BYTE);
		//cudaBenchmark.setBufferSize(GPU_Overhead_Benchmarks_OpenCL::MEGA_BYTE);
		cudaBenchmark.setBufferLength(44100);
		//cudaBenchmark.setBufferLength(44100);
		//cudaBenchmark.cuda_000_nullkernel(10, true);
		//cudaBenchmark.cuda_001_CPUtoGPU(1000, true);
		//cudaBenchmark.cuda_002_GPUtoCPU(10, true);
		//cudaBenchmark.cuda_003_CPUtoGPUtoCPU(10, true);
		//cudaBenchmark.cuda_005_cpymemory(10, true);
		//cudaBenchmark.cuda_006_cpymemorykernel(10, true);
		//cudaBenchmark.cuda_008_simplebufferprocessing(1000, true);
		//cudaBenchmark.cuda_011_complexbuffersynthesis(1, true);

		//cudaBenchmark.cuda_cputogpu_standard(10, true);
		//cudaBenchmark.cuda_cputogpu_mappedmemory(10, true);
		//cudaBenchmark.cuda_cputogpu_pinned(10, true);

		//cudaBenchmark.runGeneralBenchmarks(10);

		//cudaBenchmark.cuda_devicetransferkernel_standard(10, true);
	}
	else
		std::cout << "CUDA device or support no present to benchmark CUDA" << std::endl;

	char haltc;
	std::cin >> haltc;

	return 0;
}