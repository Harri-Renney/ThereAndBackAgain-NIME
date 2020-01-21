#ifndef GPU_OVERHEAD_BENCHMARKS_HPP
#define GPU_OVERHEAD_BENCHMARKS_HPP

#include <windows.h>
#include <cstdint>

class GPU_Overhead_Benchmarks
{
private:
protected:

public:
	GPU_Overhead_Benchmarks() {}
	virtual ~GPU_Overhead_Benchmarks() {}
	virtual void runGeneralBenchmarks(uint64_t aNumRepetitions) = 0;
	virtual void runRealTimeBenchmarks(uint64_t aFrameRate) = 0;
};

#endif