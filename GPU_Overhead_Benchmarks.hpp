#ifndef GPU_OVERHEAD_BENCHMARKS_HPP
#define GPU_OVERHEAD_BENCHMARKS_HPP

#include <windows.h>
#include <cstdint>

class GPU_Overhead_Benchmarks
{
private:
protected:
	static const size_t bufferSizesLength = 13;
	uint64_t bufferSizes[bufferSizesLength];

	typedef float datatype;
	uint32_t sampleRate_ = 44100;
	uint64_t bufferSize_ = 1024;
	uint64_t bufferLength_ = bufferSize_ / sizeof(datatype);
public:
	GPU_Overhead_Benchmarks() {}
	virtual ~GPU_Overhead_Benchmarks() {}
	virtual void runGeneralBenchmarks(uint64_t aNumRepetitions) = 0;
	virtual void runRealTimeBenchmarks(uint64_t aFrameRate) = 0;
};

#endif