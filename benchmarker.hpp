#ifndef BENCHMARKING_HPP
#define BENCHMARKING_HPP

#include <iostream>
#include <chrono>

class Benchmarker
{
private:
	std::map<std::string, std::chrono::time_point<std::chrono::steady_clock>> startTimers;
	std::map<std::string, std::chrono::time_point<std::chrono::steady_clock>> endTimers;
	std::map<std::string, std::chrono::duration<double>> elapsedTimers;
	//std::chrono::time_point<std::chrono::steady_clock> end;
	//std::chrono::duration<double> elapsed;

	std::map<std::string, uint32_t> cntTimersAverage;
public:
	void startTimer(const std::string aTimer)
	{
		++cntTimersAverage[aTimer];
		startTimers[aTimer] = std::chrono::steady_clock::now();
	}
	void pauseTimer(const std::string aTimer)
	{
		endTimers[aTimer] = std::chrono::steady_clock::now();
		elapsedTimers[aTimer] += endTimers[aTimer] - startTimers[aTimer];
	}
	void endTimer(const std::string aTimer)
	{
		endTimers[aTimer] = std::chrono::steady_clock::now();
		elapsedTimers[aTimer] += endTimers[aTimer] - startTimers[aTimer];
	}
	void elapsedTimer(const std::string aTimer)
	{
		//auto diff = end - start;
		std::cout << "Total time to complete: " << std::chrono::duration<double>(elapsedTimers[aTimer]).count() << "s" << std::endl;
		std::cout << "Total time to complete: " << std::chrono::duration <double, std::milli>(elapsedTimers[aTimer]).count() << "ms" << std::endl;
		std::cout << "Total time to complete: " << std::chrono::duration <double, std::nano>(elapsedTimers[aTimer]).count() << "ns" << std::endl;

		if (cntTimersAverage[aTimer] > 1)
		{
			double avgElapsed = std::chrono::duration <double, std::milli>(elapsedTimers[aTimer]).count() / cntTimersAverage[aTimer];
			std::cout << "Average time to complete each buffer: " << avgElapsed << "ms" << std::endl;
		}

		cntTimersAverage[aTimer] = 0;
		elapsedTimers[aTimer] = std::chrono::duration<double>(0).zero();
	}
};

#endif