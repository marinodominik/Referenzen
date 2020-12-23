#pragma once

#include <chrono>

typedef std::chrono::high_resolution_clock timer_clock;
typedef std::chrono::high_resolution_clock::time_point timer_tp;

timer_tp timer_now()
{
	return timer_clock::now();
}

float timer_elapsed(const timer_tp &start, const timer_tp &end)
{
	return std::chrono::duration<double, std::milli>(end - start).count();
}

void print_time(float cpu_comp_time, float gpu_comp_time) {
	std::cout << "\n" << std::endl;
	std::cout << "Computing Time for CPU: " << cpu_comp_time << std::endl;
	std::cout << "Computing Time for GPU: " << gpu_comp_time << std::endl;
	
	if (cpu_comp_time < gpu_comp_time) {
		std::cout << "CPU is faster" << std::endl;
	
	} else if (gpu_comp_time < cpu_comp_time) {
		std::cout << "GPU is faster" << std::endl;
	
	} else {
		std::cout << "Computing time is equal" << std::endl;
	}
}