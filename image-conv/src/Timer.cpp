#include "Timer.h"

Timer::Timer()
{
    start = std::chrono::steady_clock::now();
}
Duration Timer::elapsed() 
{
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<Duration>(now - start);
}