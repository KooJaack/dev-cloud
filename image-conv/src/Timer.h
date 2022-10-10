#include <chrono>

using Duration = std::chrono::duration<double>;

class Timer{
public:
    Timer();
    Duration elapsed();
private: 
    std::chrono::steady_clock::time_point start;
};