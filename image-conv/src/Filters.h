#include <vector>

class CnnFilters
{
public:
    CnnFilters();
    std::vector<std::vector<std::vector<std::vector<int>>>> arr;

private:
    void InitializeFirstLayer();
}