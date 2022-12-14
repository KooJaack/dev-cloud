#include "Filters.h"

CnnFilters::CnnFilters()
{
    InitializeFirstLayer();
}

void CnnFilters::InitializeFirstLayer()
{
    arr =
        {
            {{{48, 103, 59},
              {67, 127, 80},
              {38, 87, 59}},
             {{-19, -28, -29},
              {-30, -51, -40},
              {-18, -35, -14}},
             {{-13, -39, -21},
              {-35, -87, -46},
              {-37, -84, -42}}},
            {{{-1, 1, 3},
              {0, 5, 2},
              {0, -1, 3}},
             {{-18, -32, 13},
              {-2, 4, 13},
              {-2, 3, 1}},
             {{-110, -127, -31},
              {-6, -8, 17},
              {2, 1, 6}}},
            {{{17, 36, 1},
              {22, 42, -6},
              {14, 30, -19}},
             {{31, 37, 32},
              {55, 82, 62},
              {41, 67, 42}},
             {{-55, -94, -32},
              {-78, -127, -51},
              {-46, -77, -26}}},
            {{{53, -105, 49},
              {58, -114, 65},
              {27, -45, 10}},
             {{59, -119, 72},
              {58, -127, 74},
              {33, -46, 12}},
             {{48, -60, 14},
              {40, -54, 10},
              {19, 16, -25}}},
            {{{37, 112, 71},
              {-7, -10, -12},
              {-39, -117, -72}},
             {{13, 12, 15},
              {9, 45, 25},
              {-13, -15, -18}},
             {{-41, -127, -89},
              {3, -6, -4},
              {41, 115, 85}}},
            {{{-40, -14, 60},
              {-33, -3, 75},
              {-36, 11, 79}},
             {{13, 41, 53},
              {1, 16, 38},
              {-24, -18, -5}},
             {{28, -35, -127},
              {38, -26, -118},
              {50, 17, -63}}},
            {{{15, 10, -11},
              {-44, 59, -20},
              {-14, 45, -19}},
             {{-35, 57, -29},
              {-104, 127, -36},
              {-68, 92, -36}},
             {{-19, 44, -28},
              {-67, 94, -36},
              {-37, 64, -37}}},
            {{{-13, 0, 6},
              {4, -11, 0},
              {-12, -55, -36}},
             {{1, 12, 9},
              {-19, -14, 3},
              {-122, -127, -98}},
             {{0, 13, 3},
              {-11, 11, 3},
              {-14, 3, -13}}},
            {{{-33, -116, -92},
              {27, 61, 65},
              {-4, 18, 6}},
             {{-37, -127, -95},
              {43, 113, 101},
              {-10, 6, -12}},
             {{-2, -39, -30},
              {23, 79, 75},
              {-2, -1, -14}}},
            {{{-27, -34, -35},
              {43, 51, 53},
              {-18, -16, -21}},
             {{54, 76, 58},
              {-75, -127, -89},
              {30, 44, 41}},
             {{-25, -35, -23},
              {36, 52, 40},
              {-15, -17, -23}}},
            {{{12, 39, 29},
              {26, 66, 55},
              {15, 40, 32}},
             {{-29, -87, -66},
              {-48, -127, -96},
              {-31, -92, -70}},
             {{15, 43, 31},
              {26, 71, 50},
              {15, 45, 35}}},
            {{{1, 13, 9},
              {25, 47, 17},
              {-41, -84, -76}},
             {{-8, -21, 6},
              {60, 127, 84},
              {25, 37, 9}},
             {{-66, -123, -56},
              {-3, -15, 6},
              {12, 19, 4}}},
            {{{-3, 90, -9},
              {57, 75, 22},
              {89, 62, 88}},
             {{-64, 68, -95},
              {-34, 67, -102},
              {7, 58, -26}},
             {{-83, 37, -93},
              {-84, 37, -127},
              {-39, 59, -67}}},
            {{{18, 69, -98},
              {19, 82, -91},
              {0, 66, -88}},
             {{19, 67, -102},
              {31, 62, -127},
              {15, 64, -122}},
             {{-28, 85, -7},
              {-8, 89, -40},
              {10, 97, -37}}},
            {{{14, 6, 21},
              {127, 70, 81},
              {103, 51, 73}},
             {{1, 12, 14},
              {19, 14, 53},
              {5, -3, 53}},
             {{2, -6, -3},
              {-5, -8, 27},
              {-12, -12, 25}}},
            {{{-8, -27, -21},
              {31, 76, 54},
              {-14, -39, -24}},
             {{-24, -60, -50},
              {51, 127, 82},
              {-36, -78, -50}},
             {{-12, -32, -27},
              {30, 80, 58},
              {-17, -44, -25}}},
            {{{-8, -2, -7},
              {24, 13, -6},
              {-11, -2, -29}},
             {{46, 36, 29},
              {127, 87, 63},
              {24, 15, -19}},
             {{-28, -9, -5},
              {22, 17, 7},
              {11, 1, -24}}},
            {{{-53, -65, -29},
              {-16, -4, -20},
              {60, 75, 55}},
             {{25, 23, 5},
              {76, 114, 76},
              {-73, -127, -80}},
             {{31, 32, 16},
              {-42, -84, -53},
              {-1, 25, 35}}},
            {{{6, 1, 4},
              {-1, -24, -13},
              {0, 7, 12}},
             {{-11, -38, -36},
              {-72, -127, -108},
              {9, -8, -2}},
             {{-3, 16, 0},
              {-7, -2, -12},
              {-5, 11, 11}}},
            {{{-110, 8, 107},
              {-75, 16, 75},
              {60, -1, -64}},
             {{-125, 8, 127},
              {-1, 9, 1},
              {117, -17, -121}},
             {{-65, 0, 70},
              {74, -3, -73},
              {119, -19, -120}}},
            {{{5, -6, -22},
              {-17, -88, -113},
              {-13, -105, -127}},
             {{11, 67, 77},
              {25, 76, 71},
              {6, -21, -28}},
             {{-25, -21, 14},
              {9, 59, 81},
              {-1, 39, 51}}},
            {{{35, -60, 22},
              {43, -99, 59},
              {29, -55, 24}},
             {{34, -81, 41},
              {39, -127, 90},
              {36, -80, 47}},
             {{26, -40, 10},
              {38, -74, 39},
              {20, -45, 17}}},
            {{{-3, 1, 8},
              {10, -1, 3},
              {5, 2, 3}},
             {{-3, 4, 6},
              {11, -13, -2},
              {9, 0, 5}},
             {{10, -10, -4},
              {-62, -127, -103},
              {2, -22, -11}}},
            {{{-31, -66, -45},
              {59, 127, 89},
              {-25, -62, -49}},
             {{3, 3, 1},
              {0, -12, -2},
              {-2, 5, 0}},
             {{26, 59, 41},
              {-48, -119, -78},
              {20, 60, 44}}},
            {{{-2, 46, 104},
              {-20, 9, 113},
              {-23, -58, -51}},
             {{-44, -12, 105},
              {-68, -99, -22},
              {8, -44, -125}},
             {{-60, -51, 1},
              {10, -21, -91},
              {119, 127, -63}}},
            {{{5, 4, 17},
              {54, 118, 64},
              {35, 82, 11}},
             {{-59, -122, -56},
              {-22, -58, -34},
              {51, 102, 51}},
             {{-1, 5, 30},
              {-59, -127, -63},
              {0, -18, -10}}},
            {{{-7, 0, 11},
              {-9, 3, 2},
              {-1, -5, -6}},
             {{-14, 2, 35},
              {-33, -3, 25},
              {-15, 4, 9}},
             {{88, 91, 127},
              {-1, 34, 82},
              {-43, 6, 40}}},
            {{{11, 5, -16},
              {-8, 49, 83},
              {-18, -4, 38}},
             {{6, -60, -105},
              {10, 59, 65},
              {-11, 27, 58}},
             {{18, -63, -127},
              {1, -18, -29},
              {-10, 7, 34}}},
            {{{-6, 22, 84},
              {24, 40, 68},
              {61, 71, 92}},
             {{-83, -21, 47},
              {-60, -9, 10},
              {-35, 5, 40}},
             {{-109, -75, 17},
              {-127, -71, 5},
              {-57, -32, 47}}},
            {{{78, -9, 25},
              {68, -13, 64},
              {107, 87, 127}},
             {{64, -81, -44},
              {36, -125, -26},
              {43, -69, 10}},
             {{22, -99, -82},
              {30, -113, -62},
              {51, -68, -18}}},
            {{{-13, -8, -67},
              {30, 46, 9},
              {-2, -17, -94}},
             {{-1, 27, -76},
              {58, 127, 71},
              {5, 27, -75}},
             {{-64, -26, -97},
              {-30, 55, -6},
              {-38, 14, -55}}},
            {{{26, 60, 42},
              {-31, -81, -60},
              {4, 18, 18}},
             {{38, 91, 62},
              {-51, -127, -87},
              {15, 38, 33}},
             {{24, 67, 38},
              {-27, -83, -60},
              {5, 12, 16}}},
            {{{-54, -95, 124},
              {-76, -127, 46},
              {17, 25, -25}},
             {{-80, -126, 81},
              {11, -11, -4},
              {87, 127, -61}},
             {{-11, -27, 19},
              {69, 93, -56},
              {53, 105, -108}}},
            {{{92, -82, -42},
              {127, -66, -78},
              {73, -73, -22}},
             {{82, -54, -45},
              {116, -46, -83},
              {83, -65, -39}},
             {{35, -41, 7},
              {61, -50, -24},
              {40, -57, 0}}},
            {{{16, 15, -44},
              {1, 8, -6},
              {68, 77, 127}},
             {{5, 13, -53},
              {-62, -19, -77},
              {4, 29, 13}},
             {{75, 48, 22},
              {-22, 7, -19},
              {-19, 4, -26}}},
            {{{0, -46, 38},
              {51, 60, -82},
              {21, 114, -110}},
             {{-36, -113, 121},
              {30, -6, -9},
              {29, 84, -103}},
             {{-51, -114, 127},
              {-28, -68, 80},
              {6, 40, -40}}},
            {{{34, 1, -71},
              {5, -9, -1},
              {-51, 15, 127}},
             {{35, 12, -68},
              {16, -50, -80},
              {-36, 29, 127}},
             {{30, 27, -46},
              {17, -32, -75},
              {-37, 0, 81}}},
            {{{-15, -40, -28},
              {-33, -76, -54},
              {-17, -44, -23}},
             {{32, 86, 58},
              {50, 127, 87},
              {32, 82, 62}},
             {{-9, -34, -28},
              {-26, -66, -50},
              {-11, -36, -24}}},
            {{{-9, 17, -59},
              {-11, 0, -78},
              {-113, -64, -54}},
             {{57, 53, -18},
              {78, 43, -76},
              {-24, -31, -108}},
             {{78, 58, 61},
              {127, 93, 5},
              {46, 8, -86}}},
            {{{51, -39, -90},
              {54, -22, -81},
              {51, 13, -47}},
             {{74, -29, -98},
              {61, -36, -105},
              {113, 85, 5}},
             {{88, -36, -127},
              {91, 12, -68},
              {70, 55, -4}}},
            {{{-6, -8, -26},
              {32, 84, 56},
              {6, 11, 10}},
             {{27, 76, 44},
              {46, 87, 72},
              {-19, -69, -41}},
             {{2, 12, 14},
              {-27, -69, -38},
              {-59, -127, -84}}},
            {{{-116, -23, 115},
              {-125, -3, 119},
              {-74, 6, 65}},
             {{-66, 15, 59},
              {2, 19, -18},
              {60, 9, -70}},
             {{56, 18, -72},
              {127, 7, -118},
              {114, -6, -96}}},
            {{{-31, -10, 60},
              {7, 13, 18},
              {33, -45, -127}},
             {{-36, -10, 53},
              {3, 43, 74},
              {6, -57, -107}},
             {{-28, -2, 55},
              {-21, 15, 72},
              {20, -10, -36}}},
            {{{31, -20, -91},
              {-11, -82, -127},
              {0, 6, 9}},
             {{-10, -72, -108},
              {4, 5, 3},
              {-3, 58, 98}},
             {{3, 11, 12},
              {5, 65, 112},
              {-18, 26, 94}}},
            {{{27, 92, 81},
              {-3, -34, -27},
              {-9, -30, -23}},
             {{-10, -29, -20},
              {-34, -127, -108},
              {1, 25, 19}},
             {{-5, -35, -31},
              {12, 23, 12},
              {14, 102, 88}}},
            {{{-19, -4, 21},
              {-106, -17, 127},
              {-80, -24, 102}},
             {{91, 13, -103},
              {-10, 1, 8},
              {-88, -21, 106}},
             {{104, 21, -120},
              {99, 21, -121},
              {9, -3, -7}}},
            {{{92, 44, -127},
              {57, 16, -71},
              {-41, -24, 53}},
             {{89, 33, -118},
              {0, -6, 10},
              {-85, -39, 121}},
             {{40, 10, -51},
              {-58, -29, 85},
              {-83, -34, 110}}},
            {{{20, -22, 9},
              {-23, -57, 73},
              {8, -21, 13}},
             {{-14, -48, 51},
              {-70, -88, 127},
              {-11, -28, 36}},
             {{5, -28, 24},
              {-28, -55, 69},
              {13, -10, 3}}},
            {{{-7, -11, -3},
              {-41, -127, -88},
              {42, 25, 15}},
             {{34, 104, 77},
              {-45, -82, -62},
              {-37, -125, -92}},
             {{35, 102, 66},
              {21, 103, 80},
              {2, 3, 10}}},
            {{{-37, -98, -73},
              {-45, -126, -87},
              {-29, -90, -60}},
             {{-7, -7, -6},
              {-6, -3, -8},
              {3, 11, 9}},
             {{35, 101, 78},
              {49, 127, 86},
              {33, 87, 61}}},
            {{{33, 35, 31},
              {-46, -59, -53},
              {11, 20, 24}},
             {{-47, -73, -57},
              {92, 127, 96},
              {-37, -52, -43}},
             {{16, 23, 26},
              {-43, -53, -41},
              {24, 25, 18}}},
            {{{-85, 67, -5},
              {-120, 67, 34},
              {-44, 17, -11}},
             {{-91, 71, 14},
              {-127, 75, 52},
              {-32, 25, 10}},
             {{-32, 46, -13},
              {-60, 33, 18},
              {9, 1, -6}}},
            {{{22, 49, 46},
              {4, 4, 0},
              {-28, -59, -54}},
             {{-58, -111, -74},
              {-5, -16, -13},
              {67, 127, 96}},
             {{35, 63, 35},
              {5, 12, 5},
              {-41, -73, -40}}},
            {{{-55, -127, -82},
              {-5, -15, -18},
              {9, 27, 11}},
             {{5, 3, 1},
              {51, 127, 85},
              {24, 48, 34}},
             {{6, 18, 22},
              {14, 29, 25},
              {-46, -108, -78}}},
            {{{6, 26, 28},
              {-35, -85, -61},
              {31, 58, 28}},
             {{14, 41, 45},
              {-57, -127, -83},
              {47, 87, 46}},
             {{3, 22, 26},
              {-35, -83, -57},
              {27, 57, 31}}},
            {{{41, 94, 72},
              {3, 7, 4},
              {-43, -96, -74}},
             {{53, 115, 86},
              {0, 6, 1},
              {-58, -127, -85}},
             {{50, 94, 65},
              {4, 6, 2},
              {-45, -106, -68}}},
            {{{15, 104, 75},
              {61, 23, -64},
              {49, -14, -80}},
             {{33, 41, -34},
              {97, -4, -123},
              {68, -6, -91}},
             {{72, 8, -92},
              {104, -18, -127},
              {32, -45, -88}}},
            {{{-55, 18, 32},
              {65, -10, -52},
              {127, -35, -79}},
             {{-114, 39, 71},
              {-21, 16, 7},
              {111, -19, -75}},
             {{-106, 29, 57},
              {-85, 26, 51},
              {39, 6, -39}}},
            {{{76, 0, -127},
              {101, 23, -123},
              {98, 38, -82}},
             {{11, 39, 37},
              {-6, -3, -2},
              {-27, -40, -35}},
             {{-96, -39, 87},
              {-89, -13, 120},
              {-76, 8, 118}}},
            {{{30, 43, -7},
              {-34, -45, 10},
              {-105, -127, 19}},
             {{65, 79, -16},
              {59, 82, 39},
              {-70, -92, 18}},
             {{21, -5, -78},
              {63, 71, -1},
              {-18, -19, 19}}},
            {{{-44, -88, -34},
              {-2, 1, -5},
              {47, 93, 41}},
             {{-65, -127, -58},
              {-4, -4, -6},
              {67, 119, 60}},
             {{-52, -96, -43},
              {1, -1, 2},
              {54, 96, 48}}},
            {{{-15, -48, -23},
              {6, -6, 1},
              {29, 101, 75}},
             {{-9, -4, -5},
              {-33, -127, -96},
              {0, -5, -5}},
             {{28, 101, 71},
              {-5, -3, -3},
              {-12, -42, -31}}},
            {{{22, 54, 40},
              {-49, -121, -83},
              {29, 72, 48}},
             {{5, 15, 17},
              {5, -11, -14},
              {-4, -5, -2}},
             {{-31, -65, -49},
              {60, 127, 87},
              {-36, -68, -44}}},
            {{{45, 120, 53},
              {48, 93, 52},
              {-4, -22, -3}},
             {{57, 114, 71},
              {-36, -98, -73},
              {-51, -116, -57}},
             {{-13, -20, -10},
              {-61, -127, -77},
              {19, 45, 50}}}};