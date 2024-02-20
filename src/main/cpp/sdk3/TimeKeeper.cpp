#include "TimeKeeper.h"

#include <chrono>
#include <iostream>

TimeKeeper::TimeKeeper()
{}

TimeKeeper::~TimeKeeper()
{}

void TimeKeeper::setTimeStart()
{
    timestart = std::chrono::steady_clock::now();
}

double TimeKeeper::getTimeElapsed()
{
    auto timeend = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                       timeend - timestart)
                       .count();
    return static_cast<double>(elapsed); // ms
}