#include "TimeKeeper.h"

#include <iostream>
#include <windows.h>

TimeKeeper::TimeKeeper()
{
    std::cout << "constructor timekeeper..." << std::endl;
}

TimeKeeper::~TimeKeeper()
{
    std::cout << "destructor timekeeper..." << std::endl;
}

void TimeKeeper::setTimeStart()
{
    timestart = GetTickCount();
}

double TimeKeeper::getTimeElapsed()
{
    return (GetTickCount() - timestart); // ms
}