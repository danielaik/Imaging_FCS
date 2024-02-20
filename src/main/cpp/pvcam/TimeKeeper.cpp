#include "TimeKeeper.h"

#include <iostream>
#include <windows.h>

TimeKeeper::TimeKeeper()
{}

TimeKeeper::~TimeKeeper()
{}

void TimeKeeper::setTimeStart()
{
    timestart = GetTickCount();
}

double TimeKeeper::getTimeElapsed()
{
    return (GetTickCount() - timestart); // ms
}