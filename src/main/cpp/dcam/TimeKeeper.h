#ifndef _TIMEKEEPER_H__
#define _TIMEKEEPER_H__

#include <chrono>
#include <iostream>

class TimeKeeper
{
private:
    std::chrono::steady_clock::time_point timestart;

public:
    TimeKeeper();
    ~TimeKeeper();
    void setTimeStart();
    double getTimeElapsed(); // in milliseconds
};

#endif /* _TIMEKEEPER_H__ */
