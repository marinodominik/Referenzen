#include "basics/WMTime.h"

WMTime::WMTime()
{
    reset();
}

void WMTime::reset()
{
    time = -100;
    cycle = 0;
    total_cycle = 0;
}
void WMTime::reset2()
{
    time = -101;
    cycle = 0;
    total_cycle = 0;
}

bool WMTime::after_reset() const
{
    return time <= -100;
}
bool WMTime::after_reset2() const
{
    return time <= -101;
}

bool WMTime::operator==( const WMTime & t ) const
{
    return time == t.time && cycle == t.cycle;
}
bool WMTime::operator!=( const WMTime & t ) const
{
    return time != t.time || cycle != t.cycle;
}
bool WMTime::operator<=( const WMTime & t ) const
{
    return time < t.time || ( time == t.time && cycle <= t.cycle );
}

int WMTime::operator()()
{
    return time;
}

std::ostream& operator<< (std::ostream& o, const WMTime & wmtime) {
    return o << wmtime.time << "[," << wmtime.cycle <<"]";
}
