#ifndef AGENT_SRC_BASICS_WMTIME_H_
#define AGENT_SRC_BASICS_WMTIME_H_

#include <iostream>

struct WMTime {

    int time;
    int cycle;
    int total_cycle;

    WMTime();

    void reset();
    void reset2();

    bool after_reset() const;
    bool after_reset2() const;

    bool operator==(const WMTime & t) const;
    bool operator!=(const WMTime & t) const;
    bool operator<=(const WMTime & t) const;

    int operator()();
};

std::ostream& operator<< (std::ostream& o, const WMTime & wmtime) ;

#endif /* AGENT_SRC_BASICS_WMTIME_H_ */
