#ifndef AGENT_SRC_MESSAGESINFO_H_
#define AGENT_SRC_MESSAGESINFO_H_

#include "basics/wmdef.h"

class MessagesInfo
{
public:
    struct _msg
    {
        int  received;
        int  cycle;
        bool processed;
        long ms_time;
    } msg[ MESSAGE_MAX ];

    MessagesInfo();
    virtual ~MessagesInfo();

    void reset();
    void set_cycle( int c );
    void set_ms_time( long ms_time );
};

#endif /* AGENT_SRC_MESSAGESINFO_H_ */
