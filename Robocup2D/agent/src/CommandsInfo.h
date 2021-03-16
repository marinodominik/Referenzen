#ifndef AGENT_SRC_COMMANDSINFO_H_
#define AGENT_SRC_COMMANDSINFO_H_

#include "basics/Cmd.h"
#include "basics/wmdef.h"
#include "sensorbuffer.h"

class CommandsInfo
{
public:
    int last_sb_time;
    int cmd_counter[   CMD_MAX ];
    int cmd_send_time[ CMD_MAX ];

    CommandsInfo();
    virtual ~CommandsInfo();

    void reset();
    void use_msg_sense_body( const Msg_sense_body &sb );
    void set_command( int time, const Cmd &cmd );
};

#endif /* AGENT_SRC_COMMANDSINFO_H_ */
