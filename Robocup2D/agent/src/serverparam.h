#ifndef _SERVERPARAM_H_
#define _SERVERPARAM_H_

#include "server_options.h"
#include "sensorparser.h"
#include "macro_msg.h"

class ServerParam
{
private:
    static Msg_server_param  *server_param;
    static Msg_player_param  *player_param;
    static Msg_player_type  **player_types;
    static Msg_player_type    worst_case_opponent_type;

public:
    static bool incorporate_server_param_string( const char *buf );
    static bool incorporate_player_param_string( const char *buf );
    static bool incorporate_player_type_string( const char *buf );

    static bool all_params_ok();

    static bool export_server_options();

    static int  number_of_player_types();

    static Msg_player_type const* get_player_type( int type );
    static Msg_player_type const* get_worst_case_opponent_type();
};
#endif
