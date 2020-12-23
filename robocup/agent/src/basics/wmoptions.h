#ifndef _WMOPTIONS_H_
#define _WMOPTIONS_H_

#include "globaldef.h"

class WMoptions
{
public:
    static bool offline; //don't connect to the server, just start the first behavior

    static long ms_delta_max_wait_synch_see_long;
    static long ms_delta_max_wait_synch_see_short;
    static long ms_time_max_wait_after_sense_body_long;
    static long ms_time_max_wait_after_sense_body_short;
    static long ms_time_max_wait_select_interval;
    static long s_time_normal_select_interval;

    static bool foresee_opponents_positions;

    static bool use_fullstate_instead_of_see;
    static bool behave_after_fullstate;
    static bool behave_after_think;
    static bool disconnect_if_idle;
    static bool send_teamcomm;
    static bool recv_teamcomm;
    static bool ignore_fullstate;
    static bool ignore_sense_body;
    static bool ignore_see;
    static bool ignore_hear;
    static bool use_pfilter;
    static bool use_server_based_collision_detection;

    static int  max_cycles_to_forget_my_player;
    static int  max_cycles_to_forget_his_player;

    static bool save_msg_times;
    static int  his_goalie_number; //this is especially usefull for pure fullstate agents, where no goalie info is sended



    static void read_options( char const* file, int argc, char const* const * argv );

    static void set_mode_competition();
    static void set_mode_test();
    static void set_mode_synch_mode();
    static void set_mode_synch_mode_with_fullstate();
};

#endif
