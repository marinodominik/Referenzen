#ifndef _GOALIE_NECK_BMN_H_
#define _GOALIE_NECK_BMN_H_

#include "neck_cmd_bms.h"

#include "../policy/policy_tools.h"
#include "tools.h"

class GoalieNeck : public NeckBehavior
{

private:

    static bool initialized;

    NeckCmd *neck_cmd;

    Go2Ball_Steps* go2ball_list;
    struct Steps2Go
    {
        float  me;
        float  my_goalie;
        float  teammate;
        int    teammate_number;
        Vector teammate_pos;
        bool   ball_kickable_for_teammate;
        float  opponent;
        int    opponent_number;
        Vector opponent_pos;
        bool   ball_kickable_for_opponent;
    } steps2go;

    int    time_of_turn;
    int    time_attacker_seen;
    int    last_time_look_to_ball;
    double turnback2_scandir;

    void compute_steps2go();

    bool goalie_neck(   Cmd &cmd );
    bool neck_standard( Cmd &cmd );
    bool neck_lock(     Cmd &cmd );



public:

    GoalieNeck();
    virtual ~GoalieNeck();

    bool get_cmd( Cmd &cmd );

    static bool init( char const * conf_file, int argc, char const* const * argv );
};

#endif //_GOALIE_NECK_BMN_H_
