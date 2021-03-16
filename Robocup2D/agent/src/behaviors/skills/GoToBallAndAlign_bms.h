#ifndef _GOTOBALLANDALIGN_BMS_H_
#define _GOTOBALLANDALIGN_BMS_H_

#include "../../basics/Cmd.h"
#include "../base_bm.h"
#include "Vector.h"
#include "log_macros.h"
#include "basic_cmd_bms.h"
#include "globaldef.h"
#include "ws_info.h"
#include "GoToPos2016_bms.h"
#include "face_ball_bms.h"
#include "search_ball_bms.h" 
#include "tools.h"

class GoToBallAndAlign : public BodyBehavior
{

public:

    GoToBallAndAlign();
    virtual ~GoToBallAndAlign();

    static bool init( char const * conf_file, int argc, char const* const* argv );

    bool get_cmd( Cmd& cmd );

    bool get_cmd( Cmd& cmd, Vector& target );
    bool get_cmd_to_align_to_his_goal( Cmd& cmd );
    bool get_cmd( Cmd& cmd, Vector& target, int depthFactor );

    bool isPositioningFinished();

private:

    void reset_state();

    enum eState
    {
        go_to_ball_directly,
        go_to_nearing_pos,
        stop_on_nearing_pos,
        turn_to_target,
        approach_the_ball,
        positioning_done
    } pos_state;

    const static double max_go_directly_to_ball_dist;
    const static double go_to_dropped_perpendicular_foot_tolerance;
    const static double min_angle;
    const static double min_nearing_dist;

    static Vector HIS_GOAL;

    static bool initialized;

    GoToPos2016 *ivpGoToPos;
    FaceBall    *ivpFaceBall;
    SearchBall  *ivpSearchBall;
    BasicCmd    *ivpBasicCmd;

    int firstActTimeThisStdSituation;

    bool done_pos;
    bool already_nearing_on_line;

    double additionalMinDistFactor;

};

#endif
