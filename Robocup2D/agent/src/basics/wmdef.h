#ifndef _WMDEF_H_
#define _WMDEF_H_

const double PITCH_LENGTH        = 105.0;
const double PITCH_WIDTH         =  68.0;
const double PITCH_MARGIN        =   5.0;
//const double PENALTY_AREA_LENGTH = 16.5;
//const double PENALTY_AREA_WIDTH  = 40.32;

//the following values must be >= 0 (because of the encoding scheme in manual_encode_teamcomm)
const int left_SIDE    = 0;
const int right_SIDE   = 1;
const int unknown_SIDE = 2;

const int my_TEAM      = 0; //never change this value, my_TEAM must be 0
const int his_TEAM     = 1; //never change this value, his_TEAM must be 1
const int unknown_TEAM = 2; //never change this value, unknown_TEAM must be 2

const char* const TEAM_STR[] = {
        "MY_TEAM_",
        "HIS_TEAM_",
        "???_TEAM_" };

const int CMD_MAIN_MOVETO =  0;
const int CMD_MAIN_TURN   =  1;
const int CMD_MAIN_DASH   =  2;
const int CMD_MAIN_KICK   =  3;
const int CMD_MAIN_CATCH  =  4;
const int CMD_MAIN_TACKLE =  5;
const int CMD_NECK_TURN   =  6;
const int CMD_SAY         =  7;
const int CMD_VIEW_CHANGE =  8;
const int CMD_INVALID     =  9;
const int CMD_MAX         = 10;

const char* const CMD_STRINGS[ CMD_MAX ] = {
        "moveto",
        "turn",
        "dash",
        "kick",
        "catch",
        "tackle",
        "chg_neck",
        "say",
        "chg_view",
        "[invalid]" };

const int MESSAGE_SENSE_BODY         =  0;
const int MESSAGE_SEE                =  1;
const int MESSAGE_FULLSTATE          =  2;
const int MESSAGE_THINK              =  3;
const int MESSAGE_HEAR               =  4;
const int MESSAGE_INIT               =  5;
const int MESSAGE_SERVER_PARAM       =  6;
const int MESSAGE_PLAYER_PARAM       =  7;
const int MESSAGE_PLAYER_TYPE        =  8;
const int MESSAGE_CHANGE_PLAYER_TYPE =  9;
const int MESSAGE_ERROR              = 10;
const int MESSAGE_OK                 = 11;
const int MESSAGE_BEFORE_BEHAVE      = 12;
const int MESSAGE_AFTER_BEHAVE       = 13;
const int MESSAGE_CMD_LOST           = 14;
const int MESSAGE_CMD_SENT           = 15;
const int MESSAGE_DUMMY1             = 16;
const int MESSAGE_DUMMY2             = 17;
const int MESSAGE_UNKNOWN            = 18;
const int MESSAGE_MAX                = 19;

const char* const MESSAGE_STRINGS[ MESSAGE_MAX ] = {
        "[sense_body]",
        "[see]",
        "[fullstate]",
        "[think]",
        "[hear]",
        "[init]",
        "[server_param]",
        "[player_param]",
        "[player_type]",
        "[change_player_type]",
        "[error]",
        "[ok]",
        "[before_behave]",
        "[after_behave]",
        "[cmd_lost]",
        "[cmd_sent]",
        "[dummy1]",
        "[dummy2]",
        "[unknown]" };

/** playmodes
 note that these are not the same as in the MDPstate
 */
typedef enum _PlayMode
{
    PM_Null,
    PM_before_kick_off,
    PM_time_over,
    PM_play_on,
    PM_kick_off_l,
    PM_kick_off_r,
    PM_kick_in_l,
    PM_kick_in_r,
    PM_free_kick_l,
    PM_free_kick_r,
    PM_corner_kick_l,
    PM_corner_kick_r,
    PM_goal_kick_l,
    PM_goal_kick_r,
    PM_goal_l,
    PM_goal_r,
    PM_drop_ball,
    PM_offside_l,
    PM_offside_r,
    PM_goalie_catch_ball_l,
    PM_goalie_catch_ball_r,
    //following modes are not propagated, or transformed before propagation to the players
    PM_free_kick_fault_l,
    PM_free_kick_fault_r,
    PM_back_pass_l,
    PM_back_pass_r,
    //new in 9.4.2,
    PM_penalty_kick_l,
    PM_penalty_kick_r,
    PM_catch_fault_l,
    PM_catch_fault_r,
    PM_indirect_free_kick_l,
    PM_indirect_free_kick_r,
    PM_penalty_setup_l,
    PM_penalty_setup_r,
    PM_penalty_ready_l,
    PM_penalty_ready_r,
    PM_penalty_taken_l,
    PM_penalty_taken_r,
    PM_penalty_miss_l,
    PM_penalty_miss_r,
    PM_penalty_score_l,
    PM_penalty_score_r,
    PM_penalty_onfield_l,
    PM_penalty_onfield_r,
    PM_penalty_foul_l,
    PM_penalty_foul_r,
    PM_penalty_winner_l,
    PM_penalty_winner_r,
    PM_penalty_draw,
    //old
    PM_half_time,
    PM_time_up,
    PM_time_extended
} PlayMode;

/**
 * JTS 10: insert enum for card types
 */
typedef enum _Card
{
    NO_CARD,
    YELLOW_CARD,
    RED_CARD
} Card;

inline char const* show_play_mode( PlayMode pm )
{
    switch( pm )
    {
        case PM_Null :                 return "pm_null";
        case PM_before_kick_off :      return "before_kick_off";
        case PM_time_over :            return "time_over";
        case PM_play_on :              return "play_on";
        case PM_kick_off_l :           return "kick_off_l";
        case PM_kick_off_r :           return "kick_off_r";
        case PM_kick_in_l :            return "kick_in_l";
        case PM_kick_in_r :            return "kick_in_r";
        case PM_free_kick_l :          return "free_kick_l";
        case PM_free_kick_r :          return "free_kick_r";
        case PM_corner_kick_l :        return "corner_kick_l";
        case PM_corner_kick_r :        return "corner_kick_r";
        case PM_goal_kick_l :          return "goal_kick_l";
        case PM_goal_kick_r :          return "goal_kick_r";
        case PM_goal_l :               return "goal_l";
        case PM_goal_r :               return "goal_r";
        case PM_drop_ball :            return "drop_ball";
        case PM_offside_l :            return "offside_l";
        case PM_offside_r :            return "offside_r";
        case PM_goalie_catch_ball_l :  return "goalie_catch_ball_l";
        case PM_goalie_catch_ball_r :  return "goalie_catch_ball_r";
        case PM_free_kick_fault_l :    return "free_kick_fault_l";
        case PM_free_kick_fault_r :    return "free_kick_fault_r";
        case PM_back_pass_l :          return "back_pass_l";
        case PM_back_pass_r :          return "back_pass_r";
        case PM_penalty_kick_l :       return "penalty_kick_l";
        case PM_penalty_kick_r :       return "penalty_kick_r";
        case PM_catch_fault_l :        return "catch_fault_l";
        case PM_catch_fault_r :        return "catch_fault_r";
        case PM_indirect_free_kick_l : return "indirect_free_kick_l";
        case PM_indirect_free_kick_r : return "indirect_free_kick_r";
        case PM_penalty_setup_l :      return "penalty_setup_l";
        case PM_penalty_setup_r :      return "penalty_setup_r";
        case PM_penalty_ready_l :      return "penalty_ready_l";
        case PM_penalty_ready_r :      return "penalty_ready_r";
        case PM_penalty_taken_l :      return "penalty_taken_l";
        case PM_penalty_taken_r :      return "penalty_taken_r";
        case PM_penalty_miss_l :       return "penalty_miss_l";
        case PM_penalty_miss_r :       return "penalty_miss_r";
        case PM_penalty_score_l :      return "penalty_score_l";
        case PM_penalty_score_r :      return "penalty_score_r";
        case PM_penalty_onfield_l :    return "penalty_onfield_l";
        case PM_penalty_onfield_r :    return "penalty_onfield_r";
        case PM_penalty_foul_l :       return "penalty_foul_l";
        case PM_penalty_foul_r :       return "penalty_foul_r";
        case PM_penalty_winner_l :     return "penalty_winner_l";
        case PM_penalty_winner_r :     return "penalty_winner_r";
        case PM_penalty_draw :         return "penalty_draw";
        case PM_half_time :            return "half_time";
        case PM_time_up :              return "time_up";
        case PM_time_extended :        return "time_extended";
    }
    return "error_unknown";
}

#endif
