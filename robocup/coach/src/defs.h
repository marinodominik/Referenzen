#ifndef _DEFS_H_
#define _DEFS_H_

/****************************************************************************
 *
 * Author: Manuel "Sputnick" Nickschas
 *
 * Global definitions for use with SputCoach.
 *
 ****************************************************************************/

#include "Vector.h"
#include <math.h>
#include "angle.h"

#define PI M_PI
typedef double Angle;

const int NUM_PLAYERS= 11;

#define MAX_PLAYER_TYPES 18
#define TEAM_SIZE 11
#define SPEED_PROGRESS_MAX 15

#define MAX_SAY_TYPES 5  // Info, Advice, Define, Meta, Freeform

///defines the max. length of an identifier 
const int MAX_NAME_LEN= 512;

const double FIELD_BORDER_X = 52.5;
const double FIELD_BORDER_Y = 34.0;
const Vector MY_GOAL_LEFT_CORNER= Vector(-FIELD_BORDER_X,7.32);
const Vector MY_GOAL_RIGHT_CORNER= Vector(-FIELD_BORDER_X,-7.32);
const Vector HIS_GOAL_LEFT_CORNER= Vector(FIELD_BORDER_X,7.32);
const Vector HIS_GOAL_RIGHT_CORNER= Vector(FIELD_BORDER_X,-7.32);
const Vector HIS_GOAL_CENTER= Vector(FIELD_BORDER_X,0.0);
#define RAD2DEG(x)((int)(x/PI*180.))
#define DEG2RAD(x)((double)(x/180.*PI))
#define KICK_RANGE          1.085

const double PITCH_LENGTH        = 105.0;
const double PITCH_WIDTH         =  68.0;
const double PITCH_MARGIN        =   5.0;
const double PENALTY_AREA_LENGTH =  16.5;
const double PENALTY_AREA_WIDTH  = 40.32;

// JTS10 definitions for pointto, must be in alignment with pointto-behavior
const Angle PT_STAMINA_CAP_ANG_INC = DEG2RAD(1.f);
const Angle PT_STAMINA_CAP_START_ANG = 5.0*PT_STAMINA_CAP_ANG_INC;
const int PT_STAMINA_CAP_DISCRETIZATION_STEP = 400;
const int PT_STAMINA_CAP_DISCRETIZATION_MAX = int(fabs( (2.f * PI -   2.f * PT_STAMINA_CAP_START_ANG)
                                                         / PT_STAMINA_CAP_ANG_INC
                                                         * PT_STAMINA_CAP_DISCRETIZATION_STEP ));
//should be read from server parameters => TODO
const int PLAYER_STAMINA_CAPACITY = 130600;
                                                         
//the following values must be >= 0 (because of the encoding scheme in manual_encode_teamcomm)
const int left_SIDE    = 0;
const int right_SIDE   = 1;
const int unknown_SIDE = 2;

const int my_TEAM      = 0;
const int his_TEAM     = 1;
const int unknown_TEAM = 2;

const char * const TEAM_STR[]= { " MY_TEAM_", "HIS_TEAM_", "???_TEAM_" };

const char * const MSG_TYPE_STR[]= {"info","advice","define","meta","freeform","dummy","before_kick_off"};
enum MSG_TYPES{MSGT_INFO,MSGT_ADVICE,MSGT_DEFINE,MSGT_META,MSGT_FREEFORM};

//other teams, must be the same as in the agent's file globaldef.h
#define  TEAM_IDENTIFIER_BASE_CODE   ('a'-1)
#define  TEAM_IDENTIFIER_ATHUMBOLDT  'a'
#define  TEAM_IDENTIFIER_WRIGHTEAGLE 'b'
#define  TEAM_IDENTIFIER_HELIOS      'c'
#define  TEAM_IDENTIFIER_HERMES      'd'
#define  TEAM_IDENTIFIER_GLIDERS     'e'
#define  TEAM_IDENTIFIER_OXSY        'f'
#define  TEAM_IDENTIFIER_CYRUS       'g'

/** playmodes
    note that these are not the same as in the MDPstate
*/
enum PlayMode {   // RM == referee message (no play mode!)
  PM_Null,PM_before_kick_off,PM_play_on,PM_time_over,PM_kick_off_l,PM_kick_off_r,        
  PM_kick_in_l,PM_kick_in_r,PM_free_kick_l,PM_free_kick_r,PM_corner_kick_l,PM_corner_kick_r,     
  PM_goal_kick_l,PM_goal_kick_r,/*PM_goal_l,*/ /*PM_goal_r,*/ /*PM_drop_ball,*/         
  PM_offside_l,PM_offside_r,PM_indirect_free_kick_l,PM_indirect_free_kick_r,/*PM_MAX,*/
  RM_goal_l,RM_goal_r,RM_foul_l,RM_foul_r,RM_goalie_catch_ball_l,RM_goalie_catch_ball_r,
  RM_time_up_without_a_team,RM_time_up,RM_half_time,RM_time_extended,RM_drop_ball,
  RM_foul_charge_l,RM_foul_charge_r,
  RM_MAX             
};

const int PM_MAX = 16;

const char *const playmodeStrings[] = {"pm_null","before_kick_off","play_on","time_over",
				       "kick_off_l","kick_off_r","indirect_free_kick_r"/*"kick_in_l"*/,"kick_in_r",
				       "free_kick_l","free_kick_r","corner_kick_l",
				       "corner_kick_r","goal_kick_l","goal_kick_r",
				       /*"goal_l",*/ /*"goal_r",*/ /*"drop_ball",*/
				       "offside_l","offside_r","indirect_free_kick_l","indirect_free_kick_r",
				       "xxxxxxxxxxx",
				       "goal_l_","goal_r_","foul_l","foul_r",
				       "goalie_catch_ball_l","goalie_catch_ball_r",
				       "time_up_without_a_team","time_up",
				       "half_time","time_extended",
				       "drop_ball",
                                       "foul_charge_l", "foul_charge_r"};


#endif
