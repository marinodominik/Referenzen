#ifndef _GLOBALDEF_H_
#define _GLOBALDEF_H_

#include "Vector.h"
#include "Player.h"
#include "geometry2d.h"
#include <math.h>

#ifndef SQUARE
#define SQUARE( x ) ( ( x ) * ( x ) )
#endif

#define RAD2DEG( x ) ( ( x / PI * 180. ) )
#define DEG2RAD( x ) ( ( double ) ( x / 180. * PI ) )


typedef double Angle;

class   Player;
typedef Player const* PPlayer;

#define PI M_PI
const double PI_MINUS_EPS = M_PI - 0.001;

const int NUM_PLAYERS = 11;

const int MAX_NAME_LEN = 512; // defines the max. length of an identifier

const int UNKNOWN_TEAM = 0;
const int MY_TEAM      = 1;
const int HIS_TEAM     = 2;

//player types
const int PT_DEFENDER = 0;
const int PT_MIDFIELD = 1;
const int PT_ATTACKER = 2;
const int PT_GOALIE   = 3;
const int PT_UNKNOWN  = 4;

const char* const PLAYERTYPE_STR[] = {
        "defender",
        "midfield",
        "attacker",
        "goalie",
        "unknown"
};

//attack roles
const int NUM_ATTACK_ROLES = 6;
const int ATTACK_ROLE_UNDEFINED         = -1;
const int ATTACK_ROLE_LEFT_ATTACKER     =  0;
const int ATTACK_ROLE_CENTER_ATTACKER   =  1;
const int ATTACK_ROLE_RIGHT_ATTACKER    =  2;
const int ATTACK_ROLE_LEFT_MIDFIELDER   =  3;
const int ATTACK_ROLE_CENTER_MIDFIELDER =  4;
const int ATTACK_ROLE_RIGHT_MIDFIELDER  =  5;

//other teams
#define TEAM_IDENTIFIER_BASE_CODE   ('a'-1)
#define TEAM_IDENTIFIER_ATHUMBOLDT  'a'
#define TEAM_IDENTIFIER_WRIGHTEAGLE 'b'
#define TEAM_IDENTIFIER_HELIOS      'c'
#define TEAM_IDENTIFIER_HERMES      'd'
#define TEAM_IDENTIFIER_GLIDERS     'e'
#define TEAM_IDENTIFIER_OXSY        'f'
#define TEAM_IDENTIFIER_CYRUS       'g'

const int NARROW = 0;
const int NORMAL = 1;
const int WIDE   = 2;

const int LOW  = 0;
const int HIGH = 1;

const int SYNCH_SEE = 1;
const int BS03_SEE  = 0;

//opponent team's goalie's types
const int HIS_GOALIE_DEFENSIVE = 0;
const int HIS_GOALIE_OFFENSIVE = 1;


const int PM_Unknown               =  0;
const int PM_my_BeforeKickOff      =  1;
const int PM_his_BeforeKickOff     =  2;
const int PM_TimeOver              =  3;
const int PM_PlayOn                =  4;
const int PM_my_KickOff            =  5;
const int PM_his_KickOff           =  6;
const int PM_my_KickIn             =  7;
const int PM_his_KickIn            =  8;
const int PM_my_FreeKick           =  9;
const int PM_his_FreeKick          = 10;
const int PM_my_CornerKick         = 11;
const int PM_his_CornerKick        = 12;
const int PM_my_GoalKick           = 13;
const int PM_his_GoalKick          = 14;
const int PM_my_AfterGoal          = 15;
const int PM_his_AfterGoal         = 16;
const int PM_Drop_Ball             = 17;
const int PM_my_OffSideKick        = 18;
const int PM_his_OffSideKick       = 19;
const int PM_Half_Time             = 20;
const int PM_Extended_Time         = 21;
const int PM_my_GoalieFreeKick     = 22;
const int PM_his_GoalieFreeKick    = 23;
const int PM_my_BeforePenaltyKick  = 24;
const int PM_his_BeforePenaltyKick = 25;
const int PM_my_PenaltyKick        = 26;
const int PM_his_PenaltyKick       = 27;
const int PM_MAX                   = 27;

const char* const PLAYMODE_STR[] = {
    "PM_Unknown",
    "PM_my_BeforeKickOff",
    "PM_his_BeforeKickOff",
    "PM_TimeOver",
    "PM_PlayOn",
    "PM_my_KickOff",
    "PM_his_KickOff",
    "PM_my_KickIn",
    "PM_his_KickIn",
    "PM_my_FreeKick",
    "PM_his_FreeKick",
    "PM_my_CornerKick",
    "PM_his_CornerKick",
    "PM_my_GoalKick",
    "PM_his_GoalKick",
    "PM_my_AfterGoal",
    "PM_his_AfterGoal",
    "PM_Drop_Ball",
    "PM_my_OffSideKick",
    "PM_his_OffSideKick",
    "PM_Half_Time",
    "PM_Extended_Time",
    "PM_my_GoalieFreeKick",
    "PM_his_GoalieFreeKick",
    "PM_my_BeforePenaltyKick",
    "PM_his_BeforePenaltyKick",
    "PM_my_PenaltyKick",
    "PM_his_PenaltyKick",
    "PM_MAX"
};

const char* play_mode_str( int play_mode );
 
const double FIELD_BORDER_X      = 52.5;
const double FIELD_BORDER_Y      = 34.0;

const double PENALTY_AREA_LENGTH = 16.5;
const double PENALTY_AREA_WIDTH  = 40.32;


const XYRectangle2d FIELD_AREA(
        Vector( 0, 0 ),
        FIELD_BORDER_X * 2.0,
        FIELD_BORDER_Y * 2.0
);

const XYRectangle2d LEFT_HALF_AREA(
        Vector( -FIELD_BORDER_X, -FIELD_BORDER_Y ),
        Vector(               0,  FIELD_BORDER_Y )
);

const XYRectangle2d LEFT_PENALTY_AREA(
        Vector( -FIELD_BORDER_X,                       -PENALTY_AREA_WIDTH * 0.5 ),
        Vector( -FIELD_BORDER_X + PENALTY_AREA_LENGTH,  PENALTY_AREA_WIDTH * 0.5 )
);

const XYRectangle2d RIGHT_HALF_AREA(
        Vector(              0, -FIELD_BORDER_Y ),
        Vector( FIELD_BORDER_X,  FIELD_BORDER_Y )
);

const XYRectangle2d RIGHT_PENALTY_AREA(
        Vector( FIELD_BORDER_X - PENALTY_AREA_LENGTH, -PENALTY_AREA_WIDTH * 0.5 ),
        Vector( FIELD_BORDER_X,                        PENALTY_AREA_WIDTH * 0.5 )
);


const Vector MY_GOAL_CENTER        = Vector( -FIELD_BORDER_X,  0.0  );
const Vector MY_GOAL_LEFT_CORNER   = Vector( -FIELD_BORDER_X,  7.01 );
const Vector MY_GOAL_RIGHT_CORNER  = Vector( -FIELD_BORDER_X, -7.01 );
const Vector HIS_GOAL_CENTER       = Vector(  FIELD_BORDER_X,  0.0  );
const Vector HIS_GOAL_LEFT_CORNER  = Vector(  FIELD_BORDER_X,  7.01 );
const Vector HIS_GOAL_RIGHT_CORNER = Vector(  FIELD_BORDER_X, -7.01 );

#ifdef TRAINING
#define DPASS 1.0
#else
#define DPASS 5
#endif
#define DPASS_DIAG DPASS / 1.414

const Vector Pass_Direct        = Vector(           0,           0 );
const Vector Pass_Forward       = Vector(       DPASS,           0 );
const Vector Pass_Backward      = Vector(      -DPASS,           0 );
const Vector Pass_Left          = Vector(           0,       DPASS );
const Vector Pass_Right         = Vector(           0,      -DPASS );
const Vector Pass_LeftForward   = Vector(  DPASS_DIAG,  DPASS_DIAG );
const Vector Pass_RightForward  = Vector(  DPASS_DIAG, -DPASS_DIAG );
const Vector Pass_LeftBackward  = Vector( -DPASS_DIAG,  DPASS_DIAG );
const Vector Pass_RightBackward = Vector( -DPASS_DIAG, -DPASS_DIAG );
const Vector Pass_LongForward   = Vector(   2 * DPASS,           0 );

const Vector Stay          = Vector(           0,           0 );
const Vector Forward       = Vector(       DPASS,           0 );
const Vector Backward      = Vector(      -DPASS,           0 );
const Vector Left          = Vector(           0,       DPASS );
const Vector Right         = Vector(           0,      -DPASS );
const Vector LeftForward   = Vector(  DPASS_DIAG,  DPASS_DIAG );
const Vector RightForward  = Vector(  DPASS_DIAG, -DPASS_DIAG );
const Vector LeftBackward  = Vector( -DPASS_DIAG,  DPASS_DIAG );
const Vector RightBackward = Vector( -DPASS_DIAG, -DPASS_DIAG );

#define DDRIBBLE 5
#define DDRIBBLE_DIAG DDRIBBLE / 1.414

const Vector Dribble_Forward       = Vector(       DDRIBBLE,              0 );
const Vector Dribble_Backward      = Vector(      -DDRIBBLE,              0 );
const Vector Dribble_Left          = Vector(              0,       DDRIBBLE );
const Vector Dribble_Right         = Vector(              0,      -DDRIBBLE );
const Vector Dribble_LeftForward   = Vector(  DDRIBBLE_DIAG,  DDRIBBLE_DIAG );
const Vector Dribble_RightForward  = Vector(  DDRIBBLE_DIAG, -DDRIBBLE_DIAG );
const Vector Dribble_LeftFForward  = Vector(       DDRIBBLE,  DDRIBBLE_DIAG );
const Vector Dribble_RightFForward = Vector(       DDRIBBLE, -DDRIBBLE_DIAG );
const Vector Dribble_LeftBackward  = Vector( -DDRIBBLE_DIAG,  DDRIBBLE_DIAG );
const Vector Dribble_RightBackward = Vector( -DDRIBBLE_DIAG, -DDRIBBLE_DIAG );



#endif
