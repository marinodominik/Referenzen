#ifndef BS2K_BEHAVIORS_MYGOALKICK2016_BMP_H_
#define BS2K_BEHAVIORS_MYGOALKICK2016_BMP_H_

#include "base_bm.h"

#include "../policy/positioning.h"

#include "blackboard.h"
#include "../basics/intention.h"
#include "../policy/policy_tools.h"
#include "mdp_info.h"


#include "wball06_bmp.h"
#include "skills/basic_cmd_bms.h"
#include "skills/face_ball_bms.h"
#include "skills/neuro_kick05_bms.h"
#include "skills/neuro_intercept_bms.h"
#include "skills/neuro_go2pos_bms.h"
#include "skills/GoToBallAndAlign_bms.h"
#include "skills/one_step_kick_bms.h"

// Derzeit führt nur der Goalie den GoalKick aus, obwohl theoretisch auch ein Spieler den GoalKick ausführen könnte.

class MyGoalKick2016: public BodyBehavior
{

public:
    MyGoalKick2016();
    virtual ~MyGoalKick2016();

    static bool init( char const *conf_file, int argc, char const* const* argv );

    bool get_cmd( Cmd &cmd );

private:
    static bool cvInitialized;

    static bool cvDebug;

    void reset_state();

    bool get_cmd_continued( Cmd &cmd );

    bool get_goalie_cmd( Cmd &cmd );
    bool get_player_cmd( Cmd &cmd );

    int getTimeLeft();
    bool getPassIntention( Intention &intention );

    Vector getTargetNonStartPlayer();
    bool getCmdReactOnPassInfo( Cmd&, PPlayer, Vector& );
    Vector getBestDirectScorePos();

    BasicCmd         *ivpBasicCmd;
    Wball06          *ivpWBall;
    FaceBall         *ivpFaceBall;
    NeuroKick05      *ivpNeuroKick;
    NeuroIntercept   *ivpIntercept;
    GoToPos2016      *ivpGoToPos;
    GoToBallAndAlign *ivpGoToBallAndAlign;
    OneStepKick      *ivpOneStepKick;

    bool goalKickAlreadyDone;

    int ivLastActTime;
    int ivActGoalKickStartTime;
    int ivActGoalKickDuration;
    int ivTimeLeft;

    Intention ivLastIntention;
    bool   ivContinueWBallBehavior;
    double ivLastKickSpeed;
    Vector ivLastKickTarget;
    int    ivLastTimeOfRequestToOfferMySelf;

    int ivConsecKicks;
    static const int cvMAX_CONSEC_KICKS = 3;

    static const int cvPASS_ANNOUNCE_TIME = 3;

    static const int cvPANICKICK_TRESHOLD = 10;

    static const int cvMAX_BALL_AGE_WHEN_KICK = 4;

    int ivStandardWaitTime;
    static int cvDefaultWaitTime;
    static int cvMinWaitTime;
    static int cvMaxWaitTime;
    static int cvWaitOnLowStaminaTime;

    Vector ivMyPanickickLeftCornerTarget;
    Vector ivMyPanickickRightCornerTarget;
    static Vector cvZeroVector;

    static const int cvTIME_LOW_THRESHOLD = 20; // Muss noch ein sinnvoller Wert für gesucht werden!!
    static const int cvLIMITED_DASH_POWER = 45;

    static const double cvHOMEPOS_TOLERANCE;
    static int cvHomeposStaminaMin;

    static double cvDummyVal;
    static int    cvDummyInt;
    static Vector cvDummyVec;

};

#endif /* BS2K_BEHAVIORS_REAL_GOAL_KICK_BMP_H_ */
