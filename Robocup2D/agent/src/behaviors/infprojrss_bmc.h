#ifndef _IPRSS_BMC_H_
#define _IPRSS_BMC_H_

#include "base_bm.h"
#include "skills/score05_sequence_bms.h"
#include "line_up_bmp.h"
#include "skills/tingletangle.h"
#include "skills/neuro_dribble_2017.h"
#include "skills/intercept_ball_bms.h"
#include "skills/neuro_kick_wrapper_bms.h"
#include "skills/search_ball_bms.h"
#include "skills/GoToPos2016_bms.h"
#include "skills/Dribble2018.h"

//Fuer Beschreibungen und Kommentare siehe Datei infprojrss_bmc.c

class InfProjRSS: public BodyBehavior
{
    static bool cvInitialized;
    LineUp *ivpLineUpBehavior;
    TingleTangle *ivpTingleTangleBehavior;
    NeuroDribble2017 *ivpNeuroDribble2017Behavior;
    InterceptBall *ivpInterceptBallBehavior;
    NeuroKickWrapper *ivpNeuroKickBehavior;
    SearchBall *ivpSearchBallBehavior;
    GoToPos2016 *ivpGo2PosBehavior;
    OneStepKick *ivpOneStepKickBehavior;
    NeuroGo2Pos *ivpNeuroGo2Pos;
    Dribble2018 *ivpDribble2018;
    int ivState;

public:
    static bool init(char const * conf_file, int argc, char const* const* argv);
    InfProjRSS();
    virtual ~InfProjRSS();
    bool get_cmd(Cmd & cmd);
};

#endif
