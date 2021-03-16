//
// Created by dominik on 03.06.18.
//
#include "math.h"
#include "../../basics/tools.h"
#include "../../basics/Cmd.h"
#include "base_bm.h"
#include "mystate.h"
#include "log_macros.h"
#include "one_step_kick_bms.h"
#include "oneortwo_step_kick_bms.h"
#include "../../basics/ws_info.h"
#include "angle.h"
#include <Vector.h>
#include <behaviors/base_bm.h>
#include "tools.h"
#include "../base_bm.h"
#include "basic_cmd_bms.h"
#include "Cmd.h"

#include "base_bm.h"
#include "skills/one_step_kick_bms.h"
#include "skills/GoToPos2016_bms.h"
#include "score05_sequence_bms.h"
#include "tingletangle.h"
#include "../line_up_bmp.h"
#include "neuro_kick_wrapper_bms.h"




#ifndef C_RESOURCES_DRIBBLE2018_H
#define C_RESOURCES_DRIBBLE2018_H

class Dribble2018 : public BodyBehavior {
    static bool initialized;
    GoToPos2016 *ivpGo2PosBehavior;
    OneStepKick *ivpOneStepKickBehavior;
private:

    bool is_safe_to_kick();
    Vector currentPositionPlayer;
    Vector nextPositionPlayer;
    Vector currentPostitionBall;
    Vector nextPostionBall;

    static bool cvInitialized;
    LineUp *ivpLineUpBehavior;
    TingleTangle *ivpTingleTangleBehavior;
    NeuroKickWrapper *ivpNeuroKickBehavior;
    int ivState;


public:

    Dribble2018();
    ~Dribble2018();

    static bool init(char const * conf_file, int argc, char const* const* argv);

    void reset();

    bool get_cmd(Cmd &cmd);

    void set_target(ANGLE targetDir);
    void set_target(Vector targetPos);

    bool is_safe();

    bool ballOutOfControll();


    ANGLE kickAngle(Vector target);
    Vector nextBallPosition(Vector pos, Vector vel);
    Vector nextPlayerPosition(Vector pos, Vector vel);


    void stopBall(Cmd &cmd);
    void get_ws_state(MyState &state);
    MyState get_cur_state();
    double get_kick_decay(const MyState &state);


    //Neu
    bool checkBlindSpot();
    bool checkNearSideForDribble(Vector links, Vector rechts);
    bool checkBallInTheWay();
    bool check2stepKick();
    bool checkNotSaveForDribbleBecause2FarAway();
    bool checkPerfektBallPosition();
    Vector createVector(double winkel, double distanz);
    ANGLE angle2Target(Vector target);

    float bogen2Grad(double winkel);
};




#endif //C_RESOURCES_DRIBBLE2018_H
