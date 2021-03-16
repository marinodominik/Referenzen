#ifndef _TRAIN_BEHAVIORS_BMC_H_
#define _TRAIN_BEHAVIORS_BMC_H_

#include "base_bm.h"
#include "noball03_attack_bmp.h"
#include "learn_wball_bmp.h"
#include "noball03_bmp.h"
#include "goal_kick_bmp.h"
#include "standard_situation_bmp.h"
#include "goalie_bs03_bmc.h"
#include "skills/neuro_go2pos_bms.h"
#include "line_up_bmp.h"

#define TRAINING

class TrainBehaviors: public BodyBehavior {
  static bool initialized;
  Noball03_Attack *noball03_attack;
  Noball03 *noball03;
  LearnWball *learn_wball;
  GoalKick *goalkick;
  StandardSituation * standardSit;
  FaceBall *faceball;
  Goalie_Bs03 *goalie_bs03;
  NeuroGo2Pos *go2pos;
  LineUp *line_up;
  
  bool do_standard_kick;
  bool do_goal_kick;

public:

    TrainBehaviors();
    virtual ~TrainBehaviors();

    static bool init( char const * conf_file, int argc, char const* const * argv );

    bool get_cmd( Cmd & cmd );

    void reset_intention();
};

#endif
