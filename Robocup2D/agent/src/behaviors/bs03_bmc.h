#ifndef _BS03_BMC_H_
#define _BS03_BMC_H_

#include "base_bm.h"
#include "wball06_bmp.h"
#include "noball03_bmp.h"
#include "noball03_attack_bmp.h"
#include "goal_kick_bmp.h"
#include "his_goalie_freekick_bmp.h"
#include "standard_situation_bmp.h"
#include "goalie_bs03_bmc.h"
#include "skills/neuro_go2pos_bms.h"
#include "skills/one_step_pass_bms.h"
#include "skills/intercept_ball_bms.h"
#include "line_up_bmp.h"
#include "overcome_offside_08_wball_bmc.h" 
#include "foul2010_bmp.h"
#include "MyGoalKick2016_bmp.h"
#include "penalty_bmp.h"

class Bs03: public BodyBehavior {
  static bool initialized;
  Noball03 *noball03;
  Noball03_Attack *noball03_attack;
  Wball06 *wball06;
  OneStepPass *onesteppass;
  InterceptBall *interceptball;
  GoalKick *goalkick;
  HisGoalieFreeKick *hisgoaliefreekick;
  StandardSituation *standardSit07;
  FaceBall *faceball;
  Goalie_Bs03 *goalie_bs03;
  NeuroGo2Pos *go2pos;
  LineUp *line_up;
  OvercomeOffside08Wball *overcomeOffside08wball;
  Foul2010 *foul2010;
  MyGoalKick2016 *myGoalKick;
  Penalty *avengers_Player;

  bool do_standard_kick;
  long ivLastOffSideTime;
public:
  static int cvHisOffsideCounter;
  Bs03();
  virtual ~Bs03();
  static bool init( char const * conf_file, int argc, char const* const* argv );
  void reset_intention();
  bool get_cmd( Cmd &cmd );
  void log_cmd_main( Cmd &cmd );
  void select_relevant_teammates();
  int determineCurrentPlayMode();
};


#endif
