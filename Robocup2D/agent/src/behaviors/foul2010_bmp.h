#ifndef _FOUL2010_BMP_H_
#define _FOUL2010_BMP_H_

/*
Thomas Huber, 2010
This behavior will decide whether to foul or not

*/
#include <stdio.h>
#include <stdlib.h>

#include "../basics/Cmd.h"
#include "log_macros.h"
#include "tools.h"
#include "options.h"
#include "ws_info.h"
#include "log_macros.h"
#include "ws_memory.h"
#include "skills/basic_cmd_bms.h"

class Foul2010 : public BodyBehavior{

 private:
  BasicCmd  *basiccmd;
  ANGLE     ivFoulAngle;
  double    ivMaxBallVelAfterFoul;
  int       ivLastFoulExecuted;
  int       ivLastExecutedDecideForSituation;
  
 protected:
  bool decide_situation();
  Vector evaluateFoulDirection( double checkAngle,
                                ANGLE  myAngle,
                                Vector myPos,
                                Vector ballPos,
                                Vector ballVel,
                                int    foulGoal );
  bool getMinimalBallVelAfterFoul(bool ballHasToBeFree);
  bool stop_ball();
  bool get_ball();
 
 
 public: 
  static bool initialized;
  bool get_cmd(Cmd &cmd);
  static bool init(char const * conf_file, int argc, char const* const* argv);
  void  set_maxBallVelAfterFoul(double maxBallVelAfterFoul);
  double  get_maxBallVelAfterFoul();
  bool foul_situation();
  bool foul_situation(double minProbability);
  Foul2010();
  virtual ~Foul2010();
};

#endif
