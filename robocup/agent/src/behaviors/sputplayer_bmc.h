#ifndef _SPUTPLAYER_BMC_H_
#define _SPUTPLAYER_BMC_H_

#include "base_bm.h"
#include "skills/neuro_go2pos_bms.h"
#include "skills/neuro_kick_bms.h"
#include "skills/one_step_kick_bms.h"
#include "skills/oneortwo_step_kick_bms.h"
#include "skills/intercept_ball_bms.h"
#include "skills/basic_cmd_bms.h"
#include "skills/dribble_straight_bms.h"
#include "skills/face_ball_bms.h"
#include "skills/search_ball_bms.h"
#include "skills/onetwo_holdturn_bms.h"
#include "skills/selfpass_bms.h"
#include "skills/neuro_intercept_bms.h"

#include "ws_info.h"
#include "log_macros.h"
#include "tools.h"
#include "ws_memory.h"

class SputPlayer: public BodyBehavior {
  static bool initialized;

  NeuroKick *neurokick;
  NeuroGo2Pos *go2pos;
  OneStepKick *onestepkick;
  OneOrTwoStepKick *oneortwo;
  NeuroIntercept *intercept;
  BasicCmd *basic;
  OneTwoHoldTurn *holdturn;
  DribbleStraight *dribblestraight;
  SearchBall *searchball;
  FaceBall *faceball;
  Selfpass *selfpass;

  bool flg;
  
public:
  static bool init(char const * conf_file, int argc, char const* const* argv);

  SputPlayer();
  virtual ~SputPlayer();

  bool get_cmd(Cmd & cmd);
};

#endif
