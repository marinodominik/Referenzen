#ifndef _ATTACK_MOVE1_WB_BMC_H_
#define _ATTACK_MOVE1_WB_BMC_H_

#include "base_bm.h"
#include "skills/basic_cmd_bms.h"
#include "skills/neuro_kick_bms.h"
#include "skills/selfpass_bms.h"
#include "skills/dribble_straight_bms.h"
#include "skills/one_step_kick_bms.h"
#include "skills/onetwo_holdturn_bms.h"
#include "skills/oneortwo_step_kick_bms.h"
#include "skills/score_bms.h"
#include "../policy/abstract_mdp.h"
#include "intention.h"
#include "neuro_wball.h"

class Attack_Move1_Wb: public BodyBehavior {
 public:
  Attack_Move1_Wb();
  virtual ~Attack_Move1_Wb();
  bool get_cmd(Cmd & cmd);
  static bool do_move();
};


#endif
