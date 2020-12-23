#ifndef ONEOR2STEP_HOLDTURN_BMS_H_
#define ONEOR2STEP_HOLDTURN_BMS_H_

#include "../../basics/Cmd.h"
#include "../base_bm.h"
#include "mystate.h"
#include "log_macros.h"


class HoldTurnItrActions {
  /* set params */

  static const double kick_pwr_min;
  static const double kick_pwr_inc;
  static const double kick_pwr_steps;
  static const double kick_fine_pwr_inc;
  static const double kick_fine_pwr_steps;
  static const double kick_ang_min;
  static const double kick_ang_inc;
  static const double kick_ang_steps;
  static const double kick_fine_ang_inc;
  static const double kick_fine_ang_steps;

  double pwr_min,pwr_inc,pwr_steps,ang_steps;
  ANGLE ang_min,ang_inc;
  
  Cmd_Body action;
  int ang_done,pwr_done;
  double pwr;ANGLE ang;
 public:
  void reset(bool finetune=false,double orig_pwr=0,ANGLE orig_ang= ANGLE(0)) {
    if(!finetune) {
      ang_min = ANGLE(kick_ang_min);
      ang_inc = ANGLE(kick_ang_inc);
      ang_steps = kick_ang_steps;
      pwr_min = kick_pwr_min;
      pwr_inc = kick_pwr_inc;
      pwr_steps = kick_pwr_steps;
    } else {
      ang_min = orig_ang- ANGLE(.5*kick_fine_ang_steps*kick_fine_ang_inc);
      ang_inc = ANGLE(kick_fine_ang_inc);
      ang_steps = kick_fine_ang_steps + 1;
      pwr_min = orig_pwr-(.5*kick_fine_pwr_steps*kick_fine_pwr_inc);
      pwr_inc = kick_fine_pwr_inc;
      pwr_steps = kick_fine_pwr_steps + 1;
    }
    ang_done = 0;pwr_done=0;
    ang = ang_min; pwr = pwr_min;
  }
  
  Cmd_Body *next() {
    if(pwr_done<pwr_steps && ang_done<ang_steps) {
      action.unset_lock();
      action.unset_cmd();
      action.set_kick(pwr,ang.get_value_mPI_pPI());
      ang+=ang_inc;
      if(++ang_done>=ang_steps) {
	ang=ang_min;ang_done=0;
	pwr+=pwr_inc;
	pwr_done++;
      }
      return &action;
    }
    return NULL;
  }
};

class OneTwoHoldTurn : public BodyBehavior {
  static bool initialized;
 private:
  Cmd current_cmd;
  long current_cmd_valid_at;
  long holdturn_not_possible_at;
  bool relaxed_checking;
  double success_sqrdist;
  double op_min_sqrdist;
  Vector look2pos;
  bool check_1stepturn2goal(Cmd & cmd, const MyState &state);

  bool get_2step_cmd(Cmd & cmd, double &sqrdist, int &steps, const MyState &state);
  bool compute_2step_cmd(Cmd & cmd, double &sqrdist, int &steps, const MyState &state);

  bool get_emergency_cmd(Cmd & cmd, const MyState &state);

  bool ballpos_ok(const MyState &state);
  bool ballpos_optimal(const MyState &state);
  bool get_cmd(Cmd &cmd, const Vector tmp_look2pos, const bool relaxed_check);

 public:
  bool get_cmd(Cmd &cmd);
  bool get_cmd_relaxed(Cmd &cmd);
  bool get_cmd(Cmd & cmd, ANGLE target_dir);
  bool is_holdturn_safe();
 
  OneTwoHoldTurn();
  virtual ~OneTwoHoldTurn() {};
  static bool init(char const * conf_file, int argc, char const* const* argv); 
};

#endif
