#ifndef _DRIBBLE_STRAIGHT_BMS_H_
#define _DRIBBLE_STRAIGHT_BMS_H_

#include "../../basics/Cmd.h"
#include "base_bm.h"
#include "mystate.h"
#include "log_macros.h"
#include "one_step_kick_bms.h"
#include "oneortwo_step_kick_bms.h"

class DribbleStraightItrActions {
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
  ANGLE  ang_min,ang_inc;
  
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

class DribbleStraight : public BodyBehavior {
  static bool initialized;
 private:
  Cmd current_cmd;
  Cmd_Body cached_cmd;
  bool cached_res;
  long current_cmd_valid_at;
  long holdturn_not_possible_at;
  //static double success_sqrdist;
  double op_min_sqrdist;
  Vector look2pos;

  bool ballpos_ok(const MyState &state);
  bool ballpos_optimal(const MyState &state);

  OneOrTwoStepKick *onetwokick;
  OneStepKick *onestepkick;

 public:
  long last_calc;  //set to 0 to force recalculation of cmd ("empty cache")

  bool get_cmd(Cmd &cmd);
  //bool get_cmd(Cmd &cmd, const Vector tmp_look2pos);
  //bool get_cmd(Cmd & cmd, ANGLE target_dir);
  bool is_dribble_safe(int opp_time2react=1);

  bool is_dribble_safe( const AState & state, int opp_time2react );

  bool is_dribble_safe_old();

  bool calc_next_cmd(Cmd&,const MyState&);

  DribbleStraight();
  virtual ~DribbleStraight() {delete onetwokick;delete onestepkick;}
  static bool init(char const * conf_file, int argc, char const* const* argv);
};

#endif
