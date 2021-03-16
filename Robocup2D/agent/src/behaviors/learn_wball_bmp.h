#ifndef _LEARN_WBALL_BMC_H_
#define _LEARN_WBALL_BMC_H_

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

class LearnWball: public BodyBehavior {
  NeuroKick *neurokick;
  DribbleStraight *dribblestraight;
  Selfpass *selfpass;
  BasicCmd *basiccmd;
  OneStepKick *onestepkick;
  OneOrTwoStepKick *oneortwo;
  OneTwoHoldTurn *onetwoholdturn;
  Score *score;
  NeuroWball *neuro_wball;

 private:
  bool in_penalty_mode;
  bool reconsider_goalshot;
  int wait_and_see_patience;
  double wait_and_see_clearanceline;
  int at_ball_patience;
  long last_at_ball;
  int at_ball_for_cycles;
  double flank_param;
  int lastTimeLookedForGoalie;
  int lastTimeLookedForGoal;
  float intendedLookDirection;
  int lasttime_in_waitandsee;
  int cyclesI_looked2goal;
  int cycles_in_waitandsee;
  long last_waitandsee_at;
  int evaluation_mode;
  int check_action_mode;
  int action_set_type;
  int exploration_mode;
  float exploration;
  float success_threshold;
  float dribble_success_threshold;
  int my_role;
  PPlayer closest_opponent; // warning: might be 0!

  struct{
    Intention pass_or_dribble_intention;
    Intention intention;
    NeckRequest neckreq;
  } my_blackboard;


  struct{
    long valid_at;
    ANGLE targetdir;
    Vector ipos;
    double speed;
    int steps;
    Vector op_pos;
    int op_number;
  } best_selfpass;

  bool is_planned_pass_a_killer;
  bool get_turn_and_dash(Cmd &cmd);
  bool intention2cmd(Intention &intention, Cmd &cmd);

  bool test_solo(Intention &intention);
  bool test_advance_slowly(Intention &intention);
  bool test_advance(Intention &intention);
  bool test_priority_pass(Intention &intention);
  bool test_default(Intention &intention);
  bool test_holdturn(Intention &intention);
  bool test_kicknrush(Intention &intention);
  bool test_two_teammates_control_ball(Intention &intention);
  bool test_in_trouble(Intention &intention);
  bool test_opening_seq(Intention &intention);

  bool get_best_panic_selfpass(const double testdir[],const int num_dirs,double &speed, double &dir);


  bool I_can_advance_behind_offside_line();
  void get_best_selfpass();
  bool selfpass_dir_ok(const ANGLE dir );

  bool check_selfpass(const ANGLE targetdir, double &ballspeed, Vector &target, int &steps,
		      Vector &op_pos, int &op_num);
  void set_neck_selfpass(const ANGLE targetdir, const Vector &op_pos);

  bool is_dribblestraight_possible();

  bool test_pass_or_dribble(Intention &intention);
  bool check_previous_intention(Intention prev_intention, Intention  &new_intention);

  void check_write2blackboard();
  bool get_pass_or_dribble_intention(Intention &intention);
  bool get_pass_or_dribble_intention(Intention &intention, AState &state);
  
  void aaction2intention(const AAction& aaction, Intention &intention);


  // auxillary functions for offensive_move:
  void get_onestepkick_params(double &speed, double &dir);
  void get_kickrush_params(double &speed, double &dir);
  void get_kickrush_params(double &speed, double &dir, Vector &ipos, int &advantage,
			   int & closest_teammate);
  void get_opening_pass_params(double &speed, double &dir, Vector &ipos, int &advantage,
			       int & closest_teammate);

  void get_clearance_params(double &speed, double &dir);
  bool check_kicknrush(double &speed, double &dir, bool &safe, Vector &resulting_pos);


  bool get_opening_seq_cmd( const float  speed, const Vector target,Cmd &cmd);
  double adjust_speed(const Vector ballpos, const double dir, const double speed);
  bool is_pass_a_killer();


 protected:
 public:
  bool get_intention(Intention &intention);
  static bool init(char const * conf_file, int argc, char const* const* argv) {
    return (
	    NeuroKick::init(conf_file,argc,argv) &&
	    DribbleStraight::init(conf_file,argc,argv) &&
	    Selfpass::init(conf_file,argc,argv) &&
	    BasicCmd::init(conf_file,argc,argv) &&
	    OneStepKick::init(conf_file,argc,argv) &&
	    OneOrTwoStepKick::init(conf_file,argc,argv) &&
	    OneTwoHoldTurn::init(conf_file,argc,argv)
	    );
  }
  LearnWball();
  virtual ~LearnWball();
  bool get_cmd(Cmd & cmd);
  void reset_intention();
};


#endif
