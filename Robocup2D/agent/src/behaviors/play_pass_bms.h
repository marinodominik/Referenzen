#ifndef _PLAYPASS_BMS_H_
#define _PLAYPASS_BMS_H_

#include "base_bm.h"
#include "skills/basic_cmd_bms.h"
#include "skills/neuro_kick05_bms.h"
//#include "skills/selfpass_bms.h"
//#include "skills/dribble_quickly_bms.h"
//#include "skills/dribble_bms.h"
//#include "skills/dribble_straight_bms.h"
#include "skills/one_step_kick_bms.h"
#include "skills/onetwo_holdturn_bms.h"
#include "skills/oneortwo_step_kick_bms.h"
//#include "skills/score_bms.h"
#include "../policy/abstract_mdp.h"
#include "intention.h"
#include "neuro_wball.h"

class PlayPass: public BodyBehavior {
  NeuroKick05 *neurokick;
  //DribbleQuickly *dribblequickly;
  //Dribble *dribble;
  //DribbleStraight *dribblestraight;
  //Selfpass *selfpass;
  BasicCmd *basiccmd;
  OneStepKick *onestepkick;
  OneOrTwoStepKick *oneortwo;
  OneTwoHoldTurn *onetwoholdturn;
  //Score *score;
  NeuroWball *neuro_wball;

 private:
  bool reconsider_goalshot;
  int wait_and_see_patience;
  double wait_and_see_clearanceline;
  int at_ball_patience;
  long last_at_ball;
  int at_ball_for_cycles;
  double flank_param;
  bool use_handicap_goalshot_test;
  float handicap;
  float safety_handicap_sub;
  float riskyDisToPole5;
  float riskyDisToPole30;
  float decisionHandicap;
  int lastTimeLookedForGoalie;
  int lastTimeLookedForGoal;
  float intendedLookDirection;
  bool use_handicap_selfpasses;
  float selfpasses_handicap;
  float selfpasses_safety_level;
  float selfpasses_SHORT_safety_level;
  float selfpasses_SHORT_max_speed;
  bool do_refine_selfpasses;
  //int lasttime_in_waitandsee;
  int cyclesI_looked2goal;
  int cycles_in_waitandsee;
  long last_waitandsee_at;
  int evaluation_mode;
  //int check_action_mode;
  //int action_set_type;
  //int exploration_mode;
  //float exploration;
  float success_threshold;
  float dribble_success_threshold;
	int current_advantage;  // CIJAT OSAKA
  int my_role;
  PPlayer closest_opponent; // warning: might be 0!

  struct{
    Intention pass_or_dribble_intention;
    Intention intention;
    NeckRequest neckreq;
  } my_blackboard;


  bool is_planned_pass_a_killer;
  bool get_turn_and_dash(Cmd &cmd);
  bool intention2cmd(Intention &intention, Cmd &cmd);

  bool test_priority_pass(Intention &intention);
  bool test_default(Intention &intention);
  bool test_two_teammates_control_ball(Intention &intention);
  bool test_in_trouble(Intention &intention);
  bool test_opening_seq(Intention &intention);


  bool I_can_advance_behind_offside_line();

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
	inline int get_advantage(){return current_advantage;} 	// CIJAT OSAKA
  static bool init(char const * conf_file, int argc, char const* const* argv) {
    return (
	    NeuroKick05::init(conf_file,argc,argv) &&
	    BasicCmd::init(conf_file,argc,argv) &&
	    OneStepKick::init(conf_file,argc,argv) &&
	    OneOrTwoStepKick::init(conf_file,argc,argv) &&
	    OneTwoHoldTurn::init(conf_file,argc,argv)
	    );
  }
  PlayPass();
  virtual ~PlayPass();
  bool get_cmd(Cmd & cmd);
  //void reset_intention();
};


#endif
