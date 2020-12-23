#ifndef _NEURO_GO2POS_BMS_H_
#define _NEURO_GO2POS_BMS_H_

#include "../../basics/Cmd.h"
#include "base_bm.h"
#include "angle.h"
#include "Vector.h"
#include "n++.h"
#include "macro_msg.h"
#include "basic_cmd_bms.h"

class NeuroGo2PosItrActions {
  /* neurogo2pos uses dashs */
  static const double dash_power_min;
  static const double dash_power_inc;
  static const double dash_power_max;
  static const double dash_power_steps;
  
  /* neurogo2pos uses turns */
  static const double turn_min;  // important: start with strongest turn to the right
  static const double turn_max;  // and end with strongest turn to the left!
  static const double turn_inc;
  static const int    turn_steps;

  Cmd_Body action;
  Angle turn;
  int turn_counter;
  double dash;
  int dash_counter;
  double move;
 public:
  //ItrActions();
  static bool init();

  void reset() {
    turn= turn_min;
    dash= dash_power_min;
    turn_counter= 0;
    dash_counter= 0;
  }

  Cmd_Body * next() {
    if ( turn_counter < turn_steps ) {
      action.unset_lock();
      action.unset_cmd();
      // a bit tricky: make sure that maximum turn angles are contained in action set !!!!
      if(turn <= -(M_PI - 0.01))  // angle is smaller or approx. equal than max neg. turn 
	action.set_turn(-(M_PI - 0.00001/180. * PI));
	//action.set_turn(-M_PI);
      else if(turn >=(M_PI - 0.01))// angle is larger or approx. equal than max pos. turn 
	action.set_turn((M_PI - 0.00001/180. * PI));
	//action.set_turn(M_PI);
      else
	action.set_turn( turn );
      turn += turn_inc;
      turn_counter++;
      return &action;
    }
    
    if ( dash_counter < dash_power_steps ) {
      action.unset_lock();
      action.unset_cmd();
      if(dash>dash_power_max)
	dash=dash_power_max;
      action.set_dash( dash );
      dash += dash_power_inc;
      dash_counter++;
      return &action;
    }
    return 0;
  }
};





class NeuroGo2Pos: public BodyBehavior {

  BasicCmd *basic_cmd;
  int obstacle_found;

  static const double MAX_GO2POS_DISTANCE;
  static bool initialized;

  static int op_mode;
  static bool use_regular_states;

  struct State {
    Vector my_vel;
    Vector my_pos;
    ANGLE  my_angle;
    Vector opnt1_pos;
  };

  static const double costs_failure;
  static const double costs_success;
  static double target_tolerance;
  static int consider_obstacles;
  static int use_old_go2pos;
  static double costs_per_action;

  // important: do init first, since net loading might be aborted
  static Net * net;

  NeuroGo2PosItrActions itr_actions;
  Vector target_pos;

  void get_cur_state( State & state);
  void get_features( State const& state, Vector target, float * net_in);
  
  double evaluate( State const& state, Vector const& target);
  bool is_failure( State const& state,  Vector const& target);
  bool is_success( State const& state,  Vector const& target);
  bool neuro_decide(Cmd & cmd);

  // learning stuff
  #define STATE_MEMORY_SIZE 5000
  #define NUM_FEATURES 8
  float uparams[10];


  struct PatternSet{
    long ctr;
    float input[STATE_MEMORY_SIZE][NUM_FEATURES],target[STATE_MEMORY_SIZE];
  };

  static int num_stored;
  static int num_epochs;
  static int store_per_cycle;
  static int repeat_mainlearningloop;
  static int state_memory_ctr;

  typedef struct{
    State state;
    Vector target_pos;
  }   StateMemoryEntry;

  StateMemoryEntry state_memory[STATE_MEMORY_SIZE];
  PatternSet training_set;
   
  bool learn(Cmd &cmd);
  void check_cmd(Cmd &cmd);
  void store_state();
  void print_memory();
  void generate_training_patterns();
  void train_nn();
  double get_value_of_best_successor(State const &state, Vector const& target);


 public:
  NeuroGo2Pos();
  static bool init(char const * conf_file, int argc, char const* const* argv);
  virtual ~NeuroGo2Pos();

  void set_target(Vector target, double target_tolerance = 1.0, int cons_obstacles = 1, int use_old = 0);
  bool get_cmd(Cmd & cmd);
};

#endif
