#ifndef _NEURO_INTERCEPT_BMS_H_
#define _NEURO_INTERCEPT_BMS_H_

#include "../../basics/Cmd.h"
#include "base_bm.h"
#include "angle.h"
#include "Vector.h"
#include "n++.h"
#include "macro_msg.h"
#include "mystate.h"

#include "intercept_ball_bms.h"
#include "onetwostep_intercept_bms.h"

class NeuroInterceptItrActions {
  /* set params */
  
  /* neuro_intercept uses dashs */
  static const double dash_power_min;
  static const double dash_power_inc;
  static const double dash_power_max;
  static const double dash_power_steps;
  
  /* neuro_intercept uses turns */
  static const double turn_min;  // important: start with strongest turn to the right
  static const double turn_max;  // and end with strongest turn to the left!
  //  static const double turn_inc= 2*M_PI/16.0; // this gives 16 increments
  //static const double turn_inc= 2*M_PI/72.0;
  static const double turn_inc;
  //static const double turn_inc;
  static const int turn_steps;

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


class NeuroIntercept: public BodyBehavior {
  static const double MAX_INTERCEPT_DIST;
  static bool initialized;

  InterceptBall *intercept;
  OneTwoStep_Intercept *onetwostep_intercept;

  static int op_mode;
  static int init_mode;
  static int learn12step;
  static double init_param;
  static double learn_param;
  static double vball_angle_min;
  static double vball_angle_max;
  static bool use_regular_states;
  static bool do_net_init;
  static bool do_pi;
  static bool do_reference;
  static bool do_stochastic;
  static bool adjust_targets;
  static int num_testrepeats;
  static int neuro_maxcycles;

  static const double costs_failure;
  static const double costs_success;
  static double safety_margin;
  static double stress;
  static double costs_per_action;

  // important: do init first, since net loading might be aborted
  static Net * net;

  NeuroInterceptItrActions itr_actions;

  void get_cur_state( MyState & state);
  void get_features( MyState const& state, float * net_in);
  
  double evaluate( MyState const& state);
  bool is_failure( MyState const& state);
  bool is_success( MyState const& state);
  bool neuro_decide(MyState const& state, Cmd & cmd);

  bool check_onestep(Cmd & cmd, const MyState &state);

  // learning stuff
  #define STATE_MEMORY_SIZE_ICPT 500000
  #define TEST_MEMORY_SIZE_ICPT 50000
  #define NUM_ICPT_FEATURES 6
  float learn_params[10];
  int train_loops_ctr;
  char save_name_suffix[500];

  struct PatternSet{
    long ctr;
    //    float input[STATE_MEMORY_SIZE_ICPT][NUM_ICPT_FEATURES],target[STATE_MEMORY_SIZE_ICPT];
    float **input, *target;
  };

  static int num_stored;
  static int num_epochs;
  static long num_trainstates;
  static long num_teststates;
  static double prob_easy;
  static int store_per_cycle;
  static int repeat_mainlearningloop;
  static int state_memory_ctr;
  static int test_memory_ctr;
  static int max_sequence_len;

  typedef struct{  
    MyState state;
  }   MyStateMemoryEntry;

  typedef struct{
    Vector ballpos, ballvel;
    Vector my_pos, my_vel;
    ANGLE my_angle;
    int valid_at;
  }   VirtualState;

  VirtualState virtual_state;

  //  MyStateMemoryEntry state_memory[STATE_MEMORY_SIZE_ICPT];
  MyStateMemoryEntry *state_memory, *test_memory;
  double test_memory_result[TEST_MEMORY_SIZE_ICPT][2];
  PatternSet training_set;
  std::ofstream resultfile;
   
  void generate_test();
  bool learn(Cmd &cmd);
  void check_cmd(Cmd &cmd);
  void store_state();
  void print_memory();
  void generate_training_patterns();
  void train_nn();
  double get_value_of_best_successor(MyState const &state, MyState &successor_state);
  void generate_test_state(const Vector mypos, const Vector myvel,const Vector ballpos,
			   const Vector ballvel,const Angle myangle );
  void do_test();
  void do_sequence(const MyState & initial_state, double *result, const int N);
  void out(const MyState &state);

 public:
  NeuroIntercept();
  virtual ~NeuroIntercept();
  static bool init(char const * conf_file, int argc, char const* const* argv);
  void set_target(Vector target);
  bool get_cmd(Cmd & cmd);
  int get_steps2intercept();
  int get_steps2intercept(double & resultingBallDist);
  void set_virtual_state(Vector const ballpos, Vector const ballvel);
  void set_virtual_state(Vector const mypos, Vector const myvel, ANGLE myang, Vector const ballpos, Vector const ballvel);

};

#endif
