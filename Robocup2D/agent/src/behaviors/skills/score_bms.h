#ifndef _SCORE_BMS_H_
#define _SCORE_BMS_H_

#include "../../basics/Cmd.h"
#include "../base_bm.h"
#include "log_macros.h"
#include "intention.h"
#include "oneortwo_step_kick_bms.h"

class Score : public BodyBehavior {
  static bool initialized;

  OneOrTwoStepKick *oneortwo;

 private:

  int nr_of_looks_to_goal;
  double saved_velocity;
  double saved_direction;
  Vector saved_target;
  int saved_multi_step;
  int risky_goalshot_possible;
  int last_time_look_to_goal;
  float goalshot_param1;
  int goalshot_mode;


  // auxillary functions for shoot2goal:
  float get_orientation_and_speed_handicap_add(Vector target);
  double player_action_radius_at_time(int time, PPlayer player,
				     double player_dist_to_ball, int player_handicap);
  int intercept_opponents(double direction, double b_v, int max_steps);
  double goalie_action_radius_at_time(int time, double goalie_size, int goalie_handicap);
  int intercept_goalie(Vector ball_pos, Vector ball_vel, Vector goalie_pos, double goalie_size);
  bool goalie_needs_turn_for_intercept(int time, Vector initial_ball_pos, Vector initial_ball_vel, 
				       Vector b_pos, Vector b_vel, double goalie_size);
  Vector intersection_point(Vector p1, Vector steigung1, Vector p2, Vector steigung2);
  Vector point_on_line(Vector steigung, Vector line_point, double x);
  bool is_pos_in_quadrangle(Vector pos, Vector p1, Vector p2, Vector p3, Vector p4);
  bool is_pos_in_quadrangle(Vector pos, Vector p1, Vector p2, double width);

  int get_goalshot_chance(int &multi_step, double &velocity,
			  double &direction, Vector &target, int &best_index);
  void consider_special_cases(int goalie_age, int goalie_vel_age, Vector goalie_vel,
			      double &goalie_size, Vector &goalie_pos, PPlayer goalie);
  int select_best_kick(int *kick_possible, int nr_of_targets);
  void fill_target_arrays(Vector *test_targets, double *test_dirs, int nr_of_targets, Vector ball_pos);
  void fill_velocity_arrays(double *test_vels_1step, double *test_vels_multi,
			    Vector *test_targets, int nr_of_targets);

  // end of auxillary functions for shoot2goal:


 public:

  Score();
  virtual ~Score();

  bool test_shoot2goal(Intention &intention);

  bool get_cmd(Cmd &cmd);         
				    
  static bool init(char const * conf_file, int argc, char const* const* argv) {
    if(initialized) return true;
    initialized = true;
    std::cout << "\nScore behavior initialized.";
    return OneOrTwoStepKick::init(conf_file, argc, argv);
  }
};

#endif // _SCORE_BMS_H_
