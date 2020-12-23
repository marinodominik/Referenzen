#ifndef _POS_TOOLS_H_
#define _POS_TOOLS_H_

#include "ws_info.h"

struct PosValue {
  //PosValue () {};
  Vector pos;

  bool valid;

  PPlayer first;
  double first_time_to_pos;
  
  PPlayer second;
  double second_time_to_pos;

  PPlayer first_opponent;
  double first_opponent_time_to_pos;

  double initial_ball_vel_to_get_to_pos_in_time_of_first;
  double initial_ball_vel_to_get_to_pos_in_time_of_second;
  double initial_ball_vel_to_get_to_pos_in_time_of_first_opponent;
};


class PosSet {
  double time_to_pos( Vector & pos, PPlayer );
  void evaluate_position( PosValue & pos, PlayerSet & pset );
public:
  PosSet() {
    require_my_team_to_be_first= true;
    require_me_to_be_first= false;
    require_my_team_to_be_at_least_second= true;
    require_me_to_be_at_least_second= false;
  }

  static const int max_num= 120;
  PosValue position[max_num]; 
  int num;

  void reset_positions() { num= 0; }

  //following parameter can customize (and accelerate) the computations
  bool require_my_team_to_be_first;
  bool require_me_to_be_first;
  bool require_my_team_to_be_at_least_second;
  bool require_me_to_be_at_least_second;

  void draw_positions() const;
  void evaluate_positions( PlayerSet & pset );
  bool add_grid(Vector pos, int res1, Vector & dir1, int res2, Vector & dir2);
  void add_his_goal_area();
};



#endif
