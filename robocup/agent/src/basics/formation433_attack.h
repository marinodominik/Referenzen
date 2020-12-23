#ifndef _FORMATION433_ATTACK_H_
#define _FORMATION433_ATTACK_H_

#include "globaldef.h"
#include "formation.h"

#define NUM_FORMATIONS 4
#define STANDARD 0
#define CLOSE2GOAL 1
#define LEFT_WING 2
#define RIGHT_WING 3

class Formation433Attack : BaseFormation {
  struct Home {
    Vector pos;
    double delta_x;  // deviation from standard homepos
    double delta_y;  //
    int role;
  };
  Home home[NUM_FORMATIONS][NUM_PLAYERS+1];
  int boundary_update_cycle;
  double defence_line, offence_line; //this values are set in get_boundary, and can be used as cached values in the same cycle
  int formation_state_updated_at;
  int previous_formation_state, current_formation_state;
  int get_formation_state();

 public:
  bool init(char const * conf_file, int argc, char const* const* argv);
  int get_role(int number);
  Vector get_grid_pos(int number);
  bool   need_fine_positioning(int number);
  Vector get_fine_pos(int number);
  void get_boundary(double & defence, double & offence);
};

#endif
