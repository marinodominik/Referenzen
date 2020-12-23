#ifndef _NEURO_POSITIONING_H_
#define _NEURO_POSITIONING_H_

#include "../policy/abstract_mdp.h"

class NeuroPositioning {

 private:

  static const int PLAYERS = 11;
  static const int MAXACTIONS = 50;

  int num_pj_attackers;
  double min_dist2teammate;

  struct {  // jas1 stands for joint action set
    int n[PLAYERS]; // number of actions of player i
    AAction a[PLAYERS][MAXACTIONS];
    int my_idx;  // indicates my index in array
  } jas1;
  
  void get_permutation(const int m, const int no_sets, const int n[], int idx[]);
  
  float evaluate(const AState& state, const AAction jointaction[]);

  void get_jointaction(const int idx[], AAction jointaction[]);
  
  bool is_relevant(const AState& state, int player);
  
  int determine_all_jointactions(const AState& state, const XYRectangle2d *constraints_P, const Vector *home_positions_P);
  
  //bool position_check(const Vector targetpos, XYRectangle &allowed_area_P);
  
 public:
  
  NeuroPositioning();
  
  bool shall_player_do_neuropositioning(int number_P, Vector homepos_P);
  
  bool am_I_neuroattacker();
  
  bool get_neuro_position(Vector &target_P, XYRectangle2d *constraints_P, Vector *home_positions_P);

};

#endif // _NEURO_POSITIONING_H_
