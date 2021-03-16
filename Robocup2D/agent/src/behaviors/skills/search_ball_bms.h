#ifndef _SEARCH_BALL_BMS_H_
#define _SEARCH_BALL_BMS_H_

/* This behavior looks for the ball if we don't know where it is, and
   makes appropriate body turns to find it. NECK_REQ_SCANFORBALL is set,
   and the neck behavior must care for this request!
   
   If we DO know where the ball is (i.e. WSinfo::is_ball_pos_valid() is true),
   no cmd is set and get_cmd returns false!
   If you want to face the ball no matter if we know its position, use the
   FaceBall behavior (it will call SearchBall if needed).
*/

#include "../../basics/Cmd.h"
#include "../base_bm.h"
#include "log_macros.h"

class SearchBall : public BodyBehavior {
  static bool initialized;

  //int number_of_turns;
  static long time_of_turn;
  static long ms_time_of_last_sb;
  static bool turn_left;
  //bool ball_invalid;

  static int looked_to_ball,looked_left,looked_right;
  static ANGLE cur_vang_ball,cur_vang_left,cur_vang_right;
  static ANGLE next_vang_ball,next_vang_left,next_vang_right;
  static ANGLE last_updated_angle,my_angle_to_ball;
  static Vector new_ball_pos;
  bool got_update;
  
 public:
  void start_search();            // reset behavior
  bool get_cmd(Cmd &cmd);  // returns false if we already know the ball position or do not need to turn

  bool is_searching();     // returns true if we are currently searching the ball
  
  static bool init(char const * conf_file, int argc, char const* const* argv) {
    if(initialized) return true;
    initialized = true;
    if (!initialized) { std::cout << "\nSearchBall behavior NOT initialized!!!"; }
    return initialized;
  }
  SearchBall() { time_of_turn = -2; ms_time_of_last_sb=-1000;}
  virtual ~SearchBall() {}
};

#endif
