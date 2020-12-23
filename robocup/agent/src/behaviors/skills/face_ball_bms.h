#ifndef _FACE_BALL_BMS_H_
#define _FACE_BALL_BMS_H_

/* This behavior faces the ball and at the same time turns the body into a desired direction.
   
   Note that get_cmd only returns true if all conditions are met (i.e. we should have the body in
   the desired direction AND we should be able to see the ball next cycle).

   Note also that the neck behavior has to react to NECK_REQ_FACEBALL - we won't ensure a correct
   neck turn here!

   If the ball position is invalid, this behavior calls SearchBall first before doing any turns.
*/

#include "../../basics/Cmd.h"
#include "../base_bm.h"
#include "log_macros.h"
#include "search_ball_bms.h"

class FaceBall : public BodyBehavior {
  static bool initialized;

  SearchBall *search_ball;
  ANGLE desired_dir;
  bool turn_body;
  bool lock_neck;
  
  float abs_turnangle(float);
  
 public:
  /* neck_req = false: do not set neck request (-> turn_neck is free) */
  void turn_to_ball(bool neck_req=true);  // turn body and neck to ball
  void look_to_ball();                     // turn only neck to ball if possible, main cmd is not set
  void turn_in_dir(ANGLE dir, bool neck_req=true); // turn neck to ball, body in dir
  void turn_to_point(Vector target,bool neck_req=true);
                                                    // turn neck to ball, body in direction of target
  
  bool get_cmd(Cmd &cmd);          /* returns true if move is finished, false otherwise (e.g. if we
				      need to search the ball first, or if it is not possible to
				      turn the body into the desired direction while facing the ball
				      at the same time). You should call get_cmd every cycle
				      until it returns true!
				    */

  bool get_cmd_turn_to_ball(Cmd &cmd, bool neck_req = true);                  // CR: added for one line usage
  bool get_cmd_turn_to_point(Cmd &cmd, Vector &target, bool neck_req = true); // CR: added for one line usage

  static bool init(char const * conf_file, int argc, char const* const* argv) {
    if(initialized) return true;
    initialized = SearchBall::init(conf_file,argc,argv);
    if(!initialized) { std::cout << "\nFaceBall behavior NOT initialized!!!"; }
    return initialized;
  }
  FaceBall() { search_ball = new SearchBall; }
  virtual ~FaceBall() { delete search_ball; }
};

#endif
