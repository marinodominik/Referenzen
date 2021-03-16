#ifndef _BLACKBOARD_
#define _BLACKBOARD_

#include "Cmd.h"
#include "intention.h"
#include "ws_info.h"

class Blackboard {

  /* stuff for view behavior */
  struct ViewInfo {
    long ang_set;
    int next_view_angle;
    int next_view_qty;
    Vector last_ball_pos;
    long ball_pos_set;
    long last_lowq_cycle;
    bool guess_synched;
  };
  static ViewInfo viewInfo;
  
  
 public:
  static NeckRequest neckReq;
  static AttentionToRequest cvAttentionToRequest;
  static PassRequestRequest cvPassRequestRequest; //TGpr
  static Intention main_intention;
  static Intention pass_intention;
  static bool get_pass_info(const int current_time, double &speed, Vector & target, int & target_player_number);

  /* stuff for view behavior */
  static int    get_next_view_angle();        // planned view angle for next cycle
  static int    get_next_view_quality();      // planned view quality for next cycle
  static bool   get_guess_synched();          // check if we guessed synch_see
  static void   set_guess_synched();          // set that we successfully guessed synch_see mode
  static void   set_next_view_angle_and_quality(int,int);
  static long   get_last_lowq_cycle();       // last cycle where we had only LOW QUALITY view
  static void   set_last_lowq_cycle(long);
  static Vector get_last_ball_pos();       // last position ball had when we saw something
  static void   set_last_ball_pos(Vector);

  static bool force_highq_view;            // interrupt any view sync behavior
  static bool force_wideang_view; 
 
  /* stuff for neck behavior */
  static void set_neck_request(int req_type, double param = 0);
  static int  get_neck_request(double &param);    // returns NECK_REQ_NONE if not set

  /* stuff for attentionto behavior */
  static void set_attention_to_request(int plNr);
  static int  get_attention_to_request();

  //TGpr: begin
  /* stuff for say behavior (as part of the attentionto behavior) */ 
  static void set_pass_request_request(int plNr, int passAngle);
  static bool get_pass_request_request(int &plNr, int &passAngle);
  //TGpr: end
  //
  //TGdoa: begin
  /* stuff for say behavior */
  static int  cvAcceptedDirectOpponentAssignment;
  static long cvTimeOfAcceptingDirectOpponentAssignment;
  static bool get_direct_opponent_assignment_accepted( long time, int & ass );  
  static void set_direct_opponent_assignment_accepted( int ass );
  //TGdoa: end
  
  /* stuff for goal kick */
  static bool need_goal_kick;
  
  static void init();
};


#endif
