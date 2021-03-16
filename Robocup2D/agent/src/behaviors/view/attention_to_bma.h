#ifndef _ATTENTION_TO_BMA_H_
#define _ATTENTION_TO_BMA_H_

/* This is a port of the BS02_View_Strategy (in advanced mode).
   This strategy has been used since Fukuoka '02 and tries to get
   maximum view efficency by changing the view angle to normal whenever possible
   without losing cycles.

   The original view strategy has been heavily improved while porting to the
   behavior structure; the agent now usually (i.e. as long as we have no network probs)
   only needs to sync directly after connecting to the server and never loses
   synchronisation with the server afterwards.
*/

#include "../base_bm.h"
#include "log_macros.h"
#include <iostream>
#include "../../basics/Cmd.h"

class AttentionTo : public AttentionToBehavior {
  static bool initialized;

  bool teammate_controls_ball(int &number, Vector & pos);
  void generate_players4communication(Cmd & cmd_form);
  PlayerSet generate_players4communication4OpponentAwareDefense();
  void construct_say_message(Cmd & cmd);
  void set_attention_to(Cmd & cmd);

#if 0
  bool get_teammate(int number, PPlayer & p);
  int relevant_teammate[11];
  int num_relevant_teammates;

  void set_relevant_teammates(const int t1=0,const int t2=0,const int t3=0,const int t4=0,
			      const int t5=0,const int t6=0,const int t7=0,const int t8=0,
			      const int t9=0,const int t10=0,const int t11=0);
#endif

#if 0
  void check_communicate_ball_and_mypos(Cmd & cmd);
  void communicate_players(Cmd & cmd_form);
#endif

 public:
  bool get_cmd(Cmd &cmd);
  
  static bool init(char const * conf_file, int argc, char const* const* argv);

  AttentionTo();
};

#endif // _ATTENTION_TO_BMA_H_
