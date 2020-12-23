#ifndef _BS03_VIEW_BMV_H_
#define _BS03_VIEW_BMV_H_

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

class BS03View : public ViewBehavior {
  static bool initialized;

  int cur_view_width;
  int next_view_width;
  int init_state;
  long last_normal;
  long lowq_counter;
  long lowq_cycles;
  long missed_commands;
  long cyc_cnt;
  bool need_synch;

  int time_of_see();
  double get_delay();
  bool can_view_normal();
  bool can_view_normal_strict();

  void change_view(Cmd &cmd, int width, int quality);
  
 public:
  bool get_cmd(Cmd &cmd);
  
  static bool init(char const * conf_file, int argc, char const* const* argv);

  BS03View();
  virtual ~BS03View();
};

#endif
