#ifndef _SYNCH_VIEW08_BMV_H_
#define _SYNCH_VIEW08_BMV_H_

/* This is a re-implementation of the BS03View Behavior for server
 * version 12.0.0 due to the fact that there now exists a synchronous
 * view model (03/2008). 
 *
 * This version can thus gurantee that the see message arrives 30 ms
 * after sense_body (as long as synch_mode is turned on)
 * 
 * ATTENTION : This implementation only uses NARROW_VIEW since we do want
 * to fetch view information every cycle!
 * 
 * If you have any questions contact: jspringe@uos.de
 */

#include "../base_bm.h"
#include "log_macros.h"
#include "bs03_view_bmv.h" //<< include old view behavior
#include <iostream>
#include "../../basics/Cmd.h"

class SynchView08 : public ViewBehavior {
  static bool initialized;
  
  // old BS03 view behavior in order to be able to switch
  BS03View *asynch_view;
  int next_view_width;
  int init_state;
  long cyc_prob_synched; 
  long goalie_last_normal;
  long missed_commands;
  long cyc_cnt;

  int time_of_see();
  double get_delay();
  void change_view(Cmd &cmd, int width, int quality);
  
 public:
  bool get_cmd(Cmd &cmd);
  
  static bool init(char const * conf_file, int argc, char const* const* argv);
  
  // constructor
  SynchView08();
  
  // destructor
  virtual ~SynchView08() {
  	if(asynch_view)
  		delete asynch_view;
  }
};

#endif
