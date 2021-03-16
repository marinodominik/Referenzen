#ifndef _NECK_CMD_BMS_H_
#define _NECK_CMD_BMS_H_

/* This contains the interface to the base commands of soccer server.
   
*/

#include "../../basics/Cmd.h"
#include "../base_bm.h"
#include "log_macros.h"
#include "mdp_info.h"

class NeckCmd : public BodyBehavior {
  static bool initialized;

  double moment;
  
 public:
  void set_turn_neck_abs(double abs_dir_P) {
    moment = mdpInfo::moment_to_turn_neck_to_abs( abs_dir_P );
  }
  
  void set_turn_neck_rel(double rel_dir_P) {
    moment = mdpInfo::moment_to_turn_neck_to_rel( rel_dir_P );
  }

  void set_turn_neck(double moment_P) {
    moment = moment_P;
  }

  bool get_cmd(Cmd &cmd) {
    if(!cmd.cmd_neck.is_lock_set()) {    
      cmd.cmd_neck.set_turn( moment );
    } else {
      ERROR_OUT << " ERROR IN NECK_CMD BEHAVIOR: NECK CMD WAS ALREADY SET!!!";
    }
    return true;
  }
  
  static bool init(char const * conf_file, int argc, char const* const* argv) {
    if(initialized) return true;
    initialized = true;
    return true;
  }
  NeckCmd() {}
  virtual ~NeckCmd() {}
};


#endif // _NECK_CMD_BMS_H_
