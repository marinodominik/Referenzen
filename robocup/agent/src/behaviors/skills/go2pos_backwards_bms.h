/** \file move_go2pos_backwards.h
    We define a very simple move to go backwards...
*/

#include "../../basics/Cmd.h"
#include "../base_bm.h"
#include "Vector.h"
#include "basic_cmd_bms.h"

#ifndef _GO2POSBACKWARDS_BMS_H_
#define _GO2POSBACKWARDS_BMS_H_

class Go2PosBackwards : public BodyBehavior {
  static bool initialized;

  BasicCmd *basic_cmd;

  /** target position */
  Vector target;
  
  /** accuracy at which we want to reach the position */
  double accuracy;

  /** accuracy for the angle to the target */
  double angle_accuracy;

  double dash_power_needed_for_distance(double dist);

public:
  /** constructor */
  void set_params(double p_x=-52, double p_y=0, double p_accuracy=1.0, double p_angle_accuracy=0.05);

  bool get_cmd(Cmd &cmd);
  
  static bool init(char const * conf_file, int argc, char const* const* argv) {
    if ( initialized ) 
      return true;
    initialized = true;

    return BasicCmd::init(conf_file, argc, argv);
  }

  Go2PosBackwards();
  virtual ~Go2PosBackwards();

};

#endif //_GO2POSBACKWARDS_BMS_H_







