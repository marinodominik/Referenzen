#ifndef _SELFPASS_BMS_H_
#define _SELFPASS_BMS_H_

#include "basic_cmd_bms.h"
#include "oneortwo_step_kick_bms.h"

class Selfpass: public BodyBehavior {
  static bool initialized;
  BasicCmd *basic_cmd;
  OneOrTwoStepKick *onetwostepkick;
  int optime2react[50];

 public:
  void determine_optime2react(const ANGLE targetdir, const int max_dashes);
  bool is_selfpass_safe(const ANGLE targetdir, double &speed, Vector &ipos, int &steps, Vector &attacking_op,
			int & op_number, const int max_dashes = 10, double op_time2react =0);
  //op_time2react is the time that is assumed the opponents need to react. 0 is worst case, that they
  // are maximally quick, 1 assumes
  // that they need 1 cycle to react. This is already pretty aggressive and (nearly) safe

  bool is_selfpass_safe( const AState & state, const ANGLE targetdir, double &speed, Vector &ipos, int &steps,
			 Vector &attacking_op, int & op_number, const int max_dashes,
			 double op_time2react );
  void determine_optime2react( const AState & state, const ANGLE targetdir, const int max_dashes );


  bool get_cmd(Cmd &cmd);
  bool get_cmd(Cmd & cmd, const ANGLE targetdir, const double speed, const Vector target);
  bool get_cmd(Vector my_pos,Vector my_vel,ANGLE my_ang,Vector ball_pos,Vector ball_vel,
Vector opp_pos,ANGLE opp_ang,Cmd & cmd, const ANGLE targetdir, const double speed, const Vector target);

  static bool init(char const * conf_file, int argc, char const* const* argv);
  Selfpass();
  virtual ~Selfpass();
};

#endif
