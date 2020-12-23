#ifndef _LINE_UP_H_
#define _LINE_UP_H_

#include "base_bm.h"
#include "skills/basic_cmd_bms.h"
#include "skills/neuro_go2pos_bms.h"
#include "skills/face_ball_bms.h"
#include "../policy/positioning.h"

class LineUp : public BodyBehavior {
  static bool initialized;
  
  NeuroGo2Pos *go2pos;
  BasicCmd *basic_cmd;
  FaceBall *face_ball;
  void adaptMyFormPosition(Vector& p, double& delta);

 protected:

  //float home_kick_off[12][2];

 public:

  LineUp();
  virtual ~LineUp();
  ANGLE ivSectorAngle;

  static bool init(char const * conf_file, int argc, char const* const* argv) {
    if ( initialized )
      return true;
    initialized= true;
    return (
	    NeuroGo2Pos::init(conf_file, argc, argv) &&
	    BasicCmd::init(conf_file, argc, argv) &&
	    FaceBall::init(conf_file, argc, argv)
	    );
  }

  bool get_cmd(Cmd & cmd);
  bool get_cmd(Cmd & cmd, bool sectorBased);

};

#endif //_LINE_UP_H_

