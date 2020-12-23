#ifndef _ARTPLAYER_BMC_H_
#define _ARTPLAYER_BMC_H_

#include "base_bm.h"
#include "skills/neuro_go2pos_bms.h"
#include "skills/intercept_ball_bms.h"

/** This is a test player, solely for Artur's needs */

class ArtPlayer: public BodyBehavior {
  static bool initialized;
  NeuroGo2Pos *go2pos;
  InterceptBall *intercept;
public:
  static bool init(char const * conf_file, int argc, char const* const* argv);

  ArtPlayer();
  virtual ~ArtPlayer();
  bool get_cmd(Cmd & cmd);
};


#endif
