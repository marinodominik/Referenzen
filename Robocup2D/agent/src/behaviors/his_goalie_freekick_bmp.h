#ifndef _HIS_GOALIE_FREEKICK_H
#define _HIS_GOALIE_FREEKICK_H

/* 
   
*/

#include "../basics/Cmd.h"
#include "base_bm.h"
//#include "types.h"
#include "log_macros.h"
#include "skills/basic_cmd_bms.h"
#include "skills/search_ball_bms.h"
#include "skills/neuro_go2pos_bms.h"


class HisGoalieFreeKick : public BodyBehavior 
{
  static bool initialized;


  BasicCmd    *ivpBasicCmdBehavior;
  NeuroGo2Pos *ivpGo2PosBehavior;
  SearchBall  *ivpSearchBallBehavior;
  
  PPlayer ivpMyDirectOpponent;
  Vector  ivCurrentTargetPosition;

  bool get_goalie_cmd(Cmd &cmd);
  bool get_player_cmd(Cmd &cmd);

  bool cover_direct_opponent(Cmd & cmd, PPlayer dirOpp);
  bool scan_field(Cmd &cmd);
  bool turn_to_direct_opponent(Cmd & cmd, PPlayer dirOpp);
  bool search_ball(Cmd & cmd);
  bool get_player_cmd_fallback(Cmd & cmd);

  bool tryToScanField(Cmd &cmd);
  bool tryToOptimizeCovering(Cmd &cmd);
  bool tryToSearchBall(Cmd &cmd);
  bool tryToCoverDirectOpponent(Cmd &cmd);
  bool tryToSearchDirectOpponent(Cmd &cmd);
  
  void correctTargetPosition( Vector & targetPosition );  

 public:
  bool get_cmd(Cmd &cmd);
  
  static bool init(char const * conf_file, int argc, char const* const* argv);

  HisGoalieFreeKick();
  virtual ~HisGoalieFreeKick() 
  {
    delete ivpBasicCmdBehavior; 
    delete ivpSearchBallBehavior;
    delete ivpGo2PosBehavior;
  }
};

#endif
