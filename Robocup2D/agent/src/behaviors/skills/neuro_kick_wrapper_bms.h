#ifndef _NEURO_KICK_WRAPPER_BMS_H_
#define _NEURO_KICK_WRAPPER_BMS_H_

/* This behavior is a wrapper to get the best out of NeuroKick05
 * and NeuroKick08. Depending on the desired target velocity, it
 * chooses the respective bahvior.
 *  
 * Thomas Gabel, 2008
 */

#include "../base_bm.h"
#include "angle.h"
#include "Vector.h"
#include "macro_msg.h"
#include "ws_info.h"
#include "log_macros.h"
#include "mystate.h"

#include "neuro_kick05_bms.h"
#include "neuro_kick08_bms.h"


//============================================================================
// CLASS NeuroKickWrapper
//============================================================================
class NeuroKickWrapper: public BodyBehavior 
{
  //--------------------------------------------------------------------------
  private:
  //--------------------------------------------------------------------------

  //variables
  
  static bool    cvInitialized;

  long           ivInitInCycle;  
  double         ivTargetKickVelocity;
  
  NeuroKick05  * ivpNeuroKick05;
  NeuroKick08  * ivpNeuroKick08;
  
  //--------------------------------------------------------------------------
  public:
  //--------------------------------------------------------------------------
  //constructor
  NeuroKickWrapper();
  //destructor
  virtual ~NeuroKickWrapper();
  //static methods
  static 
    bool init(char const *confFile, int argc, char const* const* argv);
  //non-static methods
  /** This makes it possible to "fake" WS information.
   *  This must be called _BEFORE_ any of the kick functions, and is valid for 
   *  the current cycle only.
   */
  void set_state( const Vector &myPos, 
                  const Vector &myVel,
                  const ANGLE  &myAng,
                  const Vector &ballPos,
                  const Vector &ballVel);
  /** Resets the current state to that found in WS.
   *  This must be called _BEFORE_ any of the kick functions.
   */
  void reset_state();
  //kick initializing methods
  void kick_to_pos_with_initial_vel( double        vel,
                                     const Vector &pos );
  void kick_to_pos_with_final_vel  ( double        vel,
                                     const Vector &pos );
  void kick_to_pos_with_max_vel    ( const Vector &pos ); 
  void kick_in_dir_with_initial_vel( double        vel,
                                     const ANGLE  &dir );
  void kick_in_dir_with_max_vel    ( const ANGLE  &dir );
  //'main' method: get command
  bool get_cmd(Cmd & cmd);
};
    
#endif
