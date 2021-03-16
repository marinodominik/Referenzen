#ifndef _NEURO_KICK08_BMS_H_
#define _NEURO_KICK08_BMS_H_

/* This behavior is the re-learnt RL-based kick behavior, written in
 * May/June 2008. It features improved performance, in particular for
 * high-speed kicks, compared to the NeuroKick-Behavior, while showing
 * identical performance for slower kicks.
 * 
 * It currently makes use of 4 neural nets which where trained for
 * different speeds (the decision which net to use for which desired
 * kick velocity is made in method chooseCurrentNet()).
 * 
 * The OneStep- and OneTwoStepKick-Behaviors are fully integrated;
 * they were also incorporated during learning already.
 * 
 * The behavior is capable of reading relevant parameters from a
 * config file
 *  - maximal_kick_velocity_deviation [0.1]
 *  - maximal_kick_direction_deviation [10degrees]
 * however, the behavior at hand has been optimized for the default
 * values.
 * Furthermore, the behavior has been optimized for the current settings
 * of constant class variables
 *  - cvcMinimalDesiredBallPlayerDistanceRelative  = 0.1
 *  - cvcMaximalDesiredBallPlayerDistanceRelative  = 1.0
 * which determine allowed ball regions within the player's kickable
 * area.
 * 
 * Note, that the learning process has always assumed relative ball-player
 * distances (within [0.0; 1.0], corresponding to [0.0m, kick_radius]) so
 * that an abstraction from heterogeneous player types (with varying kick
 * radii from 0.7 to 0.9) could be introduced.
 * 
 * The 'interface' to this class (public methods) has intentionally been
 * made identical to NeuroKick's interface.
 * 
 * Thomas Gabel, 2008
 */

#include "../../basics/Cmd.h"
#include "base_bm.h"
#include "angle.h"
#include "Vector.h"
#include "tools.h"
#include "n++.h"
#include "macro_msg.h"
#include "valueparser.h"
#include "options.h"
#include "ws_info.h"
#include "log_macros.h"
#include "oneortwo_step_kick_bms.h"
#include "one_step_kick_bms.h"
#include "mystate.h"

// MACRO DEFINITIONS
#define NUMBER_OF_NEURAL_NETS_NK08  5
#define DEFAULT_PLAYER_KICKABLE_AREA_DURING_TRAINING 0.7

//============================================================================
// CLASS NeuroKick08ItrActions 
//============================================================================
class NeuroKick08ItrActions 
{
  /* set parameterss */
  static const double ivKickPowerMin;
  static const double ivKickPowerInc;
  static const double ivKickPowerSteps;
  static const double ivFinetuneKickPowerInc;
  static const double ivFinetuneKickPowerSteps;
  static const double ivKickAngleMin;
  static const double ivKickAngleInc;
  static const double ivKickAngleSteps;
  static const double ivFinetuneKickAngleInc;
  static const double ivFinetuneKickAngleSteps;

  double ivPowerMin,
         ivPowerInc,
         ivPowerSteps,
         ivAngleSteps;
  ANGLE  ivAngleMin,
         ivAngleInc;
  
  Cmd_Body ivAction;
  int      ivAngleDone,
           ivPowerDone;
  double   ivPower;
  ANGLE    ivAngle;
  
  //--------------------------------------------------------------------------
  public:
  //--------------------------------------------------------------------------
  void reset( bool finetune = false, double orig_pwr = 0, ANGLE orig_ang = ANGLE( 0 ) );
  
  Cmd_Body* next();
};

//============================================================================
// CLASS NeuroKick08State
//============================================================================
class NeuroKick08State
{
  private:
  public:
    //pre-processed state information (to be used as neural net input)
    double ivBallPositionRelativeAngle; //relative angle of current ball pos
                                        //to my body orientation
    double ivRelativeBallDistance; //relative means relative to my kick raidus
    Vector ivRelativeBallVelocity;
    double ivRelativeBodyToKickAngle; //relative angle of kick direction
                                      //to my current body orientation
    //world information
    Vector ivMyWorldPosition;
    Vector ivMyWorldVelocity;
    ANGLE  ivMyWorldANGLE;
    Vector ivBallWorldPosition;
    Vector ivBallWorldVelocity;
    //opponent information
    Vector ivOpponentPosition;
    ANGLE  ivOpponentBodyAngle;
    int    ivOpponentBodyAngleAge;
    
    NeuroKick08State();
};

//============================================================================
// CLASS NeuroKick08
//============================================================================
class NeuroKick08: public BodyBehavior 
{
#if 0
  struct State {
    Vector my_vel;
    Vector my_pos;
    ANGLE  my_ang;
    Vector ball_pos;
    Vector ball_vel;
  };
#endif

  //--------------------------------------------------------------------------
  private:
  //--------------------------------------------------------------------------

  //variables
  
  static bool   cvInitialized;
  static double cvMaximalKickVelocityDeviation;
  static double cvMaximalKickDirectionDeviation;
  
  static const double cvcMinimalDesiredBallPlayerDistanceRelative;
  static const double cvcMaximalDesiredBallPlayerDistanceRelative;

  static Net   * cvpNetArray[NUMBER_OF_NEURAL_NETS_NK08];
  Net          * ivpCurrentNet;
  
  NeuroKick08ItrActions   ivActionInterator;
  OneStepKick           * ivpOneStepKickBehavior;
  OneOrTwoStepKick      * ivpOneOrTwoStepKickBehavior;
  
  long           ivInitInCycle;
  double         ivTargetKickVelocity;
  ANGLE          ivTargetKickAngle;
  Vector         ivTargetKickPosition;
  bool           ivDoTargetTracking; 
  bool           ivDoFineTuning;
  
  NeuroKick08State   ivFakeState;
  long               ivFakeStateTime;
  
  //methods
  Net            * chooseCurrentNet();
  bool             checkIfStateIsASuccessState( NeuroKick08State & succState,
                                                double           & deviation);
  void             completeState(NeuroKick08State &state);
  double           computeInitialVelocityForFinalVelocity(double distance,
                                                          double finalVelocity);
  bool             decideForAnAction(Cmd &cmd);
  double           evaluateState(NeuroKick08State & state);
  NeuroKick08State getCurrentState();  
  void             getWSState(NeuroKick08State &s);
  bool             isFailureState(const NeuroKick08State & state);
  void             setNeuralNetInput( NeuroKick08State & state,
                                      double             targetVelocity,
                                      ANGLE              targetDir,
                                      float            * net_in );

  //--------------------------------------------------------------------------
  public:
  //--------------------------------------------------------------------------
  //constructor
  NeuroKick08();
  //destructor
  virtual ~NeuroKick08();
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
  double getTargetKickVelocity()    {return ivTargetKickVelocity;};
  void kick_to_pos_with_max_vel    ( const Vector &pos ); 
  void kick_in_dir_with_initial_vel( double        vel,
                                     const ANGLE  &dir );
  void kick_in_dir_with_max_vel    ( const ANGLE  &dir );
  //'main' method: get command
  bool get_cmd(Cmd & cmd);
};
    
#endif
