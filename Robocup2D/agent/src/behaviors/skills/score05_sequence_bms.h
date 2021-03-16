#ifndef _SCORE05_SEQUENCE_BMS_H_
#define _SCORE05_SEQUENCE_BMS_H_

#include "../base_bm.h"
#include "log_macros.h"
#include "intention.h"
#include "intercept_ball_bms.h"
#include "one_step_kick_bms.h"
#include <limits.h>
#include "../../basics/Cmd.h"

#define NUMBER_OF_VIA_POINTS 4
#define VIA_POINT_MARGIN     0.17

class Score05_Sequence
  : public BodyBehavior 
{
  private:
    static bool ivInitialized;
    OneStepKick   * ivpOneStepKickBehavior;
    InterceptBall * ivpInterceptBallBehavior;

    Vector ivViaPoints[NUMBER_OF_VIA_POINTS][3];
    ANGLE  ivCrossLineStartAngle, 
           ivCrossLineEndAngle, 
           ivCrossLineAngleStep;
    int    ivNumberOfCrosslines;
    int    ivCurrentlyPreferredViaPointIndex;
    double ivPointsOnCrosslineDelta;
    Cmd    ivCommandForThisCycle;
    Vector ivTargetForNextShot;
    double ivVelocityForNextShot;
    double ivKickPowerForNextShot;
    Angle  ivKickAngleForNextShot;
    int    ivDesiredTackling;
    int    ivDesiredAngularTackling;
    Vector ivInternalHisGoaliePosition;
    int    ivInternalHisGoalieAge;
    long   ivInternalHisGoalieLastDeterminedAtTime;
    Vector ivInternalHisGoaliePositionWhenDashP100;
    Vector ivInternalHisGoaliePositionWhenDashM100;
    bool   ivBallWithHighProbabilityInKickrangeOfAnOpponentPlayer;
    double ivGOALIE_CATCH_RADIUS;
    
    bool        ivLastTestShoot2GoalResult;
    long        ivLastTimeTestShoot2GoalHasBeenInvoked;
    Intention   ivIntentionForThisCycle;
    
    Ball    ivInternalBallObject;
    Ball   *ivpInternalBallObject;
    Player  ivInternalMeObject;
    Player *ivpInternalMeObject;

    enum  { MODE_FIRST_PLANNING=0, MODE_TO_VIA_POINT=1, 
            MODE_TO_CROSSLINE_POINT=2, MODE_FINAL_KICK=3  };
    int    ivMode;

  public:

  static const int cvcNO_V12_SCORING_TACKLE_POSSIBLE = 1000;
  Score05_Sequence();
  virtual ~Score05_Sequence();

  bool   init(char const * conf_file, int argc, char const* const* argv);
  bool  get_cmd(Cmd &cmd);         

  void   updateViaPoints();
  bool   checkKickSequencesViaViaPoints( int & bestViaPointIndex,
                                        int & nStepKick );
  bool   evaluateScorePositionAfterTCycles(int afterCycles, 
                                          int viaPointIndex);
  bool   getBestOneStepKickToViaPoint(   int       viaPointIndex,
                                        Cmd     & firstCmd,
                                        Vector  & resultingBallVelocity );
  bool   getBestTwoStepKickToViaPoint(   int       viaPointIndex,
                                        Cmd     & firstCmd,
                                        Cmd     & secondCmd,
                                        Vector  & resultingBallVelocity );
  bool   checkIfPointOnCrossLineIsViable( Vector p );
  Vector getMyPositionInTCyclesAssumingNoDash(int t);
  Vector getMyVelocityInTCyclesAssumingNoDash(int t);
  bool   checkIfGoalIsForSureWhenShootingInTCyclesFromViaPoint
                                                       ( int    inCycles, 
                                                         int    viaPointIndex,
                                                         Vector ballVel,
                                                         double &finalKickVel );
  bool   checkIfGoalIsForSureWhenShootingInTCyclesFromViaPointIntoDirection
           ( int    inTCycles, 
             int    viaPointIndex,
             Vector ballVel,
             ANGLE  targetDirection,
             double &maximalReachableVelocity,
             bool   debug=false );
  bool   performFinalKick();
  bool   checkIfGoalIsForSureWhenShootingNow( ANGLE &  bestDirection,
                                              Vector & correspondingBestTargetPosition);
  bool   checkIfGoalIsForSureWhenShootingNowIntoDirection
                                            ( ANGLE  targetDirection,
                                              double &returnedMaxKickVelocity,
                                              bool debug );
  bool   checkIfGoalIsForSureWhenTacklingNow( int tackDir, bool debug=false );
  bool   checkIfGoalIsForSureWhenV12TacklingNow( int & angularTackDir, 
                                                 bool  debug );
  bool   checkIfGoalIsForSureWhenV12TacklingNowIntoDirection
                                               ( int    angularTackDir,
                                                 Vector &resultingBallMovement,
                                                 bool   relaxedChecking,
                                                 bool   debug );
  double evaluateGoalShotWithRespectToGoalieAndPosts
                  ( const ANGLE  & targetDirection,
                    const Vector & ballPos,
                    Vector       & shotCrossesTorausLinie,
                    const double & initialBallVel );

  void   checkForNearPlayersRegardingCurrentMode();
  ANGLE  determineGoodLookDirectionIfGoalieIsPresent();
  bool   enemyMayReachPointInTCycles(Vector p, int inTCycles);
                         
                                        
  bool test_shoot2goal( Intention & intention,
                        Cmd       * currentCommand = NULL,
                        bool        allowRecursiveness=true);
  int  test_tackle2goal( Vector myPos, Vector myVel, ANGLE myAng,
                         Vector ballPos, Vector ballVel, bool debug=false );
  void determineGoaliePositionAndAge(Vector & gPos, int & gAge);
  void getPotentialGoaliePositions( int maxSearchDepth,
                                    int currentSearchDepth,
                                    Vector * goaliePositionsArray );
  bool isPointInOneOfRecentViewCones( Vector p, int fromT, int toT );
  void setAssumedGoaliCatchRadius();
  void updateInternalObjects( Cmd * currentCommandPointer );



  ///////////////// I M P O R T E ////////////////////////////////////////////
  int  intercept_goalie(   Vector ball_pos, 
                                      Vector ball_vel, 
                                      Vector goalie_pos, 
                                      double goalie_size,
                                      int    timeOffset,
                                      bool debug=false ); 
  bool goalie_needs_turn_for_intercept (  int    time, 
                                          Vector initial_ball_pos, 
                                          Vector initial_ball_vel, 
                                          Vector b_pos, 
                                          Vector b_vel, 
                                          double goalie_size);
  double goalie_action_radius_at_time ( int    time,
                                        double goalie_size,
                                        int    goalie_handicap);
  int  intercept_opponents ( double direction,
                             double b_v,
                             int    max_steps,
                             Vector startBallPosition,
                             int    timeOffset );
  double player_action_radius_at_time ( int     time,
                                        PPlayer player, 
                                        double  player_dist_to_ball,
                                        int     player_handicap);
  Vector  intersection_point ( Vector p1,  Vector steigung1, 
                               Vector p2,  Vector steigung2);
  Vector  point_on_line(Vector steigung, Vector line_point, double x);
  bool    is_pos_in_quadrangle ( Vector pos, Vector p1, Vector p2, 
                                             Vector p3, Vector p4);
  bool    is_pos_in_quadrangle(Vector pos, Vector p1, Vector p2, double width);

            
};

#endif // _SCORE05_SEQUENCE_BMS_H_
