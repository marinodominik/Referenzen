#include "score05_sequence_bms.h"
#include "../../basics/mdp_info.h"
#include "../../policy/policy_tools.h"
#include "../../basics/ws_memory.h"

//////////////////////////////////////////////////////////////////////////////
// Score05_Sequence
//
// Skill to move the ball to a 'good' position within the kickable area from
// which a scoring success is likely.
//
//////////////////////////////////////////////////////////////////////////////

#if 0
#define   TGs05sLOGPOL(YYY,XXX)        LOG_POL(YYY,XXX)
#else
#define   TGs05sLOGPOL(YYY,XXX)
#endif

#define LOGBASELEVEL  3
#define DRAWBASELEVEL 0 //10
#define GOALIE_INITIAL_SIZE 0.55
#define GOALIE_CATCH_RADIUS 1.1 //2.062 //TG08 ZUI: 1.3
#define GOALIE_CATCH_RADIUS_PENALTY 1.3 //TG09: alt: 1.8
#define TACKLE_FORWARD   100
#define TACKLE_BACKWARD -100
#define TACKLE_NONE        0
bool Score05_Sequence::ivInitialized = false;

//============================================================================
// init()
//============================================================================
bool 
Score05_Sequence::init(char const * conf_file, int argc, char const* const* argv) 
{
    if (ivInitialized) 
      return true;
    ivInitialized = true;
//    std::cout << "\nScore05_Sequence behavior initialized.";
    return    OneStepKick::init(conf_file, argc, argv)
           && InterceptBall::init(conf_file, argc, argv);
}


//============================================================================
// Constructor
//============================================================================
Score05_Sequence::Score05_Sequence()
{
  ivNumberOfCrosslines = 5;
  ivPointsOnCrosslineDelta = 0.2;
  ivDesiredTackling = TACKLE_NONE;
  ivDesiredAngularTackling = cvcNO_V12_SCORING_TACKLE_POSSIBLE;
  ivGOALIE_CATCH_RADIUS = GOALIE_CATCH_RADIUS;
  
  ivpInternalBallObject = NULL;
  ivpInternalMeObject   = NULL;
  
  ivpOneStepKickBehavior   = new OneStepKick();
  ivpInterceptBallBehavior = new InterceptBall();
  
  ivInternalHisGoaliePosition.setXY( FIELD_BORDER_X - 3.0, 0.0 );
  ivInternalHisGoalieAge = 3;
  ivInternalHisGoalieLastDeterminedAtTime = -1;
  ivBallWithHighProbabilityInKickrangeOfAnOpponentPlayer = false;
  
  ivLastTestShoot2GoalResult = false;
  ivLastTimeTestShoot2GoalHasBeenInvoked = -1;
}

//============================================================================
// Destructor
//============================================================================
Score05_Sequence::~Score05_Sequence()
{
  if (ivpOneStepKickBehavior) delete ivpOneStepKickBehavior;
}


//============================================================================
// updateViaPoints()
//============================================================================
void
Score05_Sequence::checkForNearPlayersRegardingCurrentMode()
{
  switch (ivMode)
  {
    case MODE_FIRST_PLANNING:
      //mode is first planning anyway, this behavior will not become active
      break;
    case MODE_TO_CROSSLINE_POINT:
      //i must be sure that NO enemy gets into ball contact within TWO cycles
      if (
             enemyMayReachPointInTCycles( ivTargetForNextShot, 1 )
           || 
             enemyMayReachPointInTCycles
             ( ivViaPoints[ivCurrentlyPreferredViaPointIndex][2], 2 )
         )
        ivMode = MODE_FIRST_PLANNING;
      break;
    case MODE_TO_VIA_POINT:
      //i must be sure that NO enemy gets into ball contact within ONE cycles
      if (
          enemyMayReachPointInTCycles
            (  ivViaPoints[ ivCurrentlyPreferredViaPointIndex ] [ 1 ],    1 )
         )
        ivMode = MODE_FIRST_PLANNING;
      break;
    case MODE_FINAL_KICK:
      //mode is final kick, i will get active definitely!
      break;
  }
}

bool
Score05_Sequence::checkIfGoalIsForSureWhenShootingNow
                            ( ANGLE & bestDirection,
                              Vector & correspondingBestTargetPosition )
{
  Vector ballPos = ivpInternalBallObject->pos;
  Angle minAngDevFromPost = PI*(2.5/180.0);
  Angle  minShootDirection 
    = (HIS_GOAL_RIGHT_CORNER - ballPos).ARG().get_value_mPI_pPI() + minAngDevFromPost;
  Angle  maxShootDirection
    = (HIS_GOAL_LEFT_CORNER - ballPos).ARG().get_value_mPI_pPI() - minAngDevFromPost;
  int shootTestSteps = 40;
  double safestPossibleShotGoalieValue = 0.0;
  int indexForSafestPossibleShot = -1;
  Angle  shootTestIncrement = (maxShootDirection - minShootDirection)
                              / (double)shootTestSteps;
  bool foundAGoodKickDirection = false;
  bool debug = false;
  for (int i=0; i<=shootTestSteps; i++)
  {
    if (0) debug=true; else debug=false;
    Angle shootTestAngle = minShootDirection + (shootTestIncrement*i);
    ANGLE targetDirection(shootTestAngle);
    double currentMaxKickVel;
    bool currentCheck
      = checkIfGoalIsForSureWhenShootingNowIntoDirection
           ( targetDirection,  currentMaxKickVel,  debug );
    if (currentCheck)
    {
      Vector goalPoint(targetDirection);
      goalPoint.normalize(HIS_GOAL_CENTER.distance(ballPos));
      goalPoint += ballPos;
      foundAGoodKickDirection = true;
      TGs05sLOGPOL(LOGBASELEVEL+DRAWBASELEVEL,<<_2D<<VL2D(ballPos,
                           goalPoint, "00ffff"));

      Vector shotCrossesTorausLinie;
      double currentShotValue
        = evaluateGoalShotWithRespectToGoalieAndPosts( targetDirection,
                                                       ballPos,
                                                       shotCrossesTorausLinie,
                                                       currentMaxKickVel );

      double minimalDesiredDistanceFromPosts
        = ivpInternalBallObject->pos.distance(shotCrossesTorausLinie) 
          * 0.03 * currentMaxKickVel;
      if (   indexForSafestPossibleShot == -1
          || (    safestPossibleShotGoalieValue < currentShotValue
               && HIS_GOAL_LEFT_CORNER.distance(shotCrossesTorausLinie)
                  > minimalDesiredDistanceFromPosts
               && HIS_GOAL_RIGHT_CORNER.distance(shotCrossesTorausLinie)
                  > minimalDesiredDistanceFromPosts
             )
         )
      {
        safestPossibleShotGoalieValue = currentShotValue;
        indexForSafestPossibleShot = i;
      }
    }
  }
  bestDirection = ANGLE(minShootDirection 
                  + (shootTestIncrement*indexForSafestPossibleShot));
  correspondingBestTargetPosition
    = point_on_line(  Vector(bestDirection), 
                      ballPos, 
                      FIELD_BORDER_X);
  return foundAGoodKickDirection; 
}

bool
Score05_Sequence::checkIfGoalIsForSureWhenShootingNowIntoDirection
           ( ANGLE  targetDirection,
             double &returnedMaxKickVelocity,
             bool debug )
{
  MyState assumedState;
  assumedState.my_pos = ivpInternalMeObject->pos;
  assumedState.my_vel = ivpInternalMeObject->vel;
  assumedState.ball_pos = ivpInternalBallObject->pos;
  assumedState.ball_vel = ivpInternalBallObject->vel;
  assumedState.my_angle = ivpInternalMeObject->ang;
  assumedState.op_pos = Vector(0.0,0.0);
  assumedState.op_bodydir = ANGLE(0);
  assumedState.op_bodydir_age = 0;
  assumedState.op = NULL;

  //how fast will the final kick be?
  double maxFinalKickVelocity
    = ivpOneStepKickBehavior->get_max_vel_in_dir( assumedState,
                                                  targetDirection );
  returnedMaxKickVelocity = maxFinalKickVelocity;
  Vector finalBallVelocity;
  finalBallVelocity.init_polar(maxFinalKickVelocity, targetDirection);

  //will the goalie intercept the ball?
    //be careful with respect to an opponent goalie
    //  * that is alive, but invalid
    //  * that has eventually segfaulted
  int goalieInterceptResult
    = intercept_goalie(   ivpInternalBallObject->pos, 
                          finalBallVelocity, 
                          ivInternalHisGoaliePosition, 
                          GOALIE_INITIAL_SIZE,
                          ivInternalHisGoalieAge, //time offset /*TG_OSAKA*/
                          debug);
  if (goalieInterceptResult > 0)
  {
    TGs05sLOGPOL(LOGBASELEVEL+3,<<"Score05_Sequence: check kick success -> "
      <<"NOPE, because goalie intercepts, time="<<goalieInterceptResult
      <<",ballVel="<<finalBallVelocity<<")");
    return false; //goalie intercepts
  }
    
  //will an opponent player intercept the ball?
  int opponentInterceptResult
    = intercept_opponents ( targetDirection.get_value_mPI_pPI(), //ball's direction
                            finalBallVelocity.norm(), //ball's speed
                            12, //note: value is not used!
                            ivpInternalBallObject->pos,
                            0); 
  if (opponentInterceptResult > 0)
  {
    TGs05sLOGPOL(LOGBASELEVEL+3,<<"Score05_Sequence: check kick success -> "
      <<"NOPE, because opponent intercepts, res="<<opponentInterceptResult
      <<",ballVel="<<finalBallVelocity<<")");
    return false; //an opponent player will intercept
  }
  return true;

}

bool
Score05_Sequence::checkIfGoalIsForSureWhenShootingInTCyclesFromViaPoint
           ( int    inTCycles, 
             int    viaPointIndex,
             Vector ballVel,
             double &maximalReachableVelocity )
{
  
  //having this via point we now must model potential final kicks the
  //player can make
  Angle  minShootDirection 
    = (HIS_GOAL_RIGHT_CORNER - ivViaPoints[viaPointIndex][inTCycles]).ARG().get_value_mPI_pPI();
  Angle  maxShootDirection
    = (HIS_GOAL_LEFT_CORNER - ivViaPoints[viaPointIndex][inTCycles]).ARG().get_value_mPI_pPI();
  int shootTestSteps = 20;
  Angle  shootTestIncrement = (maxShootDirection - minShootDirection)
                              / (double)shootTestSteps;
  bool foundAGoodKickDirection = false;
  bool debug = false;
  for (int i=0; i<=shootTestSteps; i++)
  {
    if (i<2 || i>shootTestSteps-1) debug=true; else debug=false;
    Angle shootTestAngle = minShootDirection + (shootTestIncrement*i);
    ANGLE targetDirection(shootTestAngle);
    bool currentCheck
      = checkIfGoalIsForSureWhenShootingInTCyclesFromViaPointIntoDirection
           ( inTCycles, 
             viaPointIndex,
             ballVel,
             targetDirection,
             maximalReachableVelocity,
             debug );
//    TGs05sLOGPOL(LOGBASELEVEL+1,<<"FINAL KICK CHECK: dir="<<RAD2DEG(targetDirection.get_value_mPI_pPI())
//      <<" vel="<<maximalReachableVelocity<<" => OK? "<<(currentCheck?"YEP":"NOPE"));
    if (currentCheck)
    {
      Vector goalPoint(targetDirection);
      goalPoint.normalize(HIS_GOAL_CENTER.distance(ivViaPoints[viaPointIndex][inTCycles]));
      goalPoint += ivViaPoints[viaPointIndex][inTCycles];
      foundAGoodKickDirection = true;
      TGs05sLOGPOL(LOGBASELEVEL+DRAWBASELEVEL,<<_2D<<VL2D(ivViaPoints[viaPointIndex][inTCycles],
                           goalPoint, "00ffff"));
    }
  }
  return foundAGoodKickDirection; 
}

bool
Score05_Sequence::checkIfGoalIsForSureWhenShootingInTCyclesFromViaPointIntoDirection
           ( int    inTCycles, 
             int    viaPointIndex,
             Vector ballVel,
             ANGLE  targetDirection,
             double &maximalReachableVelocity,
             bool   debug )
{
  MyState assumedState;
  assumedState.my_pos = getMyPositionInTCyclesAssumingNoDash(inTCycles);
  assumedState.my_vel = getMyVelocityInTCyclesAssumingNoDash(inTCycles);
  assumedState.ball_pos = ivViaPoints[viaPointIndex][inTCycles];
  assumedState.ball_vel = ballVel;
  assumedState.my_angle = ivpInternalMeObject->ang;
  assumedState.op_pos = Vector(0.0,0.0);
  assumedState.op_bodydir = ANGLE(0);
  assumedState.op_bodydir_age = 0;
  assumedState.op = NULL;

  //how fast will the final kick be?
  double maxFinalKickVelocity
    = ivpOneStepKickBehavior->get_max_vel_in_dir( assumedState,
                                                  targetDirection );
  maximalReachableVelocity = maxFinalKickVelocity;
  if (maxFinalKickVelocity < 1.0)
    return false;
  Vector finalBallVelocity;
  finalBallVelocity.init_polar(maxFinalKickVelocity, targetDirection);

  //will the goalie intercept the ball?
  int goalieInterceptResult
    = intercept_goalie(   ivViaPoints[viaPointIndex][inTCycles], 
                          finalBallVelocity, 
                          ivInternalHisGoaliePosition,
                          GOALIE_INITIAL_SIZE,
                          inTCycles, //Tools::max(Tools::min(1, inTCycles-1), 0),
                          debug);
  //check for immediate catch by goalie when ball @viapoint
  if (    inTCycles > 0 //shot via via point
       && ivInternalHisGoaliePosition.distance(ivInternalMeObject.pos) < 5.0
       && ivInternalHisGoalieAge < 6
       &&     (  ivViaPoints[viaPointIndex][inTCycles]
               - ivInternalHisGoaliePosition).norm() 
            - goalie_action_radius_at_time( 0 + inTCycles, //be optimistic: use 0 instead of ivInternalHisGoalieAge
                                            GOALIE_INITIAL_SIZE, 
                                            0 )
          < ivGOALIE_CATCH_RADIUS 
     )
  {
    goalieInterceptResult = 1; //goalie intercepts immediately :-(
    TGs05sLOGPOL(2,<<"Score05_Sequence: WARNING: Goalie intercepts ball at via point "<<viaPointIndex<<"!");
  }

  if (goalieInterceptResult > 0)
  {
    TGs05sLOGPOL(LOGBASELEVEL+6,<<"Score05_Sequence: check kick success -> NOPE, because goalie intercepts, [inTCycles="<<inTCycles<<"] time="<<goalieInterceptResult);
    return false; //goalie intercepts
  }
    
  //will an opponent player intercept the ball?
  int opponentInterceptResult
    = intercept_opponents ( targetDirection.get_value_mPI_pPI(), //ball's direction
                            maximalReachableVelocity, //ball's speed
                            12, //note: value is not used!
                            ivViaPoints[viaPointIndex][inTCycles],
                            inTCycles );
  if (opponentInterceptResult > 0)
  {
    TGs05sLOGPOL(LOGBASELEVEL+6,<<"Score05_Sequence: check kick success -> NOPE, because opponent intercepts, res="<<opponentInterceptResult);
    return false; //an opponent player will intercept
  }
  return true;

}


bool
Score05_Sequence::checkIfGoalIsForSureWhenTacklingNow( int  tackDir,
                                                       bool debug )
{
  MyState assumedState;
  assumedState.my_pos = ivpInternalMeObject->pos;
  assumedState.my_vel = ivpInternalMeObject->vel;
  assumedState.ball_pos = ivpInternalBallObject->pos;
  assumedState.ball_vel = ivpInternalBallObject->vel;
  assumedState.my_angle = WSinfo::me->ang; 
  assumedState.op_pos = Vector(0.0,0.0);
  assumedState.op_bodydir = ANGLE(0);
  assumedState.op_bodydir_age = 0;
  assumedState.op = NULL;

  Vector finalBallVelocity, tackleVector;
  if ( tackDir == TACKLE_FORWARD )
    tackleVector.init_polar( ServerOptions::ball_speed_max,
                             ivpInternalMeObject->ang );
  else
    tackleVector.init_polar( ServerOptions::ball_speed_max,
                             ivpInternalMeObject->ang + ANGLE(PI) );
  finalBallVelocity = tackleVector + ivpInternalBallObject->vel;
  if ( finalBallVelocity.norm() > ServerOptions::ball_speed_max )
    finalBallVelocity.normalize( ServerOptions::ball_speed_max );

  if (debug)
  {
    Vector dist20 = finalBallVelocity;
    dist20.normalize(20.0);
    dist20+=ivpInternalBallObject->pos;
    TGs05sLOGPOL(LOGBASELEVEL+6,<<_2D<<VL2D(ivpInternalBallObject->pos,
      dist20,"#000077"));
  }
  
  //would the ball go towards the goal at all?
  ANGLE targetDirection = ANGLE(finalBallVelocity.getX(),finalBallVelocity.getY());
  Vector ballPos = ivpInternalBallObject->pos;
  Angle minAngDevFromPost = PI*(2.0/180.0);
  Angle  minShootDirection 
    = (HIS_GOAL_RIGHT_CORNER - ballPos).ARG().get_value_mPI_pPI() + minAngDevFromPost;
  Angle  maxShootDirection
    = (HIS_GOAL_LEFT_CORNER - ballPos).ARG().get_value_mPI_pPI() - minAngDevFromPost;
  if (    targetDirection.get_value_mPI_pPI() > maxShootDirection
       || targetDirection.get_value_mPI_pPI() < minShootDirection )
  {
    TGs05sLOGPOL(LOGBASELEVEL+4,<<"Score05_Sequence: check tackle ("<<tackDir
      <<") success -> NOPE, because the ball would not go into the goal!");
    return false; //goalie intercepts
  }
  
  //will the goalie intercept the ball?
  int goalieInterceptResult
    = intercept_goalie(   ivpInternalBallObject->pos, 
                          finalBallVelocity, 
                          ivInternalHisGoaliePosition, 
                          GOALIE_INITIAL_SIZE,
                          ivInternalHisGoalieAge, //time offset /*TG_OSAKA*/
                          debug);
  if (goalieInterceptResult > 0)
  {
    TGs05sLOGPOL(LOGBASELEVEL+4,<<"Score05_Sequence: check tackle ("<<tackDir
      <<") success -> NOPE, because goalie intercepts, time="<<goalieInterceptResult);
    return false; //goalie intercepts
  }
    
  //will an opponent player intercept the ball?
  int opponentInterceptResult
    = intercept_opponents ( targetDirection.get_value_mPI_pPI(), //ball's direction
                            finalBallVelocity.norm(), //ball's speed
                            12, //note: value is not used!
                            ivpInternalBallObject->pos,
                            0); 
  if (opponentInterceptResult > 0)
  {
    TGs05sLOGPOL(LOGBASELEVEL+4,<<"Score05_Sequence: check tackle ("<<tackDir
      <<") success -> NOPE, because opponent intercepts, res="<<opponentInterceptResult);
    return false; //an opponent player will intercept
  }
  
  TGs05sLOGPOL(LOGBASELEVEL,<<"Score05_Sequence: check tackle ("<<tackDir
    <<") success -> YEP, ###SS05###");
  
  return true;
}

bool
Score05_Sequence::checkIfGoalIsForSureWhenV12TacklingNowIntoDirection
                  ( int    angularTackDir,
                    Vector &resultingBallMovement,
                    bool   relaxedChecking,
                    bool   debug )
{
  //how fast will the final kick be?
  Vector ballNewPos, dummyBallVelAfterTackling;
  Tools::model_tackle_V12( ivpInternalMeObject->pos,
                           ivpInternalMeObject->ang,
                           ivpInternalBallObject->pos,
                           ivpInternalBallObject->vel,
                           angularTackDir,
                           ballNewPos,
                           dummyBallVelAfterTackling);
  Vector ballMovementAfterTackle = ballNewPos - ivpInternalBallObject->pos;
  resultingBallMovement = ballMovementAfterTackle; 

  //debug output
  if (debug)
  {
    Vector dist20 = ballMovementAfterTackle;
    dist20.normalize(20.0*ballMovementAfterTackle.norm());
    dist20+=ivpInternalBallObject->pos;
    TGs05sLOGPOL(LOGBASELEVEL+6,<<_2D<<VL2D(ivpInternalBallObject->pos,
      dist20,"#000077"));
  }
  
  //would the ball go towards the goal at all?
  ANGLE targetDirection = ANGLE( ballMovementAfterTackle.getX(),
                                 ballMovementAfterTackle.getY());
  Vector ballPos = ivpInternalBallObject->pos;
  Angle minAngDevFromPost = PI*(2.0/180.0);
  Angle minShootDirection = (HIS_GOAL_RIGHT_CORNER - ballPos)
                            .ARG().get_value_mPI_pPI() + minAngDevFromPost;
  Angle maxShootDirection = (HIS_GOAL_LEFT_CORNER - ballPos)
                            .ARG().get_value_mPI_pPI() - minAngDevFromPost;
  if (    targetDirection.get_value_mPI_pPI() > maxShootDirection
       || targetDirection.get_value_mPI_pPI() < minShootDirection )
  {
    TGs05sLOGPOL(LOGBASELEVEL+4,<<"Score05_Sequence: Check angular tackle ("
      <<angularTackDir
      <<") success -> NOPE, because the ball would not go into the goal!");
    return false; //ball misses the goal
  }

  //take care of relaxed checking
  if (relaxedChecking)
    ballMovementAfterTackle.normalize( ballMovementAfterTackle.norm() * 1.3 );
  
  //will the goalie intercept the ball?
  int goalieInterceptResult
    = intercept_goalie(   ivpInternalBallObject->pos, 
                          ballMovementAfterTackle, 
                          ivInternalHisGoaliePosition, 
                          GOALIE_INITIAL_SIZE,
                          ivInternalHisGoalieAge, 
                          debug);
  if (goalieInterceptResult > 0)
  {
    TGs05sLOGPOL(LOGBASELEVEL+4,<<"Score05_Sequence: Check angular tackle ("
      <<angularTackDir<<",RLX="<<relaxedChecking
      <<") success -> NOPE, because goalie intercepts, time="
      <<goalieInterceptResult);
    return false; //goalie intercepts
  }
    
  //will an opponent player intercept the ball?
  int opponentInterceptResult
    = intercept_opponents ( targetDirection.get_value_mPI_pPI(), //ball's direction
                            ballMovementAfterTackle.norm(), //ball's speed
                            12, //note: value is not used!
                            ivpInternalBallObject->pos,
                            0); 
  if (opponentInterceptResult > 0)
  {
    TGs05sLOGPOL(LOGBASELEVEL+4,<<"Score05_Sequence: check angular tackle ("
      <<angularTackDir<<",RLX="<<relaxedChecking<<") success -> NOPE, because "
      <<"opponent intercepts, res="<<opponentInterceptResult);
    return false; //an opponent player will intercept
  }
  
  TGs05sLOGPOL(LOGBASELEVEL,<<"Score05_Sequence: check angular tackle ("
    <<angularTackDir<<",RLX="<<relaxedChecking<<") success -> YEP, ###SS05###");
  return true;
}

bool
Score05_Sequence::checkIfGoalIsForSureWhenV12TacklingNow( int  & angularTackDir,
                                                          bool   debug )
{
  //define the iterations
  const int coarseAngleIncrement = 10;
  const int fineAngleIncrement   = 1;
  int       potentialDirections[360/coarseAngleIncrement];
  int       potentialDirectionCounter = 0;
  const int maxGoodDirections = 40;
  int       goodDirections[maxGoodDirections];
  Vector    goodBallMovements[maxGoodDirections];
  int       goodDirectionCounter = 0;
  //coarse main loop
  for ( int checkAngle = 0; 
        checkAngle < 360; 
        checkAngle += coarseAngleIncrement )
  {
    Vector dummyResultingBallMovement;
    if ( checkIfGoalIsForSureWhenV12TacklingNowIntoDirection
                  ( checkAngle,
                    dummyResultingBallMovement,
                    true, //relaxedChecking
                    false ) //debug
       )
    {
      potentialDirections[ potentialDirectionCounter ] = checkAngle;
      potentialDirectionCounter ++ ;
    }
  }
  //fine-grained main loop
  bool skipOuterLoop = false;
  for ( int c = 0; c < potentialDirectionCounter; c++ )
  {
    int currentBaseCheckAngle = potentialDirections[c];
    for ( int checkAngle =   currentBaseCheckAngle 
                           - (coarseAngleIncrement / 2);
          checkAngle < currentBaseCheckAngle + (coarseAngleIncrement / 2);
          checkAngle += fineAngleIncrement )
    {
      Vector resultingBallMovement;
      if ( checkIfGoalIsForSureWhenV12TacklingNowIntoDirection
                    ( checkAngle,
                      resultingBallMovement,
                      false, //relaxedChecking
                      false ) //debug
         )
      {
        goodDirections[goodDirectionCounter] = checkAngle; 
        if (checkAngle <    0) goodDirections[goodDirectionCounter] += 360;
        if (checkAngle >= 360) goodDirections[goodDirectionCounter] -= 360;
        goodBallMovements[goodDirectionCounter] = resultingBallMovement;
        goodDirectionCounter ++ ;
        if (goodDirectionCounter >= maxGoodDirections)
        {
          skipOuterLoop = true;
          break;
        }
      }
    }
    if (skipOuterLoop) break;
  }
  //select the best direction
  int bestDirectionIndex = -1;
  double bestDirectionValue = 0.0, currentDirectionValue;
  Vector dummyShotCrossesTorauslinie;
  for ( int d = 0; d < goodDirectionCounter; d++ )
  {
    currentDirectionValue = evaluateGoalShotWithRespectToGoalieAndPosts
                            ( goodBallMovements[d].ARG(),
                              ivpInternalBallObject->pos,
                              dummyShotCrossesTorauslinie,
                              goodBallMovements[d].norm() );
    if ( currentDirectionValue > bestDirectionValue )
    {
      bestDirectionValue = currentDirectionValue;
      bestDirectionIndex = d;
    }
  }
  //return the best
  if ( bestDirectionIndex != -1 )
  {
    angularTackDir = goodDirections[bestDirectionIndex];
    return true;
  }
  return false;
}

bool
Score05_Sequence::checkIfPointOnCrossLineIsViable( Vector p )
{
  if (  p.distance(getMyPositionInTCyclesAssumingNoDash(1)) 
      > WSinfo::me->kick_radius - VIA_POINT_MARGIN)
    return false;
  return true;
}

bool
Score05_Sequence::checkKickSequencesViaViaPoints( int & bestViaPointIndex,
                                       int & nStepKick )
{
  //check one step preparation kicks
  for (int i=0; i<NUMBER_OF_VIA_POINTS; i++)
  {
    TGs05sLOGPOL(LOGBASELEVEL+3,<<"Score05_Sequence: ###1### Evaluate position "<<i<<" for one step kick.");
    bool goal = evaluateScorePositionAfterTCycles(1, i);
    if (goal)
    {
      nStepKick         = 1;
      bestViaPointIndex = i;
      TGs05sLOGPOL(LOGBASELEVEL+3,<<"Score05_Sequence: check kick seq: YEP, bestVPIdx="<<bestViaPointIndex<<", nStep="<<nStepKick);
      return true;
    }
  }
  
  //check two step preparation kicks
  for (int i=0; i<NUMBER_OF_VIA_POINTS; i++)
  {
    TGs05sLOGPOL(LOGBASELEVEL+3,<<"Score05_Sequence: ###2### Evaluate position "<<i<<" for two step kick.");
    bool goal = evaluateScorePositionAfterTCycles(2, i);
    if (goal)
    {
      bestViaPointIndex = i;
      nStepKick         = 2;
      TGs05sLOGPOL(LOGBASELEVEL+3,<<"Score05_Sequence: check kick seq: YEP, bestViaPointIdx="<<bestViaPointIndex<<", nStepKick="<<nStepKick);
      return true;
    }
  }
  TGs05sLOGPOL(LOGBASELEVEL+3,<<"Score05_Sequence: Check kick sequencees via via points: NOPE (found no sequence)");
  return false;
}

void
Score05_Sequence::determineGoaliePositionAndAge(Vector & gPos, int & gAge)
{
  if (WSinfo::ws->time == ivInternalHisGoalieLastDeterminedAtTime)
    return; //do this only once per cycle
  ivInternalHisGoalieLastDeterminedAtTime = WSinfo::ws->time;
    
  PPlayer hisGoalie 
    = WSinfo::alive_opponents.get_player_by_number( WSinfo::ws->his_goalie_number );
  if ( hisGoalie == NULL ) 
  {
    TGs05sLOGPOL(LOGBASELEVEL+3,<<"Score05_Sequence: SEVERE PROBLEM: I don't "
      <<"know who/where his goalie is (nr="<<WSinfo::ws->his_goalie_number<<")."
      <<" He is not among the ALIVE opponents.");
    //there is not alive opponent goalie -> place him outside the pitch
    gPos = Vector(FIELD_BORDER_X, 40.0);
    //no goalie present -> age==0 :-)
    gAge = 0;
  }
  else
  {
    gPos = hisGoalie->pos; //where i last saw him
                           //this pos is evtl. invalid!
    //age and velocity considerations
    gAge = hisGoalie->age;
    if (gAge > 20) gAge = 20;
    if ( hisGoalie->age_vel==0 ) 
      gPos += hisGoalie->vel;
    else
      gPos += 0.4 * hisGoalie->vel;
    //PENALTY MODE CONSIDERATION
    if (    WSinfo::ws->play_mode == PM_my_PenaltyKick
         && WSinfo::me->vel.norm() < 0.2 //i am not rushing (e.g. intercepting)
       )
    {
      TGs05sLOGPOL(0,"Score05_Sequence: *** PENALTY MODE ***: Disregard goalie "
        <<"velocity in determining his velocity!");
      gPos = hisGoalie->pos + (0.4*hisGoalie->vel);
    }
  }
  
  if (WSinfo::his_goalie == NULL)
  {
    //special case: goalie is not valid, having an age>100
    bool hisGoalieHasProbablyDisconnected = true;
    for (int checkDelta = -8; checkDelta <= 8; checkDelta ++ )
    {
      Vector checkPoint
        = Vector(FIELD_BORDER_X,(double)checkDelta) - ivpInternalMeObject->pos;
      ANGLE checkAngle = checkPoint.ARG();
      if (WSmemory::last_seen_in_dir( checkAngle ) > 3)
        hisGoalieHasProbablyDisconnected = false;
    }
    
    if (hisGoalieHasProbablyDisconnected)
    {
      TGs05sLOGPOL(LOGBASELEVEL+3,<<"Score05_Sequence: GOOD NEWS: I am quite "
        <<"sure that his goalie has disconnected. :-)");
      //opponent goalie disconnected -> place him outside the pitch
      gPos = Vector(FIELD_BORDER_X, 40.0);
      //goalie disconnected -> age==0 :-)
      gAge = 0;
    }
    else
    {
      //i am quite clueless. all i know is the old goalie position where i
      //saw him for the last time (at time player->time) and where i have
      //looked recently. ==> GUESS HIS POSITION!
      const int searchDepth = 4;
      int maxGoaliePositions = (int) pow( 2.0, (double)searchDepth );
      if (   hisGoalie
          && searchDepth < MAX_VIEW_INFO )
      {
        Vector *goaliePositionsArray = new Vector [maxGoaliePositions];
        for (int i=0; i<maxGoaliePositions; i++)
        {
          goaliePositionsArray[i].setXY( -1000.0, -1000.0 );
        }
        goaliePositionsArray[0].clone( hisGoalie->pos);
        //call to recursive method: fills the array of positions
        getPotentialGoaliePositions( searchDepth,
                                     0,
                                     goaliePositionsArray );
        //decision: which guessed position is most likely?
        double minDist = 1000.0;
        Vector suggestedGoaliePosition(-1000.0,-1000.0);
        for (int i=0; i<maxGoaliePositions; i++)
        {
          double currentPositionsDistance //weighting!
            =   0.6 * goaliePositionsArray[i].distance( hisGoalie->pos )
              + 0.4 * goaliePositionsArray[i].distance( HIS_GOAL_CENTER );
            
          if (   goaliePositionsArray[i].getX() > -999.9 //ausgefuellt
              && currentPositionsDistance < minDist )
          {
            minDist = currentPositionsDistance;
            suggestedGoaliePosition = goaliePositionsArray[i];
          }
        }        
        if ( minDist < 1000.0 )
        {
          TGs05sLOGPOL(0,"Score05_Sequence: SUGGESTED goalie pos: "
            << suggestedGoaliePosition);
          TGs05sLOGPOL(0,<<_2D<<VC2D(suggestedGoaliePosition,2.0,"ffff44"));
          gPos = suggestedGoaliePosition;
          gAge = WSmemory::last_seen_in_dir( (gPos-WSinfo::me->pos).ARG() ) / 2;
          if (gAge == 0) gAge = 1;
          if (gAge > MAX_VIEW_INFO) gAge = MAX_VIEW_INFO / 2;
                 //not 0 to reflect some degree of uncertainty
        }
for (int d=0; d<maxGoaliePositions; d++)
{
    TGs05sLOGPOL(0,<<"******* "<<d<<": "<<goaliePositionsArray[d]
      <<" distToLastSeen="<<goaliePositionsArray[d].distance( hisGoalie->pos )
      <<" distToHISGC="<<goaliePositionsArray[d].distance( HIS_GOAL_CENTER )
      <<" => WEIGHTEDdist="<<(0.6 * goaliePositionsArray[d].distance( hisGoalie->pos )
              + 0.4 * goaliePositionsArray[d].distance( HIS_GOAL_CENTER )));
}
        TGs05sLOGPOL(0,<<std::flush);
        delete [] goaliePositionsArray;
        goaliePositionsArray = NULL;
      }
    }
  }
  
  TGs05sLOGPOL(LOGBASELEVEL+2,<<"Score05_Sequence: Determined his goalie's pos "
    <<gPos<<" and age (age="<<gAge<<".");
  if (WSinfo::his_goalie)
  {
    TGs05sLOGPOL(LOGBASELEVEL+2,<<"Score05_Sequence: Further, his vel="
      <<WSinfo::his_goalie->vel<<" and age_vel="<<WSinfo::his_goalie->age_vel);
  }
  
  //goalie modelling for dash(100) and dash(-100)
  ivInternalHisGoaliePositionWhenDashP100 = gPos;
  ivInternalHisGoaliePositionWhenDashM100 = gPos;
  if (WSinfo::his_goalie && WSinfo::his_goalie->age == 0
                         && WSinfo::his_goalie->age_vel == 0)
  {
    Vector oldPos = WSinfo::his_goalie->pos, oldVel = WSinfo::his_goalie->vel,
           newPosP100, newPosM100, newVelP100, newVelM100;
    ANGLE  oldANG = WSinfo::his_goalie->ang, newANG;
    int    oldStamina = 4000, newStamina;
    Cmd_Body dashP100Cmd, dashM100Cmd;
    dashP100Cmd.set_dash(100.0); dashM100Cmd.set_dash(-100.0);
   
    Tools::simulate_player( oldPos, 
                            oldVel,
                            oldANG, 
                            oldStamina,
                            dashM100Cmd,
                            newPosM100, 
                            newVelM100,
                            newANG, 
                            newStamina,
                            WSinfo::his_goalie->stamina_inc_max,
                            WSinfo::his_goalie->inertia_moment,
                            WSinfo::his_goalie->dash_power_rate, 
                            WSinfo::his_goalie->effort,
                            WSinfo::his_goalie->decay);
    Tools::simulate_player( oldPos, 
                            oldVel,
                            oldANG, 
                            oldStamina,
                            dashP100Cmd,
                            newPosP100, 
                            newVelP100,
                            newANG, 
                            newStamina,
                            WSinfo::his_goalie->stamina_inc_max,
                            WSinfo::his_goalie->inertia_moment,
                            WSinfo::his_goalie->dash_power_rate, 
                            WSinfo::his_goalie->effort,
                            WSinfo::his_goalie->decay);
    ivInternalHisGoaliePositionWhenDashP100 = newPosP100;
    ivInternalHisGoaliePositionWhenDashP100 = newPosM100;
  }
  //surprise shot if ball eventually in kickrange of me and an opponent
  PlayerSet troublePSet= WSinfo::valid_opponents;
  troublePSet.keep_players_in_circle( ivInternalBallObject.pos, 
                                      2.0*ServerOptions::kickable_area); 
  troublePSet.keep_and_sort_closest_players_to_point(1,ivInternalBallObject.pos);
  if (    troublePSet.num > 0 
       &&   ivInternalBallObject.pos.distance( troublePSet[0]->pos ) 
          > 1.0 * troublePSet[0]->kick_radius - ServerOptions::ball_size 
       &&    ivInternalBallObject.pos.distance( troublePSet[0]->pos ) 
          <= 1.0 * troublePSet[0]->kick_radius ) 
  {
    if (WSinfo::his_goalie)
    {
      gAge = WSinfo::his_goalie->age / 2;
      TGs05sLOGPOL(0,"Score05_Sequence: The opponent eventually will not"
        <<" realize that the ball is kickable for him. I consider a"
        <<" reduced goalie age of "<<gAge<<" to ENFORCE scoring.");
    }
  }
  
}

ANGLE
Score05_Sequence::determineGoodLookDirectionIfGoalieIsPresent()
{
  ANGLE lookAngle,
        hisGoalCenterAngle
          = (HIS_GOAL_CENTER - ivpInternalMeObject->pos).ARG(),
        hisGoalLeftPostAngle
          = (HIS_GOAL_LEFT_CORNER - ivpInternalMeObject->pos).ARG(),
        hisGoalRightPostAngle
          = (HIS_GOAL_RIGHT_CORNER - ivpInternalMeObject->pos).ARG();
  //determine dangerous opponents
  PlayerSet opponents = WSinfo::valid_opponents;
  opponents.keep_players_in_cone
            ( ivpInternalMeObject->pos,
              Vector(FIELD_BORDER_X,-8.5) - ivpInternalMeObject->pos,
              Vector(FIELD_BORDER_X,8.5) - ivpInternalMeObject->pos );
  opponents.keep_and_sort_players_by_age( opponents.num );
  for (int cop=0; cop<opponents.num; cop++)
  {
    TGs05sLOGPOL(0,<<"Score05_Sequence: Player in view check cone: num="
      <<opponents[cop]->number<<", age="<<opponents[cop]->age);
  }
  //determine angles to which definitely not to look
  ANGLE lastViewWidth(0.0), lastViewDir(0.0);
  Vector lastViewMyPos;
  long lastViewAbsTime;
  ANGLE excludeViewAnglesFrom, excludeViewAnglesTo;
  if ( WSmemory::get_view_info_before_n_steps( 0, 
                                               lastViewWidth,
                                               lastViewDir,
                                               lastViewMyPos,
                                               lastViewAbsTime )
       && lastViewAbsTime == WSinfo::ws->time )
  {
    ANGLE nextViewAngleWidth = Tools::next_view_angle_width();
    excludeViewAnglesFrom 
      = ANGLE(
                 lastViewDir.get_value_0_p2PI() 
               - (0.45*lastViewWidth.get_value_0_p2PI())
        //       - (0.45*(nextViewAngleWidth.get_value_0_p2PI()))
             );            //0.45 instead of 0.5 to reach
    excludeViewAnglesTo    //some extra/safety coverage
      = ANGLE(
                 lastViewDir.get_value_0_p2PI()
               + (0.45*lastViewWidth.get_value_0_p2PI())
        //       + (0.45*(nextViewAngleWidth.get_value_0_p2PI()))
             );
    TGs05sLOGPOL(0,<<"Score05_Sequence: For next viewing I will exclude"
      <<" angles from "<<RAD2DEG(excludeViewAnglesFrom.get_value_0_p2PI())
      <<" to "<<RAD2DEG(excludeViewAnglesTo.get_value_0_p2PI()));
  }
  //iteration over angles
  ANGLE nextMinViewANGLE =   (  Vector(FIELD_BORDER_X,-8.0)  
                              - ivpInternalMeObject->pos ).ARG() 
                           - ANGLE(PI*(25./180.)),
        nextMaxViewANGLE =   (  Vector(FIELD_BORDER_X,8.0) 
                              - ivpInternalMeObject->pos ).ARG()
                           + ANGLE(PI*(25./180.)),
        currentANGLE, bestANGLE;
  TGs05sLOGPOL(0,<<"Score05_Sequence: nextMinViewANGLE="
    <<nextMinViewANGLE.get_value_mPI_pPI()<<" nextMaxViewANGLE="
    <<nextMaxViewANGLE.get_value_mPI_pPI());
  int currentAngleEvaluation, highestEval = -1000, bestAngle = 0;
  for (int i=0; i<360; i+=5)
  {
    currentAngleEvaluation = 0;
    currentANGLE = ANGLE(  ((double)i)*(PI/180.)  );
    if ( Tools::could_see_in_direction( currentANGLE ) == false )
      currentAngleEvaluation = -99;
    else
    if (    currentANGLE.get_value_mPI_pPI() < nextMinViewANGLE.get_value_mPI_pPI()
         || currentANGLE.get_value_mPI_pPI() > nextMaxViewANGLE.get_value_mPI_pPI() )
      currentAngleEvaluation = -97;
    else
    if (    currentANGLE.get_value_mPI_pPI() < excludeViewAnglesTo.get_value_mPI_pPI()
         && currentANGLE.get_value_mPI_pPI() > excludeViewAnglesFrom.get_value_mPI_pPI() )
      currentAngleEvaluation = -100;
    else
    {
      long lastSeenInDirResult
        = WSmemory::last_seen_in_dir( currentANGLE );
      if (lastSeenInDirResult > WSinfo::ws->time )
        currentAngleEvaluation = MAX_VIEW_INFO;
      else
        currentAngleEvaluation = lastSeenInDirResult;
    }
    TGs05sLOGPOL(1,<<"Score05_Sequence: Look angle "<<i<<" curANG="
      <<currentANGLE.get_value_mPI_pPI()<<": init eval = "<<currentAngleEvaluation)
    for (int j=0; j<opponents.num; j++)
    {
      PPlayer anOpponent = opponents[j];
      ANGLE   angleToAnOpponent = (anOpponent->pos - ivpInternalMeObject->pos).ARG();
      if ( fabs(   angleToAnOpponent.get_value_mPI_pPI()
                 - currentANGLE.get_value_mPI_pPI() ) < PI*(15./180.) )
      {
        currentAngleEvaluation += 2 * anOpponent->age;
        if (anOpponent->number == WSinfo::ws->his_goalie_number)
          currentAngleEvaluation += 3 * anOpponent->age;
      }
    }
    TGs05sLOGPOL(1,<<"Score05_Sequence: Look angle "<<i<<" curANG="
      <<currentANGLE.get_value_mPI_pPI()<<": eval = "<<currentAngleEvaluation)
    if ( currentAngleEvaluation > highestEval)
    {
      highestEval = currentAngleEvaluation;
      bestAngle = i;
      bestANGLE = currentANGLE;
    }
  }
  //final decision!
  ANGLE goalieAngle
          = (WSinfo::his_goalie->pos - ivpInternalMeObject->pos).ARG();
      //if (   dangerousPlayer 
      //    && Tools::could_see_in_direction
      //              ((dangerousPlayer->pos-ivpInternalMeObject->pos).ARG()) )
      //  lookAngle =   (dangerousPlayer->pos-ivpInternalMeObject->pos).ARG()
      //              + deltaAngle;
      //else
  if (   Tools::could_see_in_direction(goalieAngle)
      && (    WSinfo::his_goalie->age > 1 
           || WSmemory::last_seen_in_dir(goalieAngle) > 1 ) )
    lookAngle = goalieAngle;
  else
  if (    highestEval > 0
      && Tools::could_see_in_direction( bestAngle ) )
    lookAngle = bestANGLE;
  else
  if (Tools::could_see_in_direction(hisGoalCenterAngle))
    lookAngle = hisGoalCenterAngle;
  else
  if (Tools::could_see_in_direction(hisGoalLeftPostAngle))
    lookAngle = hisGoalLeftPostAngle;
  else
  if (Tools::could_see_in_direction(hisGoalRightPostAngle))
    lookAngle = hisGoalRightPostAngle;
  
  TGs05sLOGPOL(0,<<"Score05_Sequence: Decided to look to "
    <<lookAngle<<" (highestEval="<<highestEval<<",bestAngle="
    <<bestANGLE.get_value_mPI_pPI()<<")");
 
  return lookAngle;       
}        

bool
Score05_Sequence::enemyMayReachPointInTCycles(Vector p, int inTCycles)
{
  PlayerSet consideredOpponents = WSinfo::valid_opponents;
  consideredOpponents.keep_players_in_circle( p, 3.0 * inTCycles );
  Cmd dummyCmd;
  int numberOfCyclesToIntercept = 0;
  for (int i=0; i<consideredOpponents.num; i++)
  {
    //consideration for opponent player i
    dummyCmd.cmd_body.unset_lock(); dummyCmd.cmd_body.unset_cmd();
    numberOfCyclesToIntercept = inTCycles + 1;
    ivpInterceptBallBehavior->get_cmd_arbitraryPlayer
      ( consideredOpponents[i],
        dummyCmd,
        consideredOpponents[i]->pos,
        consideredOpponents[i]->vel,
        consideredOpponents[i]->ang,
        p, //virtual ball position
        Vector(0.0,0.0), //virtual ball velocity
        numberOfCyclesToIntercept,
        inTCycles + 1 //maximal number of cycles to check
      );
    if (numberOfCyclesToIntercept <= inTCycles)
    {
      TGs05sLOGPOL(LOGBASELEVEL+1,<<"Score05_Sequence: ENEMY no. "<<consideredOpponents[i]->number
        <<" may reach the ball (at point "<<p<<") within "<<numberOfCyclesToIntercept<<" cycles! *grrrr*");
      return true;
    }
  }
  return false;
}

double
Score05_Sequence::evaluateGoalShotWithRespectToGoalieAndPosts
                  ( const ANGLE  &targetDirection,
                    const Vector &ballPos,
                    Vector       &shotCrossesTorausLinie,
                    const double &initialBallVel )
{
  Vector shotWayVector(targetDirection),
         shotWayStartPoint = ballPos;
  //calculate where the ball crosses the torauslinie
  shotCrossesTorausLinie //is a return value!
    = point_on_line(shotWayVector, shotWayStartPoint, FIELD_BORDER_X);
  double currentShotGoalieValue;
  if (WSinfo::his_goalie)
  {
    //old calculation
    currentShotGoalieValue
      = Tools::get_dist2_line( ivpInternalBallObject->pos, 
                               shotCrossesTorausLinie,
                               WSinfo::his_goalie->pos );
    Vector lotfussPoint = Tools::get_Lotfuss(ivpInternalBallObject->pos,
                                             shotCrossesTorausLinie,
                                             WSinfo::his_goalie->pos );
    if ( lotfussPoint.getX() > FIELD_BORDER_X )
      currentShotGoalieValue 
        = shotCrossesTorausLinie.distance(WSinfo::his_goalie->pos);
    //TG08: new calculation: find most critical time step
    Vector simulatedBallPos = ballPos,
           simulatedBallVel;
    simulatedBallVel.init_polar( initialBallVel, targetDirection );
    double shortestGoalieIcptDistance = INT_MAX, currentGoalieIcptDist;
    for (int i=1; i<30; i++)
    {
      simulatedBallPos += simulatedBallVel;
      simulatedBallVel *= ServerOptions::ball_decay;
      //current icpt distance
      currentGoalieIcptDist 
        =   WSinfo::his_goalie->pos.distance( simulatedBallPos )
          - (   ivGOALIE_CATCH_RADIUS
              + goalie_action_radius_at_time(i,GOALIE_INITIAL_SIZE,0) );
      if ( currentGoalieIcptDist < shortestGoalieIcptDistance )
      {
        if (currentGoalieIcptDist<0.0) currentGoalieIcptDist=0.0;
        shortestGoalieIcptDistance = currentGoalieIcptDist;
        currentShotGoalieValue     = sqrt(shortestGoalieIcptDistance);
      }
      if ( simulatedBallPos.getX() > FIELD_BORDER_X )
        break;
    }
  }
  else currentShotGoalieValue = 1.0; //no goalie found

  double currentShotLeftPostValue
          = sqrt(shotCrossesTorausLinie.distance( HIS_GOAL_LEFT_CORNER )),
        currentShotRightPostValue
          = sqrt(shotCrossesTorausLinie.distance( HIS_GOAL_RIGHT_CORNER )),
        currentShotPostValue
          = Tools::min(currentShotRightPostValue, currentShotLeftPostValue);
              
  double goalieWeight = 0.55,//0.7, //TG08
        postWeight   = 0.45;//0.3;

double currentShotValue =   goalieWeight * currentShotGoalieValue
                         + postWeight * currentShotPostValue;
TGs05sLOGPOL(/*LOGBASELEVEL+*/0,
<<RAD2DEG(targetDirection.get_value_mPI_pPI())<<": C0 g="<<currentShotGoalieValue<<" p="<<currentShotPostValue<<" w="<<currentShotValue);

  postWeight *= (0.6*sin( fabs(targetDirection.get_value_mPI_pPI()) )) + 0.4;//TG07. old: 0.8-0.2
  currentShotValue
    = goalieWeight * currentShotGoalieValue + postWeight * currentShotPostValue;

TGs05sLOGPOL(/*LOGBASELEVEL+*/0,
<<RAD2DEG(targetDirection.get_value_mPI_pPI())<<": C1 g="<<currentShotGoalieValue<<" p="<<currentShotPostValue<<" w="<<currentShotValue);
  return currentShotValue;
}

bool 
Score05_Sequence::evaluateScorePositionAfterTCycles(int afterCycles, 
                                                    int viaPointIndex)
{
  Vector ballVel;
  double finalKickVel;
  Cmd    oneStepCommand, twoStepCommand;
  if (afterCycles == 1)
  {
    if ( 
         getBestOneStepKickToViaPoint( viaPointIndex, 
                                       oneStepCommand,
                                       ballVel )
       )
    {
      TGs05sLOGPOL(LOGBASELEVEL+4,<<"Score05_Sequence: I can kick to viaPoint "<<viaPointIndex<<" in ONE step and get resulting ballVel after one step: "<<ballVel);
      if (checkIfGoalIsForSureWhenShootingInTCyclesFromViaPoint
             ( 1, viaPointIndex, ballVel, finalKickVel )  )
      {
        TGs05sLOGPOL(LOGBASELEVEL+4,<<"Score05_Sequence: WOW! Ball is for sure in goal when kicking from this viapoint "<<ivViaPoints[viaPointIndex][1]<<" ["<<viaPointIndex<<"] in the next cycle! Final kick vel = "<<finalKickVel);
        ivCommandForThisCycle = oneStepCommand;
        ivMode = MODE_TO_VIA_POINT;
        return true;
      }
      else
      {
        TGs05sLOGPOL(LOGBASELEVEL+4,<<"Score05_Sequence: HM! One step kick to via point "<<viaPointIndex<<" would not bring me in scoring position. Final kick vel = "<<finalKickVel);
      }
    }
    else
    {
      TGs05sLOGPOL(LOGBASELEVEL+4,<<"Score05_Sequence: DAMN! Did not find a One step kick to via point "<<viaPointIndex<<".");
    }
                                  
  }
  if (afterCycles == 2)
  {
    if (
         getBestTwoStepKickToViaPoint( viaPointIndex, 
                                       oneStepCommand,
                                       twoStepCommand,
                                       ballVel )
       )
    {
      TGs05sLOGPOL(LOGBASELEVEL+4,<<"Score05_Sequence: I can kick to viaPoint "<<viaPointIndex<<" in TWO steps and get resulting ballVel after TWO steps: "<<ballVel);
      if (checkIfGoalIsForSureWhenShootingInTCyclesFromViaPoint
           ( 2, viaPointIndex, ballVel, finalKickVel ) )
      {
        TGs05sLOGPOL(LOGBASELEVEL+4,<<"Score05_Sequence: WOW! Ball is for sure in goal [ballVel="<<ballVel<<"] when kicking from this ("<<ivViaPoints[viaPointIndex][2]<<") ["<<viaPointIndex<<"] point in the OVERNEXT cycle! Final kick vel = "<<finalKickVel);
        ivCommandForThisCycle = oneStepCommand;
        ivMode = MODE_TO_CROSSLINE_POINT;
        return true;
      }
      else
      {
        TGs05sLOGPOL(LOGBASELEVEL+4,<<"Score05_Sequence: HM! Two step kick to via point "<<viaPointIndex<<" would not bring me in scoring position. Final kick vel = "<<finalKickVel);
      }
    }
    else
    {
      TGs05sLOGPOL(LOGBASELEVEL+4,<<"Score05_Sequence: DAMN! Did not find a two step kick to via point "<<viaPointIndex<<".");
    }
  }
  ivMode = MODE_FIRST_PLANNING;
  return false;
}

bool 
Score05_Sequence::get_cmd( Cmd & cmd )
{
  setAssumedGoaliCatchRadius();
  int chosenViaPoint, stepsForPreparation;
  if (checkKickSequencesViaViaPoints(chosenViaPoint, stepsForPreparation))
  {
    ivCurrentlyPreferredViaPointIndex = chosenViaPoint;
    TGs05sLOGPOL(LOGBASELEVEL+2,<<"Score05_Sequence: get_cmd: check of kick sequences was successful!");
    //set command
    return true;
  }
  TGs05sLOGPOL(LOGBASELEVEL+2,<<"Score05_Sequence: get_cmd: check of kick sequences was NOT successful!");
  ivCurrentlyPreferredViaPointIndex = -1;
  return false;
}

//============================================================================
// test_shoot2goal
//============================================================================
bool
Score05_Sequence::getBestOneStepKickToViaPoint
  (
    int       viaPointIndex,
    Cmd     & firstCmd,
    Vector  & resultingBallVelocity
  )
{
  TGs05sLOGPOL(LOGBASELEVEL+4,<<"Score05_Sequence: I try to find the best one step kick ...");
  Cmd dummyCmd;
  Vector desiredKickVector
    = (ivViaPoints[viaPointIndex][1] - ivpInternalBallObject->pos);
  double desiredKickVelocity
    = desiredKickVector.norm();
  TGs05sLOGPOL(LOGBASELEVEL+4,<<"Score05_Sequence: ... kicking from: "<<ivpInternalBallObject->pos);
  TGs05sLOGPOL(LOGBASELEVEL+4,<<"Score05_Sequence: ... kicking to  : "<<ivViaPoints[viaPointIndex][1]<<"  (via point in 1 cycle)");
  TGs05sLOGPOL(LOGBASELEVEL+4,<<"Score05_Sequence: ... kicking with intitial vel : "<<desiredKickVelocity);
  TGs05sLOGPOL(LOGBASELEVEL+DRAWBASELEVEL,<<_2D<<VL2D(ivpInternalBallObject->pos,
    ivViaPoints[viaPointIndex][1],"ff00ff"));

  MyState assumedState;
  assumedState.my_pos = ivpInternalMeObject->pos;
  assumedState.my_vel = ivpInternalMeObject->vel;
  assumedState.ball_pos = ivpInternalBallObject->pos;
  assumedState.ball_vel = ivpInternalBallObject->vel;
  assumedState.my_angle = ivpInternalMeObject->ang;
  assumedState.op_pos = Vector(0.0,0.0);
  assumedState.op_bodydir = ANGLE(0);
  assumedState.op_bodydir_age = 0;
  assumedState.op = NULL;

  double maxKickVelocity
    = ivpOneStepKickBehavior->get_max_vel_in_dir( assumedState,
                                                  desiredKickVector.ARG() );
  ivpOneStepKickBehavior->reset_state();
  ivpOneStepKickBehavior->kick_to_pos_with_initial_vel
                          (
                            desiredKickVelocity,
                            ivViaPoints[viaPointIndex][1]
                          );

  if ( 
          desiredKickVelocity <= maxKickVelocity * 1.1
       && ivpOneStepKickBehavior->get_cmd(dummyCmd)
     )  
  {
    double firstKickPower;
    Angle firstKickAngle;
    dummyCmd.cmd_body.get_kick(firstKickPower, firstKickAngle);
    TGs05sLOGPOL(LOGBASELEVEL+4,<<"Score05_Sequence: Ok, I can make it in one step to the desired via point ("<<viaPointIndex<<").");
  }
  else
  {
    TGs05sLOGPOL(LOGBASELEVEL+4,<<"Score05_Sequence: It is not possible for me to play the ball to via point "<<viaPointIndex<<" with one kick (desiredVel="<<desiredKickVelocity<<",maxKickVel="<<maxKickVelocity<<").");
    return false;
  }
  Vector myPos   = ivpInternalMeObject->pos,   myVel   = ivpInternalMeObject->vel, 
         ballPos = ivpInternalBallObject->pos, ballVel = ivpInternalBallObject->vel;
  ANGLE myAng = ivpInternalMeObject->ang, myANGAfter1stKick;
  Vector myPosAfter1stKick, myVelAfter1stKick, 
         ballPosAfter1stKick, ballVelAfter1stKick;
  Tools::model_cmd_main( myPos,
                         myVel,
                         myAng,
                         ballPos,
                         ballVel,
                          dummyCmd.cmd_body,
                           myPosAfter1stKick,
                           myVelAfter1stKick,
                           myANGAfter1stKick,
                           ballPosAfter1stKick,
                           ballVelAfter1stKick);
  TGs05sLOGPOL(LOGBASELEVEL+DRAWBASELEVEL,<<_2D<<VC2D(ballPosAfter1stKick,0.3,"ff00ff"));
  firstCmd = dummyCmd;
  resultingBallVelocity = ballVelAfter1stKick;
  ivTargetForNextShot   = ivViaPoints[viaPointIndex][1];
  ivVelocityForNextShot = desiredKickVelocity;
  Angle dummyAngle;
  TGs05sLOGPOL(LOGBASELEVEL+4,<<"the one step command is: "<<firstCmd);
  dummyCmd.cmd_body.get_kick( ivKickPowerForNextShot, dummyAngle);
  return true;
}

bool
Score05_Sequence::getBestTwoStepKickToViaPoint
  (
    int       viaPointIndex,
    Cmd     & firstCmd,
    Cmd     & secondCmd,
    Vector  & resultingBallVelocity
  )
{
  TGs05sLOGPOL(LOGBASELEVEL+4,<<"Score05_Sequence: I try to find the best TWO step kick ... ");

  ivCrossLineStartAngle =   ANGLE(PI) + Tools::my_abs_angle_to(HIS_GOAL_CENTER);
  switch (viaPointIndex)
  {
    case 0:
      ivCrossLineAngleStep  =   ANGLE( - PI*(20.0/180.0) );
      break;
    case 1:
      ivCrossLineAngleStep  =   ANGLE( PI*(20.0/180.0) );
      break;
  }

  Vector currentCrossLineStart = ivViaPoints[viaPointIndex][2],
         currentCrossLineVector;
  ANGLE  currentCrossLineANGLE;
  Cmd dummyCmd1, dummyCmd2;
  //--main loop---------------------------------------------------------------
  TGs05sLOGPOL(LOGBASELEVEL+4,<<"Score05_Sequence: START LOOPING OVER CROSSLINES!");
  for (int i=0; i<ivNumberOfCrosslines; i++)
  {
    TGs05sLOGPOL(LOGBASELEVEL+4,<<"Score05_Sequence: START WITH CROSSLINE "<<i);
    currentCrossLineANGLE  = ANGLE( ivCrossLineAngleStep.get_value_0_p2PI() * i );
    currentCrossLineANGLE += ivCrossLineStartAngle;
    currentCrossLineVector = Vector(currentCrossLineANGLE);
    Vector currentCrossLineEnd = currentCrossLineVector;
    currentCrossLineEnd.normalize(2.5);
    currentCrossLineEnd += currentCrossLineStart;
    TGs05sLOGPOL(LOGBASELEVEL+DRAWBASELEVEL,<<_2D<<VL2D(currentCrossLineStart,
                         currentCrossLineEnd, "dddddd"));
    Vector currentCrossLinePoint;    
    double currentDelta = 2.6;
    bool iFoundATwoStepKick = false;
    Vector ballVelAfter2ndKick;
    while (currentDelta > 0.0)
    {
      dummyCmd1.cmd_body.unset_lock(); dummyCmd1.cmd_body.unset_cmd();
      dummyCmd2.cmd_body.unset_lock(); dummyCmd2.cmd_body.unset_cmd();
      currentCrossLinePoint = currentCrossLineVector;
      currentCrossLinePoint.normalize( currentDelta );
      currentCrossLinePoint += currentCrossLineStart;
//      TGs05sLOGPOL(LOGBASELEVEL+DRAWBASELEVEL,<<_2D<<C2D(currentCrossLinePoint.x,currentCrossLinePoint.y,0.1,"000000"));//BLACK
      currentDelta -= ivPointsOnCrosslineDelta;
      
      bool isPointOnCrossLineViable
        = checkIfPointOnCrossLineIsViable( currentCrossLinePoint );
      
      if (!isPointOnCrossLineViable)
      {
        TGs05sLOGPOL(LOGBASELEVEL+5,<<"Score05_Sequence: CrossLinePoint "<<currentCrossLinePoint<<" [cl="<<i<<",delta="<<currentDelta<<"] is not viable.");
        continue;
      }
      //TGs05sLOGPOL(LOGBASELEVEL+DRAWBASELEVEL,<<_2D<<C2D(currentCrossLinePoint.x,currentCrossLinePoint.y,0.1,"ff0000"));//RED
      
      Vector desiredKickVector = (currentCrossLinePoint - ivpInternalBallObject->pos);
      double desiredKickVelocity = desiredKickVector.norm();
      MyState assumedState1;
      assumedState1.my_pos = ivpInternalMeObject->pos;
      assumedState1.my_vel = ivpInternalMeObject->vel;
      assumedState1.ball_pos = ivpInternalBallObject->pos;
      assumedState1.ball_vel = ivpInternalBallObject->vel;
      assumedState1.my_angle = ivpInternalMeObject->ang;
      assumedState1.op_pos = Vector(0.0,0.0);
      assumedState1.op_bodydir = ANGLE(0);
      assumedState1.op_bodydir_age = 0;
      assumedState1.op = NULL;
      double maxKickVelocity
        = ivpOneStepKickBehavior->get_max_vel_in_dir( assumedState1,
                                                      desiredKickVector.ARG() );
      ivpOneStepKickBehavior->reset_state();
      ivpOneStepKickBehavior->kick_to_pos_with_initial_vel
                              (
                                desiredKickVelocity,
                                currentCrossLinePoint
                              );

      if (
              desiredKickVelocity <= maxKickVelocity
           && ivpOneStepKickBehavior->get_cmd(dummyCmd1))
      {
        ivTargetForNextShot   = currentCrossLinePoint;
        ivVelocityForNextShot = desiredKickVelocity;
        Angle dummyAngle;
        dummyCmd1.cmd_body.
          get_kick( ivKickPowerForNextShot, dummyAngle);
      }
      else
      {
        TGs05sLOGPOL(LOGBASELEVEL+5,<<"Score05_Sequence: CrossLinePoint "<<currentCrossLinePoint<<" [cl="<<i<<",delta="<<currentDelta<<"] is not ok, found no first kick. (red)");
        continue;
      }

      //TGs05sLOGPOL(LOGBASELEVEL+DRAWBASELEVEL,<<_2D<<C2D(currentCrossLinePoint.x,currentCrossLinePoint.y,0.1,"0000ff"));//BLUE
      Vector myPos   = ivpInternalMeObject->pos,   myVel   = ivpInternalMeObject->vel, 
             ballPos = ivpInternalBallObject->pos, ballVel = ivpInternalBallObject->vel;
      Vector myPosAfter1stKick, myVelAfter1stKick, 
             ballPosAfter1stKick, ballVelAfter1stKick;
      ANGLE  myAng = ivpInternalMeObject->ang, myANGAfter1stKick;
      Tools::model_cmd_main( myPos,
                             myVel,
                             myAng,
                             ballPos,
                             ballVel,
                            dummyCmd1.cmd_body,
                           myPosAfter1stKick,
                           myVelAfter1stKick,
                           myANGAfter1stKick,
                           ballPosAfter1stKick,
                           ballVelAfter1stKick);
//TGs05sLOGPOL(LOGBASELEVEL+0,<<"COMPARE: crossLinePt = "<<currentCrossLinePoint<<"  vs.  ballPosAfter1stKick = "<<ballPosAfter1stKick<<" due to cmd "<<dummyCmd1.cmd_main);      
//TGs05sLOGPOL(LOGBASELEVEL+DRAWBASELEVEL,<<_2D<<L2D(currentCrossLinePoint.x,currentCrossLinePoint.y,ballPosAfter1stKick.x,ballPosAfter1stKick.y,"0000ff"));

      desiredKickVector = (ivViaPoints[viaPointIndex][2] - ballPosAfter1stKick);
      desiredKickVelocity = desiredKickVector.norm();

      MyState assumedState2;
      assumedState2.my_pos = myPosAfter1stKick;
      assumedState2.my_vel = myVelAfter1stKick;
      assumedState2.ball_pos = ballPosAfter1stKick;
      assumedState2.ball_vel = ballVelAfter1stKick;
      assumedState2.my_angle = myANGAfter1stKick;
      assumedState2.op_pos = Vector(0.0,0.0);
      assumedState2.op_bodydir = ANGLE(0);
      assumedState2.op_bodydir_age = 0;
      assumedState2.op = NULL;

      maxKickVelocity
        = ivpOneStepKickBehavior->get_max_vel_in_dir( assumedState2,
                                                      desiredKickVector.ARG() );


      ivpOneStepKickBehavior->reset_state();
      ivpOneStepKickBehavior->set_state( myPosAfter1stKick,
                                         myVelAfter1stKick,
                                         myANGAfter1stKick,
                                         ballPosAfter1stKick,
                                         ballVelAfter1stKick,
                                         Vector(0.0, 0.0), //opponent
                                         ANGLE(0),
                                         0,
                                         NULL);
      ivpOneStepKickBehavior->kick_to_pos_with_initial_vel
                              (
                                desiredKickVelocity,
                                ivViaPoints[viaPointIndex][2]
                              );
      
      if (    desiredKickVelocity <= maxKickVelocity
           && ivpOneStepKickBehavior->get_cmd(dummyCmd2))
      {
        double secondKickPower;
        Angle secondKickAngle;
        dummyCmd2.cmd_body.get_kick(secondKickPower, secondKickAngle);
      }
      else
      {
        TGs05sLOGPOL(LOGBASELEVEL+5,<<"Score05_Sequence: CrossLinePoint "<<currentCrossLinePoint<<" [cl="<<i<<",delta="<<currentDelta<<"] is not ok, found no second kick. (blue)");
        continue;
      }
      //TGs05sLOGPOL(LOGBASELEVEL+DRAWBASELEVEL,<<_2D<<C2D(currentCrossLinePoint.x,currentCrossLinePoint.y,0.1,"00ff00"));//GREEN
      Vector myPosAfter2ndKick, myVelAfter2ndKick,
             ballPosAfter2ndKick;
      ANGLE myANGAfter2ndKick;
      Tools::model_cmd_main( myPosAfter1stKick,
                             myVelAfter1stKick,
                             myANGAfter1stKick,
                             ballPosAfter1stKick,
                             ballVelAfter1stKick,
                             dummyCmd2.cmd_body,
                           myPosAfter2ndKick,
                           myVelAfter2ndKick,
                           myANGAfter2ndKick,
                           ballPosAfter2ndKick,
                           ballVelAfter2ndKick);

      double finalKickVel;
      if ( checkIfGoalIsForSureWhenShootingInTCyclesFromViaPoint
              ( 2, viaPointIndex, ballVelAfter2ndKick, finalKickVel ) )
      {
        iFoundATwoStepKick = true;
        TGs05sLOGPOL(LOGBASELEVEL+4,<<"Score05_Sequence: CrossLinePoint "<<currentCrossLinePoint<<" [cl="<<i<<",delta="<<currentDelta<<"] is GREAT.");
        TGs05sLOGPOL(LOGBASELEVEL+4,<<"                  I was successful, found a two step kick that enables me to score. Final kick vel = "<<finalKickVel);        
        TGs05sLOGPOL(LOGBASELEVEL+0,<<_2D<<VC2D(currentCrossLinePoint,0.1,"ffffff"));
        TGs05sLOGPOL(LOGBASELEVEL+0,<<_2D<<VC2D(currentCrossLinePoint,0.15,"ff7f50"));
        TGs05sLOGPOL(LOGBASELEVEL+0,<<_2D<<VC2D(currentCrossLinePoint,0.2,"ff0000"));
      }
      else
      {
        TGs05sLOGPOL(LOGBASELEVEL+5,<<"Score05_Sequence: CrossLinePoint "<<currentCrossLinePoint<<" [cl="<<i<<",delta="<<currentDelta<<"] is not ok, final kick vel = "<<finalKickVel<<". Continue search ... (green)");
      }

      if (iFoundATwoStepKick)
        break;
    }
    if (iFoundATwoStepKick) 
    {
      firstCmd  = dummyCmd1;
      secondCmd = dummyCmd2;
      resultingBallVelocity = ballVelAfter2ndKick;
      return true;
    }
  } 
  return false; 
}

Vector
Score05_Sequence::getMyPositionInTCyclesAssumingNoDash(int t)
{
  Vector returnValue = ivpInternalMeObject->pos;
  Vector myVelocity  = ivpInternalMeObject->vel;
  for (int i=0; i<t; i++)
  {
    returnValue += myVelocity;
    myVelocity *= WSinfo::me->decay;
  }
  return returnValue;
}

Vector
Score05_Sequence::getMyVelocityInTCyclesAssumingNoDash(int t)
{
  Vector returnValue = ivpInternalMeObject->vel;
  for (int i=0; i<t; i++)
  {
    returnValue *= WSinfo::me->decay;
  }
  return returnValue;
}

void
Score05_Sequence::getPotentialGoaliePositions( int maxSearchDepth,
                                               int currentSearchDepth,
                                               Vector * goaliePositionsArray )
{
  TGs05sLOGPOL(3,<<"Score05_Sequence: getPotentialGoaliePositions: Now search for currentSearchDepth="<<currentSearchDepth);
  if (currentSearchDepth >= maxSearchDepth) return; //end of recursion
  
  ANGLE curViewDirection, curViewWidth;
  Vector curPlayerPosition;
  long dummyTime;
  if ( ! WSmemory::get_view_info_before_n_steps( currentSearchDepth,
                                                 curViewWidth,
                                                 curViewDirection,
                                                 curPlayerPosition,
                                                 dummyTime ) )
    return;
  ANGLE halfCurViewWidth( curViewWidth.get_value_0_p2PI() * 0.5 );
  //find successors
  int maxGoaliePositions = (int) pow( 2.0, (double)maxSearchDepth );
  Vector *successorGoaliePositionsArray = new Vector [maxGoaliePositions];
  int successorCnt = 0;
  for (int i=0; i<maxGoaliePositions; i++)
  {
    if ( goaliePositionsArray[i].getX() < -999.9 ) continue;
    if ( isPointInOneOfRecentViewCones( goaliePositionsArray[i], 
                                        currentSearchDepth, 
                                        currentSearchDepth ) == false )
    {
      TGs05sLOGPOL(LOGBASELEVEL+6,<<"Score05_Sequence: getPotentialGoaliePositions:  i="
        <<i<<" point goaliePositionsArray["<<i<<"]="
       <<goaliePositionsArray[i]
       <<" is NOT in view cone for t-currentSearchDepth (t-"<<currentSearchDepth<<")");
      successorGoaliePositionsArray[successorCnt] = goaliePositionsArray[i];
      successorCnt ++;
      continue;
    }
    //determine guessed position left and right of view cone
    Vector endOfViewCone_L( curViewDirection + halfCurViewWidth ),
           endOfViewCone_R( curViewDirection - halfCurViewWidth );
    endOfViewCone_L += curPlayerPosition;
    endOfViewCone_R += curPlayerPosition;
    //determine guessed position LEFT of view cone
    Vector nextGuessedGoaliePos_L 
      = Tools::get_Lotfuss( curPlayerPosition,
                            endOfViewCone_L,
                            goaliePositionsArray[i] );
    Vector goalieShift_L = nextGuessedGoaliePos_L - goaliePositionsArray[i];
    goalieShift_L.normalize( goalieShift_L.norm() + 0.3 );
    nextGuessedGoaliePos_L = goaliePositionsArray[i] + goalieShift_L;
    if (   nextGuessedGoaliePos_L.distance(ivpInternalMeObject->pos) 
         < ServerOptions::visible_distance )
    {
      goalieShift_L = nextGuessedGoaliePos_L - ivpInternalMeObject->pos;
      goalieShift_L.normalize( ServerOptions::visible_distance + 0.5 );
      nextGuessedGoaliePos_L = ivpInternalMeObject->pos + goalieShift_L;
    }
    if ( isPointInOneOfRecentViewCones( nextGuessedGoaliePos_L,
                                        0,
                                        currentSearchDepth ) == false )
    {
      TGs05sLOGPOL(LOGBASELEVEL+6,<<"Score05_Sequence: getPotentialGoaliePositions: i="
        <<i<<" point nextGuessedGoaliePos_L="
        <<nextGuessedGoaliePos_L
        <<" is NOT in one of the recent view cones (from 0 to "<<currentSearchDepth
        <<").");
      successorGoaliePositionsArray[successorCnt] = nextGuessedGoaliePos_L;
      successorCnt ++;
    }
    else
    {
      TGs05sLOGPOL(LOGBASELEVEL+6,<<"Score05_Sequence: getPotentialGoaliePositions:  i="
        <<i<<" point goaliePositionsArray["<<i<<"]="
        <<nextGuessedGoaliePos_L<<" was in one of the recent view cones.");
    }
    //determine guessed position RIGHT of view cone
    Vector nextGuessedGoaliePos_R 
      = Tools::get_Lotfuss( curPlayerPosition,
                            endOfViewCone_R,
                            goaliePositionsArray[i] );
    Vector goalieShift_R = nextGuessedGoaliePos_R - goaliePositionsArray[i];
    goalieShift_R.normalize( goalieShift_R.norm() + 0.3 );
    nextGuessedGoaliePos_R = goaliePositionsArray[i] + goalieShift_R;
    if (   nextGuessedGoaliePos_R.distance(ivpInternalMeObject->pos) 
         < ServerOptions::visible_distance )
    {
      goalieShift_R = nextGuessedGoaliePos_R - ivpInternalMeObject->pos;
      goalieShift_R.normalize( ServerOptions::visible_distance + 0.5 );
      nextGuessedGoaliePos_R = ivpInternalMeObject->pos + goalieShift_R;
    }
    if ( isPointInOneOfRecentViewCones( nextGuessedGoaliePos_R,
                                        0,
                                        currentSearchDepth ) == false )
    {
      TGs05sLOGPOL(LOGBASELEVEL+6,<<"Score05_Sequence: getPotentialGoaliePositions:  i="
        <<i<<" point nextGuessedGoaliePos_R="
        <<nextGuessedGoaliePos_R
        <<" is NOT in one of the recent view cones (from 0 to "<<currentSearchDepth
        <<").");
      successorGoaliePositionsArray[successorCnt] = nextGuessedGoaliePos_R;
      successorCnt ++;
    }
    else
    {
      TGs05sLOGPOL(LOGBASELEVEL+6,<<"Score05_Sequence: getPotentialGoaliePositions:  i="
        <<i<<" point goaliePositionsArray["<<i<<"]="
        <<nextGuessedGoaliePos_R<<" was in one of the recent view cones.");
    }
  }
  for (int i=0; i<maxGoaliePositions; i++)
  {
    if ( i < successorCnt )
      goaliePositionsArray[i] = successorGoaliePositionsArray[i];
    else
    {
      goaliePositionsArray[i].setXY( -1000.0, -1000.0 );
    }
  }
  delete [] successorGoaliePositionsArray;
  successorGoaliePositionsArray = NULL;
  // ...,,,:::|||''' recursive call '''|||:::,,,...
  getPotentialGoaliePositions( maxSearchDepth,
                               currentSearchDepth + 1,
                               goaliePositionsArray );
}

bool
Score05_Sequence::isPointInOneOfRecentViewCones( Vector p, int fromT, int toT )
{
  if ( fromT >= MAX_VIEW_INFO || toT >= MAX_VIEW_INFO) return false;
  ANGLE curViewDirection, curViewWidth;
  Vector curPlayerPosition;
  long dummyTime;
  for (int i=fromT; i<=toT; i++)
  {
    if ( ! WSmemory::get_view_info_before_n_steps( i,
                                                   curViewWidth,
                                                   curViewDirection,
                                                   curPlayerPosition,
                                                   dummyTime ) )
      continue;
    ANGLE halfCurViewWidth( curViewWidth.get_value_0_p2PI() * 0.5 );
    Cone2d curViewCone( curPlayerPosition, 
                        curViewDirection - halfCurViewWidth,
                        curViewDirection + halfCurViewWidth );
    if ( curViewCone.inside( p ) )
      return true;
  }
  return false;
}


bool
Score05_Sequence::performFinalKick()
{
  ivMode = MODE_FINAL_KICK;

  ANGLE targetDirection;
  Vector targetPosition;
  bool shallIKick
    = checkIfGoalIsForSureWhenShootingNow( targetDirection, targetPosition );
  bool shallITackleForward    = false,
       shallITackleBackward   = false,
       shallITackleV12Angular = false;
  int bestAngularTackleDirection;
  ivDesiredTackling        = TACKLE_NONE;
  ivDesiredAngularTackling = cvcNO_V12_SCORING_TACKLE_POSSIBLE;
  if (ClientOptions::server_version >= 12.0) //new tackling
  {
    shallITackleV12Angular
      = checkIfGoalIsForSureWhenV12TacklingNow( bestAngularTackleDirection, 
                                                true ); //debug
    if ( shallITackleV12Angular )
    {
      TGs05sLOGPOL(LOGBASELEVEL,"Score05_Sequence: performFinalKick(): "
        <<"Amazing! Angular tackling the ball into the goal is possible ("
        <<bestAngularTackleDirection<<").");
    }
  }
  else //old tackling
  {    
    shallITackleForward
      = checkIfGoalIsForSureWhenTacklingNow( TACKLE_FORWARD );
    shallITackleBackward
      = checkIfGoalIsForSureWhenTacklingNow( TACKLE_BACKWARD );
  }
  if (WSinfo::his_goalie && WSinfo::his_goalie->age > 0)
  {
    TGs05sLOGPOL(LOGBASELEVEL+1,<<"Score05_Sequence: performFinalKick: WARNING: Goalie is too outdated for final kick (age="<<WSinfo::his_goalie->age<<")!");
    // /*TG_OSAKA*/  We do not exlcue this case generally, instead we add a time
    // offset in checkIfGoalIsForSureWhenShootingNow which is WSinfo::his_goalie->age
    //shallIKick = false;
  }
  double einschussbreite = 0.0;
  double myDistToLinkerPfosten
          = ivpInternalBallObject->pos.distance(HIS_GOAL_LEFT_CORNER),
        myDistToRechterPfosten  
          = ivpInternalBallObject->pos.distance(HIS_GOAL_RIGHT_CORNER),
        myMinDistToPfosten
          = Tools::min( myDistToRechterPfosten, myDistToLinkerPfosten);
  Vector pfostenReferencePoint, secondReferencePoint, delta;
  if (myDistToLinkerPfosten < myDistToRechterPfosten)
  {
    pfostenReferencePoint = HIS_GOAL_LEFT_CORNER;
    delta = HIS_GOAL_RIGHT_CORNER - ivpInternalBallObject->pos;
  }
  else
  {
    pfostenReferencePoint = HIS_GOAL_RIGHT_CORNER;
    delta = (HIS_GOAL_LEFT_CORNER - ivpInternalBallObject->pos);
  }
  delta.normalize(myMinDistToPfosten);
  secondReferencePoint = ivpInternalBallObject->pos + delta;
  einschussbreite = pfostenReferencePoint.distance(secondReferencePoint);
   
  if ( einschussbreite < 0.1 * myMinDistToPfosten )
  {
    TGs05sLOGPOL(LOGBASELEVEL+1,<<"Score05_Sequence: performFinalKick: "
      <<"DANGER: My shoot angle is too low!");
    shallIKick = false;
    shallITackleBackward = false; shallITackleForward = false;
    shallITackleV12Angular = false;
  }
  if (   !shallIKick  
      && !shallITackleForward && !shallITackleBackward 
      && !shallITackleV12Angular)
  {
    TGs05sLOGPOL(LOGBASELEVEL+DRAWBASELEVEL,<<_2D
      <<VL2D(ivpInternalBallObject->pos,
            targetPosition, "aa0000"));
    TGs05sLOGPOL(LOGBASELEVEL+1,<<"Score05_Sequence: It is not safe to make the final kick! STOP IT (goalie or opponent may intercept)!");
    ivMode = MODE_FIRST_PLANNING;
    return false;
  }

  Vector bal=ivpInternalBallObject->pos, balv;
  if ( !shallIKick )
  {
    double tackleSuccessProbability
      = Tools::get_tackle_success_probability( ivpInternalMeObject->pos,
                                               ivpInternalBallObject->pos,
                                               ivpInternalMeObject->ang.get_value() );
    double requiredMinimalTackleSuccessProbability = 0.98;
    TGs05sLOGPOL(LOGBASELEVEL+1,"Score05_Sequence: I cannot make a final kick"
      <<", but a successful tackle would score! Tackling is successful with prob="
      <<tackleSuccessProbability<<" (necessary thresh = "
      <<requiredMinimalTackleSuccessProbability<<")");
    if (ClientOptions::server_version >= 12.0) //new tackling, angular
    {
      if (   shallITackleV12Angular
          && tackleSuccessProbability >= requiredMinimalTackleSuccessProbability)
      {
        ivDesiredAngularTackling = bestAngularTackleDirection;
      }
      else 
      {
        ivDesiredTackling = TACKLE_NONE;
        ivDesiredAngularTackling = cvcNO_V12_SCORING_TACKLE_POSSIBLE;
        return false; //shall i kick is false, and tacklings are too unlikely
      }
      //debug output
      Vector ballNewPos, dummyBallVelAfterTackling;
      Tools::model_tackle_V12( ivpInternalMeObject->pos,
                               ivpInternalMeObject->ang,
                               ivpInternalBallObject->pos,
                               ivpInternalBallObject->vel,
                               bestAngularTackleDirection,
                               ballNewPos,
                               dummyBallVelAfterTackling);
      Vector tackleVelocity = ballNewPos - ivpInternalBallObject->pos;
      balv = tackleVelocity; //should be less than max ball speed
      tackleVelocity.normalize(20.0);
      TGs05sLOGPOL(LOGBASELEVEL+DRAWBASELEVEL,<<_2D
        <<VL2D(ivpInternalBallObject->pos,
        (ivpInternalBallObject->pos+tackleVelocity),
        "0000dd"));
    }
    else //old tackling (power value-based, within -100..100)
    {
      Vector tackleVelocity( ivpInternalMeObject->ang );
      tackleVelocity.normalize( ServerOptions::ball_speed_max );
      if (    shallITackleBackward 
           && tackleSuccessProbability >= requiredMinimalTackleSuccessProbability) 
      {
        ivDesiredTackling = TACKLE_BACKWARD;
        tackleVelocity = (-1.0) * tackleVelocity;
      }
      else 
      if (    shallITackleForward
           && tackleSuccessProbability >= requiredMinimalTackleSuccessProbability )  
      {
        ivDesiredTackling = TACKLE_FORWARD;
      }
      else 
      {
        ivDesiredTackling = TACKLE_NONE;
        ivDesiredAngularTackling = cvcNO_V12_SCORING_TACKLE_POSSIBLE;
        return false; //shall i kick is false, and tacklings are too unlikely
      }
      balv = ivpInternalBallObject->vel + tackleVelocity;
      balv.normalize( ServerOptions::ball_speed_max );
      tackleVelocity.normalize(20.0);
      TGs05sLOGPOL(LOGBASELEVEL+DRAWBASELEVEL,<<_2D
        <<VL2D(ivpInternalBallObject->pos,
        (ivpInternalBallObject->pos+tackleVelocity),
        "0000dd"));
    }
  }
  else
  {
    MyState assumedState;
    assumedState.my_pos = ivpInternalMeObject->pos;
    assumedState.my_vel = ivpInternalMeObject->vel;
    assumedState.ball_pos = ivpInternalBallObject->pos;
    assumedState.ball_vel = ivpInternalBallObject->vel;
    assumedState.my_angle = ivpInternalMeObject->ang;
    assumedState.op_pos = Vector(0.0,0.0);
    assumedState.op_bodydir = ANGLE(0);
    assumedState.op_bodydir_age = 0;
    assumedState.op = NULL;
    double maxFinalKickVelocity
      = ivpOneStepKickBehavior->get_max_vel_in_dir( assumedState,
                                                    targetDirection );
  
    ivVelocityForNextShot = maxFinalKickVelocity;
    ivKickPowerForNextShot = 100.0;
    ivTargetForNextShot = targetPosition;
    TGs05sLOGPOL(LOGBASELEVEL+DRAWBASELEVEL,<<_2D<<VL2D(ivpInternalBallObject->pos,
      ivTargetForNextShot,"ff0000"));
  
    balv.init_polar(maxFinalKickVelocity, targetDirection.get_value_mPI_pPI());
  }

  //debug output
  for (int i=0; i<7; i++)
  {
    TGs05sLOGPOL(LOGBASELEVEL,<<_2D<<VC2D(bal,0.2,"ff0000"));
    bal+=balv;
    balv*=ServerOptions::ball_decay;
    TGs05sLOGPOL(LOGBASELEVEL,<<_2D<<VC2D(
      ivInternalHisGoaliePosition,
      ivGOALIE_CATCH_RADIUS+goalie_action_radius_at_time(i,GOALIE_INITIAL_SIZE,0),"ff7777"));
  }
  return true;
}

void
Score05_Sequence::setAssumedGoaliCatchRadius()
{
  if ( WSinfo::ws->play_mode == PM_my_PenaltyKick )
    ivGOALIE_CATCH_RADIUS = GOALIE_CATCH_RADIUS_PENALTY;
  else 
    ivGOALIE_CATCH_RADIUS = GOALIE_CATCH_RADIUS;
}

bool 
Score05_Sequence::test_shoot2goal( Intention & intention, 
                                   Cmd       * currentCommandPointer,
                                   bool        allowRecursiveness )
{
  setAssumedGoaliCatchRadius();
  TGs05sLOGPOL(0,<<"Score05_Sequence: CALL TO TEST_SHOOT2GOAL! ####### "
    <<WSinfo::ball->vel<<" ##################################");
#if LOGGING && BASIC_LOGGING
  long t1 = Tools::get_current_ms_time();
#endif
  
  this->updateInternalObjects( currentCommandPointer );
  if (   allowRecursiveness == true
      && ivLastTimeTestShoot2GoalHasBeenInvoked == WSinfo::ws->time)
  {
    TGs05sLOGPOL(LOGBASELEVEL,"Score05_Sequence: I HAVE ALREADY BEEN "
      <<"CALLED THIS CYCLE, test_shoot2goal returns the same value that"
      <<" has already been returned during the previous call => "
      <<ivLastTestShoot2GoalResult);
    intention.copyFromIntention( ivIntentionForThisCycle );
    return ivLastTestShoot2GoalResult;
  }
  if (ivpInternalMeObject == NULL || ivpInternalBallObject == NULL)
  {
    TGs05sLOGPOL(LOGBASELEVEL,"Score05_Sequence: Ball and Me object are not "
      <<"available, test_shoot2goal returns false.");
    if (allowRecursiveness) ivLastTestShoot2GoalResult = false;
    return false;
  }
  if (allowRecursiveness) 
    ivLastTimeTestShoot2GoalHasBeenInvoked = WSinfo::ws->time;
  
  PlayerSet opponentsNearTheirGoal = WSinfo::alive_opponents;
  opponentsNearTheirGoal.keep_players_in_circle( HIS_GOAL_CENTER, 18.0 );
  bool thereIsAnOpponentPlayerNearHisGoalCenter
    = (opponentsNearTheirGoal.num > 0);
  bool distanceShootDueToTooOffensiveGoalieMayBePossible
    =    WSinfo::his_goalie
      && WSinfo::his_goalie->age < 5
      && (    RIGHT_PENALTY_AREA.inside(WSinfo::his_goalie->pos) == false
           || HIS_GOAL_CENTER.distance(WSinfo::his_goalie->pos) > 16.0 )
      && ivpInternalMeObject->pos.sqr_distance(HIS_GOAL_CENTER) < 35*35
      && thereIsAnOpponentPlayerNearHisGoalCenter == false;    
  if (   ivpInternalMeObject->pos.sqr_distance(HIS_GOAL_CENTER) > 25*25
      && distanceShootDueToTooOffensiveGoalieMayBePossible == false )
  {
#if LOGGING && BASIC_LOGGING
    long t2 = Tools::get_current_ms_time();
#endif
    TGs05sLOGPOL(LOGBASELEVEL+3,<<"Wball03: [TIME] Daisy Chain test 'Score05_Sequence::test_shoot2goal' required "<<t2-t1<<"ms of time.");
    if (allowRecursiveness) ivLastTestShoot2GoalResult = false;
    return false;
  }
  //initialization
  if (WSinfo::his_goalie)
  {
    TGs05sLOGPOL(LOGBASELEVEL+DRAWBASELEVEL,<<_2D<<VC2D(WSinfo::his_goalie->pos, 0.4, "ff6666"));
    TGs05sLOGPOL(LOGBASELEVEL+DRAWBASELEVEL,<<_2D<<VSTRING2D(WSinfo::his_goalie->pos, WSinfo::his_goalie->age, "ff6666"));
    //check if a teammate has performed a goal shot and the ball has mistakenly
    //got into my kick range
    if (    WSinfo::ball->age == 0 && WSinfo::ball->age_vel == 0
         && WSinfo::ball->vel.norm() > 1.5 && WSinfo::ball->vel.getX() > 0.0
         && WSinfo::his_goalie->age <= 7//TG08 ZUI originalwert 5
         && WSmemory::ball_was_kickable4me_at < WSinfo::ws->time - 1
       )
    {
      Vector ballPos = WSinfo::ball->pos, ballVel = WSinfo::ball->vel;
      bool ballCorssesGoalLine = false;
      for (int i=0; i<10; i++)
      {
        ballPos += ballVel;
        ballVel *= ServerOptions::ball_decay;
        if (ballPos.getX() > FIELD_BORDER_X)
          continue;
        if (   ballPos.getX() > FIELD_BORDER_X
            && fabs(ballPos.getY()) < 6.9 - (((double)i)*0.05) )
          ballCorssesGoalLine = true;
      }
      if (ballCorssesGoalLine == true)
      {
        int goalieInterceptResult
          = intercept_goalie(   WSinfo::ball->pos, 
                                WSinfo::ball->vel, 
                                WSinfo::his_goalie->pos, 
                                GOALIE_INITIAL_SIZE,
                                WSinfo::his_goalie->age, 
                                false );
        int opponentInterceptResult
          = intercept_opponents ( WSinfo::ball->vel.ARG().get_value_mPI_pPI(), //ball's direction
                                  WSinfo::ball->vel.norm(), //ball's speed
                                  12, //note: value is not used!
                                  WSinfo::ball->vel,
                                  0); 
        if (opponentInterceptResult > 0)
        {
          TGs05sLOGPOL(0,<<"Score05_Sequence: I checked for letting the "
           <<"ball flutsch through me, but OPPONENT intercepts. res="
           <<opponentInterceptResult<<",ballVel="<<WSinfo::ball->vel<<")");
        }
        if (goalieInterceptResult > 0)
        {
          TGs05sLOGPOL(0,<<"Score05_Sequence: I checked for letting the "
            <<"ball flutsch through me, but GOALIE intercepts, time="
            <<goalieInterceptResult<<",ballVel="<<WSinfo::ball->vel<<")");
        }
        if ( opponentInterceptResult <= 0 && goalieInterceptResult <= 0 )
        {
          //do nothing
          TGs05sLOGPOL(0,<<"Score05_Sequence: I checked for letting the "
            <<"ball flutsch through me: *** YEP *** GOALIE and OPPONENT"
            <<" won't intercept.");
          if (allowRecursiveness)
          {
            ivLastTestShoot2GoalResult = true;
            ivIntentionForThisCycle.copyFromIntention( intention );
          }
          return true;
        }
      }
    }
    //check if goalie is very outdated
    if (   WSinfo::his_goalie->age > 4//TG08 ZUI: originalwert 2
        && ! Tools::could_see_in_direction
                    ( (WSinfo::his_goalie->pos - ivpInternalMeObject->pos).ARG() ))
    {
#if LOGGING && BASIC_LOGGING
      long t2 = Tools::get_current_ms_time();
#endif
      TGs05sLOGPOL(LOGBASELEVEL+3,<<"Wball03: [TIME] Daisy Chain test 'Score05_Sequence::test_shoot2goal' required "<<t2-t1<<"ms of time.");
      if (allowRecursiveness) ivLastTestShoot2GoalResult = false;
      return false;
    }
  }
  TGs05sLOGPOL(LOGBASELEVEL+DRAWBASELEVEL,<<_2D<<VC2D(ivpInternalBallObject->pos,0.2,"ff8c00"));
  TGs05sLOGPOL(LOGBASELEVEL+DRAWBASELEVEL,<<_2D<<VC2D(ivpInternalBallObject->pos,0.3,"ff8c00"));
  TGs05sLOGPOL(LOGBASELEVEL+DRAWBASELEVEL,<<_2D<<VC2D(ivpInternalBallObject->pos+ivpInternalBallObject->vel,0.3,"ff8c00"));

  //this is the first thing to do: check for teammates also controlling the ball 
  if (allowRecursiveness == false)
  {
    TGs05sLOGPOL(0,"Score05_Sequence: Attention, I am doing "
      <<"recursive considereations for teammate "<<ivpInternalMeObject->number<<"!");
  }
  if ( allowRecursiveness )
  {
    PlayerSet teammates = WSinfo::valid_teammates_without_me;
    teammates.keep_players_in_circle( ivpInternalBallObject->pos, 1.5);
    for (int i=0; i<teammates.num; i++)
    {
      if ( WSinfo::is_ball_kickable_for( teammates[i] ) )
      {
        ivInternalMeObject = * teammates[i];
        Intention dummyIntention;
        if (    test_shoot2goal( dummyIntention, NULL, false ) 
             && WSinfo::me->pos.getX() < teammates[i]->pos.getX() )
        {
          TGs05sLOGPOL(LOGBASELEVEL+1,<<"Score05_Sequence: STOP! STOP! A teammate"
            <<" with higher x pos will also score by sequence, I must retreat!");
          ivInternalMeObject    = * WSinfo::me; //don't forget to reset
          intention.set_waitandsee(WSinfo::ws->time);
          ivLastTestShoot2GoalResult = true;
          ivIntentionForThisCycle.copyFromIntention( intention );
          return true;
        }
        ivInternalMeObject    = * WSinfo::me; //don't forget to reset
      }
    }
  }

  updateViaPoints( );

  bool finalKick = false;
  
  if ( ivMode == MODE_TO_VIA_POINT)
  {
    TGs05sLOGPOL(LOGBASELEVEL+1,<<"Score05_Sequence: HEY, the ball is already at the VIA POINT. I make the final kick now!");
  }  
  else
  {
    TGs05sLOGPOL(LOGBASELEVEL+1,<<"Score05_Sequence: Ok, let's check whether we may score immediately (actually the old scoring behavior should do that)!");
    TGs05sLOGPOL(LOGBASELEVEL+1,<<"                  Therefore, in case no goal could be obtained at this point of time, please report to TG.");
  }  
  
  //Will our final scoring kick be successful?
  finalKick = performFinalKick(); 
  TGs05sLOGPOL(LOGBASELEVEL,<<"Score05_Sequence: performFinalKick: "<<finalKick);
  if ( ivBallWithHighProbabilityInKickrangeOfAnOpponentPlayer == true )
  {
    TGs05sLOGPOL(0,<<"Score05_Sequence: Ball in kick range of me and an"
      <<" opponent player. GIVING UP!");
    ivMode = MODE_FIRST_PLANNING;
    return false;
  }
  
  if (!finalKick)
  {
    //Ok, the final kick won't be successful, so we try to follow the 
    //'normal' skill behavior.
    this->get_cmd( ivCommandForThisCycle );
  }
  
  TGs05sLOGPOL(LOGBASELEVEL+1,<<"Score05_Sequence: DECISION MADE, MODE = "<<ivMode);
  this->checkForNearPlayersRegardingCurrentMode();  

  if (ivMode != MODE_FIRST_PLANNING)
  {
    
    //action considerations
    if (   (   ClientOptions::server_version < 12.0 
            && ivDesiredTackling != TACKLE_NONE )
        || (   ClientOptions::server_version >= 12.0
            && ivDesiredAngularTackling != cvcNO_V12_SCORING_TACKLE_POSSIBLE )
       )
    {
      if (ClientOptions::server_version >= 12.0)
      {
        TGs05sLOGPOL(LOGBASELEVEL+1,<<"Score05_Sequence: Setting the intention "
          <<"(handed over by wball): SS05-ANGULAR-TACKLING ("
          <<ivDesiredAngularTackling<<")");
        int tackleParameter 
          = (ivDesiredAngularTackling > 180) ? ivDesiredAngularTackling-360
                                             : ivDesiredAngularTackling; 
        intention.set_tackling( (double)-tackleParameter,
                                WSinfo::ws->time);
      }
      else
      {
        TGs05sLOGPOL(LOGBASELEVEL+1,<<"Score05_Sequence: Setting the intention (handed over by wball): SS05-TACKLING ("<<ivDesiredTackling<<")");
        intention.set_tackling( (double)ivDesiredTackling,
                                WSinfo::ws->time);
      }
    }
    else
    {
      TGs05sLOGPOL(LOGBASELEVEL+1,<<"Score05_Sequence: Setting the intention (handed over by wball): ivTargetForNextShot="<<ivTargetForNextShot<<" ivKickPowerForNextShot="<<ivKickPowerForNextShot);
      intention.set_score( ivTargetForNextShot, 
                           ivVelocityForNextShot,
                           WSinfo::ws->time);
      intention.immediatePass = true;
    }
    //look considerations
    ANGLE   hisGoalCenterAngle
              = (HIS_GOAL_CENTER - ivpInternalMeObject->pos).ARG(),
            hisGoalLeftPostAngle
              = (HIS_GOAL_LEFT_CORNER - ivpInternalMeObject->pos).ARG(),
            hisGoalRightPostAngle
              = (HIS_GOAL_RIGHT_CORNER - ivpInternalMeObject->pos).ARG(),
            deltaAngle,
            lookAngle;
    if (WSinfo::ws->time%4 == 1) deltaAngle.set_value(PI*(15.0/180.0));
    if (WSinfo::ws->time%4 == 2) deltaAngle.set_value(PI*(-15.0/180.0));
    if (WSinfo::his_goalie == NULL)
    {
      //OLD: mdpInfo::set_my_intention(DECISION_TYPE_INTERCEPTBALL);//look to ball
      if (   Tools::could_see_in_direction(hisGoalCenterAngle)
          && WSmemory::last_seen_in_dir(hisGoalCenterAngle) > 2)
        lookAngle = hisGoalCenterAngle;
      else
      if (   Tools::could_see_in_direction(hisGoalLeftPostAngle)
          && WSmemory::last_seen_in_dir(hisGoalLeftPostAngle) > 2)
        lookAngle = hisGoalLeftPostAngle;
      else
      if (   Tools::could_see_in_direction(hisGoalRightPostAngle)
          && WSmemory::last_seen_in_dir(hisGoalRightPostAngle) > 2)
        lookAngle = hisGoalRightPostAngle;
      if (Tools::could_see_in_direction(hisGoalCenterAngle))
        lookAngle = hisGoalCenterAngle;
      else
      if (Tools::could_see_in_direction(hisGoalLeftPostAngle))
        lookAngle = hisGoalLeftPostAngle;
      else
      if (Tools::could_see_in_direction(hisGoalRightPostAngle))
        lookAngle = hisGoalRightPostAngle;
        
      TGs05sLOGPOL(LOGBASELEVEL+1,<<"Score05_Sequence: Setting the neck intention (via Tools::set_neck_request): "<<lookAngle.get_value_mPI_pPI());
      Tools::set_neck_request(NECK_REQ_LOOKINDIRECTION, lookAngle, true );
    }
    else
    {
      lookAngle
        = determineGoodLookDirectionIfGoalieIsPresent();
        
      TGs05sLOGPOL(LOGBASELEVEL+1,<<"Score05_Sequence: Setting the neck intention (via Tools::set_neck_request): "<<lookAngle.get_value_mPI_pPI());
//      Tools::set_neck_request(NECK_REQ_DIRECTOPPONENTDEFENSE, lookAngle );
      Tools::set_neck_request(NECK_REQ_LOOKINDIRECTION, lookAngle, true );

    }
#if LOGGING && BASIC_LOGGING
    long t2 = Tools::get_current_ms_time();
#endif
    TGs05sLOGPOL(LOGBASELEVEL+3,<<"Wball03: [TIME] Daisy Chain test 'Score05_Sequence::test_shoot2goal' required "<<t2-t1<<"ms of time.");
    if (allowRecursiveness)
    {
      ivLastTestShoot2GoalResult = true;
      ivIntentionForThisCycle.copyFromIntention( intention );
    }
    return true;
  }
  else
  {
    TGs05sLOGPOL(LOGBASELEVEL+1,<<"Score05_Sequence: I AM SO SORRY, BUT I HAVE TO ADMIT THAT I DID NOT FIND ANY KICK SEQUENCE.");
#if LOGGING && BASIC_LOGGING
    long t2 = Tools::get_current_ms_time();
#endif
    TGs05sLOGPOL(LOGBASELEVEL+3,<<"Wball03: [TIME] Daisy Chain test 'Score05_Sequence::test_shoot2goal' required "<<t2-t1<<"ms of time.");
    if (allowRecursiveness) ivLastTestShoot2GoalResult = false;
    return false;
  }
}

//============================================================================
// determineGoodLookDirectionIfGoalieIsPresent
//============================================================================
//============================================================================
// test_tackle2goal
//============================================================================
int
Score05_Sequence::test_tackle2goal( Vector myPos, Vector myVel, ANGLE myAng,
                                    Vector ballPos, Vector ballVel, bool debug )
{
  //initialize relevant variables
  TGs05sLOGPOL(0,<<"Score05_Sequence: CALL TO TEST_TACKLE2GOAL!");
  this->updateInternalObjects( NULL );
  setAssumedGoaliCatchRadius();

  ivpInternalMeObject->pos   = myPos;
  ivpInternalMeObject->vel   = myVel;
  ivpInternalMeObject->ang   = myAng;
  ivpInternalBallObject->pos = ballPos;
  ivpInternalBallObject->vel = ballVel;

  //angular tackling
  if (ClientOptions::server_version >= 12.0)
  {
    int bestTackDir;
    if ( checkIfGoalIsForSureWhenV12TacklingNow( bestTackDir, debug ) )
    {
      TGs05sLOGPOL(LOGBASELEVEL,"Score05_Sequence: Amazing! Angular tackling "
        <<"the ball into the goal is possible ("<<bestTackDir<<").");
      return bestTackDir;
    }
    return cvcNO_V12_SCORING_TACKLE_POSSIBLE; //TG: TODO: Deactivated for German Open 2008.
  }

  //standard behavior for up to version 11 of the Soccer Server
  int returnValue = TACKLE_NONE;
  if ( checkIfGoalIsForSureWhenTacklingNow( TACKLE_FORWARD, debug ) )
    returnValue = TACKLE_FORWARD;
  else
  if ( checkIfGoalIsForSureWhenTacklingNow( TACKLE_BACKWARD, debug ) )
    returnValue = TACKLE_BACKWARD;
  if (returnValue != TACKLE_NONE) {
    TGs05sLOGPOL(LOGBASELEVEL,"Score05_Sequence: Amazing! Tackling the ball into the goal is possible ("<<returnValue<<").");
  }
  return returnValue;
}

//============================================================================
// get_cmd
//============================================================================
//============================================================================
// checkKickSequencesViaViaPoints
//============================================================================
void
Score05_Sequence::updateInternalObjects( Cmd * currentCmdPointer )
{
  if (ivpInternalBallObject == NULL || ivpInternalMeObject == NULL)
  {
    if (WSinfo::ball == NULL || WSinfo::me == NULL)
    {
      TGs05sLOGPOL(LOGBASELEVEL+3,"Score05_Sequence: WARNING, WSinfo::ball/me "
        <<"not available (null pointer).");
      return;
    }
    ivInternalBallObject  = * WSinfo::ball;
    ivInternalMeObject    = * WSinfo::me;
    ivpInternalBallObject = & ivInternalBallObject;
    ivpInternalMeObject   = & ivInternalMeObject;
  }
  
  if (currentCmdPointer == NULL)
  {
    TGs05sLOGPOL(LOGBASELEVEL+3,"Score05_Sequence: I assume ball/player "
      <<"positions/velocities as provided by WSinfo (pos="
      <<WSinfo::ball->pos<<",vel="<<WSinfo::ball->vel<<",age="
      <<WSinfo::ball->age<<",age_vel="<<WSinfo::ball->age_vel<<").");
    ivInternalBallObject  = * WSinfo::ball;
    ivInternalMeObject    = * WSinfo::me;
  }
  else
  {
    TGs05sLOGPOL(LOGBASELEVEL+3,"Score05_Sequence: I assume ball/player "
      <<"positions/velocities corresponding to the VIRTUAL state "
      <<"(visualised in light red!) that "
      <<"results from executing the given command (cmd).");
    Vector myPosAfterGivenCmd, myVelAfterGivenCmd, 
           ballPosAfterGivenCmd, ballVelAfterGivenCmd;
    ANGLE  myANGAfterGivenCmd;
    Tools::model_cmd_main( WSinfo::me->pos,
                           WSinfo::me->vel,
                           WSinfo::me->ang,
                           WSinfo::ball->pos,
                           WSinfo::ball->vel,
                           currentCmdPointer->cmd_body,
                           ivpInternalMeObject->pos,
                           ivpInternalMeObject->vel,
                           ivpInternalMeObject->ang,
                           ivpInternalBallObject->pos,
                           ivpInternalBallObject->vel);
    TGs05sLOGPOL(LOGBASELEVEL+DRAWBASELEVEL,<<_2D<<VC2D(ivpInternalMeObject->pos,ivpInternalMeObject->kick_radius,"ff9999"));
    Vector look; 
    look.init_polar(ivpInternalMeObject->kick_radius, ivpInternalMeObject->ang);
    look = look + ivpInternalMeObject->pos;
    TGs05sLOGPOL(LOGBASELEVEL+DRAWBASELEVEL,<<_2D<<VL2D(ivpInternalMeObject->pos,look,"ff9999"));
    TGs05sLOGPOL(LOGBASELEVEL+DRAWBASELEVEL,<<_2D<<VC2D(ivpInternalBallObject->pos,0.1,"ff9999"));
    TGs05sLOGPOL(LOGBASELEVEL+DRAWBASELEVEL,<<_2D<<VC2D(ivpInternalBallObject->pos,0.2,"ff9999"));
  }
  //update internal position and age of his goalie
  this->determineGoaliePositionAndAge( ivInternalHisGoaliePosition,
                                       ivInternalHisGoalieAge );
  PlayerSet oppsNearBy = WSinfo::valid_opponents;
  oppsNearBy.keep_players_in_circle( ivInternalBallObject.pos, 
                                     2.0*ServerOptions::kickable_area); 
  oppsNearBy.keep_and_sort_closest_players_to_point(1,ivInternalBallObject.pos);
  if (   oppsNearBy.num > 0
      && ivInternalBallObject.pos.distance( oppsNearBy[0]->pos ) 
         < 0.8 * oppsNearBy[0]->kick_radius ) //20% additional safety margin
    ivBallWithHighProbabilityInKickrangeOfAnOpponentPlayer = true;
  else
    ivBallWithHighProbabilityInKickrangeOfAnOpponentPlayer = false;
}

void 
Score05_Sequence::updateViaPoints()
{
  Vector myPosInTCycles[3];
  for (int i=0; i<3; i++)
    myPosInTCycles[i] = getMyPositionInTCyclesAssumingNoDash(i);
  Vector myVectToHisGoalInTCycles[3];
  for (int i=0; i<3; i++)
  {
    myVectToHisGoalInTCycles[i] = HIS_GOAL_CENTER - myPosInTCycles[i]; 
    myVectToHisGoalInTCycles[i].normalize
                   ( WSinfo::me->kick_radius - VIA_POINT_MARGIN );
  }
  
  for (int i=0; i<NUMBER_OF_VIA_POINTS; i++)
    for (int j=0; j<3; j++)
      ivViaPoints[i][j] = myPosInTCycles[j];
    
  Vector leftTurnedVector[3],
         rightTurnedVector[3],
         frontVector[3],
         toGoalVector[3];
  for (int i=0; i<3; i++)
  {
    leftTurnedVector[i]  = myVectToHisGoalInTCycles[i];
    rightTurnedVector[i] = myVectToHisGoalInTCycles[i];
    frontVector[i]       = Vector(WSinfo::me->ang);
    toGoalVector[i]      = myVectToHisGoalInTCycles[i];
    leftTurnedVector[i].rotate( PI/2.0 );
    rightTurnedVector[i].rotate( -PI/2.0 );
    toGoalVector[i].normalize(WSinfo::me->kick_radius * 0.55);
    frontVector[i].normalize(WSinfo::me->kick_radius * 0.55);
  }
  
  for (int i=0; i<3; i++)
  {        
    ivViaPoints[0][i] += frontVector[i];
    ivViaPoints[1][i] += toGoalVector[i];
    ivViaPoints[2][i] += rightTurnedVector[i];
    ivViaPoints[3][i] += leftTurnedVector[i];
  }
  //ivViaPoints[0].y -= WSinfo::me->kick_radius - VIA_POINT_MARGIN;
  //ivViaPoints[1].y += WSinfo::me->kick_radius - VIA_POINT_MARGIN;
  for (int i=0; i<NUMBER_OF_VIA_POINTS; i++)
  {
    //draw the via points
    TGs05sLOGPOL(LOGBASELEVEL+0,<<_2D<<VC2D(ivViaPoints[i][1],0.1,"999999"));
    TGs05sLOGPOL(LOGBASELEVEL+0,<<_2D<<VC2D(ivViaPoints[i][2],0.1,"bbbbbb"));

    TGs05sLOGPOL(LOGBASELEVEL+0,<<_2D<<VC2D(ivViaPoints[i][0],0.1,"666666"));
    TGs05sLOGPOL(LOGBASELEVEL+0,<<_2D<<VSTRING2D(ivViaPoints[i][0],i,"666666"));
  }
}

//============================================================================
// evaluateScorePositionAfterTCycles
//============================================================================
//============================================================================
// getBestOneStepKickToViaPoint
//============================================================================
//============================================================================
// getBestTwoStepKickToViaPoint
//============================================================================
//============================================================================
// checkIfPointOnCrossLineIsViable
//============================================================================
//============================================================================
// getMyPositionInTCyclesAssumingNoDash
//============================================================================
//============================================================================
// getMyVelocityInTCyclesAssumingNoDash
//============================================================================
//============================================================================
// checkIfGoalIsForSureWhenShootingInTCyclesFromViaPoint
//============================================================================
/**
 * viaPointIndex: index of the via point at which the ball will be
 * ballVel      : velocity ball will have at that via point
 * 
 */
//============================================================================
// checkIfGoalIsForSureWhenShootingInTCyclesFromViaPointIntoDirection
//============================================================================
//============================================================================
// performFinalKick
//============================================================================
/**
 * When having this method, this means
 * (a) we have successfully conducted a score sequence, and the ball should
 *     be in quite a good position for scoring now
 * (b) unfortunately, the 'normal' score routine did not recognize this
 *     and did not shoot onto the goal
 */
//============================================================================
// checkIfGoalIsForSureWhenShootingNow()
//============================================================================
//============================================================================
// checkIfGoalIsForSureWhenShootingNowIntoDirection
//============================================================================
//============================================================================
// checkIfGoalIsForSureWhenTacklingNow()
//============================================================================
//============================================================================
// checkForNearPlayersRegardingCurrentMode()
//============================================================================
//============================================================================
// enemyMayReachPointInTCycles()
//============================================================================
//============================================================================
// determineGoaliePositionAndAge
//============================================================================
//============================================================================
// getPotentialGoaliePositions
//============================================================================
/**
 * Attention: This method performs recursive calls! Don't call it with too
 * large a maxSearchDepth parameter!
 */
//============================================================================
// isPointInOneOfRecentViewCones
//============================================================================
//============================================================================
// updateInternalBallObject()
//============================================================================
//////////////////////////////////////////////////////////////////////////////
///////////////// I M P O R T E //////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
#define _DUMMY_LABEL_IMPORTE_

bool 
Score05_Sequence::goalie_needs_turn_for_intercept (  int    time, 
                                                     Vector initial_ball_pos, 
                                                     Vector initial_ball_vel, 
                                                     Vector b_pos, 
                                                     Vector b_vel, 
                                                     double goalie_size)
{
  //test if goalie has a bad angle for the intercept
  PPlayer goalie = WSinfo::his_goalie;

  if (goalie == NULL) 
    return 0;
  if (goalie->age > 1) 
    return 0;
  
  Vector goalie_ang;
  goalie_ang.init_polar(1.0, goalie->ang);
  
  Vector intersection 
    = intersection_point(goalie->pos, goalie_ang, initial_ball_pos, initial_ball_vel);
  //TGs05sLOGPOL(LOGBASELEVEL+4, << _2D << C2D(intersection.x, intersection.y, 1, "ff0000"));
  
  double dist = (intersection-initial_ball_pos).norm();
  Vector ball_in_goal 
    = Vector( point_on_line(initial_ball_vel, initial_ball_pos,
                            ServerOptions::pitch_length/2.0) );
  //LOG_DAN(0, "Ball ueberschreitet Torlinie an " << ball_in_goal.x << " " << ball_in_goal.y);
  if (    dist - 2.0 > (ball_in_goal - initial_ball_pos).norm() 
       && ((b_pos - goalie->pos).norm() > 2.5) ) 
  {
    TGs05sLOGPOL(LOGBASELEVEL+5, "Score05_Sequence: Schnittpunkt zu weit entfernt!");
    return 1;
  } 
  else 
  {
    int time_L = time;
    Vector b_pos_L = b_pos;
    Vector b_vel_L = b_vel;
    
    for (int j = 0; j < 10; ++j) 
    {
      //if ((b_pos_L - initial_ball_pos).norm() > dist - 2.0) 
      if ( (b_pos_L - intersection).norm() < 1.4 ) //TG08: 2.2->1.4
      {
        double goalie_rad = goalie_action_radius_at_time(time_L, goalie_size, 0);
        double goalie_rad2 = goalie_action_radius_at_time(time_L+1, goalie_size, 0);
        Vector g_pos = intersection-goalie->pos;
        Vector g_pos2 = g_pos;
        g_pos.normalize(goalie_rad);
        g_pos += goalie->pos;
  
        g_pos2.normalize(goalie_rad2);
        g_pos2 += goalie->pos;
  
        //if (j%2==0) TGs05sLOGPOL(LOGBASELEVEL+4, << _2D << C2D(b_pos_L.x, b_pos_L.y, 1, "ff00ff"));
        if (    (goalie_rad + ivGOALIE_CATCH_RADIUS < (intersection-goalie->pos).norm()) 
             && !is_pos_in_quadrangle(b_pos_L, goalie->pos, g_pos, 2.6) //TG08: 4.0->2.6
             && (b_pos_L.distance(g_pos) > 1.4) //TG08: 2.2->1.4 
             && !is_pos_in_quadrangle(b_pos_L+b_vel_L, goalie->pos, g_pos2, 2.6) //TG08: 4.0->2.6 
             && ((b_pos_L+b_vel_L).distance(g_pos2) > 2.2) //TG08: 2.2->1.4 
             && ((b_pos-goalie->pos).norm() > 1.6) //TG08: 2.5->1.6
             && ((goalie_rad*(intersection-goalie->pos)+goalie->pos - b_pos_L).norm() > 1.6) //TG08: 2.5->1.6
           ) 
        {
          TGs05sLOGPOL(LOGBASELEVEL+1, << "Score05_Sequence: goalie has bad angle for intercept => needs one extra step, j="<<j);
          //TGs05sLOGPOL(LOGBASELEVEL+4, << _2D << C2D(g_pos.x, g_pos.y, 1, "ffff00"));
          if ( (b_pos_L + b_vel_L - intersection).norm() > 2.2 )
            return 1;
        } 
        else 
          return 0;
      } 
      
      b_pos_L += b_vel_L;
      b_vel_L *= ServerOptions::ball_decay;
      ++time_L;
    }
    return 0;
  }
}

double
Score05_Sequence::goalie_action_radius_at_time ( int   time, 
                                                 double goalie_size,
                                                 int   goalie_handicap) 
{
  int time_L = time - goalie_handicap;

  if (time_L < 0) return 0.0;
  switch (time_L) 
  {
    case 0: 
      return 0.0;
    case 1:
      return goalie_size;
    case 2:
      return 0.65 + goalie_size; //0.7
    case 3:
      return 1.5 + goalie_size; //1.6
    case 4:
      return 2.45 + goalie_size; //2.6
    default:
      if (time_L < 0) 
        return 0.0;
      else 
        return 2.45 + 1.0 * (time_L - 4) + goalie_size; //2.6
  }
}

int 
Score05_Sequence::intercept_goalie(   Vector ball_pos, 
                                      Vector ball_vel, 
                                      Vector goalie_pos, 
                                      double  goalie_size,
                                      int    timeOffset,
                                      bool debug ) 
{
  if (ball_vel.getX() < 0.0) //goalie intercepts :-(
    return 1;
  if (ball_vel.norm() < 0.5) //goalie intercepts :-(
    return 1;

  Vector b_pos = ball_pos;
  Vector b_vel = ball_vel;
  double goal_x = ServerOptions::pitch_length/2.0;

  double goalie_action_radius;

  int time = 1;
  bool wrong_angle = false;

  int minInact = 0, maxInact;
  if (WSinfo::his_goalie != NULL)
  {
    WSinfo::get_player_inactivity_interval( WSinfo::his_goalie, minInact, maxInact );
  }

  for (int i = 1; i < 50; ++i) 
  {
    b_pos += b_vel;
    b_vel *= ServerOptions::ball_decay;
    goalie_action_radius = goalie_action_radius_at_time( time + timeOffset, 
                                                         goalie_size, 
                                                         (minInact+3*maxInact)/4 );
    if (wrong_angle) goalie_action_radius -= 0.3;
    
//if (debug&&(i==1||i==2||i==3||i==4||i==5||i==6||i==7||i==10||i==20||i==40))    
//  TGs05sLOGPOL(LOGBASELEVEL+3, << _2D << C2D(goalie_pos.x, goalie_pos.y, goalie_action_radius + ivGOALIE_CATCH_RADIUS, "#0000ff"));
//if (debug&&(i==1||i==2||i==4||i==7||i==10||i==20||i==40))    
//  TGs05sLOGPOL(LOGBASELEVEL+3, << _2D << C2D(b_pos.x, b_pos.y, 1, "00ff00"));
    
    if (  b_pos.getX() > goal_x + 0.3)
    {
      if (WSinfo::his_goalie)
      {
        TGs05sLOGPOL(LOGBASELEVEL+3,<<"Score05_Sequence: GOALIE does not intercept: b_pos="<<b_pos<<" i="<<i);
        TGs05sLOGPOL(LOGBASELEVEL+3,<<"Score05_Sequence: INFO: G->age="<<WSinfo::his_goalie->age<<" G->pos="<<WSinfo::his_goalie->pos
          <<" G->age_vel="<<WSinfo::his_goalie->age_vel<<" G->vel="<<WSinfo::his_goalie->vel);
      }
      //LOG_DAN(0, << "time is " << time);
      return -time; //no interception :-)))

    } 
    else
    if  (    i == 1 //first step
          && (   (b_pos - ivInternalHisGoaliePositionWhenDashP100).norm() < ivGOALIE_CATCH_RADIUS 
              ||
                 (b_pos - ivInternalHisGoaliePositionWhenDashM100).norm() < ivGOALIE_CATCH_RADIUS
             )
        )
    {
      return time; //goalie intercepts immediately after having dashed +/- 100
    }
    else 
    if  (     ( (b_pos - goalie_pos).norm() - goalie_action_radius < ivGOALIE_CATCH_RADIUS ) 
           && wrong_angle ) 
    {
      return time;//1; //goalie catches the ball
    } 
    else 
    if  (     ((b_pos - goalie_pos).norm() - goalie_action_radius < ivGOALIE_CATCH_RADIUS) 
           && !wrong_angle) 
    {
      if (goalie_needs_turn_for_intercept(time + timeOffset, 
                                          ball_pos, ball_vel, b_pos, 
                                          b_vel, goalie_size) ) 
      {
        wrong_angle = true;
      } 
      else 
        return time;//1; //goalie intercepts
    }
    ++time;
  }
  return time;//1; //goalie intercepts
}

int 
Score05_Sequence::intercept_opponents ( double  direction,
                                        double  b_v,
                                        int    max_steps,
                                        Vector startBallPosition,
                                        int    timeOffset ) 
{
  PlayerSet pset = WSinfo::valid_opponents;
  pset.keep_players_in_cone ( ivpInternalBallObject->pos, 
                              ANGLE(direction-DEG2RAD(20)), 
                              ANGLE(direction+DEG2RAD(20)));
  for (int j=0; j<pset.num; j++)
  {
    int oppSteps;
    PPlayer opp = pset[j];
    Vector interceptPosition,
           ballVelocity;
    ballVelocity.init_polar( b_v, direction );
    Vector consideredOpponentPosition = opp->pos;
    //try to set off the opponent if recently looked into his direction
    if (   isPointInOneOfRecentViewCones( opp->pos, 0, 0 )
        && opp->age > 0 )
    {
      ANGLE curViewDirection, curViewWidth;
      Vector curPlayerPosition;
      long dummyTime;
      if ( WSmemory::get_view_info_before_n_steps( 0,
                                                   curViewWidth,
                                                   curViewDirection,
                                                   curPlayerPosition,
                                                   dummyTime ) )
      {
        ANGLE halfCurViewWidth( curViewWidth.get_value_0_p2PI() * 0.5 );
        //determine guessed position left and right of view cone
        Vector endOfViewCone_L( curViewDirection + halfCurViewWidth ),
               endOfViewCone_R( curViewDirection - halfCurViewWidth );
        endOfViewCone_L += curPlayerPosition;
        endOfViewCone_R += curPlayerPosition;
        //determine guessed position LEFT of view cone
        Vector guessedPlayerPos_L 
          = Tools::get_Lotfuss( curPlayerPosition,
                                endOfViewCone_L,
                                opp->pos );
        Vector playerShift_L = guessedPlayerPos_L - opp->pos;
        playerShift_L.normalize( playerShift_L.norm() + 0.3 );
        guessedPlayerPos_L = opp->pos + playerShift_L;
        if (   guessedPlayerPos_L.distance(ivpInternalMeObject->pos) 
             < ServerOptions::visible_distance )
        {
          playerShift_L = guessedPlayerPos_L - ivpInternalMeObject->pos;
          playerShift_L.normalize( ServerOptions::visible_distance + 0.5 );
          guessedPlayerPos_L = ivpInternalMeObject->pos + playerShift_L;
        }
        //determine guessed position RIGHT of view cone
        Vector guessedPlayerPos_R
          = Tools::get_Lotfuss( curPlayerPosition,
                                endOfViewCone_R,
                                opp->pos );
        Vector playerShift_R = guessedPlayerPos_R - opp->pos;
        playerShift_R.normalize( playerShift_R.norm() + 0.3 );
        guessedPlayerPos_R = opp->pos + playerShift_R;
        if (   guessedPlayerPos_R.distance(ivpInternalMeObject->pos) 
             < ServerOptions::visible_distance )
        {
          playerShift_R = guessedPlayerPos_R - ivpInternalMeObject->pos;
          playerShift_R.normalize( ServerOptions::visible_distance + 0.5 );
          guessedPlayerPos_R = ivpInternalMeObject->pos + playerShift_R;
        }
        //decide for ONE of the two shifted positions
        if (   opp->pos.distance( guessedPlayerPos_L )
             < opp->pos.distance( guessedPlayerPos_R ) )
          consideredOpponentPosition = guessedPlayerPos_L;
        else 
          consideredOpponentPosition = guessedPlayerPos_R;
        TGs05sLOGPOL(0,<<"Score05_Sequence: Opponent should have been seen"
          <<" recently, but wasn't. Hence, I guess his position to be at "
          <<consideredOpponentPosition);
        TGs05sLOGPOL(0,<<_2D<<VC2D(consideredOpponentPosition, 2.0, "880000"));
      }
    }
    //end of try to set off opp
    //correct consideredOpponentPosition if recently seen
    if (opp->age == 0 && opp->age_vel == 0)
    {
      Vector hisVelocity = opp->vel;
      if (hisVelocity.norm() > opp->speed_max)
        hisVelocity.normalize(opp->speed_max);
      if (   Tools::get_dist2_line( startBallPosition, 
                                    startBallPosition + ballVelocity,
                                    opp->pos + hisVelocity )
           < Tools::get_dist2_line( startBallPosition, 
                                    startBallPosition + ballVelocity,
                                    opp->pos ) )
      {
        consideredOpponentPosition = opp->pos + hisVelocity;
        TGs05sLOGPOL(0,<<"Score05_Sequence: Opponent "
          <<opp->number<<" has been seen"
          <<" recently. Add his velocity "
          <<opp->vel<<"/"<<hisVelocity<<" to his considered position. => "
          <<consideredOpponentPosition);
        TGs05sLOGPOL(0,<<_2D<<VC2D(consideredOpponentPosition, 2.0, "880000"));
      }
    }
    Policy_Tools::intercept_min_time_and_pos_hetero( oppSteps,
                                                     interceptPosition,
ServerOptions::ball_decay * ballVelocity+ /*TG08 ZUI*/                        ballVelocity +         startBallPosition,
ServerOptions::ball_decay * /*TG08 ZUI*/                        ServerOptions::ball_decay * ballVelocity,
                                                     consideredOpponentPosition,
                                                     opp->number,
                                                     false,//my team?
                                                     -1.0,//default
                                                     -1000 );//default
    if (interceptPosition.getX() < FIELD_BORDER_X)
    {
      TGs05sLOGPOL(2,<<"Score05_Sequence: Opponent "<<opp->number<<" (pos="
        <<opp->pos<<",kr="
        <<opp->kick_radius<<")intercepts at "
        <<interceptPosition<<" after "<<oppSteps<<" steps (ballVelocity="<<ballVelocity<<").");
      //returnValue += oppSteps;
      return opp->number; //old, but never really differentiated
    }
  }  
  //TGs05sLOGPOL(1,<<"Score05_Sequence: Return value of intercept opponent "
  //  <<" check: "<<returnValue);
  //return returnValue;
  return 0;
/*  WSpset pset = WSinfo::valid_opponents;
  pset.keep_players_in_cone ( ivpInternalBallObject->pos, 
                              ANGLE(direction-DEG2RAD(20)), 
                              ANGLE(direction+DEG2RAD(20)));
  Vector player_pos, b_pos, b_vel, ball_vel;
  double player_dist_to_ball, player_action_radius;
  ball_vel.init_polar(b_v, direction);

  for (int i = 0; i < pset.num; ++i) 
  {
    if (pset[i]->number == WSinfo::ws->his_goalie_number)
      continue;
    player_pos = pset[i]->pos;
    b_pos = startBallPosition;
    b_vel = ball_vel;
    player_dist_to_ball = (pset[i]->pos - startBallPosition).norm();
    int playerAgeOffSet = Tools::max(2, pset[i]->age);

    for (int j = 1; j < 12; ++j) 
    {
      b_pos += b_vel;
      b_vel *= ServerOptions::ball_decay;
      player_action_radius 
        = player_action_radius_at_time ( j + timeOffset + playerAgeOffSet, 
                                         pset[i], 
                                         player_dist_to_ball, 
                                         0);//0.4
      
      if ((b_pos - player_pos).norm() < player_action_radius) 
      {
        if (pset[i]->number > 0) 
          return pset[i]->number; //player with this number will intercept
        else 
          return 1;
      }
    }
  }
  return 0; //nobody intercepts */
}


Vector 
Score05_Sequence::intersection_point ( Vector p1, 
                                       Vector steigung1, 
                                       Vector p2, 
                                       Vector steigung2) 
{
  double x, y, m1, m2;
  if ((steigung1.getX() == 0) || (steigung2.getX() == 0))
  {
    if (fabs(steigung1.getX()) < 0.00001)
    {
      return point_on_line(steigung2, p2, p1.getX());
    } 
    else 
      if (fabs(steigung1.getX()) < 0.00001)
      {
        return point_on_line(steigung1, p1, p2.getX());
      } 
  }
  m1 = steigung1.getY()/steigung1.getX();
  m2 = steigung2.getY()/steigung2.getX();
  if (m1 == m2) return Vector(-51.5, 0);
  x = (p2.getY() - p1.getY() + p1.getX()*m1 - p2.getX()*m2) / (m1-m2);
  y = (x-p1.getX())*m1 + p1.getY();
  return Vector (x, y);
}

bool 
Score05_Sequence::is_pos_in_quadrangle ( Vector pos, 
                                         Vector p1, 
                                         Vector p2, 
                                         Vector p3, 
                                         Vector p4) 
{
  if ( Tools::point_in_triangle(pos, p1,p2,p3) 
       ||
       Tools::point_in_triangle(pos, p1,p3,p4) ) 
  { 
    return true;
  }
  return false;
}

bool 
Score05_Sequence::is_pos_in_quadrangle(Vector pos, Vector p1, Vector p2, double width)
{
  Vector tmp= p2-p1;
  Vector norm;
  norm.setX( -tmp.getY() );
  norm.setY( tmp.getX() );
  norm.normalize(0.5*width);
  Vector g1= p1+ norm;
  Vector g2= p1- norm;
  Vector g3= p2- norm;
  Vector g4= p2+ norm;

  return is_pos_in_quadrangle(pos,g1,g2,g3,g4);
}
double
Score05_Sequence::player_action_radius_at_time ( int     time, 
                                                 PPlayer player, 
                                                 double   player_dist_to_ball,
                                                 int     player_handicap) 
{
  //used for learning, otherwise players try to shoot through static defenders
  //return 1.1;

  int time_L = time - player_handicap;

  if (player_dist_to_ball < 3.0) 
  {
    if (time_L <= 2) 
    {
      return 0.0 + player->kick_radius;
    } 
    else 
    {
      return player->speed_max * (time_L-2) * 0.8 + player->kick_radius;
    }
  } 
  else 
  {
    if (time_L <= 1) 
    {
      return 0.0 + player->kick_radius;
    } 
    else
    {
      return player->speed_max * (time_L-1) * 0.8 + player->kick_radius;
    }
  }
}


/* Berechnet den Schnittpunkt zweier Geraden */
/* Berechnet die y-Koordinate Punktes auf der Linie, der die x-Koordinate x hat
 */
Vector 
Score05_Sequence::point_on_line(Vector steigung, Vector line_point, double x)
{
  //steigung.normalize();
  steigung = (1.0/steigung.getX()) * steigung;
  if (steigung.getX() > 0)
  {
    return (x - line_point.getX()) * steigung + line_point;
  }
  if (steigung.getX() < 0)
  {
    return (line_point.getX() - x) * steigung + line_point;
  }  // Zur Sicherheit, duerfte aber nie eintreten
  return line_point;
} /* point_on_line */

