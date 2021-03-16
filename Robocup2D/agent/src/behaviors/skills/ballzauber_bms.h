#ifndef _BALLZAUBER_H_
#define _BALLZAUBER_H_

#include <sstream>

#include "basic_cmd_bms.h"
#include "one_step_kick_bms.h"

#define BALLZAUBER_KICK                 1
#define BALLZAUBER_TURN                 2
#define MAX_NUMBER_OF_PLANNING_STEPS    10
#define MAX_DEVIATION                   ((8.0/180.0)*PI)
#define MIN_DISTANCE_TO_AVOID_COLLSIONS 0.45
#define UNABLE_TO_HOLD_BALL             100

class Ballzauber:public BodyBehavior
{
  private:
    bool         ivBallInTargetAreaInTSteps[MAX_NUMBER_OF_PLANNING_STEPS];
    bool         ivBallInRelaxedTargetAreaInTSteps[MAX_NUMBER_OF_PLANNING_STEPS];
    Vector       ivBallPositionInTSteps[MAX_NUMBER_OF_PLANNING_STEPS];
    Vector       ivBallVelocityInTSteps[MAX_NUMBER_OF_PLANNING_STEPS];
    Vector       ivMyVelocityInTSteps[MAX_NUMBER_OF_PLANNING_STEPS];
    Vector       ivMyPositionInTSteps[MAX_NUMBER_OF_PLANNING_STEPS];
    int          ivRequiredTurnCommandsWhenStartingTurningInTSteps
                   [MAX_NUMBER_OF_PLANNING_STEPS];
    Quadrangle2d ivTargetAreaInTSteps[MAX_NUMBER_OF_PLANNING_STEPS];
    Vector       ivTargetAreaCentralPointInTSteps[MAX_NUMBER_OF_PLANNING_STEPS];
    Quadrangle2d ivRelaxedTargetAreaInTSteps[MAX_NUMBER_OF_PLANNING_STEPS];
    ANGLE        ivTargetDirection;
    
    int          ivAction;
    Vector       ivActionKickVelocity;
    ANGLE        ivActionTurnAngle;
    stringstream ivPlanStringStream;
    bool         ivLastResult;
    int          ivLastTimeCalled;
    int          ivNumberOfStepsToTurnToTargetDirection;
    bool         ivSafetyWithRespectToNearbyOpponents;
    
    BasicCmd   * ivpBasicCmdBehavior;
    OneStepKick* ivpOneStepKickBehavior;

    static bool cvInitialized;
      
    bool         checkForNearbyOpponents();
    bool         checkForTermination();
    
    bool         createActionToKeepBallSomehowInKickrange();
    bool         createActionToKickBallToTargetAreaInAsManyStepsINeedThenForTurning();
    bool         createActionToStopBallOrMoveItALittleTowardsTargetArea();
    bool         createActionToTurnToTargetDirection();
  
    bool         doAHoldTurn();
    double       getMaximalVelocityWhenKicking( Vector myPos,   Vector myVel,
                                                ANGLE  myAng,
                                                Vector fromPos, Vector ballVel,
                                                Vector toPos );
    void         initCycle();
    bool         isBallPositionSafe( Vector pos );
    bool         setCommand( Cmd & cmd );
                                           
  public:
  
    Ballzauber();
    ~Ballzauber();
    static bool init(char const * confFile, int argc, char const* const* argv);

    enum TargetArea {
        TA_IN_FRONT,
        TA_LEFT_BEHIND,
        TA_RIGHT_BEHIND
    } ivTargetArea;

    bool         get_cmd( Cmd & cmd );
    bool         get_cmd( Cmd & cmd, ANGLE targetDir, TargetArea ta=TA_IN_FRONT);

    int          getNumberOfStepsToTurnToTargetDirection();
    int          getNumberOfStepsToTurnToTargetDirection( ANGLE targetDir,TargetArea ta=TA_IN_FRONT );
    
    bool         isBallzauberToTargetDirectionPossible();
    bool         isBallzauberToTargetDirectionPossible( ANGLE targetDir, TargetArea ta=TA_IN_FRONT );

    void         setTargetDirection( ANGLE ang );
    void         setTargetArea(TargetArea);

};
#endif
