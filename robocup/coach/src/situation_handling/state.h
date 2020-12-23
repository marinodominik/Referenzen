#ifndef _STATE_H_
#define _STATE_H_

#include <vector>
#include <stdlib.h>
#include <limits.h>

#include "coach.h"
#include "logger.h"

#define MAX_NUMBER_OF_STATE_VARIABLES      50

#define BALL_POSITION                       0
#define BALL_VELOCITY                       1
#define MY_ATTACKERS_POSITIONS_FLTR         2
#define MY_MIDFIELDERS_POSITIONS_FLTR       3
#define MY_DEFENDERS_POSITIONS_FLTR         4
#define HIS_MIDFIELDERS_POSITIONS_FLTR      5
#define HIS_DEFENDERS_POSITIONS_FLTR        6
#define HIS_GOALIE_POSITION                 7


class State
{
  private:
    void  getFeatureIndex(int id, int & min, int & max);
  
  public:
    int   ivSize;
    float ivFeatures[MAX_NUMBER_OF_STATE_VARIABLES];

    State();
    State( const State & state );
    State( const vector<float> & vectorOfFloats );

    static const int cvcNumberOfStateComponents;
    static const int cvcStateComponents[]; 
    
    static void getCurrentState(State & s);
    static void getCurrentFullState(State & s);
    
    static vector<int> getMyAttackerIndizesFLTR();
    static vector<int> getMyMidfielderIndizesFLTR();
    static vector<int> getMyDefenderIndizesFLTR();
    
    static vector<int> getHisMidfielderIndizesFLTR( vector<int> & hisDefenders );
    static vector<int> getHisDefenderIndizesFLTR();
    static int         getHisGoalieIndex();
  
    static void        sortHisPlayerIndizesFromLeftToRight
                       ( vector<int> & plIndizes );
    
};

class Episode
{
  public:
    vector<State> ivVStates;
    vector<State> ivVFullStates;
    vector<double> ivVRewards;
    
    void   addNextState();
    void   addSuggestedImmediateReward(float r);
    bool   ballKickableForOpponent(int t);
    bool   ballKickableForTeammate(int t);
    double ballVelocityAtTime(int t);
    double getAverageDistanceToBall( int t );
    double getAverageDistanceToDirectOpponents( int t );
    double getAverageWayGoneByTeammates( int t );
    double getDistanceOfNearestPlayerToPos(int t, Vector & pos);
    int    getLength();
    int    getNumberOfOpponentPlayersAroundPoint( int t,
                                                  Vector & pos,
                                                  double    radius );
    int    getNumberOfTeammatePlayersAroundPoint( int            t,
                                                  const Vector & pos,
                                                  double          radius );
    bool   getTwoTeammatesControlBall( int t );
    Vector getBallPosition(int t);
    void   reset();
    string toStringReward(int t);
    string toStringState(int t);
    
};

#endif
