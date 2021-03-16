/** \file positioning.h
   To tackle the positioning problem we divide the position process into two steps:
   1. examin (all?) possible positions and select the n best ones for the whole team
   2. assign each available player a position according to a matching function
   It would be nice, if the matching would consider players that have special jobs like
   intercepting the ball.
   For learning purpose you should alter the function "assign_positions".
   The function "match_positions" is planned as a more or less static graph algorithm,
   thus no intelligence at all.
   For simplicity we agreed to select positions that are in between our own offside line
   and the offside line of the opponent. Our own offside line is defined by the 
   "test_position_moveup" function in class "Magnetic_Playon_Policy". 
*/

#ifndef _POSITIONING_H
#define _POSITIONING_H

#include "globaldef.h"
#include "Vector.h"

#define NEW433HACK //Art new formation

#ifdef NEW433HACK
#include "formation.h"
#include "formation433_attack.h"
#endif

#define EVALUATION_DISTANCE 0
#define EVALUATION_DISTANCE_WEIGHT 1
#define EVALUATION_DISTANCE_BALL 2

#define MATCHING_PLAYER 0
#define MATCHING_GREEDY 1
#define MATCHING_DIVIDE_THREE 2
#define MATCHING_DISTRIBUTION 3
#define MATCHING_MOVEUP 4

/* we create a special data structure that holds several informations about a position
 */

class AbstractPosition : public Vector{
 public:
};

class DashPosition : public AbstractPosition{
 public:
  DashPosition() {}
  DashPosition( const Vector & vec, double dp, double r ) { setX(vec.getX()), setY(vec.getY()), dash_power= dp, radius= r; }
  double dash_power;
  double radius;
};

class PrioPosition : public AbstractPosition{
 public:
  double priority;
  double radius;
};

class DeltaPosition : public DashPosition{
 public:
    int    role;
    double  weight;
    int    player_assigned;
};


class AbstractPositioning{
 public:
  const static int NUM_CONSIDERED_POSITIONS = 10;
  static AbstractPosition pos[NUM_CONSIDERED_POSITIONS];

  static void select_positions();

  static void match_positions();

  /* returns the position for the calling player
     this function assumes that the matching orders the positions by player,
  */
  static inline AbstractPosition get_position(int player_number){return pos[player_number];}
  
};

class Formation{
 public:
    int    number;
    char   name[50];
    double  pos[10][3];

    inline void print(){ std::cout << "\n(" << number << ")";}
};

#define MAX_NUM_FORMATIONS 20

class DeltaPositioning : public AbstractPositioning{
#ifdef NEW433HACK
  static Formation433 formation433;
  static bool use_new_formation433;
#endif
  DeltaPositioning();

 public:

  static Formation form[MAX_NUM_FORMATIONS];
  static int cnt_formations;
  static double stretch;

  static int current_formation;
  static int current_matching;
  static int current_evaluation;

  static int four_chain_established;
  static bool use_BS02_gamestates;
  static double before_goal_line;

  static double max_defense_line;
  static double min_defense_line_offset;
  static double defense_line_ball_offset;

  static double ball_weight_x_defense;
  static double ball_weight_y_defense;
  static double ball_weight_x_buildup;
  static double ball_weight_y_buildup;
  static double ball_weight_x_attack;
  static double ball_weight_y_attack;
  static int cycles_after_catch;
  static double move_defense_line_to_penalty_area;
  
  static DeltaPosition pos[NUM_CONSIDERED_POSITIONS];
  static int recommended_attentionto[NUM_PLAYERS+1];
  static void show_recommended_attentionto() {
    std::cout << "\nAttention to recommendation:\n";
    for(int i=1; i < NUM_PLAYERS + 1; i++) 
      std::cout << "\nplayer_" << i << " " << recommended_attentionto[i];
  }

  // this array can be modified by the calling function to select the players
  static bool consider_player[11];

  // this function fills the array consider_players
  static void init_players(bool *players);
  static void init_formations();

  static DashPosition get_position(int player_number);
  static Vector get_position_base(int player_number);
  static int get_role(int player_number);
  static int get_num_players_with_role(int role);
  static inline int get_num_defenders(){return get_num_players_with_role(0);}
  static inline int get_num_offenders(){return get_num_players_with_role(2);}
  static inline int get_num_midfield(){return get_num_players_with_role(3);}

  static double get_my_defence_line();
  static double get_my_offence_line();

  static double evaluate(int player, int position);

  static double get_radius(int player_number);

  static bool is_position_valid(Vector p); 

  static double get_defence_line_ball_offset();

  static Formation433Attack attack433;

};

class DashStamina{
 public:
    double stamina_offence;
    double dash_offence;
    double stamina_defence;
    double dash_defence;
};


class Stamina{
 public:

  static int stamina_management_type;
  static int state,last_update;
  static double stamina_reserve_defenders;
  static double stamina_reserve_midfielders;
  static double stamina_reserve_attackers;
  static double stamina_full_level;
  static double stamina_min_reserve_level;
  static double stamina_min_reserve;

  static DashStamina dashstamina[3];
  

  static void init();
  static double economy_level();
  static int dash_power();
  static void update_state();
  static int get_state();

};

//////////////////////////////////////////////////////////////////////////////
// EXTENSIONS by TGA 2005
//////////////////////////////////////////////////////////////////////////////
struct tInterceptInformation
{
  int     myStepsToGo;
  Vector  myPointOfBallInterception;
  double  myBallDistAtTheMomentOfInterception;
  int     myGoalieStepsToGo;
  int     teammateStepsToGo;
  int     teammateNumber;
  Vector  teammatePosition;
  Vector  teammatePointOfBallInterception;
  bool    isBallKickableForTeammate;
  int     opponentStepsToGo;
  Vector  opponentPointOfBallInterception;
  int     opponentNumber;
  Vector  opponentPosition;
  bool    isBallKickableForOpponent;
  int     ballAge;
  long    validForCycle;
};
struct tPassReceivingInformation
{
  const static int NUM_OF_STORED_FUTURE_BALL_POSITIONS = 20;
  tPassReceivingInformation() {validAtTime=-1; playerWithNewestPassInfo=NULL;};
  bool    validAtTime;
  bool    immediatePass;
  PPlayer playerWithNewestPassInfo;
  int     myStepsToGo;
  Vector  myPointOfBallInterception;
  int     teammateStepsToGo;
  int     teammateNumber;
  Vector  teammatePointOfBallInterception;
  int     opponentStepsToGo;
  int     opponentNumber;
  Vector  opponentPointOfBallInterception; 
  Vector  ballPositions[NUM_OF_STORED_FUTURE_BALL_POSITIONS];
};

class OpponentAwarePosition : public AbstractPosition
{
  public:
    OpponentAwarePosition() {}
    OpponentAwarePosition( const Vector & vec, 
                           double dp,
                           double r,
                           int oppNum ) 
    { setX(vec.getX()), setY(vec.getY()), dashPower= dp, radius= r; opponentNumber = oppNum;}
    int   opponentNumber;
    double dashPower;
    double radius;
};

class OpponentAwarePositioning : public AbstractPositioning
{
  public:
    static bool cvUseVersion2017;
  private:
    //variables
    static Formation05           cvFormation;
    static tInterceptInformation cvInterceptInformation;
    #define MAX_CRITERIA 5
    static double                cvCriteriaWeights[4][MAX_CRITERIA];
    static PPlayer               cvPlayerWithMaximalDanger;
    //methods
  public:
    //methods
    static
      bool init(char const * confFile, 
                                          int          argc,  
                                          char const* const* argv);
    static 
      bool initFormations();
    static
      void updateDirectOpponentInformation();
    static
      int  getRole(int number);
    static
      OpponentAwarePosition getHomePosition(XYRectangle2d);
    static
      OpponentAwarePosition getHomePosition();
    static 
      bool getAngleToDirectOpponent( ANGLE & anleToDirOpp);
    static 
      OpponentAwarePosition getAttackOrientedPosition(float ballApproachment=0.1);
    static
      OpponentAwarePosition getOpponentAwareDefensiveFallbackHomePosition();
    static
      OpponentAwarePosition getOpponentAwareMyKickInPosition();
    static
      OpponentAwarePosition getFormationAndDirectOpponentAwareStrategicPosition();
    static
      bool getSupportedAndSupportingPlayerForMidfielder
                            ( int       midfielderNumber,
                              PPlayer & supportedPlayer,
                              PPlayer & supportingPlayer );
    static
      OpponentAwarePosition getCoverPosition();
    static
      OpponentAwarePosition getCoverPosition( PPlayer directOpponent );
    static
      OpponentAwarePosition getCoverPosition 
                            (  const int    & directOpponentNumber,
                               const Vector & directOpponentPos,
                               const Vector & directOpponentVel,
                               const ANGLE  & directOpponentANG,
                               const int    & directOpponentAge,
                               const int    & directOpponentAgeVel,
                               const int    & directOpponentAgeAng
                            );
    static
      OpponentAwarePosition getStaminaSavingCoverPosition();
    static
      OpponentAwarePosition getStaminaSavingCoverPosition( PPlayer directOpponent );
    static
      OpponentAwarePosition getBreakthroughPlayerStopPosition(PPlayer btp,
                                                         bool  areWeAttacking);
    static
      OpponentAwarePosition getRunningDuelTargetPosition(PPlayer rpl);
    static
      OpponentAwarePosition getStrategicSweeperPosition(PPlayer btp,
                                                         bool  areWeAttacking);
    static 
      PPlayer  getDirectOpponent(int number);
    static 
      PPlayer  getResponsiblePlayerForOpponent(int number);
    static 
      void     setOpponentAwareRelevantTeammates();
    static
      bool     getDistanceFromPlayerToDirectOpponent ( PPlayer teammate,
                                                       double & distance );
    static
      bool     getDirectOpponentHeadStartToMyGoal    ( PPlayer teammate,
                                                       double & distance );
    static
      PlayerSet   getViableSurveillanceOpponentsForPlayer(PPlayer teammate);
      
    //methods for attack positioning
    static void   setInterceptInformation(tInterceptInformation & icptInfo);
    static double  getBestOfferingPosition(Vector & offeringPosition);
    static double  getBestOfferingPosition(PPlayer  teammate,
                                           Vector & offeringPosition);
    static double  evaluateOfferingPosition(PPlayer  teammate,
                                            Vector & offeringPosition,
                                            double  * criteriaValues);
    static double  evaluatePenaltyAreaOfferingPosition
                                         (PPlayer  teammate,
                                          Vector & offeringPosition,
                                          bool debug=false );
    static double  evaluateAnticipatingOfferingPosition
                                         (PPlayer  teammate,
                                          Vector & offeringPosition,
                                          double  * criteriaValues,
                                          Vector   passStartPos );
    static double  evaluateOfferingPositionRegardingXPosition
                                         (PPlayer teammate,
                                          Vector & offeringPosition,
                                          Vector   passStartPos );
    static double  evaluateOfferingPositionRegardingBallDistance
                                         (PPlayer teammate,
                                          Vector & offeringPosition,
                                          Vector   passStartPos,
                                          bool     weAreInPenaltyArea=false );
    static double  evaluateOfferingPositionRegardingDist2DirectOpp
                                         (PPlayer teammate,
                                          Vector & offeringPosition );
    static double  evaluateOfferingPositionRegardingPasswayClearance
                                         (PPlayer teammate,
                                          Vector & offeringPosition,
                                          Vector & v,
                                          bool debug = false );
    static double  evaluateOfferingPositionRegardingStamina
                                         (PPlayer teammate,
                                          Vector & offeringPosition );
    static double  evaluateOfferingPositionRegardingDistanceToOpponents
                                         (PPlayer teammate,
                                          Vector & offeringPosition );
    static double  evaluateOfferingPositionRegardingDistanceFromPenaltyAreaHomePos
                                         (PPlayer teammate,
                                          Vector & offeringPosition );
    //helper methods for attack positioning
    static double computeOpponentDangerWhenOffering
                                         ( Vector ballIcptPosition,
                                           int    stepsToInterception,
                                           Vector offeringPosition,
                                           PPlayer consideredOpponent,
                                           bool debug = false );
    static double computeTeammateDangerWhenOffering
                                         ( Vector ballIcptPosition,
                                           int    stepsToInterception,
                                           Vector offeringPosition,
                                           PPlayer consideredTeammate,
                                           bool debug = false );
    static bool  shallIExcludeOfferPositionFromConsideration
                                         ( PPlayer teammate,
                                           Vector targetPosition,
                                           Vector passStartPosition,
                                           PPlayer passStartPlayer = NULL);
    static Vector calculateExpectedPassStartPosition
                                         ( PPlayer & nextBallPossessingTeammate);
    static Vector calculateExpectedPassStartPositionForPenaltyArea
                                         ( PPlayer   consideredTeammate,
                                           PPlayer & nextBallPossessingTeammate,
                                           bool debug=false);
    static Vector calculateExpectedPassStartPositionForAnticipatingOffering
                                         ( PPlayer consideredTeammate, 
                                           PPlayer & passPlayingTeammate, 
                                           bool debug=false );
                  
    static Vector getPenaltyAreaOfferingHomePosition(int number);
    
    static bool  isPassWayFreeEnoughForThisPosition
                                         ( PPlayer teammate,
                                           Vector & offeringPosition,
                                           Vector & passStartPosition,
                                           double  & freenessScore,
                                           bool debug,
                                           bool passStartPositionIsBallPosition=false,
                                           bool useHandedOverPassStartPosition=false);
    static Vector getAttackerPositionInOpponentBackFour(double xOffset = -1.0);
    static double  getBestPenaltyAreaOfferingPosition(PPlayer  teammate,
                                                     Vector & offeringPosition);
    static double  getBestAnticipatingOfferingPosition(PPlayer teammate,
                                                      Vector & offeringPosition,
                                                      PPlayer& passGivingTeammate );
                                                      
    //NEW VARIABLES & METHODS 2007
    private:
      static double cvCurrentCriticalOffsideLine;
      static double cvOffsideLineRelaxation;
    static 
      Vector getAttackScoringAreaDefaultPositionForInterceptingLeftAttacker
                                   ( int    myCurrentRole,
                                     Vector ourTeamNextBallPossessingPoint );
    static 
      Vector getAttackScoringAreaDefaultPositionForInterceptingCenterAttacker
                                   ( int    myCurrentRole,
                                     Vector ourTeamNextBallPossessingPoint );
    static 
      Vector getAttackScoringAreaDefaultPositionForInterceptingRightAttacker
                                   ( int    myCurrentRole,
                                     Vector ourTeamNextBallPossessingPoint );
    static 
      Vector getAttackScoringAreaDefaultPositionForInterceptingLeftMidfielder
                                   ( int    myCurrentRole,
                                     Vector ourTeamNextBallPossessingPoint );
    static 
      Vector getAttackScoringAreaDefaultPositionForInterceptingCenterMidfielder
                                   ( int    myCurrentRole,
                                     Vector ourTeamNextBallPossessingPoint );
    static 
      Vector getAttackScoringAreaDefaultPositionForInterceptingRightMidfielder
                                   ( int    myCurrentRole,
                                     Vector ourTeamNextBallPossessingPoint );
    public:
    static Vector getAttackScoringAreaDefaultPositionForRole
                                   ( int    myCurrentRole,
                                     Vector ourTeamNextBallPossessingPoint,
                                     int    roleOfNextBallPossessingTeammate,
                                     double  criticalOffsideLine );
        
};



#endif //_POSITIONING_H_
