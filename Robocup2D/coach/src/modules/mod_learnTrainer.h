#ifndef _MOD_LEARNTRAINER_H_
#define _MOD_LEARNTRAINER_H_

#include "modules.h"
#include "../situation_handling/situationcollection.h"
#include "../situation_handling/situationtable.h"

#define MOD_LEARNTRAINER_STATE_CONTINUE                          0
#define MOD_LEARNTRAINER_STATE_TEAM_HAS_BALL                     1
#define MOD_LEARNTRAINER_STATE_OP_HAS_BALL                       2
#define MOD_LEARNTRAINER_STATE_TIME_OVER_CURRENT_SEQUENCE        3
#define MOD_LEARNTRAINER_STATE_GAME_INTERRUPTED                  4
#define MOD_LEARNTRAINER_STATE_BALL_LEFT_KICKRANGE               5
#define MOD_LEARNTRAINER_STATE_PASS_SUCCESS                      6
#define MOD_LEARNTRAINER_STATE_BALL_INTERCEPTED                  7
#define MOD_LEARNTRAINER_STATE_ENOUGH_PT_RESEEDINGS              8
#define MOD_LEARNTRAINER_STATE_OVERALL_TIMEOUT                   9
#define MOD_LEARNTRAINER_STATE_EVALUATION                       10
#define MOD_LEARNTRAINER_STATE_GOAL_APPROACHED                  11
#define MOD_LEARNTRAINER_STATE_BALL_TACKLABLE                   12
#define MOD_LEARNTRAINER_STATE_OPPONENT_GOAL_SHOT               13
#define MOD_LEARNTRAINER_STATE_BALL_TOO_DISTANT                 14
#define MOD_LEARNTRAINER_STATE_BALL_APPROACHED                  15

#define MOD_LEARNTRAINER_STATE_NEAR_GOAL                       101
#define MOD_LEARNTRAINER_STATE_GOAL                            102
#define MOD_LEARNTRAINER_STATE_GOALIE_CATCHED                  103
#define MOD_LEARNTRAINER_STATE_ILLEGAL_GOAL                    104

#define KICK_RANGE											1.085



class ModLearnTrainer
  : public AbstractModule
{
	private:
	
	//CLASS VARIABLES
	
	  static const int	cvMaxStringLength = 512;
	
	//INSTANCE VARIABLES
	
	  //Kontrollvariablen 
	  bool			ivDoLoopOverSituations;
	  bool			ivAutoStart;
    bool      ivCreateRandomSituations;
	  int			  ivNumberOfSituations;
	  int			  ivLearningMode;
	  //Start- und Stoppkriterien
	  bool			ivStopSequenceIfOppBall;
	  bool			ivStopSequenceIfTeamBall;
    bool      ivStopSequenceIfBallNotKickable;
  	bool      ivStopSequenceIfBallOutside;
	  bool			ivStopSequenceIfBallIntercepted;
	  bool			ivStopSequenceIfGoal;
    bool      ivStopSequenceIfTeamOffside;
    bool      ivStopSequenceIfBallTooDistant;
    float     ivBallDistanceStopThreshold;
    bool      ivStopSequenceIfGoalApproached;
    float     ivGoalApproachStopThreshold;
    bool      ivStopSequenceIfBallApproached;
    float     ivBallApproachStopThreshold;
    bool      ivStopSequenceIfBallTacklable;
    float     ivDesiredTackleSuccessProbability;
    bool      ivStopSequenceIfOpponentGoalShot;
	  bool			ivStopSequenceAfterMaxSequenceRunningTime;
	  bool			ivStartWithCornerKickLeft;
	  int		    ivStopAfterAsManySequenceRepetitions;
	  int       ivMaxSequenceRunningTime;
	  int 			ivStopAfterAsManyPlayerTypeReseedings;
	  //Evaluation
	  bool 			ivDoEvaluations;
    int       ivNumberOfEvaluationSituations;
    int       ivDoEvaluationAfterAsManyTrainSequences;
    bool			ivStartEvaluation;
    int 			ivEvaluationSequenceCounter;
      //Ausgabekontrolle
	  bool			ivWriteProtocol;
	  int			  ivStatisticsFrequency;
      //Zeitangaben (f?r Statistiken)
	  int	      ivLastProtocolledAt;
	  int       ivSequenceStartTime;
    int       ivWaitTimeAfterSequenceStart;
	  int       ivStatisticsTotalTime;
    int       ivTakenExtraTime;
    int       ivExtraTime;
	  //Zaehler
	  int 			ivWaitForOneCycle;
    int       ivSequenceStartWaitingCounter;
	  int 			ivPlayerType;
	  int       ivSequenceLoopCounter;
	  int	      ivSequenceCounter;
	  int	      ivReseedCounter;
	  int	      ivCurrentSequenceLength;
	  int	      ivStatisticsMaxSequenceLength;
      //Dateinamen	  
	  char  		ivSituationFileName[cvMaxStringLength];
	  char  		ivEvaluationSituationFileName[cvMaxStringLength];
	  char 			ivProtocolFileName[cvMaxStringLength];
	  //Dateihenkel
	  FILE * 		ivpProtocolFileHandle;
	  //Sammlung von Startsituationen
	  SituationCollection *	
	                ivpSituations;
	  SituationCollection *	
	                ivpEvaluationSituations;
	  //Variablen fuer Praktikumsaufgaben
	  Vector        ivLastPositionAPlayerToucedBall;
	  long          ivLastTimeAPlayerTouchedBall;
	
	//METHODEN
	  void 			startNewSequence();
    void      restartCurrentSequence();
	  void 			startNewEvaluationSequence();
	  void			writeOutSituationProtocol(
	                  const int terminated = MOD_LEARNTRAINER_STATE_CONTINUE);
	  int			checkSequenceStopCriterion();
	
	protected:
	
	
	public:
	//STANDARDMETHODEN eines Moduls, geerbt vom Template
	  bool init(int argc,char **argv);             /** init the module */
	  bool destroy();                              /** tidy up         */
	  
	  bool behave();                               /** called once per cycle, after visual update   */
	  bool onRefereeMessage(bool PMChange);        /** called on playmode change or referee message */
	  bool onHearMessage(const char *str);         /** called on every hear message from any player */
	  bool onKeyboardInput(const char *str);       /** called on keyboard input                     */
	  bool onChangePlayerType(bool,int,int=-1);    /** SEE mod_template.c!                          */
	  
	  static const char modName[];                 /** module name, should be same as class name    */
	  const char *getModName() {return modName;}   /** do not change this!                          */

};

#endif
