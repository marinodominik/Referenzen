#ifndef _MOD_LEARNPOSITIONINGLEARNER_H_
#define _MOD_LEARNPOSITIONINGLEARNER_H_

#include "modules.h"
#include "../situation_handling/state.h"

class ModLearnPositioningLearner
  : public AbstractModule
{
	private:
  //CLASS VARIABLES
    static const int  cvMaxStringLength = 512;
  
  //INSTANCE VARIABLES
    Episode  ivEpisode;
  
    //Ausgabekontrolle
    bool      ivWriteProtocol;
    //Zeitangaben (fuer Statistiken)
    int       ivSequenceStartTime;
    //Zaehler
    int       ivCurrentSequenceLength;
    int       ivLastPlaymode;
    int       ivSuccessEpisodesCounter;
    int       ivFailureEpisodesCounter;
    int       ivIgnoreEpisodesCounter;
    int       ivRequiredSuccessEpisodesForLearning;
    int       ivRequiredTotalEpisodesForLearning;
    //Dateinamen    
    char      ivSituationFileName[cvMaxStringLength];
    char      ivEvaluationSituationFileName[cvMaxStringLength];
    char      ivProtocolFileName[cvMaxStringLength];
    char      ivEpisodeFileName[cvMaxStringLength];
    //Dateihenkel
    FILE *    ivpEpisodeFileHandle;
    //Episodensammlung-Spezifisches
    float     ivContinuingEpisodeUsageShare;
    //Reward-Spezifisches
    double    ivBasicRewardFactor;
    double    ivFinalRewardSuccess;
    double    ivFinalRewardFailure;
    double    ivRewardWeightAverageGonePlayerWaysInMeters;
    double    ivRewardWeightAverageDistanceToDirectOpponents;
    double    ivRewardWeightPenaltyIfBallInKickrange;
    double    ivRewardWeightAverageDistanceToBall;
    bool      ivSuccessIffGoal;
    bool      ivSuccessIffGoalApproached;
    double    ivGoalApproachStopThreshold;
    //METHODEN
    void      allPlayersAliveCheck();
    double    calculateSuggestedImmediateReward();
    double    calculateSuggestedFinalReward();
    void      checkForFinishingTraining();
    bool      classifyCompletedEpisode();
    void      writeOutEpisode(int success);
      
	protected:
	
	public:
  
    ModLearnPositioningLearner();
    virtual ~ModLearnPositioningLearner() {};
  
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
