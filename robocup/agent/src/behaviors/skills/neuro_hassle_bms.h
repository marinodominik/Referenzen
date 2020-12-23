#ifndef _NEURO_HASSLE_BMS_H_
#define _NEURO_HASSLE_BMS_H_

#include "base_bm.h"
#include "angle.h"
#include "Vector.h"
#include "n++.h"
#include "basic_cmd_bms.h"
#include "tools.h"
#include "valueparser.h"
#include "log_macros.h"

#include <vector>
#include <cstring>
#include <limits.h>
#include "../../basics/Cmd.h"

#define IDENTIFIER_LENGTH 200 
#define REQUIRED_TACKLE_SUCCESS_PROBABILITY_FOR_SUCCESS 0.75

using namespace std;

class NeuroHassle: public BodyBehavior 
{
  private:
  
    //////////////////////////////////////////////////////////////////////////
    //==INNER CLASS Statistics================================================
    class Statistics
    {
      private:
        int             ivSequenceCounter;
        vector<int>     ivVSequenceSuccesses;
        vector<float>   ivVSequenceCosts;
        float           getGlidingWindowSuccessFailuresAborts(int winSize, 
                                                              int sfa);
      public:
        Statistics();
        float           getGlidingWindowSuccess(int winSize);
        float           getGlidingWindowFailures(int winSize);
        float           getGlidingWindowAborts(int winSize);
        float           getGlidingWindowRealFailures(int winSize);
        float           getGlidingWindowCosts(int winSize);
        float           getAverageSuccess();
        float           getAverageFailures();
        float           getAverageAborts();
        float           getAverageRealFailures();
        float           getAverageCosts();
        void            addEntry(int success, float costs);
        int             getSize();
        void            writeOut(ostream &stream = std::cout);
        void            clear();
    };
    //////////////////////////////////////////////////////////////////////////
  

    //////////////////////////////////////////////////////////////////////////
    // INNER CLASS STATE /////////////////////////////////////////////////////
    class State
    {
      public:
        float  ivOpponentInterceptAngle;
        float  ivOpponentInterceptDistance;
        float  ivMyVelocityX;
        float  ivMyVelocityY;
        float  ivMyAngleToOpponentPosition;
        float  ivBallPositionX;
        float  ivBallPositionY;
        float  ivOpponentBodyAngle;
        float  ivOpponentAbsoluteVelocity;
        //not used for learning!
        float  ivRealBallVelocityX;
        float  ivRealBallVelocityY;
        float  ivMyPositionX;
        float  ivMyPositionY;
        float  ivOpponentHeadStart;

      public:
        State();
        State(const State & s);
        void    setMeFromAnotherState(const State & s);
        void    setMeAccordingToThisWorldInformation
                  ( 
                    float  oppPosX,
                    float  oppPosY,
                    float  oppVelAbs,
                    ANGLE  oppANGLE,
                    float  myPosX,
                    float  myPosY,
                    float  myVelX,
                    float  myVelY,
                    ANGLE  myANGLE,
                    float  ballPosX,
                    float  ballPosY,
                    float  ballVelX,
                    float  ballVelY
                  );
        string  toString();
        string  toShortString();
        //void    exportMeIntoMyState(MyState & m);
    };
    //////////////////////////////////////////////////////////////////////////
  
    //////////////////////////////////////////////////////////////////////////
    // INNER CLASS ACTION ////////////////////////////////////////////////////
    class Action
    {
      public:
        typedef enum {KICK, TURN, DASH, NIL} LearnActionTypes;
      public:
        LearnActionTypes          ivType;
        short                     ivDashPowerIndex;
        float                     ivRealDashPower;
        short                     ivTurnAngleIndex;
        float                     ivRealTurnAngle;
        bool                      ivExplorationFlag;
        bool                      ivIsSuccessAction;
        float                     ivValue;
      public:
        //Der Aktionenraum ...
        static std::vector<int> cvVDashPowerDiscretization;
        static const int cvpDashPowerDiscretization[];
        static std::vector<int> cvVTurnAngleDiscretization;
        static const int cvpTurnAngleDiscretization[];
        //Default Constructor
        Action();
        void  performMeOnThisState(
                                    NeuroHassle::State current, 
                                    NeuroHassle::State * successor,
                                    bool debug );
        void incorporateMeIntoCommand(Cmd * cmd);
        void setMeFromCommand(Cmd &cmd, bool neuro=true);
        void setMeRandomly();
        void setNullAction();
        void setMeFromAction(Action a);        
        int  getDashPower();
        int  getTurnAngle();
        void setTurnAngle(float a);
        void setExplorationFlag(bool flag);
        bool getExplorationFlag();
        void setIsSuccessAction(bool flag);
        bool getIsSuccessAction();
        void setType(LearnActionTypes type);       
        LearnActionTypes getType(); 
        void setValue(float aValue);
        float getValue();
        string toString();
    };  
    //////////////////////////////////////////////////////////////////////////
  
  

  public:
  
    //Statistiken
    static float              cvBestTimeAborts;
    static Statistics         cvLearnStatistics;
    static Statistics         cvEvaluationStatistics;
    static char               cvLearningProtocolFileName[IDENTIFIER_LENGTH];
    static ofstream           cvLearningProtocolFileHandle;
    static char               cvEvaluationProtocolFileName[IDENTIFIER_LENGTH];
    static ofstream           cvEvaluationProtocolFileHandle;
    
    static int                cvOperationMode;
    
    static vector<bool>       cvVSuccessesOrFailures; 
    static vector<float>      cvVSuccessPercentage;
    static vector<float>      cvVSequenceCosts;
    static vector<float>      cvVAverageSequenceCosts;
    static vector<float>      cvVEvaluationSueccessPercentages;
    int                       ivEvaluationSequenceCounter;
    int                       ivSuccessfulEvaluationSequenceCounter;
  
    //andere Klassen-/Instanzenvariablen
    static bool   cvEvaluationMode;
    static int    cvFlushLearnStatisticsAfterAsManySequences;
    static int    cvNumberOfNeuralNetTrainingExamples;
    static float  cvRewardForIntermediateState;
    static int    cvLearnSequenceCounter;
    static int    cvNumberOfTrainingExamplesForNetworkTraining;
    static Net  * cvpNeuralNetwork;
    static char   cvNeuralNetworkFilename[IDENTIFIER_LENGTH];
    
    Vector        ivBallVector;
    State         ivCurrentState;
    float         ivDiscountFactor;
    bool          ivHaveCollectedEnoughTrainingExamplesForNextTrainingCycle;
    float         ivEpisodeInitialGoalDist;
    long          ivLastTimeInvoked;
    int           ivNumberOfTrainingExamplesForNetworkTraining;
    PPlayer       ivpHassleOpponent;
    State         ivPredictedSuccessorState;
    bool          ivStarted;
    int           ivTrainNeuralNetworkCounter;
    vector<Action>ivVActionSequence;
    vector< pair<State,float> >      ivVBottomNeuralNetTrainingExamples;
    vector< pair<State,float> >      ivVCurrentNeuralNetTrainingExamples;
    vector< pair<State,float> >      ivVNeuralNetTrainingExamples;
    vector< pair<State,float> >      ivVTopNeuralNetTrainingExamples;
    vector<State> ivVStateSequence;
    int           ivUnExploitedEpisodeCounter;
   
    NeuroHassle();
    virtual ~NeuroHassle();
    void    analyseCompletedEvaluationSequence();
    int     classifySequenceAsSuccessOrFailure();
    int     classifyStateAsSuccessOrFailure(State s, 
                                            bool debug );
    void    decideForAnAction(Action * bestAction);
    bool    determineCurrentStateFromWorldModel();
    void    eraseTrainingExamples(bool remainTopTrainingExamples);
    bool    get_cmd(Cmd & cmd);
    bool    get_cmd_exploit(Cmd & cmd);
    float   getValueFromFunctionApproximator( State & state, 
                                              float * netInputs);
    void    handleLearningStatistics(int successCode);
    static 
      bool  init(char const * conf_file, int argc, char const* const* argv);
    bool    initNeuralNetwork();
    bool    isUnusableSequence();
    void    learningUpdate();
    float   neuralNetworkError();
    void    setNeuralNetInputFromState(State & s);
    void    setOpponent( PPlayer opp );
    void    setVirtualBall( Vector * ballPos );
    void    setValueForFunctionApproximator(float v,
                                            State &state, 
                                            float *netInputs);
    void    trainNeuralNetwork();

};


#endif
