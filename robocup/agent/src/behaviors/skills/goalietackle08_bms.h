#ifndef _GOALIETACKLE08_H_
#define _GOALIETACKLE08_H_

#include "mdp_info.h"
#include "log_macros.h"
#include "../base_bm.h"
#include "basic_cmd_bms.h"
#include "intercept_ball_bms.h"
#include <limits.h>


class GoalieTackle08 : public BodyBehavior 
{
  private:
    //class variables
    static bool         initialized;
    static const int    cvcFutureBallSteps = 20;
    static double       cvcRealCatchRadius;
    static const double cvcExpectedPerceptionNoise;//0.06
    static const int    cvcNumberOfModelledDashes = 21;
    static const int    cvcDASH = 0,
                        cvcDOUBLEDASHP = 1, cvcDOUBLEDASHM = 2,
                        cvcTURNtoBALL = 3,
                        cvcTURNandDASHP100 = 4,
                        cvcTURNandDASHP100andDASHP100 = 5,
                        cvcCATCHORTACKLENOW = 6;

    //data structure
    struct TackleAndCatchProbabilityInformationHolder
    {
      static const int cvcDOUBLEDASHPINDEX= GoalieTackle08::cvcNumberOfModelledDashes;
      static const int cvcDOUBLEDASHMINDEX= GoalieTackle08::cvcNumberOfModelledDashes+1;
      static const int cvcTURNtoBALLINDEX = GoalieTackle08::cvcNumberOfModelledDashes+2;
      static const int cvcTURNandDASHP100INDEX
                                          = GoalieTackle08::cvcNumberOfModelledDashes+3;
      static const int cvcTURNandDASHP100andDASHP100INDEX
                                          = GoalieTackle08::cvcNumberOfModelledDashes+4;
      static const int cvcCATCHORTACKLENOWINDEX   
                                          = GoalieTackle08::cvcNumberOfModelledDashes+5;
      static const int cvcEXTRAACTIONS = 6;
      static const int cvcNUMBEROFACTIONS
        = GoalieTackle08::cvcNumberOfModelledDashes + cvcEXTRAACTIONS;
      Vector ivMyNextPosAfterDashing[cvcNumberOfModelledDashes];
      Vector ivNextBallPosition;
      ANGLE  ivMyANGLEAfterTurningToBall;
      double ivCatchProbabilities[cvcNumberOfModelledDashes+cvcEXTRAACTIONS];
      double ivTackleProbabilities[cvcNumberOfModelledDashes+cvcEXTRAACTIONS];
      void   indexToAction(int index, int & action, int & actionPar);
      void   actionToIndex(int action, int actionPar, int &index);
      void   calculateTackleAndCatchProbabilityForIndex
             (int index, Vector myPos, ANGLE myANG, Vector ballPos,
              double discountFutureActions);
    };
    
    //instance variables
    bool   ivAggressiveCatchIsAdvisable;
    bool   ivAggressiveCatchIsAdvisableNextStep;
    int    ivBestCatchIndex;
    int    ivBestTackleIndex;
    Vector ivpFutureBallPositions[cvcFutureBallSteps];
    int    ivBallCrossesMyGoalLineInNSteps;
    bool   ivImmediateTacklingIsAdvisable;
    bool   ivTacklePreparationIsAdvisable;
    double ivMyDistanceToBall;
    double ivMyDistanceToBallNextStep;
    int    ivNumberOfStepsToCatchTheBall;
    TackleAndCatchProbabilityInformationHolder
           ivTackleAndCatchProbabilityInformation;
    bool   ivIsGoalieTackleSituation;
    
    //other behaviors used
    InterceptBall * ivpInterceptBallBehavior;
    BasicCmd      * ivpBasicCmdBehavior;

    ANGLE  calculateAggressiveCatchAngle();
    bool   calculatePreferredTackleANGLE( ANGLE & ang );
    int    evaluateCatchAngle( ANGLE & ang, bool debug = false );
    double evaluateAngularTackleDirection( double checkAngle );
    bool   isBallHeadingForGoal();
    double skalarprodukt(Vector v1, Vector v2);
    void   updateCatchAndTackleInformation();
    void   updateCatchingInterceptInformation( Cmd * cmd = NULL);
    void   updateDirectActionInformation();
    void   updateForCurrentCycle();
    void   updateFutureBallPositions();

  protected:

  public:

  static bool init( char const * conf_file, int argc, 
                    char const* const* argv );
  GoalieTackle08();
  virtual ~GoalieTackle08();
  void reset_intention();
  bool get_cmd(Cmd &cmd);
  bool isGoalieTackleSituation();
};

#endif //_GOALIETACKLE08_H_

