#ifndef _ONE_STEP_PASS_BMS_H_
#define _ONE_STEP_PASS_BMS_H_

/** 
 * OneStepPass ist ein Verhalten, dass schnelle Paesse mit nur einem
 * einzigen Kick-Befehl spielt.
 * 
 * Ziele:
 * - (sehr) schnelles Passspiel
 * - dem Gegner wenig Reaktionsmoeglichkeiten einraeumen
 * - Angriffe schnell zur Vollendung bringen
 * - seltener in "Stuck"-Situationen geraten
 * 
 * Verwendung:
 * - nur in Angriffssituationen 
 * - nur, wenn Ball gerade in Kickrange gelangt ist
 * - nicht, wenn Ball im Ballkontrollbereich eines weiteren Mitspielers
 * - nicht, wenn Ball im Ballkontrollbereich eines Gegenspielers
 * - nicht, wenn ein Tor geschossen werden kann 
*/


#include "../../basics/Cmd.h"
#include "../base_bm.h"
#include "angle.h"
#include "Vector.h"
#include "tools.h"
#include "macro_msg.h"
#include "valueparser.h"
#include "options.h"
#include "ws_info.h"
#include "log_macros.h"
#include "mystate.h"
#include "score05_sequence_bms.h"
#include "one_step_score_bms.h"

class OneStepPass : public BodyBehavior 
{
  private:
  
    static bool cvInitialized;
    static const int cvsMEMORYSIZE = 3;
    
    Vector  ivBallPosition;
    Vector  ivBallVelocity;
    bool    ivForeseeMode;
    int     ivLastSuccessfulForesee;
    bool    ivOffsideConqueringChance;
    Vector  ivMyPosition;
    ANGLE   ivMyBodyAngle;
    int     ivMemoryCounter;
    int     ivpMemoryRecentBallKickableTimePoints[ cvsMEMORYSIZE ];
    Score05_Sequence          * ivpScore05SequenceBehavior;
    OneStepScore              * ivpOneStepScoreBehavior;

    bool    canPlayerReachPositionIn1Step( PlayerSet relevantOpponents,
                                           Vector position,
                                           PPlayer & icptPlayer );
    void    evaluatePosition( Vector  consideredPosition,
                              PPlayer consideredTeammate,
                              double  checkVelocity,
                              int   & teammateScore,
                              int     logLevel=1);
    bool findPassTarget( Vector& target, 
                         int   & targetPlayerNumber,
                         double& resultingVelocity,
                         double& requiredKickPower,
                         ANGLE & requiredKickDirection  );
    bool findPassTargetForVirtualState
                       ( Vector  myVirtualPosition,
                         ANGLE   myVirtualBodyAngle,
                         Vector  ballVirtualPosition,
                         Vector  ballVirtualVelocity, 
                         Vector& target, 
                         int   & targetPlayerNumber,
                         double& resultingVelocity,
                         double& requiredKickPower,
                         ANGLE & requiredKickDirection  );
    bool isOneStepPassPossibleTo( Vector target, 
                                  double& resultingVelocity,
                                  double& requiredKickPower,
                                  ANGLE & requiredKickDirection );
  
  public:
    OneStepPass();
    virtual ~OneStepPass();

    bool  foresee(Cmd &cmd);
    bool  get_cmd(Cmd &cmd);
    bool  isSuitableOneStepPassSituation();
    bool  isSuitableOneStepPassSituationForExtremePanic();
    static bool init(char const * conf_file, int argc, char const* const* argv);
    void  setScoreBehaviors( BodyBehavior* scoreBehavior, BodyBehavior* oneStepScore );
};

#endif
