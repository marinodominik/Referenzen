/*
 * one_step_score_bms.h
 *
 *  Created on: 13.06.2016
 *      Author: tgabel
 */

#ifndef ONE_STEP_SCORE_BMS_H_
#define ONE_STEP_SCORE_BMS_H_

#include "base_bm.h"
#include "angle.h"
#include "Vector.h"
#include "intention.h"
#include "one_step_kick_bms.h"


class OneStepScore : public BodyBehavior
{
  private:
    OneStepKick   *ivpOneStepKick;
    ANGLE          ivShotAngle;
    Vector         ivShotTarget;
    static bool    cvInitialized;
    static const double cvcGoalPostDistance;
    static const double cvcGoalPostDistanceExtra;
    static const double cvcAggressivenes;
    void           calculateVelocityToTarget(Vector target, double& velAtStart, double& velAtTarget, int& steps);
    double         evaluateTarget( Vector target, bool extra=false );
    int            ivShotCnt;
    double         ivAverageShotDist;
  public:
    static const double cvcMinimalGoalDistance;
	OneStepScore();
	virtual ~OneStepScore();
    bool  get_cmd(Cmd &cmd);
    bool  isSuitableOneStepScoreSituation();
    bool  test_score_now( Intention& intention );
    static bool init(char const * conf_file, int argc, char const* const* argv);
};

#endif /* ONE_STEP_SCORE_BMS_H_ */
