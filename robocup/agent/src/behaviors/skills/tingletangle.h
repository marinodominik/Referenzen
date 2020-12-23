#ifndef _TINGLE_TANGLE_H
#define _TINGLE_TANGLE_H

// vim:ts=2:sw=2:fdm=syntax:fdl=0:

#include "basic_cmd_bms.h"
#include "one_step_kick_bms.h"
#include "ballzauber_bms.h"
#include "intercept_ball_bms.h"
#include "dribble_around06.h"

class TingleTangle:public BodyBehavior
{
	private:
		static 
		TingleTangle  *ivpInstance;

		BasicCmd      *ivpBasicCmd;
		OneStepKick   *ivpOneStepKick;
		Ballzauber    *ivpBallZauber;
		InterceptBall *ivpIntercept;

		Cmd           ivCachedCmd;
		bool          ivCachedResult;
		int           ivCacheTime;

		TingleTangle();

		Vector ivTarget;

		Ballzauber::TargetArea ivTargetArea;

		Quadrangle2d getTargetArea(const Vector&);
		bool isBallPosOKForAdvancing(const Vector&,const Vector&);

		bool getCmdGetBallInTargetAreaByDashing(Cmd& cmd,int=100);

		bool getCmdPrepare(Cmd&);

		bool getCmdBodyAngOK(Cmd&,int=100);

		bool getCmdAdvance(Cmd&,int=100);

		bool isBallPositionSafe(Vector);

		void fillDashArray(Vector,Vector,ANGLE,Vector*,int,double);

		Vector getBestDestOnMeInNSteps(const int&,const Vector&,const ANGLE&,const bool&,const bool&);

		Vector getKickDestForBallPosInNCycles(const Vector&, const int, bool&);


		DribbleAround06::PlayerState ivNextMeNA;
		DribbleAround06::BallState ivNextBallNA;

		
	public:
		static TingleTangle* getInstance();

		inline void   setTarget(const Vector& v){ ivTarget = v; }
		inline Vector getTarget()const{ return ivTarget; }

		void saveCmd(Cmd&,bool);

		bool isSafe();

		bool get_cmd(Cmd&);

		
		virtual ~TingleTangle();
		static bool init(char const * conf_file, int argc, char const* const* argv) {
				return(
					     BasicCmd::init(conf_file,argc,argv)
						&& Ballzauber::init(conf_file,argc,argv)
						&& OneStepKick::init(conf_file,argc,argv)
						&& InterceptBall::init(conf_file,argc,argv));
		}
};


#endif /* _TINGLE_TANGLE_H */
