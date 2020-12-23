// vim:ts=2:sw=2:ai:fdm=marker:fml=3:filetype=cpp:fmr=>>>,<<<
#include "tingletangle.h"
#include "log_macros.h"
#include <cmath>

// Log macros >>>1
#if 1   /* 1: debugging; 0: no debugging */
#define POL(XXX)   LOG_POL(0,<<"TingleTangle: "<<XXX)
#define POL2(XXX)  LOG_POL(1,<<"TingleTangle: "<<XXX)
#define DRAW(XXX)  LOG_POL(0,<<_2D<<XXX)
#define DRAW2(XXX) LOG_POL(1,<<_2D<<XXX)
#else
#define POL(XXX) 
#define POL2(XXX) 
#define DRAW(XXX) 
#define DRAW2(XXX) 
#endif
#define V(X) " "<< #X << "="<<X
// Drawing macros >>>1
#define MARK_PPOS(P,C) DRAW(C2D((P).getX(),(P).getY(),WSinfo::me->kick_radius,#C));
#define MARK_POS(P,C) DRAW(C2D((P).getX(),(P).getY(),0.3,#C));
#define MARK_BALL(B,C) DRAW(C2D(B.pos.getX(),B.pos.getY(),0.3,#C));
#define MARK_PPLAYER(P,C) DRAW(C2D(P->pos.getX(),P->pos.getY(),P->kick_radius,#C));
// Standard macros >>>1
#define SGN(X) ((X<0)?-1:1)
#define SQR(X) (X * X)

// skill constants >>>1
#define MAX_ANGLE_TO_DEST DEG2RAD(10)

// constructor >>>1
TingleTangle::TingleTangle(){ 
	ivpBasicCmd    = new BasicCmd;
	ivpBallZauber  = new Ballzauber;
	ivpOneStepKick = new OneStepKick;
	ivpIntercept   = new InterceptBall;
	ivTarget       = HIS_GOAL_CENTER;
}

// get_cmd >>>1
bool TingleTangle::get_cmd(Cmd & cmd){
	if(ivCacheTime != WSinfo::ws->time)
		isSafe();
	cmd = ivCachedCmd;
	return ivCachedResult;
}

// isSafe >>>1
bool TingleTangle::isSafe(){
	if(ivCacheTime == WSinfo::ws->time)
	{
	    POL("TT: Just returning the cached result which is "<<ivCachedResult);
		return ivCachedResult;
	}
	bool safe = false;
	Cmd cmd;

	if(!WSinfo::is_ball_kickable()){
		bool ret = ivpIntercept->get_cmd(cmd);
		saveCmd(cmd,ret);
		return ret;
	}

	ivNextBallNA.setAssumeNoAction(WSinfo::ball);
	ivNextMeNA.setAssumeNoAction(WSinfo::me,ivNextBallNA.pos);

	ANGLE toBallNextStep;
	Vector target  = ivNextBallNA.pos;
	target        -= ivNextMeNA.pos;
	toBallNextStep = target.ARG() - ivNextMeNA.ang;

	ANGLE aToBallGlobal = (ivTarget-WSinfo::me->pos).ANGLE_to(WSinfo::ball->pos-WSinfo::me->pos);
	ivTargetArea   = (aToBallGlobal.get_value_mPI_pPI()>0)?
		 Ballzauber::TA_RIGHT_BEHIND
		:Ballzauber::TA_LEFT_BEHIND;

	ANGLE aToTarget = Tools::my_angle_to(ivTarget);
	bool bodyAngleOK  = fabs(aToTarget.get_value_mPI_pPI())<MAX_ANGLE_TO_DEST;
	
	if(bodyAngleOK){
		if(getCmdBodyAngOK(cmd,100)){
			POL("getCmdBodyAngOK -forward- successful");
			safe = true;
		}
		else if(getCmdBodyAngOK(cmd,-100)){
			POL("getCmdBodyAngOK -backward- successful");
			safe = true;
		}
	}else if(getCmdPrepare(cmd)){
		POL("Preparing ball");
		safe = true;
	}

	saveCmd(cmd,safe);

	return safe;
}

bool TingleTangle::getCmdPrepare(Cmd&cmd){
	if(ivpBallZauber->get_cmd(cmd,(ivTarget-WSinfo::me->pos).ARG(), ivTargetArea)){
		POL("getCmdPrepare: Ballzauber says it can do that.");
		return true;
	}
	Ballzauber::TargetArea ta = (ivTargetArea==Ballzauber::TA_LEFT_BEHIND)?
		Ballzauber::TA_RIGHT_BEHIND
		:Ballzauber::TA_LEFT_BEHIND;
	if(ivpBallZauber->get_cmd(cmd,(ivTarget-WSinfo::me->pos).ARG(), ta)){
		POL("getCmdPrepare: Ballzauber says it can do (other target area).");
		return true;
	}
	POL("getCmdPrepare: ballzauber fails me, bailing out.");
	return false;
}

Quadrangle2d TingleTangle::getTargetArea(const Vector& mpos){
		double distNear = WSinfo::me->radius + ServerOptions::ball_size;
		double distFar  = .9*WSinfo::me->kick_radius;
		double minDistToKickRad = .2*WSinfo::me->kick_radius;

		double addAng = (ivTargetArea == Ballzauber::TA_LEFT_BEHIND)?-90:90;
		Vector nearBasePoint;
		nearBasePoint.init_polar(distNear,
														 ANGLE(DEG2RAD(addAng)) + WSinfo::me->ang);
		Vector distantBasePoint(nearBasePoint);
		distantBasePoint.normalize(distFar);
		double alpha     = acos(distFar/WSinfo::me->kick_radius);
		double farWidth  = 2.*WSinfo::me->kick_radius * sin(alpha) - 2*minDistToKickRad;
					alpha     = acos(distNear/WSinfo::me->kick_radius);
		double nearWidth = 2.*WSinfo::me->kick_radius * sin(alpha) - 2*minDistToKickRad;
		nearBasePoint    += mpos;
		distantBasePoint += mpos;
		return Quadrangle2d( distantBasePoint, nearBasePoint, farWidth, nearWidth );
}

bool TingleTangle::isBallPosOKForAdvancing(const Vector& mpos,const Vector &bpos){
	  Quadrangle2d q2d = getTargetArea(mpos);
		return q2d.inside(bpos);
}

bool
TingleTangle::isBallPositionSafe( Vector pos )
{
  PlayerSet relevantOpponents = WSinfo::valid_opponents;
  relevantOpponents.keep_players_in_circle( ivNextMeNA.pos, 4.0 );
	if(relevantOpponents.num==0)
		return true;
	if(relevantOpponents.num==1)
	{
	    int minInact, maxInact;
        WSinfo::get_player_inactivity_interval( relevantOpponents[0], minInact, maxInact );
        if (minInact > 0) return true;
        if (   WSinfo::me->pos.getX() - WSinfo::my_team_pos_of_offside_line() > 20.0
            && (minInact+maxInact)/2 > 0 )
          return true;

		DribbleAround06::PlayerState p;

		p.setAssumeNoAction(relevantOpponents[0],pos);
		if(p.reachesPos) return false;
		if(0.8<Tools::get_tackle_success_probability(p.pos,pos,p.ang.get_value()))
			return false;

		p.setAssumeToPos(relevantOpponents[0],pos);
		if(p.reachesPos) return false;
		if(0.8<Tools::get_tackle_success_probability(p.pos,pos,p.ang.get_value()))
			return false;

		p.setAssumeToPos(relevantOpponents[0],WSinfo::me->pos);
		if(p.pos.distance(pos)<p.kick_radius+.1) return false;
		if(0.8<Tools::get_tackle_success_probability(p.pos,pos,p.ang.get_value()))
			return false;
		return true;
	}
  return 
    Tools::is_ballpos_safe( relevantOpponents, pos, true ); //consider_tackles
}

bool TingleTangle::getCmdBodyAngOK(Cmd& cmd,int speed){
	
	bool ballPosOKForAdvancing = isBallPosOKForAdvancing(WSinfo::me->pos,WSinfo::ball->pos);
	bool nextBallSafe = isBallPositionSafe(ivNextBallNA.pos);
	bool canGetBallInTargetAreaByDashing = false;

	POL(V(nextBallSafe));
	if(nextBallSafe){
		canGetBallInTargetAreaByDashing = getCmdGetBallInTargetAreaByDashing(cmd,speed);
		POL(V(canGetBallInTargetAreaByDashing));
		if(canGetBallInTargetAreaByDashing)
			return true;
	}

	if(!ballPosOKForAdvancing){
		// dashing did not help, ball still not in target area
		// let ballzauber do the hard stuff
		if(getCmdPrepare(cmd)){
			POL("getCmdBodyAngOK: Let Ballzauber prepare ball.");
			return true;
		}
		POL("getCmdBodyAngOK: Ballzauber cannot prepare ball.");
		return false;
	}

	// ball pos is now OK for advancing.
	// dashing is not an option.
	// next ball might be unsafe.
	
	if(getCmdAdvance(cmd,speed)){
		POL("getCmdBodyAngOK: advancing.");
		return true;
	}

	// next ball probably not safe.
	// change sides.
	Ballzauber::TargetArea ta = (ivTargetArea==Ballzauber::TA_LEFT_BEHIND)?
		 Ballzauber::TA_RIGHT_BEHIND
		:Ballzauber::TA_LEFT_BEHIND;
	if(ivpBallZauber->get_cmd(cmd,(ivTarget-WSinfo::me->pos).ARG(), ta)){
		POL("getCmdBodyAngOK: ballpos ok for advancing, maybe unsafe, ballzauber says it can change sides.");
		return true;
	}
	return false;
	
}

void TingleTangle::fillDashArray(
		Vector mpos, Vector mvel, ANGLE mang, Vector* dash, int num, double dashSpeed){
	Vector bpos2,bvel2;
	Vector mpos2,mvel2;
	ANGLE mang2;
	Cmd tmpCmd;
	ivpBasicCmd->set_dash(dashSpeed);
	ivpBasicCmd->get_cmd(tmpCmd);

	for(int i = 0 ; i<num; i++){ 
		Tools::model_cmd_main(mpos,mvel,mang,Vector(0,0),Vector(0,0),tmpCmd.cmd_body,mpos2,mvel2,mang2,bpos2,bvel2,false);
		mpos = mpos2;
		mvel = mvel2;
		mang = mang2;
		dash[i] = mpos;
	}
}

bool TingleTangle::getCmdAdvance(Cmd&cmd,int speed){
	// preconditions: ball pos is ok for advancing
	// dashing is not an option, have to kick
	// next ball might be unsafe.
	static const int absolutelyMaxDashes = 10;
	static Vector       mPosAfterNDashes[absolutelyMaxDashes];
	static Quadrangle2d taAfterNDashes[absolutelyMaxDashes];
	int maxDashes = 5;
	fillDashArray(ivNextMeNA.pos,ivNextMeNA.vel,ivNextMeNA.ang,&mPosAfterNDashes[0],maxDashes,speed);
	for(int i=0;i<maxDashes;i++)
		taAfterNDashes[i] = getTargetArea(mPosAfterNDashes[i]);
		
	Vector dest,destN;
	int numDashes;
	double speedAdjust = (speed>0)?1.:2.;
	for(numDashes=maxDashes;numDashes>=0;numDashes--){
		POL2("Testing dash "<<numDashes);
		if(  speedAdjust*numDashes*(abs(speed)-30) +  0.3125*ServerOptions::stamina_max //TG09: alt: 1300 
           > WSinfo::me->stamina){
			POL2("getCmdAdvance: Dash "<<numDashes<<": cannot dash, stamina low");
			continue;
		}
		destN = getBestDestOnMeInNSteps(numDashes+1,mPosAfterNDashes[numDashes],WSinfo::me->ang,speed<0,true);
		
		// avoid field borders  >>>4
		if(destN.getX()>FIELD_BORDER_X-2 && destN.getX()>WSinfo::ball->pos.getX()){
			POL2("getCmdAdvance: Dash"<<numDashes<<": too close to field border X and getting closer to it");
			continue;
		}
		if(fabs(destN.getY())>FIELD_BORDER_Y-1.5 && fabs(destN.getY())>WSinfo::ball->pos.getY()){
			POL2("getCmdAdvance: Dash"<<numDashes<<": too close to field border Y");
			continue;
		}

		// where ball shall be after kick >>>4
		bool tooFar;
		dest = getKickDestForBallPosInNCycles(destN, numDashes+1, tooFar);
		if(!isBallPositionSafe(dest)){
			POL2("getCmdAdvance: Dash"<<numDashes<<" Ball pos not safe");
			continue;
		}

		// find ball trajectory >>>4
		Vector bpos = dest;
		Vector bvel = dest-WSinfo::ball->pos;
		bool looseBallAndShouldnt=false;
		int i;
		for(i = 0 ; i <= numDashes;i++){
			bvel  = ServerOptions::ball_decay * bvel;
			bpos  = bpos + bvel;
			bool haveBall    = taAfterNDashes[i].inside(bpos);
			if(!haveBall){
				looseBallAndShouldnt = true;
				break;
			}
		} 
		if(looseBallAndShouldnt){
			POL2("getCmdAdvance: Dash "<<numDashes<<": Ball "<<i<<" falls out of target area");
			continue;
		}
		
		break;
	}
	if(numDashes<0){
		POL("getCmdAdvance: No dash sequence successful.");
		return false;
	}
	ivpOneStepKick->kick_to_pos_with_initial_vel((dest-WSinfo::ball->pos).norm(),dest);
	if(ivpOneStepKick->get_cmd(cmd)){
		for(int i=0;i<=numDashes;i++){
			MARK_PPOS(mPosAfterNDashes[i],green);
		}
		MARK_POS(destN,green);
		MARK_POS(dest,blue);
		POL("getCmdAdvance: Advancing.");
		return true;
	}
	return false;
}

// getKickDestForBallPosInNCycles() >>>1
Vector TingleTangle::getKickDestForBallPosInNCycles(const Vector& target, const int cyc, bool& tooFar){
  Vector vToTarget = target - WSinfo::ball->pos;
	double maxVel = OneStepKick::get_max_vel_to_pos(target);
  float divBy = 1;
  for(int i=1;i<=cyc; i++)
    divBy += std::pow(ServerOptions::ball_decay,(double)i);
  vToTarget.normalize(vToTarget.norm()/divBy);
  if(vToTarget.norm() > maxVel){
    vToTarget.normalize(maxVel);
    tooFar = true;
  }else
    tooFar = false;
  return vToTarget + WSinfo::ball->pos;
}

Vector TingleTangle::getBestDestOnMeInNSteps(
		const int& stepsGone,
		const Vector&mPosInNSteps, const ANGLE& mAngInNSteps, const bool& backwards, const bool& goStraight){
	
	ANGLE toBall = Tools::my_angle_to(WSinfo::ball->pos);
	bool isBallOnLeftSide = toBall.get_value_mPI_pPI()>0;
	double addAng = isBallOnLeftSide?90:-90;

	Vector v;
	v.init_polar(.6*WSinfo::me->kick_radius, mAngInNSteps+ANGLE(DEG2RAD(addAng)));
	v+=mPosInNSteps;
	return v;
	
	Vector vToBall = WSinfo::ball->pos - WSinfo::me->pos;
	Vector ahead(mAngInNSteps);
	ahead.normalize(.1);
	Quadrangle2d q2d = getTargetArea(mPosInNSteps);
	Vector bpos = mPosInNSteps + vToBall;
	if(!backwards)
		while(true){
			bpos += ahead;
			if(!q2d.inside(bpos)){
				bpos -= ahead;
				break;
			}
		}
	else
		while(true){
			bpos -= ahead;
			if(!q2d.inside(bpos)){
				bpos += ahead;
				break;
			}
		}
	return bpos;
}

bool TingleTangle::getCmdGetBallInTargetAreaByDashing(Cmd& cmd,int speed){
	bool amIOutsideField = fabs(WSinfo::me->pos.getX())>FIELD_BORDER_X
		                   ||fabs(WSinfo::me->pos.getY())>FIELD_BORDER_Y;
	int canDash = 0;
	Cmd tmpCmd;
	Vector new_my_pos,new_my_vel,new_ball_pos,new_ball_vel;
	ANGLE new_my_ang;
	bool isOutsideField;
	for(int i = 70; i<=abs(speed); i+=10){
		tmpCmd.cmd_body.unset_lock();
		ivpBasicCmd->set_dash((speed<0)?-i:i);
		ivpBasicCmd->get_cmd(tmpCmd);
		Tools::model_cmd_main(WSinfo::me->pos,WSinfo::me->vel,WSinfo::me->ang,WSinfo::ball->pos,WSinfo::ball->vel,tmpCmd.cmd_body,new_my_pos,new_my_vel,new_my_ang,new_ball_pos,new_ball_vel,false);
		isOutsideField = fabs(new_my_pos.getX())>FIELD_BORDER_X
		               ||fabs(new_my_pos.getY())>FIELD_BORDER_Y;
		if(!amIOutsideField && isOutsideField) continue; // dont dash outside field
		if(isBallPosOKForAdvancing(new_my_pos,new_ball_pos))
			canDash = i;
	}
	if(canDash==0) {
		POL("getCmdGetBallInTargetAreaByDashing: Cannot dash.");
		return false;
	}
	POL("getCmdGetBallInTargetAreaByDashing: Success, dash="<<canDash);

	ivpBasicCmd->set_dash((speed<0)?-canDash:canDash);
	ivpBasicCmd->get_cmd(cmd);

	return true;
}

void TingleTangle::saveCmd(Cmd& cmd,bool res){
	ivCachedCmd = cmd;
	ivCachedResult = res;
	ivCacheTime = WSinfo::ws->time;
}

TingleTangle::~TingleTangle(){
	delete ivpBasicCmd;
	delete ivpOneStepKick;
	delete ivpBallZauber;
	delete ivpIntercept;
}

TingleTangle* TingleTangle::getInstance(){
	if(ivpInstance == NULL)
		ivpInstance = new TingleTangle;
	return ivpInstance;
}

TingleTangle* TingleTangle::ivpInstance = NULL;
