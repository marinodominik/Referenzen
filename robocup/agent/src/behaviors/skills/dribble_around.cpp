// vim:ts=2:sw=2:ai:fdm=marker:fml=3
// includes {{{1
#include "log_macros.h"
#include "../policy/positioning.h"
#include "ws_info.h"
#include "options.h"
#include "mdp_info.h"
#include "ws_memory.h"
#include <cmath>
#include "mystate.h"
#include "intention.h"

#include "dribble_around.h"

// Log macros {{{1
#if 0   /* 1: debugging; 0: no debugging */
#define POL(XXX)   LOG_POL(0,<<"DribbleAround: "<<XXX)
#define POL2(XXX)  LOG_POL(1,<<"DribbleAround: "<<XXX)
#define DRAW(XXX)  LOG_POL(0,<<_2D<<XXX)
#define DRAW2(XXX) LOG_POL(1,<<_2D<<XXX)
#else
#define POL(XXX) 
#define POL2(XXX) 
#define DRAW(XXX) 
#define DRAW2(XXX) 
#endif
// Drawing macros {{{1
#define MARK_POS(P,C) DRAW(C2D((P).x,(P).y,0.3,#C));
#define MARK_BALL(B,C) DRAW(C2D(B.pos.x,B.pos.y,0.3,#C));
#define MARK_STATE(P,C) DRAW(C2D(P.pos.x,P.pos.y,P.kick_radius,#C));\
  __ang.init_polar(P.kick_radius,P.ang.get_value());\
  __ang+=P.pos;\
	DRAW(L2D(P.pos.x,P.pos.y,__ang.x,__ang.y,#C));
#define MARK_PPLAYER(P,C) DRAW(C2D(P->pos.x,P->pos.y,P->kick_radius,#C));
// Standard macros {{{1
#define SGN(X) ((X<0)?-1:1)
#define SQR(X) (X * X)
#define QUBE(X) (X * X * X)
// skill constants {{{1
#define MAX_ANGLE_TO_DEST DEG2RAD(10)



// Constructing an object {{{1
DribbleAround::DribbleAround(){
	basic_cmd = new BasicCmd;
	dribble_straight = new DribbleStraight;
	go2pos = new NeuroGo2Pos;
	searchball = new SearchBall;
	onestepkick = new OneStepKick;
	intercept = new InterceptBall;
	holdturn = new OneTwoHoldTurn;
	dribbleTo = HIS_GOAL_CENTER;
	request = DAREQ_NONE;
	requestTime = 0;
	didDribble = false;
	neckReqSet=false;
	dribbleInsecure=false;
}
//DribbleAround::DribbleAround(const DribbleAround&){};
//DribbleAround DribbleAround::operator=(const DribbleAround&){};
DribbleAround* DribbleAround::myInstance = NULL;

DribbleAround* DribbleAround::getInstance() {
	if(myInstance==NULL){
		myInstance = new DribbleAround();
	}
	return myInstance;
}

// set a request
void DribbleAround::setRequest(DARequest dar){
	request = dar;
	requestTime = WSinfo::ws->time;
}

// get_dribble_straight_cmd() {{{1
bool DribbleAround::get_dribble_straight_cmd(Cmd& cmd){
	POL("Let dribble_straight handle situation, I cannot");
	return dribble_straight->get_cmd(cmd);
}

// getRelevantOpponent() {{{1
PPlayer DribbleAround::getRelevantOpponent(){ 
	static const float farAway     = 10;
	static const float closeBehind = 4;
	
  PlayerSet opps = WSinfo::valid_opponents;

	Vector toFarAway = dribbleTo - thisMe.pos;
	toFarAway.normalize(farAway);
	Vector justBehind = dribbleTo-thisMe.pos;
	justBehind.normalize(closeBehind);
	opps.keep_and_sort_closest_players_to_point(6,thisMe.pos);
	opps.keep_players_in_quadrangle(thisMe.pos-justBehind,thisMe.pos+toFarAway,8,20);

	// no opp
	if(opps.num==0) return NULL;
	
	PlayerState p;
	p.setAssumeToPos(opps[0],thisMe.pos);

	// one opp, reaches my position
	if(p.pos.distance(thisMe.pos)<thisMe.kick_radius + p.kick_radius)
		return opps[0];

	// check other opps, if they reach me, choose them
	for(int i=1;i<opps.num;i++){
		p.setAssumeToPos(opps[i],thisMe.pos);
		if(p.pos.distance(thisMe.pos)<thisMe.kick_radius + p.kick_radius)
			return opps[i];
	}

	// return closest.
	return opps[0];
}

/*
float pow(int a, float y, float z){
	if(a<=0) return 1;
	return (a==1)?y:pow(a-1,y*z,z);
}
*/

// getCmdKickToDest() {{{1
bool DribbleAround::getCmdKickToDest(Cmd& cmd, const Vector& dest, bool keepInKickRadius, bool force){
	POL2("In getCmdKickToDest()");
	//float distToDest = (nextMeNA.pos - dest).norm();
	cmd.cmd_body.unset_lock();
	
	if(keepBallSafe) 
		keepInKickRadius=true;
	
	// needed for onestepkick, definitely the angle to BALL, NOT me.
	ANGLE toDest = (dest - thisBall.pos).ARG();
	static const float initSpeedStart = 0.3;
	float initSpeed = initSpeedStart;
	float bestInitSpeedToDest = 0;
	static const float speedIncr = 0.1;
	float distDestToNewBall;
	float bestDistDestToNewBall = 1E6;
	float minDistToBall = 1.25*(ServerOptions::player_size + ServerOptions::ball_size);
	float kickMax = 2.0; // TODO: !? was 1.4
	bool thruOpp = false;
	bool inOpp = false;
	Cmd tmpCmd;
	Vector new_my_pos,new_my_vel,new_ball_pos,new_ball_vel;
	ANGLE new_my_ang;

	if(fabs(dest.getX())>FIELD_BORDER_X-.2
			|| fabs(dest.getY())>FIELD_BORDER_Y-.2){
		POL2("getCmdKickToDest: dest too close to field border");
		if(!force)
		  return false;
	}

	// can't keep ball at a position -- bug in onestepkick!?
	bool tooShortKick = dest.distance(thisBall.pos)<0.1;
	if(tooShortKick) {
		POL2("Too close!?");
		// if(!force) return false;
		toDest = thisBall.vel.ARG() - ANGLE(PI);
	}
	
	//bool keepInKickRadius = (nextMeNA.pos.distance(dest)<thisMe.kick_radius);
	POL2("Calculating target kick speed, keepInKickRadius = "<< keepInKickRadius);

	bool insecure;
	bool bestIsInsecure=false;
	Vector lastNewBallPos(nextBall.pos);
	bool distDecreasing=true;
	for( ; initSpeed<=kickMax; initSpeed += speedIncr){
		insecure=false;
		tmpCmd.cmd_body.unset_lock();
		onestepkick->set_state(thisMe.pos,thisMe.vel,thisMe.ang,thisBall.pos,thisBall.vel,Vector(-100,-100),ANGLE(0),0);
		onestepkick->kick_in_dir_with_initial_vel(initSpeed,toDest);
		onestepkick->get_cmd(tmpCmd);
		Tools::model_cmd_main(thisMe.pos,thisMe.vel,thisMe.ang,thisBall.pos,thisBall.vel,tmpCmd.cmd_body,new_my_pos,new_my_vel,new_my_ang,new_ball_pos,new_ball_vel,false);
		DRAW2(C2D(new_ball_pos.x,new_ball_pos.y,0.3,"green"));

		distDecreasing=
			initSpeed==initSpeedStart
			||lastNewBallPos.sqr_distance(dest)>new_ball_pos.sqr_distance(dest);
		lastNewBallPos = new_ball_pos;
		
		if(dest.sqr_distance(new_ball_pos)>SQR(.4)
				&& !distDecreasing
				&&  initSpeed>0.7
				&& thruOpp){
			POL2("Have to kick too far ahead to avoid opponent, stopping at initSpeed = "<<initSpeed);
			if(!force)
				break;
		}
		
		if(opp && new_ball_pos.distance(nextOppToMe.pos)<nextOppToMe.kick_radius){
			POL2("nextOppToMe would get ball, ruling out initSpeed = "<<initSpeed);
			thruOpp = true; // ball goes thru kick_radius of opp
			inOpp   = true;
			if(!force) continue;  
			insecure=true;
		}
		if(opp && 0.8<Tools::get_tackle_success_probability(nextOppToMe.pos,new_ball_pos,nextOppToMe.ang.get_value())){
			POL2("nextOppToMe would be able to tackle, ruling out initSpeed = "<<initSpeed);
			if(!force) continue;
			insecure=true;
		}
		if(opp && new_ball_pos.distance(opp->pos)<opp->kick_radius){
			POL2("opp would get ball, ruling out initSpeed = "<<initSpeed);
			thruOpp = true; // ball goes thru kick_radius of opp
			inOpp   = true;
			if(!force) continue;  
			insecure=true;
		}
		if(opp && new_ball_pos.distance(nextOppNA.pos)<nextOppNA.kick_radius){
			POL2("opponentNA would get ball, ruling out initSpeed = "<<initSpeed);
			if(!force) continue;  
			insecure=true;
		}
		// TODO: better: p!?
		float hisTackleProb = (opp?Tools::get_tackle_success_probability(nextOppToMe.pos,new_ball_pos,nextOppToMe.ang.get_value()):0);
		if(hisTackleProb>0.9){
			POL2("Player to dest could tackle, ruling out initSpeed = "<<initSpeed);
			if(!force) continue;  
			insecure=true;
		}
		//POL2("opponent could not get ball, allowing initSpeed = "<<initSpeed);
		inOpp   = false;

		distDestToNewBall = dest.distance(new_ball_pos);

		// more efficient, shouldnt get any better from here on
		if(distDestToNewBall>bestDistDestToNewBall && keepInKickRadius) break;

		if(distDestToNewBall<bestDistDestToNewBall                                // closer to dest
				&& (force || nextMeNA.pos.distance(dest)>minDistToBall)               // no collision
				&&(!keepInKickRadius                                                  // dont care for kick_radius
					||(new_ball_pos.distance(nextMeNA.pos)<thisMe.kick_radius))){  // or in kick_radius
			//POL2("New bestInitSpeedToDest="<<initSpeed);
			bestInitSpeedToDest = initSpeed;
			bestIsInsecure = insecure;
			bestDistDestToNewBall = distDestToNewBall;
		}
	}

	if(inOpp || bestInitSpeedToDest == 0){
		POL2("No valid kick found.");
		return false;
	}
	
	/* TODO: Was this necessary?
	if(keepInKickRadius && distOfBestToKickRadius<0.20){
		POL2("too close to kickable margin, reducing initSpeed");
		bestInitSpeedToDest *= 0.9;
	}else if(thruOpp){
		POL2("kick thru opp, kick a little more");
		bestInitSpeedToDest += .5*speedIncr;
	}
	*/


	// Now check for insecurities
	PlayerState p;
	if(bestIsInsecure)  // do not set to false, might be true already 
		dribbleInsecure = true;
	if(tooShortKick)
		dribbleInsecure = true;
	if(opp){
		p.setAssumeToPos(opp,dest);
		bool oppStandsInMe = false && nextMeNA.pos.distance(p.pos)<2.5*ServerOptions::player_size;
		if(!oppStandsInMe && p.pos.distance(dest)<p.kick_radius){
			POL2("getCmdKickToDest: dest can be reached by opponent!");
		}
	}
	/*
	if(Tools::get_tackle_success_probability(p.pos,dest,p.ang.get_value())>.90)
		dribbleInsecure = true;
		*/

	POL2("Found position is risky: "<<dribbleInsecure);
	
	onestepkick->set_state(thisMe.pos,thisMe.vel,thisMe.ang,thisBall.pos,thisBall.vel,Vector(-100,-100),ANGLE(0),0);
	onestepkick->kick_in_dir_with_initial_vel(bestInitSpeedToDest,toDest);
	if(!onestepkick->get_cmd(cmd)){
		// TODO: Why does this return false most of the time?
		POL("OneStepKick returned false: STRANGE");
		//dribbleInsecure = true;
		//return false;
	} 
	return true;
}


// getNextAction(opp) {{{1
DribbleAround::Action DribbleAround::getNextAction(PPlayer& opp){
	ANGLE toGoal = Tools::my_angle_to(dribbleTo);
	bool bodyAngleOK = fabs(toGoal.get_value_mPI_pPI())<MAX_ANGLE_TO_DEST;
	//bool bodyAngleOffALot = !bodyAngleOK 
	//	&& fabs(toGoal.get_value_mPI_pPI())>DEG2RAD(100);

	static const float maxDistToBall = 0.80*thisMe.kick_radius;

	//bool isOppGoalie = opp->number == WSinfo::ws->his_goalie_number;

	// Are there teammates also in possession of ball?
	PlayerSet team = WSinfo::valid_teammates_without_me;
	team.keep_and_sort_closest_players_to_point(1,thisBall.pos);
	bool dontKick = 
		   team.num>0
		&& (team[0]->pos.distance(thisBall.pos)<team[0]->kick_radius)
		&& (team[0]->number > WSinfo::me->number);
	if(dontKick) {
		POL("getNextAction: I have a lower number than teammate who also controls ball -- not kicking");
	}

	// Two just ahead?
	PlayerSet opps = WSinfo::valid_opponents;
	opps.keep_and_sort_closest_players_to_point(2,thisMe.pos);

	//bool twoAgainstMe = opps.num>1
	//	&& fabs(Tools::my_angle_to(opps[0]->pos).get_value_mPI_pPI())<DEG2RAD(45)
	//	&& fabs(Tools::my_angle_to(opps[1]->pos).get_value_mPI_pPI())<DEG2RAD(45)
	//	&& thisMe.pos.distance(opps[0]->pos)<2
	//	&& thisMe.pos.distance(opps[1]->pos)<2;

	float hisTackleProb = (opp?Tools::get_tackle_success_probability(opp->pos,thisBall.pos,opp->ang.get_value()):0);
	//float myTackleProb = Tools::get_tackle_success_probability(thisMe.pos,thisBall.pos,thisMe.ang.get_value());

	if(hisTackleProb>.85){
		POL("getNextAction: Opponent can tackle -- dangerous!");
		dribbleInsecure = true;
	}
	/*
	if(hisTackleProb>.95 && myTackleProb>.8){
		POL("getNextAction: Tackle needed: Opp can tackle, I can tackle!");
		return DA_TACKLE;
	}*/
	/*
	if(twoAgainstMe
			&& myTackleProb>.8){
		POL("getNextAction: Tackle needed: two against me");
		return DA_TACKLE;
	}*/
	if(nextOppToBall.pos.distance(nextBall.pos)<nextOppToBall.kick_radius+ServerOptions::ball_size){   // opponent gets ball
		POL("getNextAction: Kick needed: Opp can reach ball (1)");
		if(!dontKick)
			return DA_KICK;
	}
	if(opp->pos.distance(nextBall.pos)<opp->kick_radius+ServerOptions::ball_size){   // opponent gets ball
		POL("getNextAction: Kick needed: Opp can reach ball (2)");
		if(!dontKick)
			return DA_KICK;
	}
	/*
	 * TODO: Why do I need this?
	if((nextOppToBall.pos-nextMeNA.pos).norm()<(thisMe.kick_radius+nextOppToMe.kick_radius-0.5)){
		POL("getNextAction: Kick needed: Overlapping kick radii");
		if(!dontKick)
			return DA_KICK;
	}
	*/
	if(Tools::get_tackle_success_probability(nextOppToBall.pos,nextBall.pos,nextOppToMe.ang.get_value())>.70){
		POL("getNextAction: Kick needed: Otherwise opp could tackle next cycle");
		if(!dontKick)
			return DA_KICK;
	}

	Vector inMyDir;
	inMyDir.init_polar(0.1,thisMe.ang);
	switch(request){
		case DAREQ_NONE:
			break;
		case DAREQ_KICK:
			POL("getNextAction: Kick needed: Was requested");
			if(!dontKick) return DA_KICK;
			break;
		case DAREQ_DASH:
			if(thisMe.pos.distance(nextBall.pos) > (thisMe.pos+inMyDir).distance(nextBall.pos)){
				POL("getNextAction: Dash needed: Was requested");
				return DA_DASH;
			}else{
				POL("getNextAction: Ignoring requested dash!");
				request=DAREQ_NONE;
			}
			break;
		case DAREQ_TURN:
			if(!bodyAngleOK){
				bool looseBall = nextBall.pos.distance(nextMeNA.pos)>0.9*thisMe.kick_radius;
				bool looseBallBehind = looseBall 
					&& fabs((dribbleTo-thisMe.pos).ANGLE_to(nextBall.pos-thisMe.pos).get_value_mPI_pPI())>DEG2RAD(100);
				if(!looseBallBehind){
					POL("getNextAction: Turn needed: Was requested");
					return DA_TURN;
				}else{
					POL("getNextAction: Ignoring requested turn!");
					request=DAREQ_NONE;
				}
			}
	}

	bool ballWillLeaveField= 
		fabs(nextBall.pos.getX())>FIELD_BORDER_X-0.5
		||fabs(nextBall.pos.getY())>FIELD_BORDER_Y-0.5;

	if(nextMeNA.pos.distance(nextBall.pos)<maxDistToBall)        
	  if(!bodyAngleOK && !ballWillLeaveField){
	    POL("getNextAction: Turn needed");
	    return DA_TURN;
	  }

	if(ballWillLeaveField){
		POL("getNextAction: Kick needed, ball will be outside field next cycle");
		if(!dontKick)
			return DA_KICK;
	}
	
	bool ballDirOK = true;
	ballDirOK = ballDirOK && fabs((thisBall.vel.ARG() - thisMe.ang).get_value_mPI_pPI())<PI/5;
	ballDirOK = ballDirOK && fabs(Tools::my_angle_to(thisBall.pos).get_value_mPI_pPI())<DEG2RAD(90);
	ballDirOK = ballDirOK && thisMe.vel.norm() >= 0.5*thisBall.vel.norm();
	if(ballDirOK && !keepBallSafe){
	  POL("getNextAction: Dash needed, ball seems to be going in right dir");
	  return DA_DASH;
	}

	/*
	if(bodyAngleOffALot
			&& thisMe.pos.distance(dribbleTo)>2){
		// I'm running in the totally wrong direction
		// --> provoke collision to make dir change easier
		POL("getNextAction: Collision kick needed");
		if(!dontKick)
			return DA_COLKICK;
	}
	*/
	
	// Can we get the ball in the next cycle?
	bool iGetBallByDoingNothing = nextMeNA.pos.distance(nextBall.pos)<maxDistToBall;
	bool iGetBallByDashing=false;
	bool iGetBallLater = fabs((thisMe.ang - thisBall.vel.ARG()).get_value_mPI_pPI())<DEG2RAD(5)
		&& thisBall.vel.norm()<1.2*WSinfo::me->speed_max;
	if(!iGetBallByDoingNothing){
		Cmd tmpCmd;
		Vector new_my_pos,new_my_vel,new_ball_pos,new_ball_vel;
		ANGLE new_my_ang;
		basic_cmd->set_dash(maxSpeed);
		basic_cmd->get_cmd(tmpCmd);
		Tools::model_cmd_main(thisMe.pos,thisMe.vel,thisMe.ang,thisBall.pos,thisBall.vel,tmpCmd.cmd_body,new_my_pos,new_my_vel,new_my_ang,new_ball_pos,new_ball_vel,false);

		iGetBallByDashing = new_my_pos.distance(new_ball_pos)<maxDistToBall;
	}
	if(!iGetBallByDashing && !iGetBallByDoingNothing && !iGetBallLater){ 
		POL("getNextAction: Kick needed, can't catch Ball next cycle");
		if(!dontKick)
			return DA_KICK;
	}

	POL("getNextAction: Dash needed");
	return DA_DASH;
}

// getNextAction() {{{1
DribbleAround::Action DribbleAround::getNextAction(){
	ANGLE toGoal = Tools::my_angle_to(dribbleTo);
	bool bodyAngleOK = fabs(toGoal.get_value_mPI_pPI())<MAX_ANGLE_TO_DEST;
	bool bodyAngleOffALot = !bodyAngleOK 
		&& fabs(toGoal.get_value_mPI_pPI())>DEG2RAD(100);

	static const float maxDistToBall = 0.80*thisMe.kick_radius;
	
	// check for other teammates in possession of ball
	PlayerSet team = WSinfo::valid_teammates_without_me;
	team.keep_and_sort_closest_players_to_point(1,thisBall.pos);
	bool dontKick = 
		   team.num>0
		&& (team[0]->pos.distance(thisBall.pos)<0.9*team[0]->kick_radius)
		&& (team[0]->number > WSinfo::me->number);
	if(dontKick) {
		POL("getNextAction: I have a lower number than teammate who also controls ball -- not kicking");
	}

	Vector inMyDir;
	inMyDir.init_polar(0.1,thisMe.ang);
	switch(request){
		case DAREQ_NONE:
			break;
		case DAREQ_KICK:
			POL("getNextAction: Kick needed: Was requested");
			if(!dontKick) return DA_KICK;
			break;
		case DAREQ_DASH:
			if(thisMe.pos.distance(nextBall.pos) > (thisMe.pos+inMyDir).distance(nextBall.pos)){
				POL("getNextAction: Dash needed: Was requested");
				return DA_DASH;
			}else{
				POL("getNextAction: Ignoring requested dash!");
				request=DAREQ_NONE;
			}
			break;
		case DAREQ_TURN:
			if(!bodyAngleOK){
				bool looseBall = nextBall.pos.distance(nextMeNA.pos)>0.9*thisMe.kick_radius;
				bool looseBallBehind = looseBall 
					&& fabs((dribbleTo-thisMe.pos).ANGLE_to(nextBall.pos-nextMeNA.pos).get_value_mPI_pPI())>DEG2RAD(100);
				if(!looseBallBehind){
					POL("getNextAction: Turn needed: Was requested");
					return DA_TURN;
				}else{
					POL("getNextAction: Ignoring requested turn!");
					request=DAREQ_NONE;
				}
			}
	}

	bool ballWillLeaveField= 
		fabs(nextBall.pos.getX())>FIELD_BORDER_X-0.5
		||fabs(nextBall.pos.getY())>FIELD_BORDER_Y;
	if(nextMeNA.pos.distance(nextBall.pos)<maxDistToBall)
		// I get ball w/o moving
		if(!bodyAngleOK && !ballWillLeaveField){
			POL("getNextAction: Turn needed, kicking first");
			return DA_KICK; // TODO: Does this work?
		}

	bool ballDirOK = fabs((thisBall.vel.ARG() - thisMe.ang).get_value_mPI_pPI())<PI/5;
	ballDirOK = ballDirOK && fabs(Tools::my_angle_to(thisBall.pos).get_value_mPI_pPI())<DEG2RAD(90);
	ballDirOK = ballDirOK && thisMe.vel.norm() >= 0.9*thisBall.vel.norm();
	if(ballDirOK){
		POL("getNextAction: Dash needed, ball seems to be going in right dir");
		return DA_DASH;
	}

	if(bodyAngleOffALot && thisMe.pos.distance(dribbleTo)>2){
		// I'm running in the totally wrong direction
		// --> provoke collision to make dir change easier
		POL("getNextAction: Collision kick needed");
		if(!dontKick)
			return DA_COLKICK;
	}

	// Can we get the ball in the next cycle?
	bool iGetBallByDoingNothing = nextMeNA.pos.distance(nextBall.pos)<maxDistToBall;
	bool iGetBallByDashing = false;
	bool iGetBallLater = fabs((thisMe.ang - thisBall.vel.ARG()).get_value_mPI_pPI())<DEG2RAD(5)
		&& thisBall.vel.norm()<1.2*WSinfo::me->speed_max;
	if(!iGetBallByDoingNothing){
		Cmd tmpCmd;
		Vector new_my_pos,new_my_vel,new_ball_pos,new_ball_vel;
		ANGLE new_my_ang;
		basic_cmd->set_dash(maxSpeed);
		basic_cmd->get_cmd(tmpCmd);
		Tools::model_cmd_main(thisMe.pos,thisMe.vel,thisMe.ang,thisBall.pos,thisBall.vel,tmpCmd.cmd_body,new_my_pos,new_my_vel,new_my_ang,new_ball_pos,new_ball_vel,false);

		iGetBallByDashing = new_my_pos.distance(new_ball_pos)<maxDistToBall;
	}
	if(!iGetBallByDashing && !iGetBallByDoingNothing && !iGetBallLater){ 
		POL("getNextAction: Kick needed, can't catch Ball next cycle");
		if(!dontKick)
			return DA_KICK;
	}
	if(ballWillLeaveField){
		POL("getNextAction: Kick needed, ball will be outside field next cycle");
		if(!dontKick)
			return DA_KICK;
	}
	
	POL("getNextAction: Dash needed");
	return DA_DASH;
}

// get_cmd() {{{1
bool DribbleAround::get_cmd(Cmd& cmd){
	
	// avoid acting upon aged kickRequests
	if(WSinfo::ws->time != requestTime+1){
		setRequest(DAREQ_NONE);
	}
	if(!didDribble){
		setRequest(DAREQ_NONE);
	}

  neckReqSet=false;
	dribbleInsecure=false;
	
	// trivial cases 1st:
	if(!WSinfo::is_ball_kickable()){
		POL("Ball not kickable -> intercept");
		return intercept->get_cmd(cmd);
	}
	/*
	if(getGoalieKickCmd(cmd)){
		POL("Trying to shoot goal");
		return true;
	}
	*/
	if((WSinfo::me->pos - WSinfo::ball->pos).norm()<0.385 && WSinfo::ball->age == 0){
		POL("Collision with ball. Danger.");
	}

	static PPlayer lastOpp = NULL;
	static Vector lastOppPos = Vector(-100,-100);
	static ANGLE lastOppAng = ANGLE(0);
	static int lastOppInMeCount = 0;
  opp = getRelevantOpponent();
	
	// experimental: assume tackle
	bool oppHasTackled=false;
	if(opp 
	 && opp->pos.distance(thisMe.pos)<thisMe.kick_radius
	 && opp->age <1 
	 && (lastOppPos - opp->pos).norm()<.05
	 && (fabs(ANGLE(lastOppAng - opp->ang).get_value_mPI_pPI())<DEG2RAD(5))){
		POL("Assuming opponent tackled since he did not move/turn --> risky.");
		oppHasTackled = true;
	}

	if(opp 
			&& !oppHasTackled
			&& opp==lastOpp 
			&& opp->pos.distance(thisMe.pos)<.8*thisMe.kick_radius)
		lastOppInMeCount++;
	else
		lastOppInMeCount=0;

	lastOpp = opp;
	if(opp){
		lastOppPos = opp->pos;
		lastOppAng = opp->ang;
	}

	if(oppHasTackled) opp=NULL;

	if(lastOppInMeCount>3){
		POL("get_cmd: Last Opp has been close to me for too long!");
		dribbleInsecure = true;
	}else{
		//POL("lastOppInMeCount = "<<lastOppInMeCount);
	}

	if(opp && WSinfo::ws->time - opp->time > 0
			   && Tools::could_see_in_direction((opp->pos-thisMe.pos).ARG())
				 && WSinfo::me->pos.sqr_distance(opp->pos)<SQR(3)){
		POL("get_cmd: Haven't seen opp in 1 cycle: Setting neck request");
		neckReqSet=true;
		neckReq   = (opp->pos-thisMe.pos).ARG();
		// Tools::set_neck_request(NECK_REQ_LOOKINDIRECTION, (opp->pos-thisMe.pos).ARG());
	}

	thisBall.setNow(WSinfo::ball);
	nextBall.setNext(thisBall);
	thisMe.setNow(WSinfo::me,WSinfo::ball);
	nextMeNA.setAssumeNoAction(thisMe,nextBall.pos);

	Vector __ang;
	MARK_STATE(thisMe,#00FFFF);

	Action nextAction;
	if(opp){
			nextOppNA.setAssumeNoAction(opp,nextBall);
			nextOppToBall.setAssumeToPos(opp,nextBall);
			nextOppToMe.setAssumeToPos(opp,nextMeNA); 
			nextOppToMe.testCollision(nextMeNA);
			MARK_PPLAYER(opp,red);
			MARK_STATE(nextOppNA,  #AAAAAA);
			MARK_STATE(nextOppToMe,#666666);
			MARK_BALL(nextBall,#AAAAAA);
			nextAction = getNextAction(opp);
	}
	else{
			MARK_BALL(nextBall,#AAAAAA);
			nextAction = getNextAction();
  }

	bool res = false;
	switch(nextAction){
		case DA_TACKLE:  res= getTackleCmd(cmd); break;
		case DA_KICK:    res= getKickCmd(cmd); break;
		case DA_TURN:    res= getTurnCmd(cmd); break;
		case DA_DASH:    res= (getDashCmd(cmd)||getKickCmd(cmd)); break;
		case DA_COLKICK: res= getColKickCmd(cmd); break;
		case DA_GOALK:   res= getGoalieKickCmd(cmd)||getKickCmd(cmd); break;
	}

	if(opp&& opp->number == WSinfo::ws->his_goalie_number){
		POL("get_cmd: Goalie direct opponent, dribbling is unsafe!");
		dribbleInsecure=true;
	}
	if(opp&&nextOppToMe.movedToAvoidCollision){
		POL("get_cmd: nextOppToMe movedToAvoidCollision, dribbling is unsafe!");
		dribbleInsecure=true;
	}
	
	// never reached!?
	return res;
}

// getCmd: Dash, Turn, Tackle {{{1
// getTurnCmd {{{2
bool DribbleAround::getTurnCmd(Cmd& cmd){
	lastActionTaken=DA_TURNING;
	POL("getTurnCmd: Turning to dribbleTo.");
	basic_cmd->set_turn_inertia(Tools::my_angle_to(dribbleTo).get_value());

	Vector new_my_pos,new_my_vel,new_ball_pos,new_ball_vel;
	ANGLE new_my_ang;
	Tools::model_cmd_main(thisMe.pos,thisMe.vel,thisMe.ang,thisBall.pos,thisBall.vel,cmd.cmd_body,new_my_pos,new_my_vel,new_my_ang,new_ball_pos,new_ball_vel,false);

	basic_cmd->get_cmd(cmd);

	// TODO: set again request if criteria match. Work out criteria
	bool isMyAngleAfterTurnOK = fabs((dribbleTo - new_my_pos).ARG().get_value_mPI_pPI())<MAX_ANGLE_TO_DEST;
	bool isBallDirOK    = fabs((thisBall.vel.ARG()-(dribbleTo-thisBall.pos).ARG()).get_value_mPI_pPI())<MAX_ANGLE_TO_DEST;
	bool isBallSpeedCatchUpable = thisBall.vel.norm()<1.;

	if(isBallDirOK && !isMyAngleAfterTurnOK && isBallSpeedCatchUpable)
		setRequest(DAREQ_TURN);
	else if(isBallDirOK)
		setRequest(DAREQ_DASH);
	
	return true;
}

// getDashCmd {{{2
bool DribbleAround::getDashCmd(Cmd& cmd){
	float distToBall;
	static const float minDistToBall = 0.15+(ServerOptions::player_size + ServerOptions::ball_size);
	static const float maxDistToBall = .8*thisMe.kick_radius;

	Vector inMyDir;
	inMyDir.init_polar(0.3,thisMe.ang);
	bool isBallDirOK    = fabs((thisBall.vel.ARG()-(dribbleTo-thisBall.pos).ARG()).get_value_mPI_pPI())<MAX_ANGLE_TO_DEST;
	bool isBallSpeedCatchUpable = thisBall.vel.norm()<2.;
	//bool isBallPosOK    = thisMe.pos.distance(nextBall.pos) > (thisMe.pos+inMyDir).distance(nextBall.pos);
	bool ballBehind = fabs(Tools::my_angle_to(thisBall.pos).get_value_mPI_pPI())>DEG2RAD(100);
	bool amIOutsideField = fabs(thisMe.pos.getX())>FIELD_BORDER_X
		                   ||fabs(thisMe.pos.getY())>FIELD_BORDER_Y;

	bool dontCareAboutMaxDist = isBallDirOK && isBallSpeedCatchUpable && !ballBehind;

	if(maxSpeed==0){
		POL("getDashCmd: Cannot dash: maxSpeed=0");
		return false;
	}
	
	if(request == DAREQ_DASH){
		POL("getDashCmd: Returning requested dash maxSpeed="<<maxSpeed);
		basic_cmd->set_dash(maxSpeed);
		return basic_cmd->get_cmd(cmd);
	}

	int canDash = 0;
	Cmd tmpCmd;
	Vector new_my_pos,new_my_vel,new_ball_pos,new_ball_vel;
	ANGLE new_my_ang;
	bool isOutsideField;
	for(int i = 70; i<=maxSpeed; i+=10){
		tmpCmd.cmd_body.unset_lock();
		basic_cmd->set_dash(i);
		basic_cmd->get_cmd(tmpCmd);
		Tools::model_cmd_main(thisMe.pos,thisMe.vel,thisMe.ang,thisBall.pos,thisBall.vel,tmpCmd.cmd_body,new_my_pos,new_my_vel,new_my_ang,new_ball_pos,new_ball_vel,false);
		isOutsideField = fabs(new_my_pos.getX())>FIELD_BORDER_X
		               ||fabs(new_my_pos.getY())>FIELD_BORDER_Y;
		if(!amIOutsideField && isOutsideField) continue; // dont dash outside field
		distToBall = (new_my_pos-new_ball_pos).norm();
		if( ((distToBall <= maxDistToBall)// play it safe
				 ||dontCareAboutMaxDist)      // vorgelegt                             
		  && distToBall > minDistToBall)  // avoid collision
			canDash = i;
	}
	if(canDash==0) {
		POL("getDashCmd: Cannot dash.");
		lastActionTaken=DA_NO_ACTION;
		return false;
	}
	bool ballDirOK = fabs((thisBall.vel.ARG() - thisMe.ang).get_value_mPI_pPI())<PI/5;
	if(ballDirOK && ballBehind && canDash<60){
		POL("getDashCmd: BallDir OK and I can dash only little --> rather kick");
		lastActionTaken=DA_NO_ACTION;
		return false;
	}
		
	POL("getDashCmd: Dashing with power "<<canDash);
	lastActionTaken=DA_DASHING;
	basic_cmd->set_dash(canDash);
	return basic_cmd->get_cmd(cmd);
}

// getColKickCmd {{{2
/*
 * Return a kick that is prone to collide with me in the next cycle
 * thus giving me the possibility to turn afterwards
 * (ball aint movin and me neither)
 */
bool DribbleAround::getColKickCmd(Cmd& cmd){
	Vector toGoal = dribbleTo - nextMeNA.pos;
	toGoal.normalize(0.5 * ServerOptions::player_size);
	Vector dest = toGoal + nextMeNA.pos;
	lastActionTaken=DA_COLLISION_KICK;
	setRequest(DAREQ_TURN);
	return getCmdKickToDest(cmd,dest,true,true);
}
// getTackleCmd {{{2
bool DribbleAround::getTackleCmd(Cmd& cmd){
	Vector tackleTo;
	tackleTo.init_polar(4,thisMe.ang.get_value());
	tackleTo+=thisBall.pos;
	if(tackleTo.distance(MY_GOAL_CENTER)> thisBall.pos.distance(MY_GOAL_CENTER))
		basic_cmd->set_tackle(100);
	else
		basic_cmd->set_tackle(-100);
	lastActionTaken=DA_TACKLING;
	return basic_cmd->get_cmd(cmd);
}
// getCmd: Kicks {{{1

// getKickCmd {{{2
bool DribbleAround::getKickCmd(Cmd& cmd){
	POL2("In getKickCmd()");
	ANGLE toGoal = Tools::my_angle_to(dribbleTo);
	bool bodyAngleOK = fabs(toGoal.get_value_mPI_pPI())<MAX_ANGLE_TO_DEST;
	if(bodyAngleOK)
		return getKickAhead(cmd);
	else
		return getKickForTurn(cmd) || getKickAhead(cmd);
}

// getKickAhead {{{2
bool DribbleAround::getKickAhead(Cmd& cmd){
	POL2("In getKickAhead()");
	ANGLE toBall = Tools::my_angle_to(thisBall.pos);
	ANGLE toOpp = opp?Tools::my_angle_to(opp->pos):ANGLE(0);
	double toBallFl = toBall.get_value_mPI_pPI();
	double toBallFlAbs = fabs(toBallFl);
	//double toOppFl = toOpp.get_value_mPI_pPI();
	double distToOpp = opp?thisMe.pos.distance(opp->pos):1000;

	bool isBallOnLeftSide = (toBallFl>0);

	bool ballChangeSidePossible = true ||
		((thisBall.pos.distance(thisMe.pos)>.4)
		&& (
				  (toBallFlAbs<DEG2RAD(50))
				||(toBallFlAbs>DEG2RAD(130))));

	bool oppOnLeftSide = opp
		&& Tools::my_angle_to(opp->pos).get_value_mPI_pPI()>0;
	bool oppStraightAhead = 
		opp && fabs(Tools::my_angle_to(opp->pos).get_value_mPI_pPI())<DEG2RAD(4);
	bool oppStraightBehind = 
		opp && fabs(Tools::my_angle_to(opp->pos).get_value_mPI_pPI())>DEG2RAD(170);
	bool oppOnMyLine = opp
		&& (oppStraightAhead || oppStraightBehind);
	bool amICloseToFieldXBorder = 
		  FIELD_BORDER_X-fabs(thisMe.pos.getX())<thisMe.kick_radius;
	bool amICloseToFieldYBorder = 
		FIELD_BORDER_Y-fabs(thisMe.pos.getX())<thisMe.kick_radius;
	bool isFieldBorderOnLeftSide = 
		  (amICloseToFieldXBorder 
			 && ((thisMe.ang.get_value_mPI_pPI()<0 && thisMe.pos.getX()>0)
			  || (thisMe.ang.get_value_mPI_pPI()>0 && thisMe.pos.getX()<0)))
		||(amICloseToFieldYBorder 
			 && ((fabs(thisMe.ang.get_value_mPI_pPI())>DEG2RAD(90) && thisMe.pos.getY()<0)
				|| (fabs(thisMe.ang.get_value_mPI_pPI())<DEG2RAD(90) && thisMe.pos.getY()>0)));

	bool keepBallOnLeftSide = 
		(!opp && isBallOnLeftSide) // no opponent and ball is on left side
		|| (   opp                 // opponent ahead and ball left
				&& oppOnMyLine
				&& isBallOnLeftSide)
		|| (   opp                 // opponent is on right side
				&& !oppOnLeftSide
				&& ( isBallOnLeftSide || ballChangeSidePossible))
		|| (   opp                 // opponent is on left side
				&& oppOnLeftSide
				&& isBallOnLeftSide 
				&& !ballChangeSidePossible)
		|| isFieldBorderOnLeftSide;

	bool haveToSwitchSides =
		  ( isBallOnLeftSide && !keepBallOnLeftSide)
		||(!isBallOnLeftSide && keepBallOnLeftSide);
	haveToSwitchSides = haveToSwitchSides && ballChangeSidePossible;

	bool switchSidesBehind = 
		haveToSwitchSides && (
				   (distToOpp<4                     // close to opp
						&&  oppStraightAhead)           // who is directly ahead
				|| (toBallFlAbs>DEG2RAD(90)         // ball behind me anyway
					  && !(oppStraightBehind          // no opponent behind me
							   && distToOpp<4))           // who is close
		 );

	Vector lot;
	Geometry2d::projection_to_line(lot, thisBall.pos, Line2d(thisMe.pos,Vector(thisMe.ang)));
	bool ballAhead = lot.distance(thisBall.pos)<1.1*ServerOptions::player_size;

	bool ballPosOKForAdvancing = 
		   !haveToSwitchSides
		&& !ballAhead
		&& toBallFlAbs<DEG2RAD(135)  // not too straight behind
		// && thisMe.pos.distance(thisBall.pos)>0.5 // TODO: is this necessary?
		// avoid kicking when ball is far to left/right
		&& (thisBall.pos.distance(thisMe.pos) < (0.9*thisMe.kick_radius)
				|| toBallFlAbs<DEG2RAD(75)
				|| toBallFlAbs>DEG2RAD(105));

	bool closestOppDirectlyBehindMe = 
		opp 
		&& fabs(Tools::my_angle_to(opp->pos).get_value_mPI_pPI())>DEG2RAD(100)
		&& !nextOppToMe.reachesPos;

	bool breakThruOffsideLine = 
		  WSinfo::his_team_pos_of_offside_line() - thisMe.pos.getX() < 1.5
		&& dribbleTo.getX() > WSinfo::his_team_pos_of_offside_line();

	bool closestOppInMeWrongDir = 
		opp
		&& fabs((opp->ang - thisMe.ang).get_value_mPI_pPI())>DEG2RAD(70)
		&& opp->pos.distance(thisMe.pos) < 1.5*thisMe.kick_radius;

	bool ignoreClosestOpp = 
		  closestOppDirectlyBehindMe
		||closestOppInMeWrongDir
		||breakThruOffsideLine;

	bool dontAdvanceChased = 
		opp 
		&& fabs(Tools::my_angle_to(opp->pos).get_value_mPI_pPI())>DEG2RAD(100)
		&& opp->pos.distance(thisMe.pos)<1.4*thisMe.kick_radius
		&& fabs((opp->vel.ARG()-thisMe.ang).get_value_mPI_pPI())<DEG2RAD(30)
		&& opp->vel.norm() > 0.4*opp->speed_max
		&& (opp->vel.norm()>thisMe.vel.norm());
	POL2("getKickAhead: Chase! Do not advance.");

	ANGLE deviate(keepBallOnLeftSide?DEG2RAD(10):DEG2RAD(-10));
	if(ballPosOKForAdvancing && !dontAdvanceChased){
		if(getKickAheadBallOK(cmd,ANGLE(0),ignoreClosestOpp)) 
			return true;
		else if(getKickAheadPrepareBall(cmd,keepBallOnLeftSide,true))
			return true;
	}

	if(ballAhead && !dontAdvanceChased){
		if(getKickAheadBallOK(cmd,deviate,ignoreClosestOpp)){
		  POL("getKickAhead: BallPos not OK, deviating from 0?");
			return true;
		}
	}

	if(getKickAheadPrepareBall(cmd,keepBallOnLeftSide,switchSidesBehind))
		return true;

	Vector safestPos, bestPos;
	getTargetsInMe(safestPos, bestPos);

	bool chased = (Tools::my_angle_to(bestPos).get_value_mPI_pPI() < PI/7                  // best dest ahead
			&& nextOppToMe.pos.distance(bestPos)<0.4
			&& fabs(Tools::my_angle_to(thisBall.pos).get_value_mPI_pPI())<PI/3); // ball ahead of me

	if(chased){
		if(getKickAheadBallOK(cmd,deviate,true)){
		  POL("getKickAhead: chased, deviating from 0?, ignoring closest opp");
			return true;
		}
	}
	

	if(getCmdKickToDest(cmd,bestPos,true,false)){
		POL("getKickAhead: Kicking to best Pos in Dir.");
		lastActionTaken = DA_KICK_AHEAD;
		MARK_POS(bestPos, blue);
		return true;
	};
	if(getCmdKickToDest(cmd,safestPos,true,false)){
		POL("getKickAhead: Kicking to safest Pos.");
		lastActionTaken = DA_KICK_AHEAD;
		MARK_POS(safestPos, blue);
		return true;
	};

	return false;
}

// getKickAheadBallOK {{{2
bool DribbleAround::getKickAheadBallOK(Cmd& cmd, ANGLE deviate, bool ignoreClosestOpp){
	POL2("In getKickAheadBallOK(), ignoreClosestOpp="<<ignoreClosestOpp);
	ANGLE kick_angle = thisMe.ang + deviate;
	Cmd tmpCmd;
	Vector new_my_pos,new_my_vel,new_ball_pos,new_ball_vel;
	Vector old_new_my_pos,old_new_my_vel;
	ANGLE new_my_ang;

	bool goingInXDir = dribbleTo.getX() > thisMe.pos.getX() + 5;
	bool haventLookedThere = 
		WSmemory::last_seen_in_dir(Tools::my_angle_to(dribbleTo))>3;

	const int maxLookAhead = (!goingInXDir || (opp&&opp->age>1))?1:8;
	Vector *afterNdash = new Vector [maxLookAhead+1];

	// kicking == doing nothing
	basic_cmd->set_dash(0);
	basic_cmd->get_cmd(tmpCmd);
	Tools::model_cmd_main(thisMe.pos,thisMe.vel,thisMe.ang,thisBall.pos,thisBall.vel,tmpCmd.cmd_body,new_my_pos,new_my_vel,new_my_ang,new_ball_pos,new_ball_vel,false);
	old_new_my_pos = new_my_pos;
	old_new_my_vel = new_my_vel;
	
	// simulate 1st dash
	basic_cmd->set_dash(maxSpeed);
	basic_cmd->get_cmd(tmpCmd);

	// before doing anything else, calculate how far we can get by
	// maxLookAhead or less dashes
	for(int i = 1 ; i<=maxLookAhead; i++){ // smaller bc we already did one!
		Tools::model_cmd_main(old_new_my_pos,old_new_my_vel,thisMe.ang,thisBall.pos,thisBall.vel,tmpCmd.cmd_body,new_my_pos,new_my_vel,new_my_ang,new_ball_pos,new_ball_vel,false);
		old_new_my_pos = new_my_pos;
		old_new_my_vel = new_my_vel;
		afterNdash[i] = new_my_pos;
	}
	
	Vector lot,tmp,dest;
	bool foundSafePos = false;
	static const double maxDistToBall = 0.7*thisMe.kick_radius;
	PlayerSet pset;

	Vector lotOnMyDirDiff;
	Vector lotOnMyDir;
	int nThDash;
	bool ballLeavesKickRadius;
	for(nThDash=maxLookAhead;nThDash>0; nThDash--){
		if(  nThDash*70 + 0.3125*ServerOptions::stamina_max //TG09: alt: 1300 
           > WSinfo::me->stamina) 
			continue;
		for(double furthest=0.8*thisMe.kick_radius;furthest>0; furthest-=.1){
			lotOnMyDirDiff.init_polar(furthest,thisMe.ang);
			lotOnMyDir = afterNdash[nThDash] + lotOnMyDirDiff; // right to me in 2 cycles

			Geometry2d::projection_to_line(lot, lotOnMyDir, Line2d(thisBall.pos,Vector(kick_angle)));
			if(!(nThDash<7 && !(fabs(thisMe.pos.getY())>FIELD_BORDER_Y-2)))
				lot = lotOnMyDir + 0.5*(lot - lotOnMyDir);

			// we know now the kick dir, figure out destination of kickCmd:
			bool tooFar;
			dest = getKickDestForBallPosInNCycles(lot,1+nThDash,tooFar);
			
			if(tooFar) continue;

			if(afterNdash[nThDash].distance(lot)<maxDistToBall){
				foundSafePos = true;
				break;
			}
		}
		if(!foundSafePos)
			continue;
		pset = WSinfo::valid_opponents;
		if(ignoreClosestOpp && opp)
			pset.remove(opp);
		pset.keep_and_sort_closest_players_to_point(4,lot);
		if(fabs(lot.getY())>FIELD_BORDER_Y-1 || fabs(lot.getX())>FIELD_BORDER_X-3){
			// Can catch ball outside field -- dont do this
			foundSafePos = false;
			continue;
		}

		// now calculate whether the ball will leave my kick_radius
		// in the course of kick+dashes
		new_ball_pos = dest;
		new_ball_vel = dest - thisBall.pos;
		ballLeavesKickRadius = false;
		for(int i=1;i<=nThDash; i++){
			new_ball_vel = ServerOptions::ball_decay*new_ball_vel;
			new_ball_pos = new_ball_pos + new_ball_vel;
			if(afterNdash[i].distance(new_ball_pos)>0.8*thisMe.kick_radius){
				ballLeavesKickRadius = true;
				break;
			}
		}
		// advance fast even if havent looked there if ball stays controlled
		if(ballLeavesKickRadius && haventLookedThere){
			foundSafePos = false;
			continue;
		}
		bool betterInterceptorFound=false;
		Vector his_lot;
		float deltaPos;
		Geometry2d::projection_to_line(his_lot, thisMe.vel, Line2d(Vector(0,0),Vector(lot - thisMe.pos)));
		float myDeltaPos = thisMe.pos.distance(his_lot);
		for(int i=0;i<pset.num&&ballLeavesKickRadius;i++){
			Geometry2d::projection_to_line(his_lot, pset[i]->vel, Line2d(Vector(0,0),Vector(lot - pset[i]->pos)));
			deltaPos = pset[i]->pos.distance(his_lot);
			if(pset[i]->pos.distance(lot)-thisMe.pos.distance(lot)<0.5+0.15*nThDash + deltaPos-myDeltaPos){
				// I loose control of the ball and my next opponent is closer to ball than I am
				betterInterceptorFound=true;
				break;
			}
		}
		if(betterInterceptorFound){
			foundSafePos=false;
			continue;
		}
		break;
	}

	if(!foundSafePos){
		POL2("getKickAheadBallOK: Advancing too risky, stopping it");
		return false;
	}

	bool isBallOnLeftSide = (Tools::my_angle_to(thisBall.pos).get_value_mPI_pPI()>0);
	Vector fallBack;
	fallBack.init_polar(0.6,thisMe.ang + (isBallOnLeftSide ? ANGLE(DEG2RAD(112)):ANGLE(DEG2RAD(-112))));
	Vector inMyDir;
	inMyDir.init_polar(0.3,thisMe.ang);
	bool canDash = true;
	bool mustKick = false;
	if(opp 
			&& nextOppToMe.pos.distance(dest)<nextOppToMe.kick_radius+ServerOptions::ball_size){
		POL2("getKickAheadBallOK: Do not advance in Opponent, moving ball to back!");
		dest = fallBack + nextMeNA.pos;
		canDash = false;
		mustKick = true;
	}
	if(opp 
			&& nextOppToMe.pos.distance(dest)<nextOppToMe.kick_radius+ServerOptions::ball_size){
		POL2("getKickAheadBallOK: Do not advance in Opponent(2), moving ball to back!");
		dest = fallBack + inMyDir + nextMeNA.pos;
		canDash = false;
		mustKick = true;
	}
	/*
	 * Do NOT do this: upper level will request dash!
	if(opp 
			&& nextOppToMe.pos.distance(dest)<nextOppToMe.kick_radius+ServerOptions::ball_size){
		POL2("getKickAheadBallOK: Do not advance in Opponent(3), last chance");
		Vector safestPos, bestPos;
		getTargetsInMe(safestPos, bestPos);
		if(getCmdKickToDest(cmd,bestPos,true,false)){
			POL("Kicking ball to best position");
			MARK_POS(safestPos, blue);
			return true;
		}
		if(getCmdKickToDest(cmd,safestPos,true,false)){
			POL("Kicking ball to safest position");
			MARK_POS(safestPos, blue);
			return true;
		}
	}
	*/

	POL2("getKickAheadBallOK: Should get ball after "<<nThDash<< " dashes after kick");
	MARK_POS(lotOnMyDir, green);
	MARK_POS(lot, green);
	DRAW(L2D(thisBall.pos.x,thisBall.pos.y,lot.x,lot.y,"green"));
	DRAW(C2D(afterNdash[nThDash].x,afterNdash[nThDash].y,thisMe.kick_radius,"green"));
    delete [] afterNdash;
    afterNdash = NULL;
	if(getCmdKickToDest(cmd,dest,false,false)){
		POL("getKickAheadBallOK: Ball moving in right direction, kicking ahead.");
		if(canDash)
			setRequest(DAREQ_DASH);
		else if(mustKick)
			setRequest(DAREQ_KICK);
		lastActionTaken = DA_KICK_AHEAD;
		MARK_POS(dest, blue);
		return true;
	};
	return false;
}
// getKickAheadPrepareBall {{{2
bool DribbleAround::getKickAheadPrepareBall(Cmd& cmd, bool keepBallOnLeftSide, bool switchSidesBehind){
	POL2("In getKickAheadPrepareBall()");
	ANGLE toBall = Tools::my_angle_to(thisBall.pos);
	double toBallFl = toBall.get_value_mPI_pPI();
	double toBallFlAbs = fabs(toBallFl);
	bool isBallOnLeftSide = (toBallFl>0);

	bool haveToSwitchSides =
		  ( isBallOnLeftSide && !keepBallOnLeftSide)
		||(!isBallOnLeftSide && keepBallOnLeftSide);

	if(!haveToSwitchSides){
		Vector dest, dest1, dest2;
		Vector inMyDir;
		inMyDir.init_polar(0.4,thisMe.ang);
		dest1.init_polar(0.8,thisMe.ang + (isBallOnLeftSide ? ANGLE(5*PI/8):ANGLE(-5*PI/8)));
		dest1+= nextMeNA.pos;
		dest2 = dest1 + inMyDir;
		dest = ((thisBall.pos.sqr_distance(dest1)<thisBall.pos.sqr_distance(dest2))?dest1:dest2);
		if(getCmdKickToDest(cmd,dest,true,false)){
			POL("getKickAheadPrepareBall: Preparing ball on same side.");
			setRequest(DAREQ_KICK);
			MARK_POS(dest, blue);
			return true;
		}
	}
	
	// now the switch sides case
	// TODO: this is always true!?
	bool ballChangeSidePossible = true ||
		((thisBall.pos.distance(thisMe.pos)>.4)
		&& (
				  (toBallFlAbs<DEG2RAD(50))
				||(toBallFlAbs>DEG2RAD(130))));
	
	if(!ballChangeSidePossible){
		bool toMyLine = (switchSidesBehind && toBallFlAbs>DEG2RAD(90))
			||(!switchSidesBehind && toBallFlAbs<DEG2RAD(90));

		Vector dest;
		if(toMyLine){
			dest.init_polar((switchSidesBehind?-0.8:0.8)*thisMe.kick_radius,thisMe.ang);
			dest+=nextMeNA.pos;
			if(getCmdKickToDest(cmd,dest,true,false)){
				POL("getKickAheadPrepareBall: Cannot change sides, preparing by kicking on my line.");
				setRequest(DAREQ_KICK);
				MARK_POS(dest, blue);
				return true;
			}
		}
		ANGLE toDest;
		if(isBallOnLeftSide)
			toDest = ANGLE(thisMe.ang + ANGLE(switchSidesBehind?-5*PI/8:-3*PI/8));
		else
			toDest = ANGLE(thisMe.ang + ANGLE(switchSidesBehind? 5*PI/8: 3*PI/8));

		dest.init_polar(0.6*thisMe.kick_radius, toDest);
		dest+=nextMeNA.pos;
		if(getCmdKickToDest(cmd,dest,true,false)){
			POL("getKickAheadPrepareBall: Cannot change sides, preparing by kicking in desired dir.");
			setRequest(DAREQ_KICK);
			MARK_POS(dest, blue);
			return true;
		}
	}

	// we have to change sides and ball change sides is possible
	Vector dest;
	ANGLE toDest;
	if(keepBallOnLeftSide)
		toDest = ANGLE(thisMe.ang + ANGLE(switchSidesBehind? 5*PI/8: 3*PI/8));
	else
		toDest = ANGLE(thisMe.ang + ANGLE(switchSidesBehind?-5*PI/8:-3*PI/8));
	dest.init_polar(0.6*thisMe.kick_radius,toDest);
	dest += nextMeNA.pos;

	if(getCmdKickToDest(cmd,dest,true,false)){
		POL("getKickAheadPrepareBall: Can change sides, doing so.");
		setRequest(DAREQ_KICK);
		MARK_POS(dest, blue);
		return true;
	}

	return false;

}

// getKickForTurn {{{2
bool DribbleAround::getKickForTurn(Cmd& cmd){
	ANGLE toBall       =     (dribbleTo-thisMe.pos).ANGLE_to(thisBall.pos - thisMe.pos);
	ANGLE toOpp        = opp?(dribbleTo-thisMe.pos).ANGLE_to(opp->pos          - thisMe.pos):ANGLE(0);
	float toOppFl      = toOpp.get_value_mPI_pPI();
	float toBallFl     = toBall.get_value_mPI_pPI();
	ANGLE aToDribbleTo = (dribbleTo - thisMe.pos).ARG();
	bool isBallOnLeftSide = toBallFl > 0;
	bool isOppLeft        = toOppFl  > 0;
	POL2("getKickForTurn: Opp will be left="<<isOppLeft << " toOpp="<<RAD2DEG(toOppFl));
	POL2("getKickForTurn: Ball is be left="<<isBallOnLeftSide << " toBall="<<RAD2DEG(toBallFl));

	//float toDestFl = Tools::my_angle_to(dribbleTo).get_value_mPI_pPI();
	//bool turnALot  = fabs(toDestFl) > DEG2RAD(80);

	Cmd tmpCmd;
	Vector new_my_pos,new_my_vel,new_ball_pos,new_ball_vel;
	Vector old_new_my_pos,old_new_my_vel;
	ANGLE new_my_ang, old_new_my_ang;

	// simulate kick
	basic_cmd->set_dash(0);
	basic_cmd->get_cmd(tmpCmd);
	Tools::model_cmd_main(thisMe.pos,thisMe.vel,thisMe.ang,thisBall.pos,thisBall.vel,tmpCmd.cmd_body,new_my_pos,new_my_vel,new_my_ang,new_ball_pos,new_ball_vel,false);
	old_new_my_pos = new_my_pos;
	old_new_my_vel = new_my_vel;
	old_new_my_ang = new_my_ang;

	//bool goingInXDir = dribbleTo.x > thisMe.pos.x + 5;

	//const int maxDashes = (!goingInXDir || opp&&opp->age>1) ? 1:8;

	// simulate turns until dir is ok
	int numTurns = 0;
	double moment;
	while(fabs((aToDribbleTo - new_my_ang).get_value_mPI_pPI())>MAX_ANGLE_TO_DEST){
		if(numTurns>5) break;
		numTurns++;
		tmpCmd.cmd_body.unset_lock();

		// calculate moment. More or less copy 'n' paste from do_turn_inertia (basic_cmd_bms.c)
		moment = (aToDribbleTo-new_my_ang).get_value_mPI_pPI();
		moment = moment * (1.+(WSinfo::me->inertia_moment * new_my_vel.norm()));
		moment = moment >  3.14 ?  3.14 : moment;
		moment = moment < -3.14 ? -3.14 : moment;

		basic_cmd->set_turn(moment);
		basic_cmd->get_cmd(tmpCmd);
		Tools::model_cmd_main(old_new_my_pos,old_new_my_vel,old_new_my_ang,thisBall.pos,thisBall.vel,tmpCmd.cmd_body,new_my_pos,new_my_vel,new_my_ang,new_ball_pos,new_ball_vel,false);
		old_new_my_pos = new_my_pos;
		old_new_my_vel = new_my_vel;
		old_new_my_ang = new_my_ang;
		POL2("getKickForTurn: Angle to dest: "<< (aToDribbleTo-new_my_ang));
	}
	POL2("getKickForTurn: Will need " << numTurns << " cycles to turn around");
	POL2("getKickForTurn: Opponent is left="<<isOppLeft);

	Vector myPosAfterTurn = new_my_pos;
	ANGLE  myAngAfterTurn = new_my_ang;
	
	Vector onMe;
	bool keepBallLeft = ( opp && !isOppLeft)
		                ||(!opp &&  isBallOnLeftSide);
	onMe.init_polar(0.5*thisMe.kick_radius, myAngAfterTurn + ANGLE(DEG2RAD(keepBallLeft?90:-90)));

	bool haventLookedThere = 
		WSmemory::last_seen_in_dir(Tools::my_angle_to(dribbleTo))>3;

	int numDashes   = 0;
	tmpCmd.cmd_body.unset_lock();
	basic_cmd->set_dash(100);
	basic_cmd->get_cmd(tmpCmd);
	PlayerSet opps;
	Vector catchPoint;
	if(!haventLookedThere) // do not consider dashing if having to turn a lot
		do{
			Tools::model_cmd_main(old_new_my_pos,old_new_my_vel,old_new_my_ang,thisBall.pos,thisBall.vel,tmpCmd.cmd_body,new_my_pos,new_my_vel,new_my_ang,new_ball_pos,new_ball_vel,false);
			opps = WSinfo::valid_opponents;
			catchPoint = new_my_pos+onMe;
			opps.keep_and_sort_closest_players_to_point(1,catchPoint);
			if(opps.num>0)
				if(opps[0]->pos.distance(catchPoint) < opps[0]->kick_radius + (numTurns+numDashes+1)*opps[0]->speed_max*1)
					break;
			if(fabs(catchPoint.getX())>FIELD_BORDER_X-1 || fabs(catchPoint.getY())>FIELD_BORDER_Y-1)
				break;
			numDashes++;
			old_new_my_pos = new_my_pos;
			old_new_my_vel = new_my_vel;
			old_new_my_ang = new_my_ang;
		}while(numDashes<10);

	POL2("getKickForTurn: Additional "<<numDashes<<" dashes are possible");

	Vector myPosAfterDashes = (numDashes==0) ? myPosAfterTurn : old_new_my_pos;
	ANGLE  myAngAfterDashes = (numDashes==0) ? myAngAfterTurn : old_new_my_ang;

	catchPoint = onMe+myPosAfterDashes;

	bool tooFar;
	Vector dest = getKickDestForBallPosInNCycles(catchPoint, 1+numTurns+numDashes, tooFar);

	bool closeToXBorder = (thisMe.pos.getX()) > FIELD_BORDER_X-thisMe.kick_radius;
	bool closeToYBorder = (thisMe.pos.getY()) > FIELD_BORDER_Y-thisMe.kick_radius;
	bool tooCloseToBorder = closeToXBorder || closeToYBorder;

	// OK, now try to get the ball there.

	if(getCmdKickToDest(cmd,dest,false,false)&&!dribbleInsecure){
		POL("getKickForTurn: 1. Kicking ball in \"good\" direction, hope to catch up");
		DRAW(C2D(myPosAfterDashes.x,myPosAfterDashes.y,thisMe.kick_radius,"green"));
		MARK_POS(catchPoint,green);
		setRequest(DAREQ_TURN);
		MARK_POS(dest, blue);
		return true;
	}
	dribbleInsecure=false;

	// it didnt work out, probably I'm in the way. Try some other angles
	// (unchecked! dangerous!)

	onMe.init_polar(0.5*thisMe.kick_radius, myAngAfterTurn + ANGLE(DEG2RAD(keepBallLeft?45:-45)));
	catchPoint = onMe+myPosAfterDashes;
	dest = getKickDestForBallPosInNCycles(catchPoint, 1+numTurns+numDashes, tooFar);
	if(getCmdKickToDest(cmd,dest,false,false) && !dribbleInsecure){
		POL("getKickForTurn: 2. Kicking ball in \"good\" direction, hope to catch up");
		setRequest(DAREQ_TURN);
		DRAW(C2D(myPosAfterDashes.x,myPosAfterDashes.y,thisMe.kick_radius,"green"));
		MARK_POS(catchPoint,green);
		MARK_POS(dest, blue);
		return true;
	}
	dribbleInsecure=false;

	onMe.init_polar(0.5*thisMe.kick_radius, myAngAfterTurn + ANGLE(DEG2RAD(keepBallLeft?135:-135)));
	catchPoint = onMe+myPosAfterDashes;
	dest = getKickDestForBallPosInNCycles(catchPoint, 1+numTurns+numDashes, tooFar);
	if(getCmdKickToDest(cmd,dest,false,false)&&!dribbleInsecure){
		POL("getKickForTurn: 3. Kicking ball in \"good\" direction, hope to catch up");
		DRAW(C2D(myPosAfterDashes.x,myPosAfterDashes.y,thisMe.kick_radius,"green"));
		MARK_POS(catchPoint,green);
		setRequest(DAREQ_TURN);
		MARK_POS(dest, blue);
		return true;
	}
	dribbleInsecure=false;

	if(tooCloseToBorder){
		POL("getKickForTurn: Too close to Border, turn-kick not safe!");
		return false;
	}
	
	if(nextOppToMe.reachesPos){
		Vector safestPos, bestPos;
		getTargetsInMe(safestPos, bestPos);
		if(getCmdKickToDest(cmd,bestPos,true,false)){
			POL("Kicking ball to best position");
			MARK_POS(safestPos, blue);
			return true;
		}
		if(getCmdKickToDest(cmd,safestPos,true,false)){
			POL("Kicking ball to safest position");
			MARK_POS(safestPos, blue);
			return true;
		}
	}

	if(holdturn->is_holdturn_safe()){
		POL("getKickForTurn: Executing holdTurn cmd");
		dribbleInsecure=true;
		return holdturn->get_cmd(cmd, (dribbleTo-thisMe.pos).ARG());
	}

	POL("getKickForTurn: Slowing ball down is not possible");
	return false;
}

// getTargetsInMe(safest,best)
#define POS_ON_CIRC_NUM  24
void DribbleAround::getTargetsInMe(Vector& safestPos, Vector& bestPos){
	static const float maxOppTackleProb = 0.8;
	float bestDestDist=-1E6;
	float bestVal = -1E6;
	float tmpDist,tmpVal,tackleProb;
	float minDistBallToOpp = 2*ServerOptions::ball_size;
	
	Vector straightAhead;
	straightAhead.init_polar(10,thisMe.ang);
	straightAhead+=nextMeNA.pos;
	Vector tmpVec;

	for(int i=0; i < POS_ON_CIRC_NUM; i++){
		// find the best position in a circle within my kick radius
		tmpVec.init_polar(0.80*thisMe.kick_radius, i*2*PI/POS_ON_CIRC_NUM);
		tmpVec += nextMeNA.pos;
		//DRAW(C2D(tmpVec.x,tmpVec.y,0.1,"black"));
		tmpDist = (nextOppToMe.pos-tmpVec).norm() - nextOppToMe.kick_radius;
		tackleProb = Tools::get_tackle_success_probability(nextOppToMe.pos,tmpVec,nextOppToMe.ang.get_value());
		tmpVal = (tmpVec-straightAhead).norm()  // prefer playing ahead
			- ((tackleProb>maxOppTackleProb)?2*tackleProb:0)
			- ((tmpDist<minDistBallToOpp)?2*tmpDist:0);

		if(tmpDist > bestDestDist){
			bestDestDist = tmpDist;
			safestPos = tmpVec;
		}

		if(tmpVal  > bestVal ){
			// ball is far from opponent,
			// gets better rating than previous best ball
			// and tackling isn't too easy for worstcase-to-me-opp
			bestVal = tmpVal;
			bestPos = tmpVec;
		}
	}
}

// getPlayerDistToBallTraj {{{1
float DribbleAround::getPlayerDistToBallTraj(const PPlayer& p, const Vector& v, int& steps){
  Vector ballPos   = thisBall.pos;
  Vector ballVel   = v;
	Vector playerPos = p->pos+p->vel;
  float lastDist=1000;
  float pDist;
  float bestpDist=1000;
  for(int i=0;i<30;i++){
    pDist = playerPos.distance(ballPos);
    if(pDist>lastDist) break;
    if(pDist<bestpDist){
      bestpDist = pDist;
			steps = i;
    }
    ballVel = ServerOptions::ball_decay * ballVel;
    ballPos = ballPos + ballVel;
  }
  return bestpDist;
}

// getGoalieKickCmd() {{{2
bool DribbleAround::getGoalieKickCmd(Cmd& cmd){

	bool closeToGoal =      thisMe.pos.getX()>FIELD_BORDER_X-10
		              && fabs(thisMe.pos.getY())<1.5*ServerOptions::goal_width;
	if(!closeToGoal) return false;

	float toLeftGoalCorner = Tools::my_angle_to(Vector(FIELD_BORDER_X,0.45*ServerOptions::goal_width)).get_value_mPI_pPI();
	float toRightGoalCorner = Tools::my_angle_to(Vector(FIELD_BORDER_X,-0.45*ServerOptions::goal_width)).get_value_mPI_pPI();
	bool lookingAtGoal = toLeftGoalCorner>0 && toRightGoalCorner<0;
	bool goalBehindMe = toLeftGoalCorner<0 && toRightGoalCorner>0;

	// Can I tackle ball into goal? {{{
	if(lookingAtGoal||goalBehindMe){ 
		ANGLE kickDir(thisMe.ang + ANGLE(lookingAtGoal?0:M_PI));
		bool lineIsFree = true;
		PlayerSet opps = WSinfo::valid_opponents;
		Vector ballVel;
		ballVel.init_polar(ServerOptions::ball_speed_max,thisMe.ang);
		float pDist;
		int steps;
		for(int i=0; i<opps.num; i++){
			pDist = getPlayerDistToBallTraj(opps[i],ballVel,steps);
			if(pDist < opps[i]->kick_radius + steps*opps[i]->speed_max){
				lineIsFree = false;
				break;
			}
			if(opps[i]->number == WSinfo::ws->his_goalie_number){
				if(pDist < ServerOptions::catchable_area_l){
					lineIsFree = false;
					break;
				}
			}
		} 

		if(lineIsFree){
			basic_cmd->set_tackle(lookingAtGoal?100:-100);
			basic_cmd->get_cmd(cmd);
			return true;
		}

	} // }}}
  
  return false;
}

// getKickDestForBallPosInNCycles() {{{1
Vector DribbleAround::getKickDestForBallPosInNCycles(const Vector& target, const int cyc, bool& tooFar){
  Vector vToTarget = target - thisBall.pos;
  float divBy = 1;
  for(int i=1;i<=cyc; i++){
    divBy += std::pow(ServerOptions::ball_decay,(double)i);
  }
  vToTarget.normalize(vToTarget.norm()/divBy);
  if(vToTarget.norm() > ServerOptions::ball_speed_max){
    vToTarget.normalize(ServerOptions::ball_speed_max);
    tooFar = true;
  }else
    tooFar = false;
  return vToTarget + thisBall.pos;
}

// public setters {{{1
void DribbleAround::set_target(const Vector& dribto){
	dribbleTo = dribto;
}
void DribbleAround::set_keepBall(bool keepBall){
	keepBallSafe = keepBall;
}
void DribbleAround::set_max_speed(int ms){
	maxSpeed = ms;
}

void DribbleAround::setDribbled(bool b){
	didDribble=b;
}

void DribbleAround::resetRequest(){
	POL("Requests resetted! was:"<<request);
	setRequest(DAREQ_NONE);
}

bool DribbleAround::isDribbleInsecure(){
	return dribbleInsecure;
}

// ~DribbleAround() {{{1
DribbleAround::~DribbleAround() {
	delete basic_cmd;
	delete go2pos;
	delete dribble_straight;
	delete onestepkick;
	delete intercept;
	delete holdturn;
}

// PlayerState/BallState stuff {{{1
void DribbleAround::PlayerState::setAssumeNoAction(const PPlayer p, const Vector& target){
  origPlayer = p;
	vel = p->vel;
	pos = p->pos + vel;
	age = WSinfo::ws->time - p->time;
	vel *= ServerOptions::player_decay; // TODO: should be needed, but makes strange errors

	ang = p->ang;
	if(p->number == WSinfo::ws->his_goalie_number)
		kick_radius = 1.2*ServerOptions::catchable_area_l; // play it safe: just hold the ball to pass to others!?
	else
		kick_radius = p->kick_radius;
	float newDistanceToPos = (pos - target).norm();
	reachesPos = (kick_radius >= newDistanceToPos);
	movedToAvoidCollision = false;
}

void DribbleAround::PlayerState::setAssumeToPos(const PPlayer p,const Vector& target){
	origPlayer = p;
	float distToPos = (p->pos - target).norm();

	// see intercept_ball_bms.c, only valid for distances < 10m.
	bool hasToTurn = (fabs((p->ang - (target-p->pos).ARG()).get_value_mPI_pPI())>asin(1/distToPos));
	bool angleOld = p->age_ang > 0;

	if(hasToTurn && !angleOld){
		setAssumeNoAction(p,target);

		// calculate new angle, assuming full turn towards me
		float toOpp = Tools::my_angle_to(pos).get_value_mPI_pPI();
		ang.set_value(toOpp + (toOpp<0) ? PI : -PI);
	}else{
		setAssumeNoAction(p,target); // law of inertia ;-)
		if(angleOld)
			ang = (target - p->pos).ARG();
		Vector noActionPos = pos;
		Vector a(cos(ang),sin(ang)); // has length 1
		float edp = p->dash_power_rate * p->effort * 100; // assume full speed
		a *= edp;
		if(a.norm() > p->speed_max) a.normalize(p->speed_max);
		Vector lot;
		Geometry2d::projection_to_line(lot, target, Line2d(noActionPos,a));
		float distToLot = (lot-noActionPos).norm();
		bool movesBackwards=false;
		if(a.norm()>distToLot){
			Vector pos1 = noActionPos+.01*a; // ahead
			Vector pos2 = noActionPos-.01*a; // backwards
			movesBackwards = (pos1.sqr_distance(target) > pos2.sqr_distance(target));
			pos = lot;
		  vel = pos - origPlayer->pos;
		}else{
			Vector pos1 = noActionPos+a; // ahead
			Vector pos2 = noActionPos-a; // backwards
			movesBackwards = (pos1.sqr_distance(target) > pos2.sqr_distance(target));
			pos = movesBackwards?pos2:pos1;
		  vel = pos - origPlayer->pos;
		}

		float hisMaxSpeedSqr = SQR(p->speed_max);
		a.normalize(0.1);
		int num=0;
		while(p->pos.sqr_distance(pos) > hisMaxSpeedSqr && num++<20){
			pos += movesBackwards?a:-1*a;
		}
		/*
		 * works very good, but makes agent too conservative:
		 * ball stays in kick_radius too long
		static const float playersMinDist=2.5*ServerOptions::player_size;
		if(pos.distance(target)<playersMinDist){
			movedToAvoidCollision = true;
			// can reach pos completely, but will probably 
			// try to avoid collision

			// this is a little ugly:
			// assume that the player does not want to "switch sides" in one step
			// (i.e. go to opposite side of object)
			bool switchesSides = fabs((p->pos-target).ANGLE_to(pos-target).get_value_mPI_pPI())>.8*PI;
			switchesSides=false;

			Vector toTar;
			toTar = target - noActionPos;
			pos=target;                                      // all the way
			toTar.normalize(playersMinDist);                 // min dist

			if(!switchesSides)
				pos -= toTar;                                    // "step back"
			else
				pos += toTar;
		}
		*/
	}
	
	// kick_radius = p->kick_radius; // is set in assumeNA, taking into account goalie catch radius
	float newDistanceToPos = (pos - target).norm();
	reachesPos = (kick_radius >= newDistanceToPos);
}

void DribbleAround::PlayerState::setAssumeToPos(const PPlayer p,const Ball* b){
	setAssumeToPos(p,b->pos);
}
void DribbleAround::PlayerState::setAssumeToPos(const PPlayer p,const BallState& b){
	setAssumeToPos(p,b.pos);
}
void DribbleAround::PlayerState::setAssumeToPos(const PPlayer p,const PlayerState& t){
	setAssumeToPos(p,t.pos);
}
void DribbleAround::PlayerState::setAssumeNoAction(const PPlayer p,const BallState& b){
	setAssumeNoAction(p,b.pos);
}
void DribbleAround::PlayerState::setAssumeNoAction(const PlayerState& p,const Vector& v){
	setAssumeNoAction(p.origPlayer,v);
}

void DribbleAround::PlayerState::setNow(const PPlayer p, const Ball* b){
	origPlayer = p;
	pos = p->pos;
	vel = p->vel;
	age = p->age;
	kick_radius = p->kick_radius;
	ang = p->ang;
	reachesPos = kick_radius>(b->pos.distance(p->pos));
}

void DribbleAround::PlayerState::testCollision(const PlayerState& p){
	double distP2B = pos.distance(p.pos);
	double r = origPlayer->radius+p.origPlayer->radius;
	if(distP2B < r){
		movedToAvoidCollision = true;
		POL("FOUND COLLISION while setting player, corrected player info, r="<<r);
		Vector dif = pos - p.pos;
		Angle th = fabs(dif.angle(vel));
		double l1 = distP2B * cos(th) ;
		double h = distP2B * sin(th) ;
		double cosp = h / r ;
		double sinp = sqrt(1.0 - SQR(cosp)) ;
		double l2 = r * sinp ;
		Vector dv = p.vel;
		dv.normalize(-(l1 + l2)) ;
		pos = p.pos+dv;
		vel = p.vel;
		vel *=-.1;
	}
}
void DribbleAround::PlayerState::setNowTestCollision(const PPlayer p, const Ball* b){
	setNow(p,b);
	double distP2B = p->pos.distance(b->pos);
	double r = p->radius+ServerOptions::ball_size;
	if(distP2B < r){
		POL("FOUND COLLISION, corrected player info, r="<<r);
		Vector dif = p->pos - b->pos;
		Angle th = fabs(dif.angle(p->vel));
		double l1 = distP2B * cos(th) ;
		double h = distP2B * sin(th) ;
		double cosp = h / r ;
		double sinp = sqrt(1.0 - SQR(cosp)) ;
		double l2 = r * sinp ;
		Vector dv = p->vel;
		dv.normalize(-(l1 + l2)) ;
		pos = p->pos+dv;
		vel = p->vel;
		vel *=-.1;
		reachesPos=pos.distance(b->pos)<kick_radius;
	}
}


void DribbleAround::BallState::setNowTestCollision(const PPlayer& p,const Ball* b){
	double distP2B = p->pos.distance(b->pos);
	double r = p->radius+ServerOptions::ball_size;
	if(distP2B < r){ // collision
		POL("FOUND COLLISION, corrected ball info");
		Vector dif = b->pos - p->pos;
		Angle th = fabs(dif.angle(b->vel));
		double l1 = distP2B * cos(th) ;
		double h = distP2B * sin(th) ;
		double cosp = h / r ;
		double sinp = sqrt(1.0 - SQR(cosp)) ;
		double l2 = r * sinp ;
		Vector dv = b->vel;
		dv.normalize(-(l1 + l2)) ;
		pos = b->pos+dv;
		vel = b->vel;
		vel *=-.1;
	}else{
		setNow(b);
	}
}
void DribbleAround::BallState::setNow(const Ball* b){
	pos = b->pos;
	vel = b->vel;
}
void DribbleAround::BallState::setNext(const BallState& b){
	vel = b.vel;
	vel *= ServerOptions::ball_decay;
	pos = b.pos + vel;
}
void DribbleAround::BallState::setAssumeNoAction(const Ball* b){
	vel = b->vel;
	vel *= ServerOptions::ball_decay;
	pos = b->pos + vel;
}
