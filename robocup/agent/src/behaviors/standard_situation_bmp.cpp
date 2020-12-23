// vim:ts=2:sw=2:ai:fdm=marker:fml=3:filetype=cpp:fmr=>>>,<<<
#include "standard_situation_bmp.h"

#include "../basics/PlayerSet.h"
#include "ws_info.h"
#include "blackboard.h"
#include "../policy/positioning.h"
#include "geometry2d.h"
#include "tools.h"
#include "../policy/policy_tools.h"
#include "mdp_info.h"

#if 0   /* 1: debugging; 0: no debugging */
#include "log_macros.h"
#define POL(XXX)   LOG_POL(0,<<"StdSit: "<<XXX)
#define POL2(XXX)  LOG_POL(1,<<"StdSit: "<<XXX)
#define DRAW(XXX)  LOG_POL(0,<<_2D<<XXX)
#define DRAW2(XXX) LOG_POL(1,<<_2D<<XXX)
#else
#define POL(XXX)
#define POL2(XXX)
#define DRAW(XXX)
#define DRAW2(XXX)
#endif
// Drawing macros >>>1
#define MARK_POS(P,C) DRAW(C2D((P).x,(P).y,0.3,#C));
#define DRAW_LINE(P,Q,C) DRAW(L2D((P).x,(P).y,(Q).x,(Q).y,#C))
// Syntactic Sugar >>>1
#define RETPOL(V,XXX) if(1){POL(XXX<<": "<<#V<<" = "<<V); return V;}
#define V(X) " "<< #X << "="<<X
#define SQR(X) ((X)*(X))
#define sgn(X) (((X)<0)?-1:1)


// Static Variables >>>1
StandardSituation* StandardSituation::svpInstance = NULL;
bool   StandardSituation::svIsInitialized = false;
int    StandardSituation::svMaxConsecKicks = 2; //changed by TG: original cijat-value was 6

double StandardSituation::homepos_tolerance = 0.8;
int    StandardSituation::homepos_stamina_min = 2250;//old: TG: 2000
double StandardSituation::clearance_radius_min = 10.0;
int    StandardSituation::max_ball_age_when_kick = 4;
Vector StandardSituation::svKickoffTarget    = Vector(-13.0, 20.0);
double StandardSituation::svKickoffSpeed     = 2.3;
int    StandardSituation::svPassAnnounceTime = 3;

bool StandardSituation::init(char const * conf_file, int argc, char const* const* argv){ // >>>1
  if(svIsInitialized)
    return true;

  bool result = true;

  result &= NeuroGo2Pos::init(conf_file,argc,argv);
  result &= Wball06::init(conf_file,argc,argv);
  result &= FaceBall::init(conf_file,argc,argv);
  result &= NeuroKick05::init(conf_file,argc,argv);

  if(result)
    svIsInitialized = true;

  return result;
}

StandardSituation::StandardSituation(){ // >>>1
  if(!svIsInitialized){
    POL("Must initialize first! -- exiting.")
    exit(1);
  }
  ivLastActivationTime = -100;
  ivpBasicCmd  = new BasicCmd;
  ivpScore     = new Score05_Sequence;
  ivpIntercept = new NeuroIntercept;
  ivpGo2Pos    = new NeuroGo2Pos;
  ivpFaceBall  = new FaceBall;
  ivpWBall     = new Wball06;
  ivpNeuroKick = new NeuroKick05;
  ivpPlayPass  = new PlayPass;
}

// main behavior function
bool StandardSituation::get_cmd(Cmd & cmd) { // >>>1
  ivMyNumber = WSinfo::me->number;
  // check if situation changed since last call
  if (WSinfo::ws->time - ivLastActivationTime > 2  
      ||(  WSinfo::ws->play_mode != PM_PlayOn 
	&& WSinfo::ws->play_mode != ivLastPlayMode)){
    POL("Restart.");
    ivActSitStartTime = WSinfo::ws->time;
    ivLastPlayMode    = (PlayMode)WSinfo::ws->play_mode;
    ivWeShallKick     = shallWeKick(ivLastPlayMode);
    ivLastKickSpeed   = 0;
    ivpLastBehavior   = NULL;
    ivConsecKicks     = 0;
  }

  ivLastActivationTime = WSinfo::ws->time;
  ivActSitDuration     = WSinfo::ws->time - ivActSitStartTime;
  ivTimeLeft           = this->getTimeLeft();

	POL(V(ivTimeLeft)<<V(ivActSitDuration));

  // we dont handle normal play behavior, but let player complete 2-step-kicks and prevent
  // self-passes just after the end of a standard situation
  if (getCmdContinued(cmd)) {
    write2blackboard();
    return true;
  }

  // set ball pos to reasonable value according to PlayMode
  ivBallPos = getCleanBallPos();

  POL("INFO: duration=" <<  ivActSitDuration << "  ,time_left=" << ivTimeLeft);

  POL("num=" <<ivMyNumber<<" act="<<0<<" pm="<<WSinfo::ws->play_mode);

  //hauke: hack TODO: check this out
  if(ivMyNumber != ivKickerNr
      && ivLastPlayMode        == PM_my_GoalKick 
      && WSinfo::ws->play_mode == PM_PlayOn)
    return false;

  ivKickerNr = getKickerNumber();

  bool res;
  if (ivMyNumber == ivKickerNr)
    res = getCmdKicker(cmd);
  else
    res = getCmdNonKicker(cmd);

  write2blackboard();

  return res;
}

bool StandardSituation::getCmdContinued(Cmd & cmd) // >>>1
{
	/*
	 * cijat bremen
  if (WSinfo::ws->play_mode != PM_PlayOn && ivConsecKicks == 0){
		POL("Cannot continue last cmd: no such thing.");
    return false;
	}
	*/

  // some deadlock !? / player thinks can kick, but can't
  if (ivConsecKicks >= svMaxConsecKicks) {
    ivConsecKicks   = 0;
    ivpLastBehavior = NULL;
    ivLastKickSpeed = 0;
		POL("Cannot continue last cmd: Too many kicks.");
    return false;
  }

  // just continue last kick, if possible
  if (ivLastKickSpeed > 0) {
    POL("PM_PlayOn, try continue kick (spd="<<ivLastKickSpeed<<",target="<<ivLastKickTarget<<")");
    ivpNeuroKick->reset_state();
    ivpNeuroKick->kick_to_pos_with_initial_vel(ivLastKickSpeed, ivLastKickTarget);
    if (ivpNeuroKick->get_cmd(cmd)){}
      ivConsecKicks++;
      ivLastIntention.confirm_intention(ivLastIntention, WSinfo::ws->time);
      POL("PM_PlayOn kick continued, int. set");
      return true;
  }

  // otherwise try to continue last behavior
	if (ivpLastBehavior == ivpScore){
		POL("PM_PlayOn, try scoring again");
		Intention intention;
		POL("ivpScore->test_shoot2goal");
		if(ivpScore->test_shoot2goal( intention )) {
			POL("ivpScore->get_cmd");
			if(ivpScore->get_cmd(cmd)){
				POL("Continue scoring...");
				ivpWBall->intention2cmd(intention,cmd);
				ivLastIntention = intention;
				return true;
			}
		}
	}
	else if ( ivpLastBehavior == ivpWBall ) {
    POL("PM_PlayOn, try pass again.");
    ivpLastBehavior = 0;
		Intention intention;
		if (getPassIntention(intention)) {
			if (ivpWBall->intention2cmd(intention,cmd)) {
				ivpLastBehavior = ivpWBall;
				ivConsecKicks++;
        intention.get_kick_info(ivLastKickSpeed, ivLastKickTarget);
				ivLastIntention = intention;
				return true;
      }
    }
		return false;
  } else if ( ivpLastBehavior == ivpNeuroKick ) {
    POL("PM_PlayOn, try neurokick again.");
    ivpLastBehavior = NULL;
    ivpNeuroKick->reset_state();
    ivpNeuroKick->kick_to_pos_with_max_vel(ivLastKickTarget);
    if ( WSinfo::is_ball_kickable() && ivpNeuroKick->get_cmd(cmd)) {
      POL("PM_PlayOn, call neuro_kick again.");
      ivConsecKicks++;
      ivpLastBehavior  = ivpNeuroKick;
      ivLastKickSpeed = ServerOptions::ball_speed_max;
      ivLastIntention.confirm_intention(ivLastIntention, WSinfo::ws->time);
      return true;
    }
  }
	if(WSinfo::ws->play_mode == PM_PlayOn){
		POL("PM_PlayOn, turn to ball.");
		ivConsecKicks = 0;
		ivpFaceBall->turn_to_ball();
		ivpFaceBall->get_cmd(cmd);
		return true;
	}
	return false;
}

void StandardSituation::write2blackboard() { // >>>1
  if (ivLastIntention.valid_at() == WSinfo::ws->time) {
    Blackboard::pass_intention = ivLastIntention;
    double sp; Vector t; int i;
    ivLastIntention.get_kick_info(sp, t, i);
    POL("blackb::p_int. set: type="<<ivLastIntention.get_type() <<",validSince="<<ivLastIntention.valid_since()<<",valid="<<ivLastIntention.valid_at()<<",sp="<<sp<<",targ="<<t<<",player="<<i);
  }
}

// sets the internally used ball position and corrects it using playmode information if necessary
Vector StandardSituation::getCleanBallPos() { // >>>1
  Vector ballPos = WSinfo::ball->pos;

  if (ivLastPlayMode == PM_my_KickOff || ivLastPlayMode == PM_his_KickOff)
    ballPos = Vector(0, 0);

  if (ivLastPlayMode == PM_my_CornerKick)
    ballPos.setX( FIELD_BORDER_X - 1.0 );

  if (ivLastPlayMode == PM_his_CornerKick)
    ballPos.setX( -(FIELD_BORDER_X  - 1.0) );

  if (ivLastPlayMode == PM_my_KickIn || ivLastPlayMode == PM_his_KickIn) {
    if (ballPos.getY() > FIELD_BORDER_Y - 2.0)
      ballPos.setY( FIELD_BORDER_Y );
    if (ballPos.getY() < -FIELD_BORDER_Y + 2.0)
      ballPos.setY( -FIELD_BORDER_Y );
  }
  return ballPos;
}

double StandardSituation::getKickerVal(const PPlayer& p){
  double distToBall = p->pos.distance(ivBallPos);
	PPlayer opp = OpponentAwarePositioning::getDirectOpponent(p->number);
	double distBallOwnOpp=0;
	if(opp)
		distBallOwnOpp = opp->pos.distance(ivBallPos);
	return distToBall+distBallOwnOpp;
}

int StandardSituation::getKickerNumber(){ // >>>1
  int kickernr = -1;
  if(!ivWeShallKick)
    RETPOL(kickernr,"We shall not kick");
	if(WSinfo::ws->play_mode == PM_my_GoalKick){
		kickernr = WSinfo::ws->my_goalie_number;
		if(kickernr<1 || kickernr>11)
			kickernr = 1;  // fallback
		RETPOL(kickernr,"PM_my_GoalKick, goalie shall kick.");
	}
  if(ivLastPlayMode == PM_my_CornerKick){
    if(ivBallPos.getY() > 0)
      kickernr = 9;
    else
      kickernr = 11;
    RETPOL(kickernr,"Corner kick");
  }
    
  PlayerSet allowed=WSinfo::valid_teammates;
	if(ivBallPos.getX()>FIELD_BORDER_X*.4)
		allowed.remove(allowed.get_player_by_number(4));

	if(ivBallPos.getX()>FIELD_BORDER_X*.2) /*begin TG*/
	{
		allowed.remove(allowed.get_player_by_number(2));//left defender
		allowed.remove(allowed.get_player_by_number(5));//right defende
	}
	if(ivBallPos.getX()>FIELD_BORDER_X*.0)
		allowed.remove(allowed.get_player_by_number(3));//central defender /*end TG*/

	PlayerSet team = allowed;
  team.keep_and_sort_by_func(1,&StandardSituation::getKickerVal,this);
  if(team.num<1)
    RETPOL(kickernr,"No closest Player found");
	kickernr = team[0]->number;

	team = allowed;
	team.keep_and_sort_closest_players_to_point(3,ivBallPos);
	if(team.num>1){
		kickernr = team[(team[0]->number>team[1]->number)?0:1]->number;
		RETPOL(kickernr,"2 players already very close to ball, greater number wins.");
	}
	team.keep_players_in_circle(ivBallPos,1);
	if(team.num>0){
		// one player is already at ball _and_ is allowed to kick
		kickernr = team[0]->number;
	}
  RETPOL(kickernr,"Closest allowed Player shall go to ball");
}
bool StandardSituation::shallWeKick(int playmode) { // >>>1
  return  ( playmode == PM_my_KickIn     || playmode == PM_my_FreeKick
	|| playmode == PM_my_CornerKick || playmode == PM_my_OffSideKick
	|| playmode == PM_my_KickOff || playmode == PM_my_GoalKick
	|| playmode == PMS_my_Kick); // should never happen (PMS_my_Kick)
}
bool StandardSituation::isStartPlayerMissing() { // >>>1
  PlayerSet players_near_ball = WSinfo::valid_teammates_without_me;
  players_near_ball.keep_players_in_circle(ivBallPos, 3.0);
  return (players_near_ball.num == 0);
}
// set value of time_left, i.e. estimated time until ball will be kicked
int StandardSituation::getTimeLeft() { // >>>1
  static const int defaultWaitPeriod = 40;

  // do not wait before kickoff
  int initWait = (ivLastPlayMode==PM_my_KickOff)?0:defaultWaitPeriod;

  // wait until recovered
  int stamina_regen = int(4000 - WSinfo::me->stamina) / 30; // 40 as compromise
  if (initWait < stamina_regen) 
    initWait = stamina_regen;

  // take goal diff into account
  int goalDiff = WSinfo::ws->my_team_score - WSinfo::ws->his_team_score;
  if (goalDiff > 0 && goalDiff < 3) 
    initWait += 30 * goalDiff;
  if (goalDiff < 0) 
    initWait += 20 * goalDiff;
  
  // cut too high/low values
  if (initWait < 0) initWait = 0;
  if (initWait > 160) initWait = 160;
  
  // calculate time of kick
  int kick_time = ivActSitStartTime + initWait;

  // kick_time should be _at_least_ now.
  if (kick_time < WSinfo::ws->time) 
    kick_time = WSinfo::ws->time;
  int timeLeft = kick_time - WSinfo::ws->time;
  RETPOL(timeLeft,"Calculated time to kick");
}
// try to kick at all circumstances, even if bad kick results
bool StandardSituation::getCmdPanicKick(Cmd & cmd)
{
  // must see ball, otherwise 2-step-kicks etc. will not function correctly
  if (WSinfo::ball->age > max_ball_age_when_kick) {
    POL("PANIC-kick: faceBall.");
    ivConsecKicks = 0;
    ivpFaceBall->turn_to_ball();
    ivpFaceBall->get_cmd(cmd);
    return true;
  }
  
  // continue started kicks
	Intention intention;
  if (getPassIntention(intention)) {
    if (ivpWBall->intention2cmd(intention,cmd)) {
			ivConsecKicks++;
      intention.get_kick_info(ivLastKickSpeed, ivLastKickTarget);
      ivLastIntention = intention;
    }
    ivpLastBehavior= ivpWBall;
    POL("PANIC-kick: pass played.");
    return true;
  }

	// TODO ins freie Feld oder so!?
	Vector target = Vector(50, 0); // just shoot at goal
  
  // kick the ball if possible 
  ivpNeuroKick->reset_state();
  ivpNeuroKick->kick_to_pos_with_max_vel(target);
  if (WSinfo::ball->age <= max_ball_age_when_kick 
			&& ivpNeuroKick->get_cmd(cmd)) {
    POL("PANIC-kick: neuroKick.");
    ivConsecKicks++;
    ivpLastBehavior  = ivpNeuroKick;
    ivLastKickTarget = target;
    ivLastKickSpeed  = ServerOptions::ball_speed_max;
    Intention intention;
    intention.set_panic_kick(target, WSinfo::ws->time);
    ivLastIntention = intention;
    return true;
  }

  // go to ball if still not reached it
  POL("PANIC-kick: interceptBall.");
  ivConsecKicks = 0;
  ivpIntercept->get_cmd(cmd);
	return true;
}
bool StandardSituation::getCmdNonKicker(Cmd& cmd){ // >>>1
  // must search ball?
  if (!WSinfo::is_ball_pos_valid()) {
    ivpFaceBall->turn_to_ball();
    ivpFaceBall->get_cmd(cmd);
    POL("search Ball (duration=" << ivActSitDuration << ",  ball_age=" << WSinfo::ball->age << ")");
    return true;
  }

	Vector target = getTargetNonStartPlayer();
  
  PPlayer p= WSinfo::get_teammate_with_newest_pass_info();
  if (p && getCmdReactOnPassInfo(cmd,p,target))
		return true;

	if (getCmdReplaceMissingStartPlayer(cmd))
		return true;


	MARK_POS(target,000000);
	DRAW_LINE(WSinfo::me->pos,target,000000);
	Vector lot;
	Geometry2d::projection_to_line(lot,ivBallPos,Line2d(WSinfo::me->pos,target-WSinfo::me->pos));

	// find out whether I need to walk thru circle to get to target >>>2
	bool targetIsJustWaypoint = false;
	Vector origTarget;
	static const double keepAwayRadius = 9.4;
	if(!ivWeShallKick
			&& lot.sqr_distance(ivBallPos)+.2 < SQR(keepAwayRadius)            // line have to go thru circle
			&& lot.sqr_distance(target)<WSinfo::me->pos.sqr_distance(target)){ // lot is ahead
		POL("would have to walk thru circle: moved target to tangent");
		Vector origTarget = target;
		target = getTangentIsct(ivBallPos,keepAwayRadius,WSinfo::me->pos, origTarget);
		targetIsJustWaypoint=true;
		DRAW_LINE(WSinfo::me->pos,target,222222);
		DRAW_LINE(target,origTarget,000000);
	} // >>>2

	// if i have to hurry up >>>2
	static const float homepos_tolerance_at_offside_line = .5;
	bool dangerousHisFreeKick = 
	       WSinfo::me->pos.distance(target) > homepos_tolerance
			&& WSinfo::ws->play_mode == PM_his_FreeKick 
			&& WSinfo::ball->pos.distance(MY_GOAL_CENTER)<35.0;

	bool iAmInOffside = 
		     WSinfo::me->pos.distance(target) > homepos_tolerance_at_offside_line
			&& WSinfo::me->pos.getX() >= WSinfo::his_team_pos_of_offside_line();

	if(dangerousHisFreeKick)
	{
		POL("Dangerous HisFreeKick, hurry up");
		ivpGo2Pos->set_target(target);
		return ivpGo2Pos->get_cmd(cmd);
	}

	if(iAmInOffside){
		POL("I am in Offside, hurry up!");
	  target.subFromX( .2 );
		ivpGo2Pos->set_target(target,homepos_tolerance_at_offside_line);
		return ivpGo2Pos->get_cmd(cmd);
	}

	double distMeTarget   = WSinfo::me->pos.distance(target);
	double distMeBall     = WSinfo::me->pos.distance(ivBallPos);
	double distTargetBall = target.distance(ivBallPos);
	bool noTimeLeftToWalkToHomePos = ivTimeLeft<svPassAnnounceTime;
  if ( distMeTarget < homepos_tolerance 
    || WSinfo::me->stamina < homepos_stamina_min
		|| (noTimeLeftToWalkToHomePos && distTargetBall > distMeBall
        && !(WSinfo::ws->play_mode==PM_my_GoalKick && LEFT_PENALTY_AREA.inside(WSinfo::me->pos) ) )) 
  {
    POL("getCmdGo2PosWithCare: near target (d="<<distMeTarget
				<<") or low stamina (="<<WSinfo::me->stamina
				<<") or no time left)--> stay here.");	    

		PPlayer opp = OpponentAwarePositioning::getDirectOpponent(WSinfo::me->number);
		if(opp){ POL("Age of my opp is "<<opp->age); }
		if(ivWeShallKick){
			if(WSinfo::ws->time%20<7)  // TODO: make this parameter
				if(getCmdScanField(cmd))
					return true;
		}else{
			if(getCmdScanField2nd(cmd))
				return true;
		}

		if(fabs(Tools::my_angle_to(ivBallPos).get_value_mPI_pPI())>DEG2RAD(5)){
			POL("turning to ball...");
			ivpFaceBall->turn_to_ball();
			ivpFaceBall->get_cmd(cmd);
			return true;
		}
		return false;
	}

	bool returnValue = targetIsJustWaypoint
		?getCmdGo2PosWithCare(cmd, target, false,WSinfo::me->pos.distance(origTarget))
		:getCmdGo2PosWithCare(cmd, target, false);

	return returnValue;
}

bool StandardSituation::getCmdReactOnPassInfo(Cmd& cmd, const PPlayer p,const Vector& origTarget){ // >>>1
	bool gotPassInfo=false;
	Vector ballpos,ballvel;
	ANGLE dir;
	dir = ANGLE(p->pass_info.ball_vel.arg());
	ballpos = p->pass_info.ball_pos;
	ballvel = p->pass_info.ball_vel;
	Vector target_pos;
	POL("got pass info from player " 
			<< p->number  
			<< " a= " << p->pass_info.age
			<< " p= " << ballpos
			<< " v= " << p->pass_info.ball_vel
			<< " at= " << p->pass_info.abs_time
			<< " rt= " << p->pass_info.rel_time
			<<" speed= "<<speed <<" dir "<<RAD2DEG(dir.get_value()));
	
	if(    false
			&& WSinfo::ws->play_mode == PM_my_CornerKick 
			&& isPassInfoTheSignal(p)){
		if(getCornerRole() == L1_M){
			POL("I heared the signal, react (L1_M)");
			Vector ballToMe = WSinfo::me->pos - ivBallPos;
			ballToMe.normalize(6);
			ballToMe += ivBallPos;
			ivpGo2Pos->set_target(ballToMe);
			return ivpGo2Pos->get_cmd(cmd);
		}
		if(getCornerRole() == L2_L){
			POL("I heared the signal, react (L2_L)");
			Vector ballToMe = WSinfo::me->pos - ivBallPos;
			ballToMe.normalize(6);
			ballToMe += ivBallPos;
			ivpGo2Pos->set_target(ballToMe);
			return ivpGo2Pos->get_cmd(cmd);
		}
	}

	// check if i should intercept
	InterceptResult ires[8];
	PlayerSet ps = WSinfo::valid_teammates;
	ps.keep_and_sort_best_interceptors_with_intercept_behavior(3,ballpos,ballvel,ires);
	for(int p=0;p<ps.num;p++) {
		POL("Best intercept: p="<<p<<",#="<<ps[p]->number <<",pos="<<ires[p].pos);
		if(ps[p]->number == WSinfo::me->number) {
			if(WSinfo::me->pos.distance(ivBallPos)>35){
				POL("Could go for pass but i am too far away, see info might be wrong");
				continue;
			}
			Vector ipos,ballpos,ballvel;
			if(Policy_Tools::check_go4pass(ipos,ballpos,ballvel) == false) {
				POL("Check_go4pass failed, i don't intercept!");
			} 
			else {
				POL("I intercept!");
				MARK_POS(ires[p].pos, 0000ff);
				POL("ires" << ires[p].pos);
				mdpInfo::set_my_intention(DECISION_TYPE_INTERCEPTBALL); 
				gotPassInfo=true;
				ballvel.normalize();
				target_pos.clone( ballpos );
				target_pos=ires[p].pos;
			}
		}
	}

	if (ivWeShallKick && gotPassInfo){
		DRAW(L2D(target_pos.x,target_pos.y,WSinfo::me->pos.x,WSinfo::me->pos.y,"aaaaaa"));
		ivpIntercept->set_virtual_state(target_pos,Vector(0,0));
		bool res = ivpIntercept->get_cmd(cmd);
		RETPOL(res,"Intercept!");
		//go_to_pos(cmd, target_pos, true);     
	}
	return false;
}

bool StandardSituation::getCmdReplaceMissingStartPlayer(Cmd&cmd){ // >>>1
  // Kein Startplayer da, aber wir haben Ball
  static const int time_replace_missing_start_player = 60;
  static const double dist_factor_replace_missing_start_player = 1.0;
  if ( ivWeShallKick 
      && isStartPlayerMissing() 
      && ivLastPlayMode != PM_my_GoalKick
      && ivActSitDuration > (time_replace_missing_start_player 
			  + dist_factor_replace_missing_start_player
			    * (WSinfo::me->pos - ivBallPos).norm())
			&& ivBallPos.distance(WSinfo::me->pos)<35) // others: do not use up your stamina
  {
    POL("i'm n-s-pl, but nobody near ball --> i will kick!!");
    ivKickerNr = ivMyNumber;
    return getCmdKicker(cmd);
  }
	return false;
}

Vector StandardSituation::getTargetNonStartPlayer(){ // >>>1
	Vector target = OpponentAwarePositioning::getCoverPosition();

	OpponentAwarePosition oap;
	//TG -> special cases (for special play modes)
	switch ( WSinfo::ws->play_mode )
	{
		case PM_my_GoalKick:
			{
				//do not use my hack, but the default solution.
				POL("TG-HACK: mode=my_GoalKick"<<flush);
				oap.clone( target );
				PlayerSet tmm = WSinfo::valid_teammates_without_me;
				tmm.keep_players_in_circle(WSinfo::me->pos, 5.0);
				if (tmm.num>0 && WSinfo::me->number > tmm[0]->number)
				{
					oap.addToX( 6.0 );
					POL("TG-HACK: too near to a teammate, change oap.x="<<oap.getX()<<flush);
				}
				if(oap.getX() > FIELD_BORDER_X-16)
					oap.setX( FIELD_BORDER_X-16 );
          
        if (WSinfo::me->number==2 || WSinfo::me->number==5)
        {
          Vector targetVector( target );
          PlayerSet opp = WSinfo::valid_opponents;
          opp.keep_players_in_circle( targetVector, 6.0 );
          if (opp.num>0)
          {
            if (target.getY() >= 0) oap.setY( target.getY() + 6.0 );
            if (target.getY() <  0) oap.setY( target.getY() - 6.0 );
            if (oap.getY() > 30.0) oap.setY(  30.0 );
            if (oap.getY() <-30.0) oap.setY( -30.0 );
          }
        }
				break;
			}
		case PM_my_KickIn:
			{
				if(ivBallPos.getX()>-FIELD_BORDER_X+35){
					Vector v = getTargetNonStartPlayerMyKickIn();
					oap.clone( v );
				}
				else
					oap = OpponentAwarePositioning::getOpponentAwareMyKickInPosition();
				POL("TG-HACK: mode=my_KickIn"<<flush);
				break;
			}
		case PM_his_GoalKick:
			{
				oap = OpponentAwarePositioning::getCoverPosition();
				if ( oap.getX() > FIELD_BORDER_X - 17.0 )
					oap.setX( FIELD_BORDER_X - 17.0 );
				break;
			}
		case PM_my_KickOff: 
		case PM_his_KickOff:
			{
				Vector homePos = OpponentAwarePositioning::getHomePosition();
				if (homePos.distance(WSinfo::me->pos) > 15.0)
				{
					oap.clone( homePos );
					POL("TG-hack: target is homeposition (kick off)");
				}
				else
				{
					oap.clone( WSinfo::me->pos );
					POL("TG-hack: target is my own position (kick off)");
				}
				break;
			}
		case PM_my_CornerKick:
			{
				oap   = OpponentAwarePositioning::getAttackOrientedPosition(0.2);
				if(getCornerRole()!=NOT_INVOLVED){
					Vector v = getTargetNonStartPlayerMyCorner();
					oap.clone( v );
				}
				break;
			}
		default:
			{
				POL("mode="<<WSinfo::ws->play_mode<<V(ivWeShallKick));
				if ( ivWeShallKick )
					oap = OpponentAwarePositioning::getAttackOrientedPosition(0.2);
				else
				{
					if (WSinfo::ball->pos.getX() < 0 || WSinfo::me->stamina < 3500)
						oap = OpponentAwarePositioning::getCoverPosition();
					else
						oap = OpponentAwarePositioning::getAttackOrientedPosition(0.0);
				}
			}
	} //end of switch

	target.clone( oap );

	// avoid field borders
	if (target.getX()> FIELD_BORDER_X-1.0) target.setX(  FIELD_BORDER_X-1.0 );
	if (target.getX()<-FIELD_BORDER_X+1.0) target.setX( -FIELD_BORDER_X+1.0 );
	if (target.getY()> FIELD_BORDER_Y-1.0) target.setY(  FIELD_BORDER_Y-1.0 );
	if (target.getY()<-FIELD_BORDER_Y+1.0) target.setY( -FIELD_BORDER_Y+1.0 );

	if(    ivBallPos.getX()<20
			&& WSinfo::ws->play_mode != PM_my_KickOff 
			&& WSinfo::ws->play_mode != PM_his_KickOff
		  && OpponentAwarePositioning::getRole(WSinfo::me->number)==PT_DEFENDER){
			// ball is (~) in our half
			// ==> move defense line behind ball towards goal
			Vector toGoal = .04*(MY_GOAL_CENTER - target);
			target += toGoal;
	}

	/*
	 * cijat: deleted: should walk around circle instead!
	static const double keepAwayRadius = 9.4;
	if(!ivWeShallKick){
		if(target.distance(ivBallPos)<keepAwayRadius){
			POL("Moving target 9.1m away from ball!");
			target.init_polar(keepAwayRadius,(WSinfo::me->pos-ivBallPos).ARG());
			target += ivBallPos;
		}
	}
	*/

	static const double keepAwayRadius = 9.4;
	if(!ivWeShallKick){
		PPlayer opp = OpponentAwarePositioning::getDirectOpponent(WSinfo::me->number);
		if(opp){
			if(opp->pos.sqr_distance(ivBallPos)<SQR(keepAwayRadius)){
				POL("'my' opp is in no-go circle, blocking his way to goal");
				target.init_polar(keepAwayRadius,(MY_GOAL_CENTER-ivBallPos).ARG());
				target+=ivBallPos;
				if((fabs(ivBallPos.getY())>FIELD_BORDER_Y-8) && (target.getX()<-50.0)){
					POL("I'm supposed to go quite close to our goal line("<<V(target.x)<<"), won't do that...");
					target.setX( -50.0 );
				}
				PlayerSet team=WSinfo::valid_teammates;
				team.keep_and_sort_closest_players_to_point(1,target);
				if(team.num && team[0]->pos.sqr_distance(target)<SQR(4)){
					POL("There's a player at the position i wanted to go to, stay clear");
					target.init_polar(keepAwayRadius,(WSinfo::me->pos-ivBallPos).ARG());
					target+=ivBallPos;
				}
			}
		}
	}
	return target;
}

void StandardSituation::modifyDashCmd(Cmd& cmd){ // >>>1
	//modify the command in case of low stamina
	if (   //we only need to modify dash commands
			cmd.cmd_body.get_type() == cmd.cmd_body.TYPE_DASH
			// my stamina is not _that_ high
			&& (WSinfo::me->stamina < 3500)
			//we must be careful when it is the opponent's standard sit and
			//it is very near our goal (>35 includes his corner kicks!)
			&& (ivWeShallKick || WSinfo::ball->pos.distance(MY_GOAL_CENTER)>35.0 )
			&& WSinfo::ws->play_mode != PM_my_GoalKick
			&& ! ( OpponentAwarePositioning::getRole(WSinfo::me->number) == PT_DEFENDER 
				     && ivWeShallKick==false ) )
	{
		double relativeMaxDash = 1.0;
		relativeMaxDash = (      ( (WSinfo::me->stamina-1250.0) / (4000.0-1250.0)) //-> in [0.3...1.0]
				+  ( Tools::max( 0.0, 1.0 - (WSinfo::me->pos.distance(WSinfo::ball->pos) / FIELD_BORDER_X) ) )//-> in [0.0..1.0]
				)  / 2.0;
		POL("TG-HACK: relative max dash power is "<<relativeMaxDash);
		double intendedDashPower, allowedDashPower;
		cmd.cmd_body.get_dash( intendedDashPower );
		POL("TG-HACK: intended dash power is "<<intendedDashPower);
		if (intendedDashPower < 0) intendedDashPower *= 2;
		if ( fabs(intendedDashPower) > relativeMaxDash * WSinfo::me->stamina_inc_max )
			allowedDashPower = relativeMaxDash * WSinfo::me->stamina_inc_max;
		else
			allowedDashPower = intendedDashPower;
		if (intendedDashPower < 0) allowedDashPower *= -0.5;
		POL("TG-HACK: setting to allowed dash power "<<allowedDashPower);
		cmd.cmd_body.unset_lock();  cmd.cmd_body.unset_cmd();  
		cmd.cmd_body.set_dash( allowedDashPower );
	}
}



bool StandardSituation::getCmdKicker(Cmd& cmd){ // >>>1

	/*
	 * Do not shoot directly!
	 *
  if (WSinfo::is_ball_kickable()){
		Intention intention;
		if(ivpScore->test_shoot2goal( intention )) {
			if(ivpScore->get_cmd(cmd)){
				ivpLastBehavior = ivpScore;
				ivLastIntention = intention;
				ivpWBall->intention2cmd(intention,cmd);
				POL("Trying to score goal.");
				return true;
			}
		}
	}
	*/

  // if for 150 cycles nothing happened, just kick somewhere
  if (WSinfo::is_ball_kickable()&&200 - ivActSitDuration < 50){ 
    if (ivLastPlayMode != PM_my_GoalKick){
      return getCmdPanicKick(cmd);
    }
    else{ 
      // GoalKick entsch?rfen (-->ecke)
      if (WSinfo::me->pos.getY()<0)
				ivpNeuroKick->kick_to_pos_with_max_vel(Vector(-55,-25));
      else 
				ivpNeuroKick->kick_to_pos_with_max_vel(Vector(-55,25));
      return ivpNeuroKick->get_cmd(cmd);
    }
  }

  // must search ball?
  if (WSinfo::ball->age >= ivActSitDuration 
   || WSinfo::ball->age >  max_ball_age_when_kick) {
    ivConsecKicks = 0;
    ivpFaceBall->turn_to_ball();
    ivpFaceBall->get_cmd(cmd);
    POL("search Ball (duration=" << ivActSitDuration << ",  ball_age=" << WSinfo::ball->age << ")");
    return true;
  }

  // kick-off: hardcoded kick
  if (ivLastPlayMode == PM_my_KickOff && WSinfo::is_ball_kickable()) {
    Vector target = svKickoffTarget;
    if (Tools::int_random(2)) target.mulYby( -1 );
    ivpNeuroKick->reset_state();
    ivpNeuroKick->kick_to_pos_with_initial_vel(svKickoffSpeed, target);
    if (ivpNeuroKick->get_cmd(cmd)) {
      ivConsecKicks++;
      
      Intention intention;
      intention.set_panic_kick(target, WSinfo::ws->time);

      ivpLastBehavior  = ivpNeuroKick;
      ivLastKickTarget = target;
      ivLastKickSpeed  = svKickoffSpeed;
      ivLastIntention  = intention;
      POL("Made neuroKick for KickOff.");
      return true;
    }
  }
	// Turn to best direct if neccessary.
	Vector turnDest = getKickerTurnTarget();
	if (WSinfo::is_ball_kickable() 
	 && fabs(Tools::my_angle_to(turnDest).get_value_mPI_pPI())>DEG2RAD(5)){
	  ivpFaceBall->turn_to_point(turnDest,false);
		ivpFaceBall->get_cmd(cmd);
		return true;
	}

  // anounce pass before actually played
	Intention intention;
  if (ivTimeLeft <= svPassAnnounceTime 
			&& WSinfo::ws->time%20 >= (7-svPassAnnounceTime)
			&& WSinfo::is_ball_kickable()) {
    POL("Check if pass to announce.");
    if (getPassIntention(intention)) {
      POL("ivpWBall delivered pass-intention");
      Blackboard::pass_intention = intention;
			
      double new_kick_speed, old_kick_speed;
      Vector new_kick_target, old_kick_target;
      intention.get_kick_info(new_kick_speed, new_kick_target);
      ivLastIntention.get_kick_info(old_kick_speed, old_kick_target);
			POL("CONTDEBUG"<<V(new_kick_target)<<V(old_kick_target)<<V(new_kick_speed)<<V(old_kick_speed));
      if ( new_kick_speed != old_kick_speed 
	      || new_kick_target.sqr_distance(old_kick_target) > 1.5) {
        ivpFaceBall->turn_to_point(new_kick_target);
        POL("New intention: look in that direction."<<V(new_kick_target)<<V(old_kick_target)<<V(new_kick_speed)<<V(old_kick_speed));
				ivLastIntention = intention;
				ivpLastBehavior = ivpWBall;
				ivpFaceBall->get_cmd(cmd);
				return true;
      }
			POL("Set Pass-intention ==> will be announced.");
      ivLastIntention = intention;
    }
		else{
			POL("Got no pass intention from wball");;
		}
		/*
		else{
			// no pass intention from wball, generate pass intention with "the Signal"
			// TODO: only in corners
			POL("I give THE SIGNAL.");
			ivpBasicCmd->set_turn(0);
			ivpBasicCmd->get_cmd(cmd);
			cmd.cmd_say.set_pass(WSinfo::me->pos,Vector(0,0),WSinfo::ws->time + 2);
			return true;
		}
		*/
  }
	
  // most important: never forget to kick the ball if possible
	// CIJAT OSAKA: changed to play safe 
	int myGoalieNum = WSinfo::ws->my_goalie_number;
	myGoalieNum= myGoalieNum<=0 ? 1 : myGoalieNum;
	if (ivTimeLeft == 0 
			&& ivLastIntention.valid_at() == WSinfo::ws->time
			&& ivpWBall->intention2cmd(intention,cmd)
			&& WSinfo::ws->time%20 >= 7) { // TODO make this parameter
		// TODO: how to play safe now??
		if(  // ivpWBall->get_advantage()>1 
		     // || 
			   // WSinfo::me->number != myGoalieNum
				 true){
			ivpLastBehavior = ivpWBall;
			ivConsecKicks++;
			intention.get_kick_info(ivLastKickSpeed, ivLastKickTarget);
			ivLastIntention = intention;
			POL("I'm startplayer, and passed!");
			return true;
		}else{
			POL("I'm startplayer, could pass, but not good enough!");
		}
	}
	
  POL("go to homepos.");

  Vector target = ivBallPos;
  double tolerance = 1.0;
   
  static const double kickBorderSetoffFact    = 0.65;
  static const double kickBorderToleranceFact = 0.3;

  if (ivBallPos.getY() > FIELD_BORDER_Y - 0.5) {
    target.setY( ivBallPos.getY() - kickBorderSetoffFact    * WSinfo::me->kick_radius );
    tolerance =                     kickBorderToleranceFact * WSinfo::me->kick_radius;
  }
  if (ivBallPos.getY() < -FIELD_BORDER_Y + 0.5) {
    target.setY( ivBallPos.getY() + kickBorderSetoffFact    * WSinfo::me->kick_radius );
    tolerance =                     kickBorderToleranceFact * WSinfo::me->kick_radius;
  }
	if(WSinfo::me->pos.distance(target)<tolerance){
		POL("I am at my homepos.");
		return false;
	}

	POL("Goto ballPos="<<ivBallPos<<"with "<<V(tolerance)<<V(target));
  ivConsecKicks = 0;
  ivpGo2Pos->set_target(target, tolerance);
  bool returnValue = ivpGo2Pos->get_cmd(cmd);
	// TG // TG // TG // TG // TG // TG // TG // TG // TG // TG 
	if (cmd.cmd_body.get_type() == cmd.cmd_body.TYPE_DASH)
	{
		double absoluteMaxDash = 100;
		if (WSinfo::me->stamina < 2300) absoluteMaxDash = WSinfo::me->stamina_inc_max;
		if (WSinfo::me->stamina < 3000) absoluteMaxDash = 1.5*WSinfo::me->stamina_inc_max;
		double intendedDashPower, allowedDashPower;
		cmd.cmd_body.get_dash( intendedDashPower );
		LOG_DEB(0, <<"StandardSit-TG-HACK: intended dash power is "<<intendedDashPower);
		if (intendedDashPower < 0) intendedDashPower *= 2;
		if ( fabs(intendedDashPower) > absoluteMaxDash  )
			allowedDashPower = absoluteMaxDash;
		else
			allowedDashPower = intendedDashPower;
		if (intendedDashPower < 0) allowedDashPower *= -0.5;
		LOG_DEB(0, <<"StandardSit-TG-HACK: setting to allowed dash power "<<allowedDashPower);
		cmd.cmd_body.unset_lock();  cmd.cmd_body.unset_cmd();  
		cmd.cmd_body.set_dash( allowedDashPower );
	}
	// TG // TG // TG // TG // TG // TG // TG // TG // TG // TG 
  return returnValue;
}

// uses go2pos-behavior, with additional stamina-management and 
// avoidance of clearance zone (if opponent has freekick e.g.)
// if dist==0 calculate dist to target, otherwise use value in dist
// returns true if cmd is set
bool StandardSituation::getCmdGo2PosWithCare(Cmd & cmd, 
                                             Vector target, 
                                             bool intercept_GoalKick, 
                                             double dist) // >>>1
{
  if (intercept_GoalKick && target.getX()<-35) target.setX( -35 );
  else if (ivLastPlayMode == PM_my_GoalKick && target.getX()<-33) target.setX( -33 );   //do not enter Strafraum
  
  
  if (   ivLastPlayMode == PM_his_GoalieFreeKick 
      || ivLastPlayMode == PM_his_GoalKick)
    if(target.getX()>33) target.setX(33);   //do not enter Strafraum
    
  Vector to_target = target - WSinfo::me->pos;
  if(dist==0)
		dist = to_target.norm();


  if (ivWeShallKick) 
  { // if so, just call go2pos
    ivpGo2Pos->set_target(target);
    bool returnValue = ivpGo2Pos->get_cmd(cmd);
    modifyDashCmd( cmd );
    return returnValue;
  }

  // check if direct line to target crosses forbidden area
  Vector unit_to_target =  (1.0/dist) * to_target;  // dist > homepos_tolerance > 0
  Vector normal_to_target(-unit_to_target.getY(), unit_to_target.getX());
  Vector to_ball = WSinfo::ball->pos - WSinfo::me->pos;
  //double dist_ball = to_ball.norm();
  double t = to_ball.getX() * normal_to_target.getX() + to_ball.getY() * normal_to_target.getY();

  Vector normal_to_ball;
  if (t > 0)
    normal_to_ball = Vector(to_ball.getY(), -to_ball.getX());
  else
    normal_to_ball = Vector(-to_ball.getY(), to_ball.getX());
  normal_to_ball.normalize();

  POL("getCmdGo2PosWithCare: me="<<WSinfo::me->pos<<", target="<<target<<", ball="<<WSinfo::ball->pos<<", dist="<<dist<<", s="<<s<<", t="<<t<<", v="<<v<<", w="<<w);

#if 0
  if ((target - WSinfo::ball->pos).norm() < clearance_radius_min) {
    //target must not be in forbbiden area around ball
    //but approach, if player still far away from clearance area
    if (dist_ball <= clearance_radius_min + 2) { // + 2 as safety
      target = WSinfo::me->pos;
      POL("getCmdGo2PosWithCare: wanted to enter clearance area-->stay here.");
      return false;
    }
  } else if (s > 0) { // only consider if player is approaching ball
    if (dist_ball < clearance_radius_min) {
      POL("getCmdGo2PosWithCare: too near to clearance area, correct target");
      target = WSinfo::me->pos + 10 * normal_to_ball;
      MARK_POS(target, f8f800);
    } else {
      if (fabs(t) < clearance_radius_min && s < dist)
        target = target + (clearance_radius_min - fabs(t)) * dist / s * normal_to_ball;
      MARK_POS(target, f0f000);
    } 
  } 
#endif


  DRAW(C2D(WSinfo::ball->pos.x, WSinfo::ball->pos.y, 9.5, "666600"));
  ivpGo2Pos->set_target(target);
  return ivpGo2Pos->get_cmd(cmd);
}

bool StandardSituation::getCmdScanField2nd(Cmd&cmd){
	if(ivKickerNr == WSinfo::me->number)
		return false;
	POL("Scanning field (2nd).");
	static int lookAwayAtModOne = drand48()<.5;
	static ANGLE dir             = ANGLE(0);
	if((lookAwayAtModOne+WSinfo::ws->time) % 3 == 0){
		if(WSinfo::ws->view_quality == Cmd_View::VIEW_QUALITY_LOW)
			dir += ANGLE(DEG2RAD(15));
		else
			dir += ANGLE(DEG2RAD(43));
		Tools::set_neck_request(NECK_REQ_SCANFORBALL);
		ivpBasicCmd->set_turn_inertia(dir.get_value_mPI_pPI());
		ivpBasicCmd->get_cmd(cmd);
	}else{
    ivpFaceBall->turn_to_ball();
    ivpFaceBall->get_cmd(cmd);
	}
	return true;
}
bool StandardSituation::getCmdScanField(Cmd&cmd){
	if(ivKickerNr == WSinfo::me->number)
		return false;
	POL("Scanning field.");
	ANGLE dir;
	if(WSinfo::ws->time_of_last_update != WSinfo::ws->time)
		dir = ANGLE(0);
	else if(WSinfo::ws->view_quality == Cmd_View::VIEW_QUALITY_LOW)
		dir = ANGLE(DEG2RAD(15));
  else
		dir = ANGLE(DEG2RAD(43));
	Tools::set_neck_request(NECK_REQ_SCANFORBALL);
	ivpBasicCmd->set_turn_inertia(dir.get_value_mPI_pPI());
  ivpBasicCmd->get_cmd(cmd);
	return true;
}

Vector StandardSituation::getTangentIsct(const Vector& cM, double cR, const Vector& mypos, const Vector& dest){
	double meToM   = mypos.distance(cM);
	Vector meToMV = cM - mypos;
	double meToT = sqrt(fabs(SQR(meToM) - SQR(cR)));
	ANGLE alpha = meToMV.ANGLE_to(dest-mypos);
	double beta  = atan(meToT/cR);
	Vector t1,t2;
	t1.init_polar(cR,(ANGLE(M_PI)+meToMV.ARG())+ANGLE(beta));
	t1 += cM;
	t2.init_polar(cR,(ANGLE(M_PI)+meToMV.ARG())-ANGLE(beta));
	t2 += cM;
	return ((t1.distance(dest)<t2.distance(dest))?t1:t2);
}

bool StandardSituation::getPassIntention(Intention& intention){
	if(!WSinfo::is_ball_kickable())
		return false;
	ivpWBall->determine_pass_option(intention,WSinfo::me->pos,WSinfo::me->vel,WSinfo::me->ang,ivBallPos,WSinfo::ball->vel);

  double sp; Vector t; int i;
  intention.get_kick_info(sp, t, i);
  PlayerSet opps = WSinfo::valid_opponents;
  opps.keep_players_in_circle( t, 6.0 );
  DRAW( C2D(t.x,t.y,6.0,"999999") );  
  
  if (    intention.valid_at() != WSinfo::ws->time  // no pass option available
       || ((    WSinfo::ws->play_mode==PM_my_GoalKick )
            && (LEFT_PENALTY_AREA.inside( t ) || opps.num>0)) )
    return false;
	return true;
}

Vector StandardSituation::getKickerTurnTarget(){
  // check different conditions for a standard situation.
	// this should be very stable, since otherwise the 
	// player will turn and not kick all the time

	// standard value: look to center of field.
	// This works especially for situations in the corners.
	Vector dest(0,0);

  if(fabs(ivBallPos.getX()) < FIELD_BORDER_X-15){
	  // no corners

		// we are at side of field
		if(fabs(ivBallPos.getY()) > FIELD_BORDER_Y-5)
			dest = Vector(ivBallPos.getX(),-ivBallPos.getY());  // other side of field
		else
			dest = Vector(FIELD_BORDER_X,0);
	}
	return dest;
}

bool StandardSituation::isPassInfoTheSignal(PPlayer p){
	bool thisIsTheSignal = p->pass_info.rel_time == 2;
	POL(V(thisIsTheSignal)<<V(p->pass_info.rel_time));
	return thisIsTheSignal;
}

StandardSituation::CornerRole 
StandardSituation::getCornerRole(){
	if(ivBallPos.getY()>0){
		switch(WSinfo::me->number){
			case  9: return L1_L; break;
			case 10: return L1_M; break;
			case 11: return L1_R; break;
			case  6: return L2_L; break;
			case  7: return L2_M; break;
			case  8: return L2_R; break;
		}
	}else{
		switch(WSinfo::me->number){
			case  9: return L1_R; break;
			case 10: return L1_M; break;
			case 11: return L1_L; break;
			case  6: return L2_R; break;
			case  7: return L2_M; break;
			case  8: return L2_L; break;
		}
	}
	return  NOT_INVOLVED;
}

Vector StandardSituation::getTargetNonStartPlayerMyKickIn(){ // >>>1
	bool onRightSide = ivBallPos.getY()<0;
	Vector lp, rp;
	double hisOffsideLine = WSinfo::his_team_pos_of_offside_line();

	double maxTX, minTX, minX,maxX;

	minX  = -FIELD_BORDER_X+35;
	maxX  = FIELD_BORDER_X;

	// defense line
	maxTX = 5;
	minTX = -FIELD_BORDER_X+35;
	lp.setX( minTX + (ivBallPos.getX() - minX)/(maxX-minX) * (maxTX-minTX) );

	// offense line
	minTX = 20;
	maxTX = FIELD_BORDER_X-3;
	rp.setX( minTX + (ivBallPos.getX() - minX)/(maxX-minX) * (maxTX-minTX) );
	rp.setX( (rp.getX()>hisOffsideLine-1)?hisOffsideLine-1:rp.getX() );

	lp.setY( -FIELD_BORDER_Y+5 );
	rp.setY( 20 );
	if(!onRightSide) {
		rp.mulYby( -1 );
		lp.mulYby( -1 );
	}
	XYRectangle2d rect(rp,lp);
	DRAW(rect);
	Vector v = OpponentAwarePositioning::getHomePosition(rect);

	PlayerSet team = WSinfo::valid_teammates;
	team.keep_and_sort_closest_players_to_point(3,ivBallPos);
	if((WSinfo::me->pos.distance(ivBallPos)<40 || v.distance(ivBallPos)<40)
				  && team.get_player_by_number(WSinfo::me->number)){
		OpponentAwarePosition oap = OpponentAwarePositioning::getAttackOrientedPosition(0.2);
		Vector target;
		target.clone( oap );
		return target;
	}
	return v;
}
Vector StandardSituation::getTargetNonStartPlayerMyFreeKick(){ // >>>1
    return Vector();
}
Vector StandardSituation::getTargetNonStartPlayerMyCorner(){ // >>>1
	Vector target;
	OpponentAwarePosition oap;

	CornerRole myCRole = getCornerRole();
	
	PlayerSet team = WSinfo::valid_teammates;
	team.keep_and_sort_closest_players_to_point(3,ivBallPos);
	POL("getTargetNonStartPlayerMyCorner: "<<V(dist)<<V(team.get_player_by_number(WSinfo::me->number)));
	if(OpponentAwarePositioning::getRole(WSinfo::me->number) == PT_DEFENDER 
			|| (WSinfo::me->pos.distance(ivBallPos)<40
				  && team.get_player_by_number(WSinfo::me->number))){
		oap = OpponentAwarePositioning::getAttackOrientedPosition(0.2);
		target.clone( oap );
		return target;
	}

	// set position assuming ivBallPos.y>0
	switch(myCRole){
		case L1_L: target = Vector(52,33);break; // kicker, doesnt matter anyways
		case L1_M: target = Vector(49,20);break;
		case L1_R: target = Vector(44,00);break;
		case L2_L: target = Vector(35,30);break;
		case L2_M: target = Vector(30,17);break;
		case L2_R: target = Vector(29,-8);break;
    case NOT_INVOLVED: break;
	}
	
	if(ivBallPos.getY()<0)
		target.mulYby( -1 );
	POL("getTargetNonStartPlayerMyCorner: "<<V(target));

	return target;
}
