/*
 * DribbleBetween skill
 *
 * @author Hannes Schulz <mail@hannes-schulz.de>
 * @version 0.9
 *
 * vim:fdm=indent
 *
 */

#include "dribble_between.h"
#if 0 /* 1: debugging; 0: no debugging */
#define POL(XXX)   LOG_POL(0,<<"DribbleBetween: "<<XXX)
#define POL2(XXX)  LOG_POL(1,<<"DribbleBetween: "<<XXX)
#define DRAW(XXX)  LOG_POL(0,<<_2D<<XXX)
#define DRAW2(XXX) LOG_POL(1,<<_2D<<XXX)
#else
#define POL(XXX)   
#define POL2(XXX) 
#define DRAW(XXX)  
#define DRAW2(XXX) 
#endif
#define KEEPQ2D(P1,P2,W1,W2) keep_players_in_quadrangle(P1,P2,W1,W2);
#define MARK_POS(P,C) DRAW(C2D((P).x,(P).y,0.3,#C));
#define MIN(X,Y) ((X<Y)?X:Y)
#define MAX(X,Y) ((X>Y)?X:Y)

#if 0
#define DRIBBLE_STATS
#endif

DribbleBetween::DribbleBetween(){
	dribbleAround = DribbleAround::getInstance();
	lastDribbleTime = 0;
	dribblingInsecure = false;
}
DribbleBetween* DribbleBetween::myInstance = NULL;
DribbleBetween* DribbleBetween::getInstance() {
	if(myInstance == NULL){
		myInstance = new DribbleBetween();
	}
	return myInstance;
}

void DribbleBetween::setRelevantOpponents(){ // {{{1
	static const float farAway = 5;
	opps = WSinfo::valid_opponents;
	int oppsInMyKickRange = 0;
	switch(opps.num){
		case 0: mode = DB_NO_OPP; return;
		case 1: mode = DB_ONE_OPP; return;
		default: break;
	}

	// find out how many players could roughly attack me in next cycle
	// DribbleAround can only deal with one at a time.
	float minDistDangerous = (3 * WSinfo::me->kick_radius);
	for(int i=0;i<opps.num;i++){
		if(opps[i]->pos.distance(WSinfo::ball->pos)<minDistDangerous)
			oppsInMyKickRange++;
	}

	if(oppsInMyKickRange>1){
		dribblingInsecure = true; // TODO: is this useful?
		//mode = DB_TOO_MANY_OPP;
		//return;
	}
	
	DRAW(C2D(dribbleTo.x,dribbleTo.y,0.2,"red"));
	//Vector toFarAway = dribbleTo - WSinfo::me->pos;
	Vector toFarAway;
	toFarAway.init_polar(farAway,WSinfo::me->ang);
	toFarAway.normalize(farAway);
	//Vector justBehind = WSinfo::me->pos - dribbleTo;
	Vector justBehind;
	justBehind.init_polar(-2*WSinfo::me->kick_radius,WSinfo::me->ang);
	justBehind.normalize(2*WSinfo::me->kick_radius);
	opps.keep_players_in_quadrangle(WSinfo::me->pos+justBehind,WSinfo::me->pos+toFarAway,4,6);
	DRAW(Quadrangle2d(WSinfo::me->pos+justBehind,WSinfo::me->pos+toFarAway,4,6));

	for(int i=0;i<opps.num;i++){
		DRAW(C2D(opps[i]->pos.x,opps[i]->pos.y,1.1*opps[i]->kick_radius,"blue"));
	}


	/*
	 * TODO: use lotfusses to find out whether two opponents are on the same side of me
	 * and remove the one further away.
	 *
	 for(int i = 0; i<maxNum;i++){
		if(opps.num>=2){
			Vector between;
			between.x = (opps[0]->pos.x + opps[1]->pos.x)/2;
			between.y = (opps[0]->pos.y + opps[1]->pos.y)/2;
			if((    WSinfo::me->pos.y > opps[0]->pos.y
						&& WSinfo::me->pos.y > opps[1]->pos.y)
					|| (WSinfo::me->pos.y < opps[0]->pos.y
						&& WSinfo::me->pos.y < opps[1]->pos.y)){
				// both are on same side:
				// remove the one further away
				opps.remove(opps[1]);  
			}
			//else if(opps[0]->pos.distance(opps[1]->pos)<(WSinfo::me->pos.distance(between))){
				// too close together
				// remove the one further away
				//opps.remove(opps[1]);  
			//}
		}else break;
	}
	*/
	switch(opps.num){
		case 0: mode = DB_NO_OPP;break;
		case 1: mode = DB_ONE_OPP;break;
		case 2: mode = DB_TWO_OPP;break;
		default: mode = DB_TOO_MANY_OPP;
	}
} // }}}1

float DribbleBetween::getValueForDribbleDir(float dribbleAngle){
	static const float maxLookAhead = 9.;
	static const float minLookAhead = -4;
	Vector toDribbleTo = (dribbleTo-WSinfo::me->pos);
	Vector notFurtherThan(toDribbleTo.ARG()+ANGLE(dribbleAngle));
	Vector atLeastThatFar(toDribbleTo.ARG()+ANGLE(dribbleAngle));
	notFurtherThan.normalize(maxLookAhead);
	atLeastThatFar.normalize(minLookAhead);
	notFurtherThan += WSinfo::me->pos;
	atLeastThatFar += WSinfo::me->pos;

	// GoStraightCheck BEGIN
	// TODO: does this work? Is it good?
  /*
	WSpset opps = WSinfo::valid_opponents;
	Vector lookAhead;
	static const float lookAheadLength = 8;
	lookAhead.init_polar(lookAheadLength,notFurtherThan.ARG());
	opps.keep_players_in_quadrangle(WSinfo::me->pos,WSinfo::me->pos+lookAhead,8,8);
	if(opps.num==0 && WSinfo::me->pos.x>-10)
		return -200.+(WSinfo::me->pos.x>20?(WSinfo::me->pos+lookAhead).distance(HIS_GOAL_CENTER):0);
    */
	// GoStraightCheck END

	opps = WSinfo::valid_opponents;
	opps += WSinfo::valid_teammates_without_me;
	Player leftBorder, rightBorder;
	leftBorder.pos.init_polar(.5*maxLookAhead,toDribbleTo.ARG()+ANGLE(90));
	leftBorder.pos += WSinfo::me->pos;
	leftBorder.vel  = Vector(0,0);
	leftBorder.number = -5;
	leftBorder.team = HIS_TEAM;
	rightBorder.pos.init_polar(.5*maxLookAhead,toDribbleTo.ARG()+ANGLE(-90));
	rightBorder.pos += WSinfo::me->pos;
	rightBorder.vel  = Vector(0,0);
	rightBorder.number = -6;
	rightBorder.team = HIS_TEAM;
	opps.append(&leftBorder);
	opps.append(&rightBorder);

	Vector lot;
	float oppDistToLot,value=1;
	float opponentValue;
	float intervalLength = maxLookAhead - minLookAhead;
	Vector posPlusVel;
	for(int i=0;i<opps.num;i++){
		posPlusVel = opps[i]->pos + opps[i]->vel;
		Geometry2d::projection_to_line(lot, posPlusVel, Line2d(WSinfo::me->pos,notFurtherThan-WSinfo::me->pos));
		// lot must be in range [atLeastThatFar,notFurtherThan] from me, only ahead
		if(notFurtherThan.distance(lot)>intervalLength) continue;
		if(WSinfo::me->pos.distance(lot)>maxLookAhead) continue;
		
		opponentValue = ((opps[i]->team == HIS_TEAM) ? 1 : 0.2);

		if(opps[i]->number == WSinfo::ws->his_goalie_number)
			opponentValue = 0.1;

		opponentValue *= (posPlusVel.distance(WSinfo::me->pos)<4)?2:1;
		//opponentValue *= (isBehindMe?-.5:1);
		oppDistToLot   =  posPlusVel.distance(lot);
		oppDistToLot  -=  8;
		oppDistToLot  *= -1;
		oppDistToLot   =  MIN(8,oppDistToLot);
		oppDistToLot   =  MAX(-0,oppDistToLot);
		oppDistToLot  *=  opponentValue;
		
		value += (oppDistToLot*oppDistToLot);
	}

	return value;
}

Vector DribbleBetween::getTargetPos(){
	float bestAngle=0;
	float value, bestValue=1E6;
	static const float maxDribbleAngle = DEG2RAD(70);
	static const int numCheckedAngles = 25;
	static const float angleIncrease = maxDribbleAngle/numCheckedAngles;
	Vector toDribbleTo = (dribbleTo-WSinfo::me->pos);
	Vector notOff;
	bool goingToCenterAllowed = WSinfo::me->pos.getX()>10||fabs(WSinfo::me->pos.getY())<8;
	bool behindGoal;
	bool toCenter;
	for(float dribbleAngle = 0; dribbleAngle<maxDribbleAngle;dribbleAngle+=angleIncrease){
		notOff.init_polar(14,toDribbleTo.ARG()+ANGLE(dribbleAngle));
		notOff+=WSinfo::me->pos;
		behindGoal = fabs(notOff.getY())<.8*ServerOptions::goal_width;
		toCenter   = fabs(notOff.getY()) < fabs(WSinfo::me->pos.getY()) - 5;
		if((fabs(notOff.getY())>FIELD_BORDER_Y-2 || fabs(notOff.getX())>FIELD_BORDER_X-2)
				&& !behindGoal){
			//POL2("getTargetPos: "<<RAD2DEG(dribbleAngle)<< "?:\t Leads out of field");
		}
		else if(!goingToCenterAllowed && toCenter){
			//POL2("getTargetPos: "<<RAD2DEG(dribbleAngle)<< "?:\t Leads to center");
		}else{
			value = getValueForDribbleDir(dribbleAngle);
			//POL2("getTargetPos: "<<RAD2DEG(dribbleAngle)<< "?:\t"<< value);
			if(value<bestValue){
				bestValue = value;
				bestAngle = dribbleAngle;
			}
		}
		notOff.init_polar(14,toDribbleTo.ARG()+ANGLE(-dribbleAngle));
		notOff+=WSinfo::me->pos;
		behindGoal = fabs(notOff.getY())<.8*ServerOptions::goal_width;
		toCenter   = fabs(notOff.getY()) < fabs(WSinfo::me->pos.getY()) - 5;
		if((fabs(notOff.getY())>FIELD_BORDER_Y-2 || fabs(notOff.getX())>FIELD_BORDER_X-2)
				&& !behindGoal){
			//POL2("getTargetPos: "<<RAD2DEG(-dribbleAngle)<< "?:\t Leads out of field");
		}else if(!goingToCenterAllowed && toCenter){
			//POL2("getTargetPos: "<<RAD2DEG(-dribbleAngle)<< "?:\t Leads to center");
		}else{
			value = getValueForDribbleDir(-dribbleAngle);
			//POL2("getTargetPos: "<<RAD2DEG(-dribbleAngle)<< "?:\t"<< value);
			if(value<bestValue){
				bestValue = value;
				bestAngle = -dribbleAngle;
			}
		}
		//if(bestValue < 2) break; // TODO: does this help?
	}

	POL("Dribble Dir has value "<<bestValue);

	Vector dest;
	dest.init_polar(20,toDribbleTo.ARG()+ANGLE(bestAngle));
	dest+=WSinfo::me->pos;

	DRAW(L2D(WSinfo::me->pos.x,WSinfo::me->pos.y,dest.x,dest.y,"green"));
	
	if(dest.distance(WSinfo::me->pos)<2){
		POL2("going for dribbleTo instead of dribbling between");
		dest = dribbleTo;
	}
	if(fabs(WSinfo::me->pos.getX())>FIELD_BORDER_X-4){
		POL2("going for HIS_GOAL_CENTER instead of dribbling between");
		dest = HIS_GOAL_CENTER;
	}

	bool amICloseToFieldBorder = 
		(fabs(lastTarget.getX())>FIELD_BORDER_X-2 ||
		 fabs(lastTarget.getY())>FIELD_BORDER_Y-2)
		&&(
		   fabs(WSinfo::me->pos.getX()) > FIELD_BORDER_X - 5
		|| fabs(WSinfo::me->pos.getY()) > FIELD_BORDER_Y - 5);
	bool isOldTargetInBadAngleButNewNot =
		      fabs(Tools::my_angle_to(lastTarget).get_value_mPI_pPI())>DEG2RAD(80)
			 && fabs(Tools::my_angle_to(dest)      .get_value_mPI_pPI())<DEG2RAD(80);

	if(WSinfo::ws->time - lastDribbleTime > 3){
		lastTarget = dest;
		dribbleAroundTargetValue = bestValue;
		dribbleAround->resetRequest();
		targetTimeCounter = 0;
	}else if((targetTimeCounter > 4 && mode!=DB_NO_OPP)
			   ||targetTimeCounter >5){
		lastTarget = dest;
		dribbleAroundTargetValue = bestValue;
		dribbleAround->resetRequest();
		targetTimeCounter = 0;
	}else if(WSinfo::me->pos.distance(lastTarget)<1){
		lastTarget = dest;
		dribbleAroundTargetValue = bestValue;
		dribbleAround->resetRequest();
		targetTimeCounter = 0;
	}else if(amICloseToFieldBorder){
		lastTarget = dest;
		dribbleAroundTargetValue = bestValue;
		dribbleAround->resetRequest();
		targetTimeCounter = 0;
	}else if(isOldTargetInBadAngleButNewNot){
		lastTarget = dest;
		dribbleAroundTargetValue = bestValue;
		dribbleAround->resetRequest();
		targetTimeCounter = 0;
	}
	else{
		dest = lastTarget;
		targetTimeCounter++;
	}
	if(iAmAttacked()){
		POL("I am attacked, going straight");
		dribbleAroundTargetValue = 10; 
		dest.init_polar(10,WSinfo::me->ang);
		dest += WSinfo::me->pos;
	}
	return dest;
}

bool DribbleBetween::is_dribble_safe(bool getcmd){

	isDribblingSafe = false;
	dribblingInsecure = false;
	
	setRelevantOpponents(); // has to be here to generate stats
	                        // otherwise move a little down
	dribbleAroundTarget = getTargetPos();

#ifdef DRIBBLE_STATS
	stats.updateTestDribbling();
	stats.getStatistics();
#endif

	if(mode==DB_TOO_MANY_OPP){
		POL("Dribblink not safe: Too many opponents in range");
		return false;
	}
	if(mode==DB_ONE_OPP||mode==DB_TWO_OPP){
		if(WSinfo::ws->time - opps[0]->time > 2 
				&& opps[0]->pos.distance(WSinfo::me->pos)<4){
			POL("Dribblink not safe: Opponent age too old.");
			dribblingInsecure=true;
			//return false;
		}
	}
	if(mode != DB_NO_OPP){
		if(opps[0]->pos.distance(WSinfo::ball->pos)< opps[0]->kick_radius){
			POL("Dribblink not safe: Opponent has ball.");
			return false;
		}
	}
	// TODO: account for catchable_area_l
	/*
	if(mode==DB_TWO_OPP 
			&& opps[0]->pos.distance(WSinfo::me->pos)< 1.0*(WSinfo::me->kick_radius+opps[0]->kick_radius)   // two very close opps
			&& opps[1]->pos.distance(WSinfo::me->pos)< 1.0*(WSinfo::me->kick_radius+opps[1]->kick_radius)){ // two very close opps
		POL("Dribblink not safe: Two opps very close to me");
		return false;
	}
	*/
	if(mode==DB_TWO_OPP 
			&& fabs(WSinfo::me->pos.getY()	)>FIELD_BORDER_X-3){
		POL("Dribblink not safe: Two opps close to me, I'm close to border -- I'm cornered!");
		dribblingInsecure=true;
	}
	isDribblingSafe=true;
	bool res;
	if(getcmd){
	  res = get_cmd(cachedCmd);
	  cachedCmdTime = WSinfo::ws->time;
	}else
		res = true;
	if(res) {
		POL("Dribblink is safe.");
	}
	return res;
}

bool DribbleBetween::is_dribble_insecure(){
	return dribblingInsecure;
}

bool DribbleBetween::iAmAttacked(){
	PlayerSet opps;
	bool oppVeryClose;
	bool oppInMyDir;
	bool thereIsOppInMyDir = false;
	bool thereIsVeryCloseOpp = false;

	static bool result = false;;
	static int time    = -1;

	if(time == WSinfo::ws->time){
		return result;
	}
	
	opps = WSinfo::valid_opponents;
	opps.keep_and_sort_closest_players_to_point(2,WSinfo::me->pos);

	for(int i=0;i<opps.num;i++){
		oppVeryClose = opps[i]->pos.distance(WSinfo::me->pos)<2;
		oppInMyDir = fabs(Tools::my_angle_to(opps[i]->pos).get_value_mPI_pPI())<PI/4;
		if(oppVeryClose)
			thereIsVeryCloseOpp = true;
		if(oppInMyDir && oppVeryClose)
			thereIsOppInMyDir = true;
	}

	time = WSinfo::ws->time;
	
	result = thereIsVeryCloseOpp && !thereIsOppInMyDir;

	return result;
}

bool DribbleBetween::get_cmd(Cmd& cmd){
	if(cachedCmdTime == WSinfo::ws->time){
		cmd = cachedCmd;
		dribbleAround->setDribbled(true);
		lastDribbleTime = WSinfo::ws->time;
		return true;
	}

#ifdef DRIBBLE_STATS
	stats.updateGetCmd();
#endif

	/*
	 * DEBUGGING PURPOSES
	set_target(Vector(50,WSinfo::me->pos.y));
	is_dribble_safe(false);
	if(!isDribblingSafe && dribbleAround->get_dribble_straight_cmd(cmd))
		return true;
	*/
	
	
	DRAW(C2D(dribbleAroundTarget.x,dribbleAroundTarget.y,WSinfo::me->kick_radius,"red"));

	// keep the ball in own kickrange if 2nd opponent ahead
	bool keepBall = (mode==DB_TWO_OPP 
			&& (opps[1]->pos.distance(WSinfo::me->pos)<4)
			&& fabs(Tools::my_angle_to(opps[1]->pos).get_value_mPI_pPI())<PI/3);

	bool isOppJustAhead = mode!=DB_NO_OPP
			&& (opps[0]->pos.distance(WSinfo::me->pos)<4)
			&& (opps[0]->pos.distance(WSinfo::me->pos)>.8*(WSinfo::me->kick_radius+opps[0]->kick_radius))
			&& fabs(Tools::my_angle_to(opps[0]->pos).get_value_mPI_pPI())<PI/2.5;

	int maxSpeed;
	if(mode==DB_NO_OPP){
		maxSpeed = 100;
	}else if(isOppJustAhead){
		maxSpeed = 100;
	}
	else{
		maxSpeed = 100;
	}

	dribbleAround->set_max_speed(maxSpeed);
	dribbleAround->set_target(dribbleAroundTarget);
	dribbleAround->set_keepBall(keepBall);
	bool ret =  dribbleAround->get_cmd(cmd);
	
	if(dribbleAround->isDribbleInsecure())
		dribblingInsecure=true;

	dribbleAround->setDribbled(false);

	return ret;
}

bool DribbleBetween::is_neck_req_set(){
	return dribbleAround->neckReqSet;
}
ANGLE DribbleBetween::get_neck_req(){
	return dribbleAround->neckReq;
}

void DribbleBetween::set_target(const Vector& t){
	dribbleTo = t;
	POL("DribbleTo set to " << t );
}
Vector DribbleBetween::get_target(){
	return dribbleTo;
}

DribbleBetween::~DribbleBetween(){
	// do not delete: could be deleted twice! (singleton!)
	// delete dribbleAround;
}

DribbleBetween::Stats::Stats(){
#ifdef DRIBBLE_STATS
	dribbleStates[0].closestOpp = NULL;
	dribbleStates[0].xDistToClosestOpp = 0;
	dribbleStates[0].actionTaken = DA_NO_ACTION;
	dribbleStates[0].actionSucceeded = false;
	dribbleStates[0].didDribble = false;
	for(int i=1;i<DRIBBLE_STATS_BUFFER_LENGTH;i++){
		dribbleStates[i] = dribbleStates[0];
	}
	bufferPos=0;
#endif
}
// #define MARK(X) cerr << "mark " << #X << endl << flush;
void DribbleBetween::Stats::updateTestDribbling(){
#ifdef DRIBBLE_STATS
	db = DribbleBetween::getInstance();
	static const int maxHorizon = 5; // cycles until an action has "failed"
	if(dribbleStates[bufferPos].didDribble){
		dribbleStates[bufferPos].actionTaken     = db->dribbleAround->lastActionTaken;
		dribbleStates[bufferPos].actionSucceeded = 
				 dribbleStates[bufferPos].didDribble
			// && WSmemory::team_last_at_ball()  // doesnt make sense since there may have been lots of time in between
			&& WSinfo::ws->time - dribbleStates[bufferPos].time<maxHorizon;
	}

	++bufferPos;
	bufferPos = bufferPos % DRIBBLE_STATS_BUFFER_LENGTH;

	dribbleStates[bufferPos].time = WSinfo::ws->time;
	dribbleStates[bufferPos].didDribble = false;
	dribbleStates[bufferPos].actionTaken = DA_NO_ACTION;

	if(db->mode != DB_NO_OPP){
		dribbleStates[bufferPos].closestOpp = db->opps[0];
		dribbleStates[bufferPos].xDistToClosestOpp = db->opps[0]->pos.x-WSinfo::me->pos.x;
	}
	else{
		dribbleStates[bufferPos].closestOpp = NULL;
		dribbleStates[bufferPos].xDistToClosestOpp = FIELD_BORDER_X;
	}
#endif
}
void DribbleBetween::Stats::updateGetCmd(){
#ifdef DRIBBLE_STATS
	dribbleStates[bufferPos].didDribble = true;
#endif
}
void DribbleBetween::Stats::getStatistics(){
#ifdef DRIBBLE_STATS
 	int pos = ((bufferPos<1)? DRIBBLE_STATS_BUFFER_LENGTH-1:bufferPos-1);

	const DribbleState ds = dribbleStates[pos];
	if(!ds.didDribble) return;
	POL("LastDribble: ("<< ds.time << ") " << dribbleActionNames[ds.actionTaken] << "\t"
			               << (ds.actionSucceeded?"Success":"Failure"));
#endif
}
