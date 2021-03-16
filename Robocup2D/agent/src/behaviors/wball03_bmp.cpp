#include "wball03_bmp.h"
#include "ws_info.h"
#include "ws_memory.h"
#include "log_macros.h"
#include "mdp_info.h"
#include <stdlib.h>
#include <stdio.h>
#include "tools.h"
#include "valueparser.h"
#include "options.h"
#include "log_macros.h"
#include "geometry2d.h"

//#include "../basics/tools.h" //get_tackle_success_probability()
#include "../policy/planning.h"
#include "../policy/policy_tools.h"
#include "../policy/positioning.h"  // get_role()
#include "globaldef.h"
#include "ws_info.h"
//#include "pos_tools.h"
#include "blackboard.h"

#define LOG_INFO2(X)
#define LOG_INFO3(X)

#if 0
#define LOGNEW_POL(LLL,XXX) LOG_POL(LLL,<<"WBALL03: " XXX)
#define LOGNEW_DRAW(LLL,XXX) LOG_POL(LLL,<<_2D <<XXX)
#define LOGNEW_ERR(LLL,XXX) LOG_ERR(LLL,XXX)
#define NEWBASELEVEL 0 // level for logging; should be 3 for quasi non-logging
#define NEWGETTIME (Tools::get_current_ms_time())
//#define LOG_DAN(YYY,XXX) LOG_DEB(YYY,XXX)
#define LOG_DAN(YYY,XXX)
#else
#define LOGNEW_POL(LLL,XXX)
#define LOGNEW_DRAW(LLL,XXX)
#define LOGNEW_ERR(LLL,XXX) 
#define NEWGETTIME (0)
#define NEWBASELEVEL 0 // level for logging; should be 3 for quasi non-logging
#define LOG_DAN(YYY,XXX)
#endif



#if 0 // previous logging; now replaced
#define DBLOG_POL(LLL,XXX) LOG_POL(LLL,<<"WBALL03: "XXX)
#define DBLOG_DRAW(LLL,XXX) LOG_POL(LLL,<<_2D <<XXX)
#define DBLOG_ERR(LLL,XXX) LOG_ERR(LLL,XXX)
#define MYGETTIME (Tools::get_current_ms_time())
#define BASELEVEL 2 // level for logging; should be 3 for quasi non-logging
#define LOG_DAN(YYY,XXX)
#else
#define DBLOG_POL(LLL,XXX)
#define DBLOG_DRAW(LLL,XXX)
#define DBLOG_ERR(LLL,XXX) 
#define MYGETTIME (0)
#define BASELEVEL 2 // level for logging; should be 3 for quasi non-logging
#define LOG_DAN(YYY,XXX)
#endif

#define MIN(X,Y) ((X<Y)?X:Y)


/* constructor method */
Wball03::Wball03() {
  /* read with ball params from config file*/
  cyclesI_looked2goal = 0;
  lastTimeLookedForGoalie = -1;
  lastTimeLookedForGoal = -1;
  check_action_mode = 0;
  last_at_ball = -100;
  at_ball_for_cycles = 0;
  at_ball_patience = 1000;
  cycles_in_waitandsee = 0;
  last_waitandsee_at = -1;
  last_heavy_attack_at = -1;

  ValueParser vp(CommandLineOptions::policy_conf,"wball03_bmp");
  vp.get("at_ball_patience",at_ball_patience);
  vp.get("check_action_mode", check_action_mode);
  
  neurokick = new NeuroKick05;
  dribblestraight = new DribbleStraight;
	dribble_between = DribbleBetween::getInstance();
  selfpass = new Selfpass;
  selfpass2 = new Selfpass2;
  basiccmd = new BasicCmd;
  onestepkick = new OneStepKick;
  oneortwo = new OneOrTwoStepKick;
  onetwoholdturn = new OneTwoHoldTurn;
  score = new Score;
  score05_sequence = new Score05_Sequence;
  neuro_wball = new NeuroWball;

  my_blackboard.pass_or_dribble_intention.reset();
  my_blackboard.intention.reset();
}

Wball03::~Wball03() {
	// do not delete dribble_between: singleton!
  delete neurokick;
  delete dribblestraight;
  delete selfpass;
  delete selfpass2;
  delete basiccmd;
  delete onestepkick;
  delete oneortwo;
  delete onetwoholdturn;
  delete score;
  delete score05_sequence;
  delete neuro_wball;
}


void Wball03::foresee(const Vector newmypos, const Vector newmyvel,const ANGLE newmyang, 
		      const Vector newballpos,const Vector newballvel, ANGLE & targetdir){

  Intention intention, tmp_intention;


  Vector const opponent_goalpos(52.5,0.);
  targetdir  = (opponent_goalpos-WSinfo::me->pos).ARG();


  
  my_role= DeltaPositioning::get_role(WSinfo::me->number);

  if((my_blackboard.pass_or_dribble_intention.valid_at() == WSinfo::ws->time)
     || (WSinfo::is_ball_kickable() == true)){ // not set yet
    LOGNEW_POL(0,<<"FORESEE: intention already valid");  // wie ist das denn m�glich??
    return;
  }
  if (score->test_shoot2goal(intention)){
    LOGNEW_POL(0,<<" FORESEE: Wow, I will score next cycle -> look 2 goal!");
    Tools::set_neck_request(NECK_REQ_LOOKINDIRECTION, 
			    (Vector(FIELD_BORDER_X,0) - WSinfo::me->pos).arg());
    return;
  }

  LOGNEW_POL(0,<<"Foresee: precompute best selfpass ! mypos: "<<newmypos
	     <<" myang "<<RAD2DEG(newmyang.get_value())<<" ballpos "<<newballpos);
  precompute_best_selfpass(newmypos,newmyvel, newmyang, WSinfo::me->stamina, newballpos,newballvel);
  LOGNEW_POL(0,<<"Foresee: is selfpass possible:"<<is_selfpass_possible());

  get_pass_or_dribble_intention(intention,newmypos, newmyvel, newballpos, newballvel);
  int type = intention.get_type();
  if(type == PASS || type == LAUFPASS){
    my_blackboard.pass_or_dribble_intention.reset();
    my_blackboard.pass_or_dribble_intention=intention;
    if(test_priority_pass2(tmp_intention) == true){ // I will play a pass next time. Look in direction
      LOGNEW_POL(0,<<"FORESEE: Priority Pass to teammate "<<intention.target_player);
      Vector target;
      double speed;
      my_blackboard.pass_or_dribble_intention.get_kick_info(speed, target);
      ANGLE neck_targetdir = (target - newballpos).ARG(); // direction of the target
      my_blackboard.neckreq.set_request(NECK_REQ_LOOKINDIRECTION,neck_targetdir.get_value());
    }
  }

  //  precompute_best_selfpass(newmypos,newmyvel, newmyang, WSinfo::me->stamina, newballpos,newballvel);
  if(is_selfpass_possible()){ 
    set_neck_selfpass2();
    LOGNEW_POL(0,<<"FORESEE: Selfpass possible, so try to get information!");
    targetdir = scheduled_selfpass.targetdir;
  }

  // do neckrequests here: OVERWRITE potential previous request, e.g. set by noball...
  if (my_blackboard.neckreq.is_set()){
    LOGNEW_POL(0,<<"FORESEE: Neck Request is SET to dir :"
	       <<RAD2DEG(my_blackboard.neckreq.get_param()));
    //Blackboard::set_neck_request(my_blackboard.neckreq.get_type(),my_blackboard.neckreq.get_param());
    Tools::set_neck_request(my_blackboard.neckreq.get_type(),my_blackboard.neckreq.get_param(), true);
  }

  check_write2blackboard(); // communicate and look
  

  /*
  // try to look to most likely attacking opponent
  WSpset pset= WSinfo::valid_opponents;
  Vector endofregion;
  double length = 30;
  const ANGLE opgoaldir = (Vector(47.,0) - WSinfo::me->pos).ARG(); //go towards goal
  endofregion.init_polar(length, opgoaldir);
  endofregion += WSinfo::me->pos;
  Quadrangle2d check_area = Quadrangle2d(WSinfo::me->pos, endofregion, 5.0, 1.5 * length);
  DBLOG_DRAW(0, check_area );
  pset.keep_players_in(check_area);
  PPlayer attacker= pset.closest_player_to_point(WSinfo::me->pos);
  if(attacker!=0){
    LOGNEW_POL(0,<<"FORESEE: looking to most likely attacker number "<<attacker->number);
    Tools::set_neck_request(NECK_REQ_LOOKINDIRECTION, (attacker->pos - WSinfo::me->pos).ARG());
  }
  */
}

void Wball03::reset_intention() {
  my_blackboard.pass_or_dribble_intention.reset();
  my_blackboard.intention.reset();
  //ERROR_OUT << "  wball03 reset intention, cycle " << WSinfo::ws->time;
  DBLOG_POL(0, << "wball03 reset intention");
}

bool Wball03::get_cmd(Cmd & cmd) {
  Intention intention;

#if 0 // test special behaviour
  test_holdturn(intention);
  intention2cmd(intention,cmd);
  return true;
#endif


  in_penalty_mode = false;
  //  if(WSinfo::ws->play_mode == PENALTY_PLAYON){
  if(WSinfo::ws->play_mode == PM_my_PenaltyKick){
    //  if(0){ // test only
    in_penalty_mode = true;
  }

  LOG_POL(BASELEVEL, << "In WBALL03_BMC : ");
  if(!WSinfo::is_ball_kickable())
    return false;
  switch(WSinfo::ws->play_mode) {
  case PM_PlayOn:
    if(get_intention(intention)){
      intention2cmd(intention,cmd);
      if(cmd.cmd_body.get_type() == Cmd_Body::TYPE_KICK ||
	 cmd.cmd_body.get_type() == Cmd_Body::TYPE_TURN)
	last_waitandsee_at = WSinfo::ws->time;
      Blackboard::main_intention = intention; // ridi 05: publish main intention in Blackboard
      //DBLOG_POL(BASELEVEL, << "WBALL03: intention was set! ");
      return true;
    }
    else{
      DBLOG_POL(BASELEVEL, << "WBALL03: WARNING: NO CMD WAS SET");
      return false;
    }
    break;
  default:
    return false;  // behaviour is currently not responsible for that case
  }
  return false;  // behaviour is currently not responsible for that case
}


bool Wball03::get_intention(Intention &intention){

#if LOGGING && BASIC_LOGGING
  long ms_base_time= NEWGETTIME;
  long ms_time;
#endif
  DBLOG_POL(BASELEVEL+2, << "Entering Wball03");
  LOGNEW_POL(NEWBASELEVEL, << "************************* Wball03*******************************");

  if(last_at_ball == WSinfo::ws->time -1)
    at_ball_for_cycles ++;
  else
    at_ball_for_cycles =1;
  last_at_ball = WSinfo::ws->time;

  if(last_waitandsee_at == WSinfo::ws->time -1) // waitandsee selected again
    cycles_in_waitandsee ++;
  else
    cycles_in_waitandsee = 0; // reset

  I_am_heavily_attacked_since();  //* call once to update


  if(cycles_in_waitandsee >0){
    DBLOG_POL(BASELEVEL,<<"cylces in wait and see: "<<cycles_in_waitandsee);
  }


  my_role= DeltaPositioning::get_role(WSinfo::me->number);
  //  my_role= 2; // test only !!!!
  // determine closest opponent (frequently needed)
  PlayerSet pset = WSinfo::valid_opponents;
  pset.keep_and_sort_closest_players_to_point(1, WSinfo::me->pos);
  if(pset.num >0)
    closest_opponent = pset[0];
  else
    closest_opponent = NULL;

  if(onestepkick->can_keep_ball_in_kickrange() == false) {
    DBLOG_POL(0,<<"WBALL03: CANNOT keep ball in kickrange");
  }


  Intention tmp, tmp_previous, alternative_pass_intention;

  tmp_previous = my_blackboard.pass_or_dribble_intention; // copy old intention

  if(my_blackboard.intention.get_type() == OPENING_SEQ){
    my_blackboard.pass_or_dribble_intention.reset(); // 
    DBLOG_POL(0,<<"Current intention is opening_seq: invalidated original pass intention");
  }
  else if(check_previous_intention(my_blackboard.pass_or_dribble_intention,
				   my_blackboard.pass_or_dribble_intention) == true){
    DBLOG_POL(0,<<"Pass or Dribble Intention was already set, still ok, just take it!");
    DBLOG_POL(0,<<"But: GET ALTERNATIVE intention!");
    get_pass_or_dribble_intention(alternative_pass_intention);
  }
  
  if(my_blackboard.pass_or_dribble_intention.valid_at() != WSinfo::ws->time)
  { // not set yet
    if(get_pass_or_dribble_intention(tmp))
    { // check for a new intention.
      if(tmp_previous.valid_at() == WSinfo::ws->time -1)
      {// if previous intention was set in last cycle
	DBLOG_POL(0,"previous pass intention was set, dir "<<RAD2DEG(tmp_previous.kick_target.ARG().get_value()));
	if(tmp_previous.kick_target.ARG().diff(tmp.kick_target.ARG()) < 20./180. *PI){
	  tmp.set_valid_since(WSinfo::ws->time-1);  // indicate
	  DBLOG_POL(0,"current and previous pass direction have not changed too much, current: "
		    <<RAD2DEG(tmp.kick_target.ARG().get_value()));
	}
      } // previous intention was valid last cycle
      my_blackboard.pass_or_dribble_intention = tmp;
    }
    else
      my_blackboard.pass_or_dribble_intention.reset();      
  }
  DBLOG_POL(0,<<"Pass or Dribble Intention valid since "<<my_blackboard.pass_or_dribble_intention.valid_since());

#if LOGGING && BASIC_LOGGING
  ms_time = NEWGETTIME - ms_base_time;
#endif
  LOGNEW_POL(0, << "***************+WBALL03: Before Dribble computation. Already needed  " << ms_time );

	int idbp = is_dribblebetween_possible();
	switch(idbp){
		case 0: is_dribble_between_ok = false;
						is_dribble_between_insecure = false;
						break;
		case 1: is_dribble_between_ok = true;
						is_dribble_between_insecure = true;
						break;
		case 2: is_dribble_between_ok = true;
						is_dribble_between_insecure = false;
						break;
	}
  is_dribble_straight_ok = is_dribble_between_ok;
  

  is_dribble_ok = 
    WSinfo::me->stamina > MIN_STAMINA_DRIBBLE &&
    (is_dribble_straight_ok || is_dribble_between_ok);

#if LOGGING && BASIC_LOGGING
  ms_time = NEWGETTIME - ms_base_time;
#endif
  LOGNEW_POL(0, << "***************+WBALL03: Before Precompute selfpasses. Already needed  " << ms_time );

  precompute_best_selfpass();    // precompute best selfpass. Important for decision of priority pass.

  bool result = false;


#if LOGGING && BASIC_LOGGING
  ms_time = NEWGETTIME - ms_base_time;
#endif
  LOGNEW_POL(0, << "***************+WBALL03: Before DAISY Chain. Already needed  " << ms_time );

  // daisy
  if ( 0 )
  {}
  else if ( (result = score05_sequence->test_shoot2goal( intention )) ){
    LOGNEW_POL(NEWBASELEVEL+0,"Daisy chain: test SCORE05_SEQUENCE successful");
  }
  else if ( (result = score->test_shoot2goal(intention)) ){
    LOGNEW_POL(NEWBASELEVEL+0,"Daisy chain: test SCORE successful");
  }
  else if ( (result = check_previous_intention(my_blackboard.intention, intention)) ){
    LOGNEW_POL(NEWBASELEVEL+0,"Daisy chain: test PREVIOUS INT. successful -> continue prev. intention");
  }
  else if ( (result = test_two_teammates_control_ball(intention))){
    LOGNEW_POL(NEWBASELEVEL+0,"Daisy chain: test TWO TEAMMATES AT BALL successful");
  }

  else if ( (result = test_in_trouble(intention))){
    LOGNEW_POL(NEWBASELEVEL+0,"Daisy chain: test IN TROUBLE successful");
  }

  /* test only: */
  //  else if ( (result = test_selfpasses(intention))){
  //    LOGNEW_POL(NEWBASELEVEL+0,"Daisy chain: test SELFPASSES successful");
  //  }


  else if ( in_penalty_mode == false && (result = test_priority_pass2(intention))){
    LOGNEW_POL(NEWBASELEVEL+0,"Daisy chain: test PRIORITY PASS successful");
  }
  else if ( (result = test_selfpasses(intention))){
    LOGNEW_POL(NEWBASELEVEL+0,"Daisy chain: test SELFPASSES successful");
  }
  else if ( (result = test_dribbling(intention)) ){
    LOGNEW_POL(NEWBASELEVEL+0,"Daisy chain: test DRIBBLING successful");
  }
  else if ( (result = test_holdturn2(intention))){
    LOGNEW_POL(NEWBASELEVEL+0,"Daisy chain: test HOLDTURN successful");
  }
  else if ( in_penalty_mode == false && (result = test_pass_or_dribble(intention))){
    LOGNEW_POL(NEWBASELEVEL+0,"Daisy chain: test PASS successful");
  }
  else result = test_default(intention); // modified version

  LOGNEW_POL(0, << "***************+WBALL03: After Daisy chain. Already needed  " << ms_time );


  // ridi 04

  if(intention.get_type()!=PASS 
      && intention.get_type()!= LAUFPASS 
      && intention.get_type()!= KICKNRUSH){
    // no pass is currently intended, so probably switch to another alternative
    if(alternative_pass_intention.valid_at() == WSinfo::ws->time){
      DBLOG_POL(0,"ALTERNATIVE V:"<<alternative_pass_intention.V<<" CURRENT V: "
		<<my_blackboard.pass_or_dribble_intention.V);
      if(alternative_pass_intention.V < my_blackboard.pass_or_dribble_intention.V){
	DBLOG_POL(0,"I found a besser passing alternative, SO CHANGE!");
	my_blackboard.pass_or_dribble_intention = alternative_pass_intention;
      }
    }
  }
  else{
    LOGNEW_DRAW(NEWBASELEVEL+0, VL2D(WSinfo::ball->pos,
				    intention.kick_target,
				    "orange"));
  }


  LOGNEW_POL(0, << "***************+WBALL03: After check alternative passes  " << ms_time );

  my_blackboard.intention = intention;
  check_write2blackboard();

#if LOGGING && BASIC_LOGGING
  ms_time = NEWGETTIME - ms_base_time;
#endif
  LOGNEW_POL(0, << "***************+WBALL03 policy needed " << ms_time << "millis to decide****************");

  return result;
}

bool Wball03::selfpass_dir_ok(const ANGLE dir){

  const ANGLE opgoaldir = (Vector(47.,0) - WSinfo::me->pos).ARG(); //go towards goal

  if(WSinfo::me->pos.getX() > 45. || dir.diff(ANGLE(0))<25/180.*PI )
    return true;
  if (dir.diff(opgoaldir)<20/180.*PI)
    return true;
  return false;
}

bool Wball03::aggressive_selfpass_dir_ok(const ANGLE dir){

  //  const ANGLE opgoaldir = (Vector(47.,0) - WSinfo::me->pos).ARG(); //go towards goal

  if(WSinfo::me->pos.getX() < 35 && dir.diff(ANGLE(0))>25/180.*PI )
    return false;

  PlayerSet pset= WSinfo::valid_opponents;
  Vector endofregion;
  double scanrange = 5;
  scanrange = 5;
  endofregion.init_polar(scanrange, dir);
  endofregion += WSinfo::me->pos;
  Quadrangle2d check_area = Quadrangle2d(WSinfo::me->pos, endofregion, scanrange/2., scanrange);
  //  DBLOG_DRAW(0, check_area );
  pset.keep_players_in(check_area);
  if(pset.num >0){
    if(pset.num <= 1 && pset[0] == WSinfo::his_goalie){
      DBLOG_POL(0,<<"AGGRESSIVE SELFPASS: Only goalie before me -> do not reduce steps");
    }
    else{
      DBLOG_POL(0,<<"AGGRESSIVE SELFPASS: targetdirection not possible");
      return false;
    }
  }
  return true;
}


void Wball03::set_neck_selfpass2(){
  if (is_selfpass_possible() == false)
    return;
  int opage = -1;
  ANGLE opdir = scheduled_selfpass.targetdir;  // reasonable default (just in case...)
  // precompute attacker information:
  LOGNEW_POL(0, " Set Neck Selfpass: attacking op: "<<scheduled_selfpass.attacker_num);
  PlayerSet opset= WSinfo::valid_opponents;
  PPlayer attacker = opset.get_player_by_number(scheduled_selfpass.attacker_num);
  if(attacker != NULL){
    //LOGNEW_DRAW(0, C2D(attacker->pos.x, attacker->pos.y, 1.3, "orange"));
    Tools::display_direction(attacker->pos, attacker->ang, 3.0);
    opdir = (attacker->pos  - WSinfo::me->pos).ARG();
    opage = attacker->age;
  }
  else{ // no attacker known
    LOGNEW_POL(0,<<"set neck selfpass: no attacker known. Turn 2 targetdir ");
    my_blackboard.neckreq.set_request(NECK_REQ_LOOKINDIRECTION,scheduled_selfpass.targetdir.get_value());
    return;
  }

  // from here on, opdir and opage should be reasonaybl defined
  bool look2opponent = true;

  if(opage >=1){
    LOGNEW_POL(0,<<"set neck selfpass: haven'nt looked to opponent since "
	       <<WSmemory::last_seen_in_dir(opdir)
	       <<" turn neck 2 opponent in dir "<<RAD2DEG(opdir.get_value()));
    look2opponent = true;
  }
  else if(WSmemory::last_seen_in_dir(scheduled_selfpass.targetdir) >=2){
    LOGNEW_POL(0,<<"set neck selfpass: haven'nt looked in targetdir since "
	       <<WSmemory::last_seen_in_dir(scheduled_selfpass.targetdir)
	       <<" turn neck 2 targetdir");
    look2opponent = false;
  }
  else{  // targetdir is ok, now look at opponent
    look2opponent = true;    
  }

  ANGLE look2dir; // default: look 2 opponent

  if(look2opponent == true){
    look2dir = opdir;
    LOGNEW_POL(0,<<"set neck selfpass. Turn 2 opdir ");
    if(Tools::could_see_in_direction(look2dir) == false){
      LOGNEW_POL(0,<<"set neck selfpass. turn2 opdir NOT POSSIBLE: turn 2 targetdir instead ");
      look2dir = scheduled_selfpass.targetdir;  // look at least in targetdir
    }
    my_blackboard.neckreq.set_request(NECK_REQ_LOOKINDIRECTION,look2dir.get_value());
  }
  else{
    LOGNEW_POL(0,<<"set neck selfpass. Turn 2 targetdir ");
    look2dir = scheduled_selfpass.targetdir;
    my_blackboard.neckreq.set_request(NECK_REQ_LOOKINDIRECTION,look2dir.get_value());    
  }
}




void Wball03::set_neck_selfpass(const ANGLE targetdir, const Vector &op_pos){
  if (my_blackboard.neckreq.is_set()){
      DBLOG_POL(0,<<"Check Selfpass: Neck Request already set (probably priority pass) ");
      return;
  }

  ANGLE opdir = (op_pos - WSinfo::me->pos).ARG();

  if((WSmemory::last_seen_in_dir(opdir) >0) && (WSinfo::me->pos.distance(op_pos)) 
     >  ServerOptions::visible_distance ){
    // havent looked for longer time an opponent is not in feelrange
    if(targetdir.diff(opdir) <110./180.*PI){ // look 2 opponent only if I have a chance to see him
      my_blackboard.neckreq.set_request(NECK_REQ_LOOKINDIRECTION,opdir.get_value());
      DBLOG_POL(0,<<"SET NECK: haven'nt looked to op. at "<<op_pos<<" direction "<<RAD2DEG(opdir.get_value())
		<<" since "<<WSmemory::last_seen_in_dir(opdir)
		<<" and opponent not close: turn neck");
      DBLOG_DRAW(0, C2D(op_pos.x, op_pos.y, 1.3, "orange"));
      return;
    }
    else{
      DBLOG_POL(0,<<"SET NECK: haven'nt looked to op. at "<<op_pos<<" direction "<<RAD2DEG(opdir.get_value())
		<<" since "<<WSmemory::last_seen_in_dir(opdir)
		<<" and opponent not close: BUT I CAN'T !!!!!!!!");
    }
  }

  if((WSmemory::last_seen_in_dir(targetdir) >0 &&
      WSinfo::me->pos.getX() < 36.) ||
     (WSmemory::last_seen_in_dir(targetdir) >0 &&
      targetdir.diff(ANGLE(0)) > 20/180.*PI) || 
     (WSmemory::last_seen_in_dir(targetdir) >2)){
    my_blackboard.neckreq.set_request(NECK_REQ_LOOKINDIRECTION,targetdir.get_value());
    DBLOG_POL(0,<<"selfpass: haven'nt looked in targetdir since "<<WSmemory::last_seen_in_dir(targetdir)
	      <<" turn neck 2 targetdir");
  }

}

bool Wball03::check_selfpass(const ANGLE targetdir, double &ballspeed, Vector &target, int &steps,
			     Vector &op_pos, int &op_num){

  //  double op_time2react = 1.0;
  double op_time2react = 0.0;
  /* op_time2react is the time that is assumed the opponents need to react. 0 is worst case, that they
     are maximally quick. This means that the game is less aggressive and much less effective
     1 assumes
     that ops. need 1 cycle to react. This is already pretty aggressive and (nearly) safe.
     Maybe this could be improved by selecting op_time2react depending on the position on the field
     or by using dribbling in hard cases.
  */


  if(my_role == 0){// I'm a defender: 
    if(WSinfo::me->pos.getX() > -5)
      return false;

    int stamina = Stamina::get_state();
    //DBLOG_POL(0,<<"Check selfpass: check stamina state "<<staminaa);
    if((stamina == STAMINA_STATE_ECONOMY || stamina == STAMINA_STATE_RESERVE )
       && WSinfo::me->pos.getX() >-35){// stamina low
      DBLOG_POL(0,<<"check selfpass: I'm a defender and stamina level not so good -> do not advance");
      return false;
    }

  }

  int max_dashes = 7; // be a little cautious anytime

  PlayerSet pset;


  if(WSinfo::me->pos.getX() + 3. < WSinfo::his_team_pos_of_offside_line()){
    max_dashes = 4;

    pset= WSinfo::valid_opponents;
    Vector endofregion;
    double scanrange = 20;
    for(int i=0; i<2; i++){
      if(i==0)
	scanrange = 20;
      else	
	scanrange = 10;
      endofregion.init_polar(scanrange, targetdir);
      endofregion += WSinfo::me->pos;
      Quadrangle2d check_area = Quadrangle2d(WSinfo::me->pos, endofregion, scanrange/2., scanrange);
      //      DBLOG_DRAW(1, check_area );
      pset.keep_players_in(check_area);
      if(pset.num >0){
	if(pset.num == 1 && pset[0] == WSinfo::his_goalie){
	  DBLOG_POL(3,<<"Only goalie before me -> do not reduce steps");
	  max_dashes = 100;
	}
	else{
	  if(i==0)
	    //max_dashes = 3;
	    max_dashes = 2; // ridi04 : be more cautios
	  else
	    max_dashes = 2;
	}
      }
    }
    DBLOG_POL(2,"Selfpass targetdir "<<RAD2DEG(targetdir.get_value())<<" I'm behind offside line;reducing max_dashes to "<<max_dashes);
  }

  // check if I should risk something
  if(WSinfo::me->pos.getX()  > 0 && targetdir.diff(ANGLE(0)) <90./180.*PI  ){
    double scanrange = 80;
    Vector endofregion;
    endofregion.init_polar(scanrange, targetdir);
    endofregion += WSinfo::me->pos;
    Quadrangle2d check_area = Quadrangle2d(WSinfo::me->pos, endofregion, 5., scanrange);
    //DBLOG_DRAW(0, check_area );
    pset.keep_players_in(check_area);
    if(pset.num == 1){
      if(pset[0] == WSinfo::his_goalie){
	op_time2react = 1.0;
	//op_time2react = 2.0;
	DBLOG_POL(3,"Selfpass targetdir "<<RAD2DEG(targetdir.get_value())
		  <<" Only goalie is before me -> risk something ");
      }
    }
    else if(pset.num == 0){
      if(WSinfo::his_goalie){// if goalie pointer is defined
	if(WSinfo::his_goalie->age > 2){
	  DBLOG_POL(3,"Selfpass targetdir "<<RAD2DEG(targetdir.get_value())
		    <<" goalie age > 2 -> reduce max_dashes ");
	  max_dashes = 4;
	}
      }
      else{
	DBLOG_POL(3,"Selfpass targetdir "<<RAD2DEG(targetdir.get_value())
		  <<" goalie pointer not defined/ not known) -> reduce max_dashes ");
	max_dashes = 4;
      }
      op_time2react = 1.0;
      DBLOG_POL(3,"Selfpass targetdir "<<RAD2DEG(targetdir.get_value())
		<<" Nobody is before me -> risk something ");
    }
  }

  if(WSinfo::me->pos.getX() + 3. > WSinfo::his_team_pos_of_offside_line()){
    DBLOG_POL(3,"Selfpass targetdir "<<RAD2DEG(targetdir.get_value())
	      <<" I'm at offside line -> risk something ");
    op_time2react = 1.0;
  }

  if(WSinfo::me->ang.diff(targetdir) >90./180.*PI ||
     WSmemory::last_seen_in_dir(targetdir) >1){
    max_dashes = 1;
    DBLOG_POL(3,"Selfpass targetdir "<< RAD2DEG(targetdir.get_value())
	      << " Bad body dir or see too old "<<WSmemory::last_seen_in_dir(targetdir)
	      <<" reduce number of dashes to "<<max_dashes);
  }


  op_time2react = 0.0; // ridi: reset, because of self determination in selfpass
  //if(my_role == 0){// I'm a defender: 
  if(my_blackboard.pass_or_dribble_intention.valid_at() == WSinfo::ws->time){
    op_time2react = -2.0; // make the opponent a little more fat to play safe passes!
    DBLOG_POL(0,<<"I can pass, CHECK SELFPASS set op_time2react to "<<op_time2react);
  }

  bool result = selfpass->is_selfpass_safe(targetdir, ballspeed, target, steps, op_pos, op_num,
					   max_dashes,op_time2react);
  pset= WSinfo::valid_teammates_without_me;
  pset.keep_players_in_circle(target,2);
  if (pset.num >0){
    if (result == true) {
      DBLOG_POL(3,"Selfpass ok, but target too close to teammate");
    }
    return false;
  }

  if(my_role == 0){// I'm a defender: 
    if(target.getX() >-10)
      return false;
  }

  if (result == false)
    return false;
  return true;
}


bool Wball03::check_previous_intention(Intention prev_intention, Intention  &new_intention){
  LOGNEW_POL(0,<<"Entering Check previous intention; valid at: "<<prev_intention.valid_at());
  if(prev_intention.valid_at() < WSinfo::ws->time -1){
    LOGNEW_POL(0,<<"Check previous intention: Previous intention not valid!");
    return false;
  }
  if (test_two_teammates_control_ball(prev_intention) == true){
    LOGNEW_POL(0,<<"Check previous intention: Two teammates control ball, previous intention is invalidated!");
    return false;
  }
  /*
  if(is_planned_pass_a_killer){
    if(prev_intention.get_type() != PASS &&
       prev_intention.get_type() != LAUFPASS){
      DBLOG_POL(0,<<"Check previous intention: planned pass is a killer -> RESET intention");
      return false;
    }
  }
  */
  
  // toga05: check if 'i am in trouble' (meaning: ball is in kick area of me _and_ opponent player)
  PlayerSet ballControllingOpponents = WSinfo::valid_opponents;
  ballControllingOpponents.keep_players_in_circle(WSinfo::ball->pos, 2.0*ServerOptions::kickable_area);
  ballControllingOpponents.keep_and_sort_closest_players_to_point(1, WSinfo::ball->pos);
  if (    ballControllingOpponents.num > 0
       &&   WSinfo::ball->pos.distance(ballControllingOpponents[0]->pos)
          < 1.05 * ballControllingOpponents[0]->kick_radius ) //5% additional safety margin
  {
    LOGNEW_POL(0,<<"Check previous intention: I am in trouble since an opponent has ball control, also!");
    return false;
  }
  // ridi04: check if I am attacked
  /* ridi 05: not used.
  if(am_I_attacked() || is_my_passway_in_danger()){
    if(prev_intention.get_type() != PASS &&
       prev_intention.get_type() != LAUFPASS){
      DBLOG_POL(0,<<"Check previous intention: I am attacked or my passway is in danger!");
      return false;
    }
  }
  */

  int targetplayer_number = prev_intention.target_player;
  if (targetplayer_number >11) targetplayer_number = 0; // invalid then
  float speed = prev_intention.kick_speed;
  Vector target = prev_intention.kick_target;
  Angle dir = (target - WSinfo::me->pos).arg(); // direction of the target
  ANGLE targetdir = prev_intention.target_body_dir; // for selfpasses
  Vector ballpos = WSinfo::ball->pos;
  //  Vector mypos = WSinfo::me->pos;
  Vector interceptpos, playerpos;  // return parameters; not considered here
  int advantage, number;
  //int mytime2intercept;  // return parameters; not considered here
  bool myteam; // return pararamter, not considered here
  Vector receiverpos; // used 
  //  int steps; // return parameter for selfpass, not used
  Vector op_pos; // return parameter for selfpass, not used
  bool stillok = false;
  // double ballspeed;
  //int op_number;
  int risky_pass = prev_intention.risky_pass;
  double new_kickspeed;
  int attacker_num;
  int tmp_advantage;

  switch(prev_intention.get_type()){
  case PASS:
    receiverpos = target; // default setting
    if(targetplayer_number >0){
      DBLOG_POL(BASELEVEL,
		<<"WBall03: Check previous intention: PASS: correct target, old: "<<target);
      PPlayer p= WSinfo::get_teammate_by_number(targetplayer_number);
      if(p){
	receiverpos = WSinfo::get_teammate_by_number(targetplayer_number)->pos;
	if(receiverpos.getX() > WSinfo::his_team_pos_of_offside_line()){
	  DBLOG_POL(BASELEVEL,
		    <<"WBall03: Check Pass considered harmful, offside of receiver -> RESET");
	  stillok = false;
	  break;
	}
	dir = (receiverpos - WSinfo::me->pos).arg(); // direction of the target
      }
      if(risky_pass == false)// regular evaluation
	stillok = Planning::is_pass_successful_hetero(ballpos, speed, dir, p, interceptpos, advantage); // use player info
      else{ // risky direct passes are evaluated differently
	DBLOG_POL(0,"REevaluating RISKY DIRECT PASS");
	stillok = Planning::is_laufpass_successful(ballpos,speed, 
						   dir,interceptpos,advantage,
						   number,playerpos,risky_pass);
      }
    } // target player known
    else{ // player number not known
      stillok = Planning::is_pass_successful(ballpos, speed, dir, receiverpos, interceptpos, advantage); // 
    }
    if(stillok == false){
      DBLOG_POL(BASELEVEL,
		<<"WBall03: Check previous intention: Pass considered harmful -> RESET");
    }
    else{
      DBLOG_POL(BASELEVEL,
		<<"WBall03: Pass considered ok, speed "<<speed<<" dir "<<RAD2DEG(dir));
    }
    target = receiverpos;
    break;
  case LAUFPASS:
    stillok = Planning::is_laufpass_successful2(ballpos,speed, dir);
    if(stillok == false && WSinfo::me->pos.getX() >35. && fabs(WSinfo::me->pos.getY()) <20.){
      // second chance for penaltyareapasses
      stillok = Planning::is_penaltyareapass_successful(ballpos,speed,dir);
    }

    if(stillok == false) {
      DBLOG_POL(BASELEVEL,
		<<"WBall03: Check previous intention: Pass considered harmful -> RESET");
    }
    break;
  case KICKNRUSH:
    stillok = true; // kicknrush remains ok, since this is default action!
    break; 
  case OPENING_SEQ:
    stillok = true; // kicknrush remains ok, since this is default action!
    if(onetwoholdturn->is_holdturn_safe() == false){
      DBLOG_POL(BASELEVEL, <<"WBALL 03: opening is interrupt, since holdturn not safe");
      stillok = false; 
    }  
    Policy_Tools::earliest_intercept_pos(WSinfo::me->pos,speed,dir,
					 interceptpos, myteam, number, advantage);
    if(interceptpos.getX() < WSinfo::his_team_pos_of_offside_line()){
      DBLOG_POL(BASELEVEL, <<"WBALL 03: opening is interrupt, since ipos of pass < offsideline");
      stillok = false; 
    }
    break; 
  case SELFPASS:
    if(is_selfpass_possible()){  // found a new selfpass that is possible
      if(Tools::evaluate_pos_analytically(scheduled_selfpass.targetpos) > 
	 Tools::evaluate_pos_analytically(prev_intention.kick_target) + 3.){
	// only stop a successful selfpass if a new selfpass improves considerably
	stillok = false;
	LOGNEW_POL(0,<<"Check previous intention: New Selfpass improves evaluation. Abort");
	break;
      }
    }
    if(selfpass2->is_selfpass_still_safe(tmp_advantage,prev_intention.target_body_dir, new_kickspeed, attacker_num)==true){
      speed = new_kickspeed;       // speed now has new kickspeed ; it is set below,  (see new_intention.correct_speed(speed))
      stillok = true;
      // set selfpass information for neck policy
      scheduled_selfpass.valid_at = WSinfo::ws->time;
      scheduled_selfpass.kickspeed = new_kickspeed;
      scheduled_selfpass.attacker_num = attacker_num;
      set_neck_selfpass2();
      LOGNEW_POL(0,<<"Check previous intention: SELFPASS to dir "
		 <<RAD2DEG(prev_intention.target_body_dir.get_value())<<" is still ok"<<" speed "<<speed);
    }
    else{
      stillok = false;
      LOGNEW_POL(0,<<"Check previous intention: Selfpass not safe. Aborting");
    }
    break;
  case IMMEDIATE_SELFPASS:
    stillok = false;
    DBLOG_POL(BASELEVEL,
	      <<"WBall03: Check previous intention: Selfpass ALWAYS reconsidered harmful -> RESET");
    break;
  case SCORE:
    stillok = true;
    if (prev_intention.immediatePass==true)
    {
      stillok = false;
      DBLOG_POL(BASELEVEL,
         <<"WBall03: Check previous intention: Score05_Sequence ALWAYS reconsidered harmful -> RESET");
    }
    break;    
  case DRIBBLE:
  case DRIBBLE_QUICKLY:
    stillok = false;
    break;
  case HOLDTURN:
  case TURN_AND_DASH:
    stillok = false;
    break;
  default:
    stillok = false;
    break;
  }
  if(stillok){
    DBLOG_POL(0,<<"Previous Intention still ok, just take it!");
    new_intention.confirm_intention(prev_intention,WSinfo::ws->time );
    new_intention.correct_target(target);
    new_intention.correct_speed(speed);
    DBLOG_DRAW(BASELEVEL, L2D(WSinfo::ball->pos.x,WSinfo::ball->pos.y,target.x, target.y,"lightblue"));
    return true;
  }
  return false;
}


bool Wball03::get_pass_or_dribble_intention(Intention &intention){
  AState current_state;
  AbstractMDP::copy_mdp2astate(current_state);

  return get_pass_or_dribble_intention(intention, current_state);
}

bool Wball03::get_pass_or_dribble_intention(Intention &intention, const Vector newmypos, const Vector newmyvel,
				     const Vector newballpos,const Vector newballvel){
  AState current_state;
  AbstractMDP::copy_mdp2astate(current_state);

  current_state.ball.pos = newballpos;
  current_state.ball.vel = newballvel;
  current_state.my_team[current_state.my_idx].pos = newmypos;
  current_state.my_team[current_state.my_idx].vel = newmyvel;



  return get_pass_or_dribble_intention(intention, current_state);
}



bool Wball03::get_pass_or_dribble_intention(Intention &intention, AState &state){
  AAction best_aaction;


  long ms_time= NEWGETTIME;

  intention.reset();

  //  if (neuro_wball->evaluate_passes_and_dribblings(best_aaction, state) == false){ // nothing found
  // new Osaka version:
  if (neuro_wball->evaluate_passes(best_aaction, state) == false){ // nothing found
    LOGNEW_POL(NEWBASELEVEL+0,<<"Get Pass Intention: No pass found");
    ms_time = NEWGETTIME - ms_time;
    LOGNEW_POL(0, << "Get Pass intention needed " << ms_time << " millis to decide****************");
    return false;
  }

  if(best_aaction.action_type == AACTION_TYPE_WAITANDSEE)
    last_waitandsee_at = WSinfo::ws->time;
  
  DBLOG_POL(0,<<"BEST PASS: Value: "<<best_aaction.V);
  aaction2intention(best_aaction, intention);

  LOGNEW_POL(NEWBASELEVEL+0,<<"Get Pass Intention: Found a pass (black). Taget "<<intention.kick_target
	     <<" speed "<<intention.kick_speed<<" Value: "<<best_aaction.V);

  LOGNEW_DRAW(NEWBASELEVEL+0, VL2D(WSinfo::ball->pos,
				  intention.kick_target,
				  "black"));

    ms_time = NEWGETTIME - ms_time;
    LOGNEW_POL(0, << "Get Pass intention needed " << ms_time << " millis to decide****************");

  // found a pass or dribbling
  return true;
}


bool Wball03::test_opening_seq(Intention &intention)
{
  if(onetwoholdturn->is_holdturn_safe() == false){
    DBLOG_POL(BASELEVEL, <<"WBALL 03: Opening not possible, since holdturn not safe");
    return false;
  }

  if (my_blackboard.pass_or_dribble_intention.valid_at() == WSinfo::ws->time){ // pass or dribble int. is set
    if(my_blackboard.pass_or_dribble_intention.kick_target.getX()>WSinfo::me->pos.getX()){
      DBLOG_POL(BASELEVEL+0,<<"WBALL03 No Kickrush, Pass is planned, and kick target before me!"
		<<my_blackboard.pass_or_dribble_intention.kick_target);
      return false;
    }
  }

#if 0
  if(WSinfo::me->pos.x <-10){
    DBLOG_POL(BASELEVEL+0,<<"WBALL03 No Opening, too far back");
    return false;
  }
#endif
  if(WSinfo::me->pos.getX() >ServerOptions::pitch_length - 20){
    DBLOG_POL(BASELEVEL+0,<<"WBALL03 No Opening, too far advanced");
    return false;
  }

  //  DBLOG_POL(BASELEVEL, <<"WBALL 03: Start Opening Seq");

  Vector target;
  double speed, dir;
  Vector ipos; // intercept position
  int advantage, closest_teammate;
  
  get_opening_pass_params(speed,dir,ipos,advantage,closest_teammate);
  if(speed <0)
    return false; // no kickrush params found

  if(ipos.getX() < WSinfo::his_team_pos_of_offside_line()){
    DBLOG_POL(BASELEVEL+0,<<"WBALL03 Opening sequence NOT possible, best ipos "<<ipos
	      <<" behind offsideline "<<WSinfo::his_team_pos_of_offside_line());
    return false;
  }
    
  target.init_polar(speed/(1-ServerOptions::ball_decay),dir);
  target += WSinfo::me->pos;
  
  //  speed = 3.0; // artifical speed to communicate opening seq.
  intention.set_opening_seq(target,speed, WSinfo::ws->time);
  DBLOG_POL(BASELEVEL+0,<<"WBALL03  Start Opening sequence to "<<target<<" dir "
	    <<RAD2DEG(dir)<<" w.speed "<<speed
	    <<" best ipos "<<ipos
	    <<" before offsideline "<<WSinfo::his_team_pos_of_offside_line());
  //DBLOG_POL(0, << _2D << C2D(target.x, target.y, 1, "#00ffff"));
  return true;
}





/** default move */
bool Wball03::test_default(Intention &intention)
{

  if(onestepkick->can_keep_ball_in_kickrange() == false){
    PlayerSet pset= WSinfo::valid_opponents;
    Vector endofregion;
    endofregion.init_polar(5, WSinfo::ball->vel.ARG());
    endofregion += WSinfo::me->pos;
    Quadrangle2d check_area = Quadrangle2d(WSinfo::me->pos, endofregion, 3);
    //DBLOG_DRAW(0, check_area );
    pset.keep_players_in(check_area);
    if(pset.num == 0){
      intention.set_holdturn(ANGLE(0.), WSinfo::ws->time); // this will maximally stop the ball
      LOGNEW_POL(NEWBASELEVEL+0,<<"DEFAULT Move - Stop Ball");
      return true;
    }
  }

  Vector opgoalpos = Vector (52.,0); // opponent goalpos

  if(cycles_in_waitandsee < 50 && onetwoholdturn->is_holdturn_safe() == true){
    ANGLE target_dir = (opgoalpos - WSinfo::me->pos).ARG();
    LOGNEW_POL(NEWBASELEVEL+0,<<"DEFAULT Move - Hold Turn is Safe");
    intention.set_holdturn(target_dir, WSinfo::ws->time);
    return true;
  }

  Vector target;
  double speed, dir;
  get_onestepkick_params(speed,dir);
  if(speed > 0){
    target.init_polar(speed/(1-ServerOptions::ball_decay),dir);
    target += WSinfo::ball->pos;
    intention.set_kicknrush(target,speed, WSinfo::ws->time);
    LOGNEW_POL(NEWBASELEVEL+0,<<"DEFAULT Clearance to "<<target<<" dir "<<RAD2DEG(dir)<<" w.speed "<<speed);
    return true;
  }

  LOGNEW_POL(NEWBASELEVEL+0,<<"DEFAULT Move - Hold Turn is NOT Safe, but no other alternative, TRY");
  intention.set_holdturn((opgoalpos - WSinfo::me->pos).ARG(), WSinfo::ws->time);
  return true;
}

// cijat
int Wball03::is_dribblebetween_possible(){
#define POL(XXX) LOGNEW_POL(1,<<"isDribblePossible:"<<XXX)

	bool insecure=false;
	bool isPossible=true;


	bool isTargetVeryCool = false;
	if(my_role == 0){// I'm a defender: 
    int stamina = Stamina::get_state();
    //DBLOG_POL(0,<<"Check selfpass: check stamina state "<<staminaa);
    if((stamina == STAMINA_STATE_ECONOMY || stamina == STAMINA_STATE_RESERVE )
       && WSinfo::me->pos.getX() >-35){// stamina low
      LOGNEW_POL(NEWBASELEVEL+0,<<"Check is dribble possible: I'm a defender and stamina level not so good -> dribble not possible");
			isPossible=false;
    }
  }


	// migrated checks from DribbleBetween BEGIN
	// Check: My Offside Line
	if (WSinfo::me->pos.getX() - WSinfo::my_team_pos_of_offside_line() < 5.0 && !isTargetVeryCool)
	{
		POL("Dribblink not safe: too close to my offside line");
		insecure=true;
	}
  if(WSinfo::me->pos.getX() <-FIELD_BORDER_X+16 && fabs(WSinfo::me->pos.getY())<16){
    POL("Dribblink not safe: in own penalty-area");
    insecure=true;
  }
	// Check: My Stamina
	if (
	      (WSinfo::me->stamina < (MIN_STAMINA_DRIBBLE) && WSinfo::me->pos.getX()>30)
	    ||(WSinfo::me->stamina < MIN_STAMINA_DRIBBLE*1.0 )
	   )
	{
		POL("Dribblink not safe: too weak (Stamina="<<WSinfo::me->stamina<<")");
		insecure=true;
	}
	// Check: My Role and Position
	if(OpponentAwarePositioning::getRole(WSinfo::me->number) == PT_DEFENDER
			&& WSinfo::me->pos.getX()>0.25*FIELD_BORDER_X){
		POL("Dribblink not safe: I'm a defender, don't attack!");
			insecure=true;
	}
	if (WSinfo::me->pos.distance(HIS_GOAL_CENTER) < 15.0 ) {     //CHANGE BY TG
		if (WSinfo::his_team_pos_of_offside_line() - WSinfo::me->pos.getX() > 4.0 && !isTargetVeryCool){
			POL("Dribblink not safe: too close to opponent goal");
			insecure = true;
		}
	}
	// migrated checks from DribbleBetween END

	
  // now estimate the optimal target 
  Vector target;
  if(FIELD_BORDER_X-WSinfo::me->pos.getX() < 3*MIN(0,fabs(WSinfo::me->pos.getY())-12)){
    //LOGNEW_POL(1,<<"Check is dribble possible: In 45 deg angle from goal.");
    target = Vector(50,FIELD_BORDER_X*0.8);
  } 
  else  
    target = Vector(50,WSinfo::me->pos.getY());

  dribble_between->set_target(target);
  bool isSafe = dribble_between->is_dribble_safe();
	bool isInsecure = dribble_between->is_dribble_insecure();
	     isSafe = isSafe && isPossible;

  LOGNEW_POL(0,<<"Check is dribble possible: is_dribble_safe() returns "<<isSafe<<", insecure="<<isInsecure);
	
	insecure = insecure || isInsecure;
	if(isSafe && !insecure) return 2;
	if(isSafe &&  insecure) return 1;
	if(!isSafe            ) return 0;
	return 0; // ridi05: default (to avoid compiler warnings)
#undef POL
}

// cijat
bool Wball03::test_dribbling(Intention& intention){
	// I just assume that the target has been set by is_dribblebetween_possible().
	if(!is_dribble_between_ok)
		return false;
	intention.set_dribble(dribble_between->get_target(), WSinfo::ws->time);
	if(dribble_between->is_neck_req_set())
		Tools::set_neck_request(NECK_REQ_LOOKINDIRECTION, dribble_between->get_neck_req());
	return true;
}



/** selects one player who is responsible for the ball */
bool Wball03::test_two_teammates_control_ball(Intention &intention){
  // test if Ball is in kickrange of me and a teammate!

  Vector ballpos = WSinfo::ball->pos;
  PPlayer p_tmp = WSinfo::valid_teammates_without_me.closest_player_to_point(ballpos);
  if (p_tmp == NULL) return false;
  int teammate = p_tmp->number;
  float dist = (p_tmp->pos - ballpos).norm();

  if((dist < .95*ServerOptions::catchable_area_l) && (teammate == WSinfo::ws->my_goalie_number)){
    LOG_ERR(0,<<"WBALL03: Our goalie has the ball, I could kick - Maybe I should back up?!");
    DBLOG_POL(BASELEVEL,<<"WBALL03: Our goalie has the ball, I could kick - Maybe I should back up?! (not yet implemented)");
  }

  if ((dist < ServerOptions::kickable_area) )
  {
    //we should risk nothing and just kick the ball away if we are too near to 
    //our offside line!
    if ( WSinfo::ball->pos.getX() - WSinfo::my_team_pos_of_offside_line() < 15.0 )
    {
      intention.set_panic_kick( HIS_GOAL_CENTER, WSinfo::ws->time);
      return true;
    }

    if (WSinfo::me->pos.getX() > p_tmp->pos.getX())
    {
      LOG_ERR(0,<<"WBALL03: Another teammate with smaller x position has the ball, too => my x pos is higher, I can kick!");
      DBLOG_POL(BASELEVEL,<<"WBALL03: I could kick Another teammate with lower number has the ball => my number is higher, I can kick! ");
      //i return false so that i can pass
      return false;
    }
    else
    {
      //intention.set_backup(WSinfo::ws->time);
      DBLOG_POL(BASELEVEL,<<"WBALL03: I could kick but another teammate with larger x pos has the ball, also => my x pos is smaller, I must retreat! ");
      return true;
    }
  }
  return false;
}



bool Wball03::I_can_advance_behind_offside_line(){
  // removal candidate 
  return false;
}

bool Wball03::is_pass_a_killer(){
  if(my_blackboard.pass_or_dribble_intention.valid_at() != WSinfo::ws->time)
    return false;

  bool result = false;

  // possibility 1: behind offside line
  Vector respos = my_blackboard.pass_or_dribble_intention.resultingpos;
  if(result == false && I_can_advance_behind_offside_line() == false &&
     respos.getX() > WSinfo::his_team_pos_of_offside_line()){
    DBLOG_POL(0,<<"Planned pass is a killer -> resulting pos is behind offside line and I can not advance behind offsideline");
    result = true;
  }

  if(result == false && Tools::is_a_scoring_position(respos) == true){
    DBLOG_POL(0,<<"Planned pass is a killer -> in scoring position");
    result = true;
  }


  PlayerSet pset;

  // possibility 2: teammate can advance
  if(result == false && WSinfo::me->pos.getX() > 30. &&
     fabs(WSinfo::me->pos.getY()) > fabs(respos.getY())){
    pset = WSinfo::valid_opponents;
    Vector endofregion;
    endofregion.init_polar(15, 0);
    endofregion += respos;
    Quadrangle2d check_area = Quadrangle2d(respos, endofregion, 5.);
    //DBLOG_DRAW(0, check_area );
    pset.keep_players_in(check_area);
    if(pset.num == 0 &&
       respos.getX() >= WSinfo::ball->pos.getX() -2){
      DBLOG_POL(0,<<"Planned pass is a killer -> from resulting pos teammate can advance ");
      result = true;;
    }
  }

  if(result == false && I_can_advance_behind_offside_line() == false &&
     am_I_attacked() == true &&
     (my_blackboard.pass_or_dribble_intention.resultingpos.getX() > WSinfo::me->pos.getX() ||
      fabs(my_blackboard.pass_or_dribble_intention.resultingpos.getY()) < fabs(WSinfo::me->pos.getY())) &&
     (fabs(my_blackboard.pass_or_dribble_intention.resultingpos.getY())<13 &&
      my_blackboard.pass_or_dribble_intention.resultingpos.getX()>35)){
    DBLOG_POL(BASELEVEL, <<" I cannot advance, teammate is in better position -> pass");
    result = true;
  }


  if(result == true){
    Vector target;
    double speed;
    my_blackboard.pass_or_dribble_intention.get_kick_info(speed, target);
    ANGLE targetdir = (target-WSinfo::me->pos).ARG();
    if(WSmemory::last_seen_in_dir(targetdir) >0){ 
      my_blackboard.neckreq.set_request(NECK_REQ_LOOKINDIRECTION,targetdir.get_value());
      DBLOG_POL(0,<<"Planned pass is a killer, but I have to look in target direction first "<<RAD2DEG((target-WSinfo::me->pos).arg()));
    }
    return true;
  }
  return false;
}



bool Wball03::is_my_passway_in_danger(){
  if (my_blackboard.pass_or_dribble_intention.valid_at()  != WSinfo::ws->time){
    DBLOG_POL(BASELEVEL, <<"check passing corridor: pass intention currently not valid !");
    return false;
  }

  float width = 10.0;
  if(my_role == 0)
    width = 10.0;


  PlayerSet pset = WSinfo::valid_opponents;
  Vector respos = my_blackboard.pass_or_dribble_intention.resultingpos;
  Quadrangle2d check_area = Quadrangle2d(WSinfo::me->pos, respos,width);
  //DBLOG_DRAW(0,check_area);
  pset.keep_players_in(check_area);
  if(pset.num>0){
     DBLOG_POL(0, <<"Someone enters passing corridor!");
    return true;
  }
  return false;
}

bool Wball03::am_I_attacked(){
  PlayerSet pset= WSinfo::valid_opponents;
  double radius_of_attacked_circle =  2 * ServerOptions::kickable_area + 2 * ServerOptions::player_speed_max;
  //DBLOG_DRAW(0,C2D(WSinfo::me->pos.x,WSinfo::me->pos.y,radius_of_attacked_circle,"red"));
  //DBLOG_DRAW(0,C2D(WSinfo::me->pos.x,WSinfo::me->pos.y,radius_of_attacked_circle-+.5,"black"));
  pset.keep_players_in_circle(WSinfo::me->pos,radius_of_attacked_circle); 
  //if(pset.num>0)
  //  DBLOG_POL(0,"I AM ATTACKED!");

  return  (pset.num >0);

}

bool Wball03::am_I_attacked2(){
  PlayerSet pset= WSinfo::valid_opponents;
  double radius_of_attacked_circle =  2 * ServerOptions::kickable_area;
  //DBLOG_DRAW(0,C2D(WSinfo::me->pos.x,WSinfo::me->pos.y,radius_of_attacked_circle,"red"));
  //DBLOG_DRAW(0,C2D(WSinfo::me->pos.x,WSinfo::me->pos.y,radius_of_attacked_circle-+.5,"black"));
  pset.keep_players_in_circle(WSinfo::me->pos,radius_of_attacked_circle); 
  //if(pset.num>0)
  //  DBLOG_POL(0,"I AM ATTACKED!");

  return  (pset.num >0);

}

int Wball03::I_am_heavily_attacked_since(){
  // returns -1, if no attacking occured. Must be called at least once every cycle to be effective.
  PlayerSet pset= WSinfo::valid_opponents;
  double radius_of_attacked_circle =  2*ServerOptions::kickable_area;
  pset.keep_players_in_circle(WSinfo::me->pos,radius_of_attacked_circle); 

  if(pset.num ==0){ //* no attack
      last_heavy_attack_at = -1; // reset
    return -1;
  }
  // I am heavily attacked this cycle
  if(last_heavy_attack_at <0){
    last_heavy_attack_at = WSinfo::ws->time; // remember start of heavy attack
    return 1;
  }
  return ((WSinfo::ws->time - last_heavy_attack_at) +1);
}



bool Wball03::test_holdturn(Intention &intention){
  if(cycles_in_waitandsee >= at_ball_patience){
    DBLOG_POL(BASELEVEL, <<"WBALL 03: HoldTurn NOT desired. Wait and see patience expired");
    return false;
  }
  if(onestepkick->can_keep_ball_in_kickrange() == false){
    DBLOG_POL(BASELEVEL, <<"WBALL 03: HoldTurn NOT possible. Can not keep ball in kickrange");
    return false;
  }

#if 0 // test only
  intention.set_holdturn(0, WSinfo::ws->time);
  return true;
#endif

  ANGLE targetdir;

  if(WSinfo::me->pos.getX() > 42.0){  // if I'm close to goal, then turn2goal
    targetdir = (Vector(47,0) - WSinfo::me->pos).ARG();
  }
  else{
    targetdir = ANGLE(0.);  // turn straight ahead
  }


  int targetplayer_age= 1000;

  PPlayer targetplayer = WSinfo::get_teammate_by_number(my_blackboard.pass_or_dribble_intention.target_player);
  if(targetplayer != NULL){
    targetplayer_age = targetplayer->age;
    DBLOG_POL(0,"Holdturn: Check pass candidate: Age of targetplayer "<<targetplayer_age);
  }

  PlayerSet pset= WSinfo::valid_opponents;
  pset.keep_players_in_circle(WSinfo::me->pos, 3.); 

  if (my_blackboard.pass_or_dribble_intention.valid_at() == WSinfo::ws->time){ 
    // pass or dribble int. is set
    if((pset.num >0) && // opponent is close and pass intention communicated
       (my_blackboard.pass_or_dribble_intention.valid_since() < WSinfo::ws->time)){ 
      DBLOG_POL(BASELEVEL, <<"WBALL 03: HoldTurn ok, but pass set and op. close -> no holdturn");
      return false;
    }
    if((my_blackboard.pass_or_dribble_intention.valid_since() < WSinfo::ws->time) // I had 1 cycle time to communicate
       && targetplayer_age <=1 // targetplayer is reasonably young
       //	 && ((WSinfo::me->pos.x > 20) || // I'm advanced, so play quickly
       && (my_blackboard.pass_or_dribble_intention.kick_target.getX()>WSinfo::me->pos.getX() - 3.)){
      // or kick target is before me
      DBLOG_POL(BASELEVEL, <<"WBALL 03: HoldTurn possible, but pass set since "
		<<WSinfo::ws->time<<" and advancing pass -> play");
      return false;
    }
    if(Tools::get_abs_angle(targetdir.get_value() - WSinfo::me->ang.get_value()) <15./180. *PI){
      if(my_blackboard.pass_or_dribble_intention.valid_since() < WSinfo::ws->time-1){
	// I had 2 cycles time to communicate
	DBLOG_POL(BASELEVEL, <<"WBALL 03: HoldTurn possible, but pass set since "
		  <<"WSinfo::ws->time"<<"-> play");
	return false;
      }
    }
  }
    
  if(onetwoholdturn->is_holdturn_safe() == false){
    DBLOG_POL(BASELEVEL, <<"WBALL 03: HoldTurn NOT possible. Not Safe");
    return false;
  }

  DBLOG_POL(BASELEVEL, <<"WBALL 03: Do HOLD and TURN.");

  intention.set_holdturn(targetdir, WSinfo::ws->time);
  return true;
}

bool Wball03::test_holdturn2(Intention &intention){
  if(onetwoholdturn->is_holdturn_safe() == false){
    LOGNEW_POL(NEWBASELEVEL+0, <<"HoldTurn NOT Safe");
    return false;
  }

  if(cycles_in_waitandsee >= 50){
    LOGNEW_POL(NEWBASELEVEL+0, <<"TEST HOLDTURN. wait and  see patience expired");
    return false;
  }
  if(onestepkick->can_keep_ball_in_kickrange() == false){
    LOGNEW_POL(NEWBASELEVEL+0, <<"WBALL 03: HoldTurn NOT possible. Can not keep ball in kickrange");
    return false;
  }

  ANGLE targetdir;

  if(WSinfo::me->pos.getX() > 42.0){  // if I'm close to goal, then turn2goal
    targetdir = (Vector(47,0) - WSinfo::me->pos).ARG();
  }
  else{
    targetdir = ANGLE(0.);  // turn straight ahead
  }
  intention.set_holdturn(targetdir, WSinfo::ws->time);
  return true;
}



/** What to do if two opponents attack me */
bool Wball03::test_in_trouble(Intention &intention){
  // test for situation 'a': Ball is in kickrange of me and opponent!

  Vector target;


  PlayerSet pset= WSinfo::valid_opponents;
  pset.keep_players_in_circle(WSinfo::ball->pos, 2.0*ServerOptions::kickable_area); 
  // considr only close ops. for correct
  pset.keep_and_sort_closest_players_to_point(1,WSinfo::ball->pos);
  if ( pset.num == 0 )
    return false; // no op. has ball in kickrange
  else
  {
    if ( WSinfo::ball->pos.distance( pset[0]->pos ) 
         > 1.05 * pset[0]->kick_radius ) //5% additional safety margin
      return false;
  }


  DBLOG_POL(BASELEVEL, << "WBALL03: Ball is in kickrange of me and opponent "<<pset[0]->number);
  if(   WSinfo::me->pos.getX() >-10
     || WSinfo::me->pos.getX() - WSinfo::my_team_pos_of_offside_line() < 10.0 )
  { 
    DBLOG_POL(BASELEVEL, << "WBALL03: I dont care for trouble. I'm too far advanced, just cont.");
    return false;
  }

  //////////////////////
  // ADD-ON by TG 04/05
  //////////////////////
  if (   WSinfo::me->pos.sqr_distance( MY_GOAL_CENTER ) < 20*20 )
  {
    double tackSuccProb = Tools::get_tackle_success_probability( WSinfo::me->pos, WSinfo::ball->pos, WSinfo::me->ang.get_value_0_p2PI() );
    PlayerSet kickCapableOpponents = WSinfo::valid_opponents;
    kickCapableOpponents.keep_and_sort_closest_players_to_point( 1, WSinfo::ball->pos );
    double oppKickPower = 0.0,
          ownKickPower = 0.0;
    if ( kickCapableOpponents.num > 0 )
    {
      oppKickPower = 100.0;
      Vector oppBallVec = (WSinfo::ball->pos - kickCapableOpponents[0]->pos);
      double  oppBallDistNetto = oppBallVec.norm() - kickCapableOpponents[0]->radius - ServerOptions::ball_size;
      double  oppBallAngle = (oppBallVec.ARG() - kickCapableOpponents[0]->ang).get_value_mPI_pPI();
      oppKickPower *= ServerOptions::kick_power_rate *
        (1 - 0.25*fabs(oppBallAngle)/PI - 0.25*oppBallDistNetto/(kickCapableOpponents[0]->kick_radius - kickCapableOpponents[0]->radius - ServerOptions::ball_size));
    }
    ownKickPower = 100.0;
    Vector ownBallVec = (WSinfo::ball->pos - WSinfo::me->pos);
    double  ownBallDistNetto = ownBallVec.norm() - WSinfo::me->radius - ServerOptions::ball_size;
    double  ownBallAngle = (ownBallVec.ARG() - WSinfo::me->ang).get_value_mPI_pPI();
    ownKickPower *= ServerOptions::kick_power_rate *
        (1 - 0.25*fabs(ownBallAngle)/PI - 0.25*ownBallDistNetto/(WSinfo::me->kick_radius - WSinfo::me->radius - ServerOptions::ball_size));

    //NOTE: A tackling is only advisable if
    //      (a) it has a very high success probability
    //      (b) I am quite weaker in shooting currently
    //      (c) when tackling I am much stronger than the opponent
    double kickPowerDelta = 0.5, powerWhenTackling = 100*ServerOptions::kick_power_rate;

    if (    tackSuccProb > 0.9 
         && oppKickPower - ownKickPower > kickPowerDelta
         && powerWhenTackling - oppKickPower > kickPowerDelta)  
    {
      double tacklePower;
      if ( fabs( WSinfo::me->ang.get_value_mPI_pPI() ) < PI*0.5 )
        tacklePower = 100;
      else
        tacklePower = -100;
      intention.set_tackling( tacklePower, WSinfo::ws->time );
      DBLOG_POL(BASELEVEL, << "WBALL03: TG: Tackle the ball away (prob="<<tackSuccProb<<").");
      return true;
    }
    DBLOG_POL(BASELEVEL, << "WBALL03: TG: I would like to tackle the ball, but tackle success probability is "<<tackSuccProb<<" (<0.9).");
  }
  //////////////////////

  // Danger: Opponent can also kick -> kick ball straight away
  if(WSinfo::me->pos.getX() <10){ // I am far from opponents goal
    target = Vector(52.5,0);
      //////////////////////
      // ADD-ON by TG 04/05
      //////////////////////
     if (   WSinfo::me->pos.sqr_distance( MY_GOAL_CENTER ) < 20*20 )
     {
       PlayerSet kickCapableOpponents = WSinfo::valid_opponents;
       kickCapableOpponents.keep_and_sort_closest_players_to_point( 1, WSinfo::ball->pos ); 
       if ( kickCapableOpponents.num > 0 )
       {
         //assuming the opponent will also kick, we ought to aim at a position which
         //results in a ball movement away from the opponent (when both ball accelaration vectors are combined)
         if (   kickCapableOpponents[0]->pos.getY() + kickCapableOpponents[0]->vel.getY()*kickCapableOpponents[0]->decay
              > WSinfo::me->pos.getY() + WSinfo::me->vel.getY()*WSinfo::me->decay )
           target = Vector(52.5, -40.0);
         else
           target = Vector(52.5, 40.0);
         DBLOG_POL(BASELEVEL, << "WBALL03: TG: I modified the target of my panic kick to (52.5,+-40.0)."); 
       }
     }
     //////////////////////
  }
  else if(WSinfo::me->pos.getX() <43.5){ // I have more than 7 meters to goal
    if(WSinfo::me->pos.getY() >0)
      target = Vector(52.5,ServerOptions::pitch_width/2.-5.);
    else
      target = Vector(52.5,-ServerOptions::pitch_width/2.+5.);
  }
  else{// I have less than 17 meters to goal)
    //target = Vector(35.,0);
    if(WSinfo::me->pos.getY() >0)
      target = Vector(52.5,ServerOptions::goal_width/2.-.5);
    else
      target = Vector(52.5,-ServerOptions::goal_width/2.+.5);
  }
  DBLOG_POL(BASELEVEL, << "WBALL03: panic: Kick Ball with maxpowert to target "<<target);
  intention.set_panic_kick(target, WSinfo::ws->time);
  return true;
}


bool Wball03::get_best_panic_selfpass(const double testdir[],const int num_dirs,double &speed, double &dir){
  Vector ipos;
  double tmp_speed;
  double best_speed = 0;
  double best_dir = 2*PI;
  
  int stamina = Stamina::get_state();
  //DBLOG_POL(0,<<"Check selfpass: check stamina state "<<staminaa);
  if(stamina == STAMINA_STATE_RESERVE )
    return false;


  for(int i=0;i<num_dirs;i++){
    if(fabs(testdir[i]) > 90/180.*PI)
      continue;
    onestepkick->reset_state();
    bool speed_ok = false;
    tmp_speed =.4;
    while(!speed_ok){
      tmp_speed +=.1;
      onestepkick->kick_in_dir_with_initial_vel(tmp_speed,ANGLE(testdir[i]));
      speed_ok = onestepkick->get_vel(tmp_speed);
      if(tmp_speed >1.3)
	speed_ok = true;
    }
    if (tmp_speed >1.3)
      continue; // no acceptable speed possible in that dir; check next directions

    int myadvantage = Policy_Tools::get_selfpass_advantage( tmp_speed, testdir[i], ipos);
#if 0
    DBLOG_POL(0,<<"Test panic selfpass in dir "<<RAD2DEG(testdir[i])<<" speed "<<tmp_speed
	      <<" advantage "<<myadvantage);
#endif

    if(myadvantage<1)
      continue;
    if(fabs(testdir[i]) < fabs(best_dir)){
      best_speed = tmp_speed;
      best_dir = testdir[i];
    }     
  }
  if(best_speed >0){
    speed = best_speed;
    dir = best_dir;
    return true;
  }
  return false;
}

int Wball03::howmany_kicks2pass(){
  if (my_blackboard.pass_or_dribble_intention.valid_at() != WSinfo::ws->time){
    return 0;
  }

  // now, check for turn neck request
  int pass_type = my_blackboard.pass_or_dribble_intention.get_type();
  if(pass_type == PASS || pass_type == LAUFPASS || pass_type == KICKNRUSH){
    // turn neck to pass receiver
    Vector target;
    double speed;
    my_blackboard.pass_or_dribble_intention.get_kick_info(speed, target);

    oneortwo->kick_to_pos_with_initial_vel(speed,target);
    double speed1, speed2;
    oneortwo->get_vel(speed1,speed2);
    DBLOG_POL(0,"how many steps to pass to "<<target<<" speed "<<speed<<" : speed1 "<<speed1<<" speed2: "<<speed2);
    if(fabs(speed1 -speed)<.1){
      return 1;
    }
    else{
      return 2;
    }	
  }
  return 0;
}



void Wball03::get_onestepkick_params(double &speed, double &dir){
  double tmp_speed;
  Vector final;
  Vector ballpos = WSinfo::ball->pos;
  const int max_targets = 360;
  double testdir[max_targets];
  double testspeed[max_targets];

  int num_dirs = 0;

  //  for(ANGLE angle=ANGLE(0);angle.get_value()<2*PI;angle+=ANGLE(5./180.*PI)){
  //  for(float ang=0.;ang<PI;ang+=5./180.*PI){
  for(float ang=0.;ang<PI/2.;ang+=5./180.*PI){ //ridi 05: allow only forward directions!
    for(int sign = -1; sign <= 1; sign +=2){
      ANGLE angle=ANGLE((float)(sign * ang));
      tmp_speed = onestepkick->get_max_vel_in_dir(angle);
      if(tmp_speed <0.1){
	final.init_polar(1.0, angle);
	final += ballpos;
	//DBLOG_DRAW(0,L2D(ballpos.x, ballpos.y, final.x, final.y, "000000"));
      }
      else{
	testdir[num_dirs] = angle.get_value();
	testspeed[num_dirs] = neuro_wball->adjust_speed(WSinfo::ball->pos, angle.get_value(),tmp_speed);
	if(num_dirs < max_targets)
	  num_dirs ++;
	else {
	  DBLOG_POL(0,"test onestep_kicks: Warning: too many targets");
	}
	final.init_polar(tmp_speed, angle);
	final += ballpos;
	//DBLOG_DRAW(0,L2D(ballpos.x, ballpos.y, final.x, final.y, "aaaaaa"));
      }
    }
  }


  int advantage;
  Vector ipos;
  int closest_teammate;
  Policy_Tools::get_best_kicknrush(WSinfo::me->pos,num_dirs,testdir,
				   testspeed,speed,dir,ipos,advantage, closest_teammate);

/*  if(   ipos.x < WSinfo::me->pos.x -20
     || ( ipos.distance(WSinfo::me->pos) > 20.0 && ipos.x - WSinfo::my_team_pos_of_offside_line() < 25.0 ) ) TG_OSAKA
  {
    speed = 0;
    return;
  }*/

  if(WSinfo::me->pos.getX() >0){
    if(advantage<1 || closest_teammate == WSinfo::me->number || in_penalty_mode == true){
      DBLOG_POL(0,<<"Wball03: Check Panic Selfpasses ");
      if(get_best_panic_selfpass(testdir,num_dirs,speed,dir)) {
        DBLOG_POL(0,<<"Wball03: found a panic selfpass speed "<<speed<<" dir "<<RAD2DEG(dir));
      }
    }
  }

  if(speed >0){
    DBLOG_POL(0,<<"Wball03: found onestepkick with advantage "<<advantage<<" resulting pos : "
	      <<ipos<<" closest teammate "<<closest_teammate);
    //DBLOG_DRAW(0,C2D(ipos.x,ipos.y,1.0,"00FFFF"));
    //DBLOG_DRAW(0,L2D(WSinfo::ball->pos.x,WSinfo::ball->pos.y,ipos.x,ipos.y,"00FFFF"));
  }
  else{
    DBLOG_POL(0,<<"Wball03: NO onestepkick found ");
  }
}

void Wball03::get_clearance_params(double &speed, double &dir){
  int advantage, closest_teammate;
  Vector ipos;

  return get_opening_pass_params(speed,dir, ipos, advantage, closest_teammate);
}


void Wball03::get_opening_pass_params(double &speed, double &dir, Vector &ipos, int &advantage,
				      int & closest_teammate){
  float min_angle = -50;
  float max_angle = 50;

  const int max_targets = 200;
  double testdir[max_targets];
  double testspeed[max_targets];

  int i= 0;
  float tmp_speed;
  Angle tmp_dir;
  for(float angle=min_angle;angle<max_angle;angle+=5){
    for(float speed=2.5;speed<2.6;speed+=.5){
      if (i>=max_targets)
	break;
      tmp_speed = speed;
      tmp_dir = angle/180. *PI;
      if(Planning::is_kick_possible(tmp_speed,tmp_dir) == false){
	continue;
      }
      testdir[i]= tmp_dir;
      testspeed[i]= tmp_speed;
      i++;
    }
  }

  Policy_Tools::get_best_kicknrush(WSinfo::me->pos,i,testdir,
				   testspeed,speed,dir,ipos,advantage, closest_teammate);
  if(speed >0){
    DBLOG_POL(0,<<"found kicknrush with advantage "<<advantage<<" resulting pos : "
	      <<ipos<<" closest teammate "<<closest_teammate);
    //DBLOG_DRAW(0,C2D(ipos.x,ipos.y,1.0,"00FFFF"));
    //DBLOG_DRAW(0,L2D(WSinfo::ball->pos.x,WSinfo::ball->pos.y,	     ipos.x,ipos.y,"00FFFF"));
  }
  else{
    DBLOG_POL(0,<<"Wball03: NO kicknrush found ");
  }


}


bool Wball03::is_dribblestraight_possible(){
  ANGLE opgoaldir = (Vector(52,0) - WSinfo::me->pos).ARG(); //go towards goal


  if(my_role == 0){// I'm a defender: 
    if(WSinfo::me->pos.getX() > -5)
      return false;
    int stamina = Stamina::get_state();
    //DBLOG_POL(0,<<"Check selfpass: check stamina state "<<staminaa);
    if((stamina == STAMINA_STATE_ECONOMY || stamina == STAMINA_STATE_RESERVE )
       && WSinfo::me->pos.getX() >-35){// stamina low
      DBLOG_POL(0,<<"DRIBBLE STRAIGHT: I'm a defender and stamina level not so good -> do not advance");
      return false;
    }
  }

  if((WSinfo::me->ang.diff(ANGLE(0))>20/180.*PI || WSinfo::me->pos.getX() > 45) &&
     (WSinfo::me->ang.diff(opgoaldir)>20/180.*PI)){
    return false;
  }

  //  DBLOG_POL(0,<<"DRIBBLE STRAIGHT: me->ang "<<RAD2DEG(WSinfo::me->ang.get_value())<<" OK for Dribble");

  if(dribblestraight->is_dribble_safe(0)){
    if(my_blackboard.pass_or_dribble_intention.valid_since() < WSinfo::ws->time){
      // check area before me
      // modified by cijat: be more careful.
      // especially overtaking opponents should be taken into account
      // or taken care of by dribble_around.
      PlayerSet pset = WSinfo::valid_opponents;
      Vector endofregion;
      Vector startofregion;
      startofregion.init_polar(-.5, WSinfo::me->ang);
      startofregion += WSinfo::me->pos;
      endofregion.init_polar(4, WSinfo::me->ang);
      endofregion += WSinfo::me->pos;
      Quadrangle2d check_area = Quadrangle2d(startofregion, endofregion, 3);
      //DBLOG_DRAW(0, check_area );
      pset.keep_players_in(check_area);
      if(pset.num >0){
	DBLOG_POL(0,<<"DRIBBLE STRAIGHT: Player directly before me and pass intention is set -> do not dribble");
	return false; // go on in any case
      }
    }
    DBLOG_POL(0,<<"DRIBBLE STRAIGHT: possible");
    return true;
  }
  return false;
}



bool Wball03::test_pass_or_dribble(Intention &intention){
  if(my_blackboard.pass_or_dribble_intention.valid_at() == WSinfo::ws->time){
    intention = my_blackboard.pass_or_dribble_intention;
    return true;
  }
  return false;
}

void Wball03::check_write2blackboard(){
  int main_type = my_blackboard.intention.get_type();
  bool main_type_is_pass = false;
  int potential_pass_type = my_blackboard.pass_or_dribble_intention.get_type();

  if (main_type == PASS || main_type == LAUFPASS || 
      main_type == KICKNRUSH) 
    main_type_is_pass = true;
  
  // check for turn neck requests first
  if (my_blackboard.neckreq.is_set()){
    LOGNEW_POL(0,<<"W2BLACKBOARD: Neck Request is SET to dir :"
	       <<RAD2DEG(my_blackboard.neckreq.get_param()));
    //Blackboard::set_neck_request(my_blackboard.neckreq.get_type(),my_blackboard.neckreq.get_param());
    Tools::set_neck_request(my_blackboard.neckreq.get_type(),my_blackboard.neckreq.get_param());
  }
  else {
    LOGNEW_POL(0,<<"W2BLACKBOARD: No neck request has been set in my_blackboard.");
  }
  
  // check communication request for main type
  if(main_type == OPENING_SEQ || main_type == KICKNRUSH){
    Blackboard::pass_intention = my_blackboard.intention;
    DBLOG_POL(0,<<"W2BLACKBOARD:: intention opening seq / kicknrush Set blackboard intention: valid at "
	      << Blackboard::pass_intention.valid_at());
    return;
  }

  if (my_blackboard.pass_or_dribble_intention.valid_at() != WSinfo::ws->time){
    DBLOG_POL(0,<<"W2BLACKBOARD: Check write2blackboard: no pass or dribble intention is set");
    return;
  }

  // now, check if turn neck 2 passreceiver is possible/ required
  if (main_type_is_pass == true){ // no doubt -> look to pass receiver
    Vector target;
    double speed;
    my_blackboard.pass_or_dribble_intention.get_kick_info(speed, target);
    ANGLE ball2targetdir = (target - WSinfo::ball->pos).ARG(); // direction of the target
    if(WSmemory::last_seen_in_dir(ball2targetdir) >=0){ // probably change this to > 0 -> sometimes look!!!
      DBLOG_POL(0,<<"W2BLACKBOARD: Intention is to play pass-> look 2 receiver "<<RAD2DEG(ball2targetdir.get_value()));
      Tools::set_neck_request(NECK_REQ_LOOKINDIRECTION, ball2targetdir.get_value());
    }
  }

  if(my_blackboard.neckreq.is_set()== false){
    if(potential_pass_type == PASS || potential_pass_type == LAUFPASS || potential_pass_type == KICKNRUSH){
      DBLOG_POL(0,<<"W2BLACKBOARD: no neck req. is set, check 2 look to POTENTIAL pass receiver");  
      Vector target;
      double speed;
      my_blackboard.pass_or_dribble_intention.get_kick_info(speed, target);
      ANGLE ball2targetdir = (target - WSinfo::ball->pos).ARG(); // direction of the target
      if(WSmemory::last_seen_in_dir(ball2targetdir) >0){ // probably change this to >= 0 -> always look!!!
	DBLOG_POL(0,<<"W2BLACKBOARD: Intention is to play pass-> look 2 receiver "
		  <<RAD2DEG(ball2targetdir.get_value()));
	Tools::set_neck_request(NECK_REQ_LOOKINDIRECTION, ball2targetdir.get_value());
      }
    }
  }
  else {
    LOGNEW_POL(0,<<"W2BLACKBOARD: No Neck Request has been set in my_blackboard.");
  }
  


  PlayerSet pset = WSinfo::valid_opponents;
  Vector endofregion;
  const double scanrange = 7;
  endofregion.init_polar(scanrange,(my_blackboard.intention.kick_target -  WSinfo::me->pos).ARG());
  endofregion += WSinfo::me->pos;
  Quadrangle2d check_area = Quadrangle2d(WSinfo::me->pos, endofregion, 5., scanrange);
  pset.keep_players_in(check_area);
  bool opps_in_targetdir = (pset.num >0);

  // if opponent could get in my kickrange, then keep pass or dribble intention  
  if((main_type == SELFPASS || main_type == DRIBBLE) && // I'm dribbling
     is_pass_a_killer() == false && // do not reset killer passes 
     am_I_attacked() == false && opps_in_targetdir == false){ // I have enough space around me
    DBLOG_POL(0,<<"W2BLACKBOARD: selfpass or dribble and not attacked -> do not communicate AND reset pass intent.");
    my_blackboard.pass_or_dribble_intention.reset();
    return;
  }


  // now, check for communication request
  if(potential_pass_type == PASS || potential_pass_type == LAUFPASS || potential_pass_type == KICKNRUSH){
    DBLOG_POL(0,<<"W2BLACKBOARD: potential pass intention is set");
    // for now, communication is done indirectly; should be improved
    Blackboard::pass_intention = my_blackboard.pass_or_dribble_intention;
		//my_blackboard.intention = my_blackboard.pass_or_dribble_intention;
  }
}




void Wball03::aaction2intention(const AAction &aaction, Intention &intention){

  double speed = aaction.kick_velocity;
  Vector target = aaction.target_position;
  double kickdir = aaction.kick_dir;
  int selfpass_steps = aaction.advantage;

  Vector op_pos = aaction.actual_resulting_position;

  switch(aaction.action_type){
  case  AACTION_TYPE_PASS:
    intention.set_pass(target,speed, WSinfo::ws->time, aaction.targetplayer_number, 0, 
		       aaction.actual_resulting_position, aaction.potential_position );
    break;
  case  AACTION_TYPE_LAUFPASS:
    intention.set_laufpass(target,speed, WSinfo::ws->time, aaction.targetplayer_number, 0, 
			   aaction.actual_resulting_position, aaction.risky_pass,  aaction.potential_position);
    break;
  case AACTION_TYPE_SELFPASS:
    if (selfpass_steps > 2){ // I can go more than 2 steps -> reset pass intention
      DBLOG_POL(1,"Advance w. Selfpass: can go more than two steps -> reset pass intention");
      my_blackboard.pass_or_dribble_intention.reset();
    } 
    DBLOG_POL(0,"Advance w. Selfpass");
    set_neck_selfpass(ANGLE(kickdir), op_pos);

    intention.set_selfpass(ANGLE(kickdir), target, speed, WSinfo::ws->time);
    break;
  case AACTION_TYPE_IMMEDIATE_SELFPASS:
    target.init_polar(5.,kickdir);
    target += WSinfo::ball->pos;
    intention.set_immediateselfpass(target,speed, WSinfo::ws->time);
    break;
  case  AACTION_TYPE_WAITANDSEE:
    intention.set_waitandsee(WSinfo::ws->time);
    break;
  case AACTION_TYPE_TURN_AND_DASH:
    intention.set_turnanddash(WSinfo::ws->time);
    break;
  case  AACTION_TYPE_DRIBBLE:
    if(closest_opponent)
      set_neck_selfpass(WSinfo::me->ang, closest_opponent->pos);
    else
      set_neck_selfpass(WSinfo::me->ang, Vector(52.0,0) );

    intention.set_dribble(target, WSinfo::ws->time);
    break;
  default:
    DBLOG_POL(0,<<"WBALL03 aaction2intention: AActionType not known");
    LOG_ERR(0,<<"WBALL03 aaction2intention: AActionType not known");
  }
  intention.V = aaction.V;
}

bool Wball03::get_turn_and_dash(Cmd &cmd){
  // used for turn_and_dash
  Vector dummy1, dummy3;
  Vector mynewpos,ballnewpos;
  double dummy2;
  int dash = 0;  
  Cmd_Body testcmd;
  double required_turn;
  bool dash_found;

  required_turn = 
    Tools::get_angle_between_null_2PI(WSinfo::ball->vel.arg() - 
				      WSinfo::me->ang.get_value());
  // If I can get ball by dashing, dash
  dash_found=false;
  if(Tools::get_abs_angle(required_turn)<40./180. *PI){
    for(dash=100;dash>=30;dash-=10){
      testcmd.unset_lock();
      testcmd.unset_cmd();
      testcmd.set_dash(dash);
      Tools::model_cmd_main(WSinfo::me->pos, WSinfo::me->vel, WSinfo::me->ang.get_value(), 
			    WSinfo::ball->pos,
			    WSinfo::ball->vel,
			    testcmd, mynewpos, dummy1, dummy2, ballnewpos, dummy3);
      if((mynewpos-ballnewpos).norm()<0.8*ServerOptions::kickable_area){
	dash_found=true;
	break;
      }
      }
  }
  if(dash_found ==true){
    LOG_POL(BASELEVEL,<<"TURN_AND_DASH: I found a dash "<<dash);
    basiccmd->set_dash(dash);
    basiccmd->get_cmd(cmd);
    return true;
    }
  
  if(Tools::get_abs_angle(required_turn)>10./180. *PI){
    LOG_POL(BASELEVEL,<<"Intention2cmd: Turn and Dash: turn 2 balldir "
	    <<RAD2DEG(WSinfo::ball->vel.arg())<<" Have to turn by "
	    <<RAD2DEG(required_turn));
    basiccmd->set_turn_inertia(required_turn);
    basiccmd->get_cmd(cmd);
    return true;
  }
  // dash forward
  basiccmd->set_dash(100);
  basiccmd->get_cmd(cmd);
  return true;
}

bool Wball03::get_opening_seq_cmd( const float  speed, const Vector target,Cmd &cmd){

  Vector ipos;
  bool myteam;
  int number;
  int advantage;
  Angle dir = (target-WSinfo::ball->pos).arg();
  DBLOG_POL(BASELEVEL+0,<<"Get Command for Opening sequence to "<<target<<" dir "
	    <<RAD2DEG(dir)<<" w.speed "<<speed);
  //DBLOG_DRAW(0, C2D(target.x, target.y, 1, "#00ffff"));

  Policy_Tools::earliest_intercept_pos(WSinfo::me->pos,speed,dir,
				       ipos, myteam, number, advantage);


  if((advantage >0) && (my_blackboard.intention.valid_since() < WSinfo::ws->time)){
    // pass possible and I decided it at least one cycle ago -> already communicated 
    my_blackboard.intention.set_kicknrush(target,speed, WSinfo::ws->time); // directly set new intent.
    DBLOG_POL(BASELEVEL+0,<<"WBALL03 Opening sequence DO LAUFPASS, advantage ok "<<advantage
	      <<target<<" dir "
	      <<RAD2DEG(dir)<<" w.speed "<<speed<<" receiver "<<number);
    //DBLOG_DRAW(0, C2D(target.x, target.y, 1, "#00ffff"));
    neurokick->kick_to_pos_with_initial_vel(speed,target);
    neurokick->get_cmd(cmd);
    return true;
  }

  DBLOG_POL(0,<<"wball03: get opening seq cmd: do holdturn");
  onetwoholdturn->get_cmd(cmd);
  return true;  
}

bool Wball03::intention2cmd(Intention &intention, Cmd &cmd){
  double speed;
  Vector target;
  speed = intention.kick_speed;
  target = intention.kick_target;
  ANGLE targetdir = intention.target_body_dir; // for selfpasses
  //  int targetplayer_number = intention.target_player;
  //  double ball2targetdir = (target - WSinfo::ball->pos).arg(); // direction of the target
  bool cankeepball = onestepkick->can_keep_ball_in_kickrange();
  //double speed1step = Move_1Step_Kick::get_vel_in_dir(speed, ball2targetdir);
  //bool need_only1kick = (fabs(speed -speed1step) < .2);
  double opposite_balldir;
  double kick_dir;
  oot_intention = intention;//hauke

  double toRightCornerFlag = fabs((Vector(FIELD_BORDER_X,-FIELD_BORDER_Y) - WSinfo::me->pos).ARG().get_value_mPI_pPI());
  double toLeftCornerFlag = fabs((Vector(FIELD_BORDER_X,FIELD_BORDER_Y) - WSinfo::me->pos).ARG().get_value_mPI_pPI());
  double toGoal = fabs(WSinfo::me->ang.get_value_mPI_pPI());
    

  //tell the blackboard's pass intention whether this pass is really
  //intended to be actively played (i.e. a kick is already initiated
  //within the current cycle
  if (
         intention.get_type() == PASS
      || intention.get_type() == LAUFPASS
      || intention.get_type() == KICKNRUSH
      || intention.get_type() == PANIC_KICK
     )
    Blackboard::pass_intention.immediatePass = true;
  else
    Blackboard::pass_intention.immediatePass = false;

  switch(intention.get_type()){
  case  PASS:
  case  LAUFPASS:
    speed = intention.kick_speed;
    target = intention.kick_target;
    /*
    LOGNEW_POL(NEWBASELEVEL+0,<<"WBALL03: Intention2cmd: passing to teammate "<<targetplayer_number
	      //<<" onestepkick is possible (0=false) "<<need_only1kick
	      <<" speed "<<speed<<" to target "<<target.x<<" "<<target.y);
    */
    if(intention.risky_pass == true){
      LOGNEW_POL(NEWBASELEVEL+0,<<"WBALL03: RISKY PASS!!!");
    }
    neurokick->kick_to_pos_with_initial_vel(speed,target);
    neurokick->get_cmd(cmd);
    return true;
    break;
  case  OPENING_SEQ:
    speed = intention.kick_speed;
    target = intention.kick_target;
    /*
    LOGNEW_POL(NEWBASELEVEL+0,<<"WBALL03: Intention2cmd: Opening Seq for teammate "<<targetplayer_number
	      //<<" onestepkick is possible (0=false) "<<need_only1kick
	      <<" speed "<<speed<<" to target "<<target.x<<" "<<target.y);
    */
    return get_opening_seq_cmd(speed,target,cmd);
    break;
  case SELFPASS:
    LOGNEW_POL(NEWBASELEVEL+0,<<"Intention type SELFPASS2 to target "<<target
	       <<" w. speed "<<speed<<" targetdir "<<RAD2DEG(intention.target_body_dir.get_value()));
    selfpass2->get_cmd(cmd,intention.target_body_dir, intention.kick_target, intention.kick_speed);
    return true;
    break;

    // pre-Osaka 05:
    /*
    LOGNEW_POL(NEWBASELEVEL+0,<<"WBALL03: Intention type SELFPASS in dir "<<RAD2DEG(targetdir.get_value()));
    //selfpass->set_params(speed,target);
    return selfpass->get_cmd(cmd, targetdir, speed, target);
    
    break;
    */
  case IMMEDIATE_SELFPASS:
    speed = intention.kick_speed;
    target = intention.kick_target;
    kick_dir = (target - WSinfo::ball->pos).arg();
    mdpInfo::set_my_intention(DECISION_TYPE_IMMEDIATE_SELFPASS,
			      speed,target.getX(),target.getY(),0,
			      0);  
    LOGNEW_POL(NEWBASELEVEL+0,<<"Intention2cmd: Immediate Selfpass with speed "
	      <<speed<<" to target "<<target.getX()<<" "<<target.getY());
    if((Tools::get_abs_angle(WSinfo::ball->vel.arg() - kick_dir) 
	<10/180. *PI) && fabs(WSinfo::ball->vel.norm() - speed) < 0.05){
      // Ball already has desired dir and speed
      LOGNEW_POL(NEWBASELEVEL+0,<<"turn and dash");
      return get_turn_and_dash(cmd);
    }
    LOGNEW_POL(NEWBASELEVEL+0,<<"Intention2cmd: Immediate Selfpass with speed "
	      <<speed<<" to target "<<target.getX()<<" "<<target.getY());
    neurokick->kick_to_pos_with_initial_vel(speed,target);
    neurokick->get_cmd(cmd);
    return true;
    break;
  case  WAITANDSEE:
    LOGNEW_POL(NEWBASELEVEL+0,<<"Intention2cmd: Wait and see ");
    if(cankeepball){
      mdpInfo::set_my_intention(DECISION_TYPE_WAITANDSEE);
      onetwoholdturn->get_cmd(cmd);
      last_waitandsee_at = WSinfo::ws->time;
      return true;
    }
    else{
      opposite_balldir = WSinfo::ball->vel.arg() + PI;
      opposite_balldir = opposite_balldir - WSinfo::me->ang.get_value();
      LOG_ERR(0,<<"Can't keep the ball in my kickrange, and I do not plan to pass. Kick to "
	      <<RAD2DEG(opposite_balldir));
      basiccmd->set_kick(100,ANGLE(opposite_balldir));
      basiccmd->get_cmd(cmd);
      return true;
    }

    break;
  case TURN_AND_DASH:
    LOGNEW_POL(NEWBASELEVEL+0,<<"Intention2cmd: Turn and Dash ");
    return get_turn_and_dash(cmd);
    break;
  case  DRIBBLE:  
    //cijat
    if(false && is_dribble_straight_ok
	&& toGoal< toRightCornerFlag
	&& toGoal< toLeftCornerFlag
	&& (!am_I_attacked()||!is_dribble_between_ok)
	&& WSinfo::me->pos.getX()<FIELD_BORDER_X-10) {
      LOGNEW_POL(NEWBASELEVEL+0,"DRIBBLING STRAIGHT");
      dribblestraight->get_cmd(cmd);
    }
    else{
      LOGNEW_POL(NEWBASELEVEL+0,"DRIBBLING BETWEEN ");
      dribble_between->set_target(intention.player_target);
      dribble_between->get_cmd(cmd);
    }
    return true;
    break;
  case  DRIBBLE_QUICKLY:  
    LOGNEW_POL(NEWBASELEVEL+0,"DRIBBLING STRAIGHT");
    dribblestraight->get_cmd(cmd);
    return true;
    break;
  case SCORE:
    speed = intention.kick_speed;
    target = intention.kick_target;
    if (intention.immediatePass == false)
    {
      mdpInfo::set_my_intention(DECISION_TYPE_SCORE, speed, 0, target.getX(), target.getY(),0);
      LOGNEW_POL(NEWBASELEVEL+0,<<"intention2cmd: try to score w speed "<<speed<<" to target "<<target);
      neurokick->kick_to_pos_with_initial_vel(speed,target);
      neurokick->get_cmd(cmd);
    }
    else
    {
      LOGNEW_POL(NEWBASELEVEL+0,<<"intention2cmd: try to score by sequence, using onetstepkick with speed "<<speed<<" to target "<<target);
      onestepkick->kick_to_pos_with_initial_vel(speed,target);      
      onestepkick->get_cmd(cmd);
    }
    return true;
    break;
  case KICKNRUSH:
    speed = intention.kick_speed;
    target = intention.kick_target;
    mdpInfo::set_my_intention(DECISION_TYPE_KICKNRUSH, speed,0,0,0, 0);
    LOGNEW_POL(NEWBASELEVEL+0,<<"intention2cmd: kicknrush w speed "<<speed<<" to target "<<target);
    neurokick->kick_to_pos_with_initial_vel(speed,target);
    neurokick->get_cmd(cmd);
    return true;
    break;
  case PANIC_KICK:
    LOGNEW_POL(NEWBASELEVEL+0,<< "intention2cmd: PANIC KICK.");
    target = intention.kick_target;
    kick_dir = (target - WSinfo::me->pos).arg() - WSinfo::me->ang.get_value();
    basiccmd->set_kick(100,ANGLE(kick_dir));
    basiccmd->get_cmd(cmd);
    return true;
    break;
  case TACKLING:
    speed = intention.kick_speed;
    LOGNEW_POL(NEWBASELEVEL+0,<< "intention2cmd: Make a Tackling (TG).");
    basiccmd->set_tackle( speed );
    basiccmd->get_cmd(cmd);
    return true;
    break;
  case BACKUP:
    LOGNEW_POL(NEWBASELEVEL+0,<<"intention2cmd: back up (two teammates at ball)  not yet implemented");
    LOG_ERR(BASELEVEL,<<"WBALL03 intention2cmd: back up (two teammates at ball)  not yet implemented");
    //ridi03: todo
    return false;
    break;
  case HOLDTURN:
    LOGNEW_POL(NEWBASELEVEL+0,<<"intention2cmd: holdturn in dir "<<RAD2DEG(intention.target_body_dir.get_value()));
    if(cankeepball){
      if(onetwoholdturn->is_holdturn_safe() == false){
	LOGNEW_POL(NEWBASELEVEL+0,<<" intention2cmd: holdturn NOT safe, relaxed trial (should only occur in troubled sits)");
	last_waitandsee_at = WSinfo::ws->time;
	onetwoholdturn->get_cmd_relaxed(cmd);
	return true;
      }
      last_waitandsee_at = WSinfo::ws->time;
      onetwoholdturn->get_cmd(cmd,intention.target_body_dir);
      return true;
    }
    else{
      opposite_balldir = WSinfo::ball->vel.arg() + PI;
      opposite_balldir = opposite_balldir - WSinfo::me->ang.get_value();
      LOGNEW_POL(NEWBASELEVEL+0,<<" intention2cmd: holdturn. Can't keep the ball in my kickrange, and I do not plan to pass. Kick to "<<RAD2DEG(opposite_balldir));
      LOG_ERR(0,<<"Can't keep the ball in my kickrange, and I do not plan to pass. Kick to "
	      <<RAD2DEG(opposite_balldir));
      basiccmd->set_kick(100,ANGLE(opposite_balldir));
      basiccmd->get_cmd(cmd);
      return true;
    }


    return true;
  default:
    LOGNEW_POL(NEWBASELEVEL+0,<<"intention2cmd: Intention not known");
    LOG_ERR(0,<<"WBALL03 intention2cmd: AActionType not known");
    return false;
  }
  return false;
}

/**************************************************/

/* NEW ROUTINES FOR PRIORITY PASSES: 05 */

/**************************************************/


bool Wball03::test_priority_pass2(Intention &intention){

  if(my_blackboard.pass_or_dribble_intention.valid_at() != WSinfo::ws->time) // no pass scheduled
    return false;

  Vector pass_target;
  double pass_speed;
  my_blackboard.pass_or_dribble_intention.get_kick_info(pass_speed, pass_target);
  Vector pass_resulting_pos = my_blackboard.pass_or_dribble_intention.resultingpos;
  //  Vector pass_potential_pos = my_blackboard.pass_or_dribble_intention.potential_pos;  this is often not realistic
  Vector pass_potential_pos = my_blackboard.pass_or_dribble_intention.resultingpos;
  ANGLE pass_dir = (pass_target-WSinfo::me->pos).ARG();

  LOGNEW_POL(NEWBASELEVEL+0,<<"Test priority pass (red). Taget "<<my_blackboard.pass_or_dribble_intention.kick_target
	     <<" speed "<<my_blackboard.pass_or_dribble_intention.kick_speed
	     <<" Respos: "<<pass_resulting_pos<<" potentialpos: "<<pass_potential_pos);

  LOGNEW_DRAW(NEWBASELEVEL+0, VL2D(WSinfo::ball->pos,
				  my_blackboard.pass_or_dribble_intention.kick_target,
				  "red"));


  // compute all the predicates first
  bool overcome_offside_tm = Tools::can_advance_behind_offsideline(pass_potential_pos);
  bool attacked = am_I_attacked2();
  bool potential2advance = is_dribble_ok || is_selfpass_possible();
  bool passway_in_danger = is_my_passway_in_danger();
  // ranges of pass quality:  ok, cool, supercool, killer
  bool pass_quality_ok = false;
  bool pass_quality_cool =false;

  // predicades about pass information state
  bool pass_is_announced = my_blackboard.pass_or_dribble_intention.valid_since() < WSinfo::ws->time;
  bool I_can_look_in_passdir = Tools::could_see_in_direction(pass_dir);
  int kicks2pass = howmany_kicks2pass();
  int last_looked_in_dir = WSmemory::last_seen_in_dir(pass_dir);
  bool view_information_is_fresh = false;
  
  if((last_looked_in_dir == 0) ||  // very recent view information
     (last_looked_in_dir == 1 && kicks2pass <= 1) ){ // recent view information and quick pass possible
    // this guarantees, that at the time I play the pass, the view information is at most 2 cyles old
    view_information_is_fresh = true;
  }

  //  Vector my_potential_pos = Tools::check_potential_pos(WSinfo::me->pos, 5.0); // check potential advance with max. 5 m
  Vector my_potential_pos;
  if(is_selfpass_possible()){
    //    my_potential_pos = Tools::check_potential_pos(scheduled_selfpass.targetpos); 
    my_potential_pos = scheduled_selfpass.targetpos; 
  }
  else{
    //    my_potential_pos = Tools::check_potential_pos(WSinfo::me->pos, 3.0); // check potential advance with max. 3 m
    my_potential_pos = WSinfo::me->pos;  // my pot. pos might be harmful
  }

  // determine principle pass quality:
  double evaluation_delta;

  LOGNEW_POL(NEWBASELEVEL, <<"Compare  potential paspos:"<<pass_potential_pos
	     <<" to my potential pos "<<my_potential_pos
	     <<Tools::compare_positions(pass_potential_pos, my_potential_pos,evaluation_delta)
	     <<" eval. delta: "<<evaluation_delta);
  
  if(Tools::compare_positions(pass_potential_pos, my_potential_pos,evaluation_delta) == EQUAL){
    // positions are in the same equivalence class
    if(pass_resulting_pos.getX()>WSinfo::me->pos.getX()){
      // I don't have to play a back pass to get this potential position.
      if(evaluation_delta >=0.){
	LOGNEW_POL(NEWBASELEVEL+1, <<"I can at least improve a bit by passing.  QUALITY ok.");
	pass_quality_ok= true;
      }
      if(evaluation_delta >10.){
	LOGNEW_POL(NEWBASELEVEL+1, <<"My potential looks not so good, can improve signf: QUALITY: cool");
	pass_quality_cool= true;
      }
    } // resulting pos is ok.
    // enables many passes:, avoids selfish play
    if(attacked == true && evaluation_delta >0)
      pass_quality_cool = true;


  }  // pass position and my position are EQUAL

  LOGNEW_POL(NEWBASELEVEL, " I can score "<< Tools::can_score(WSinfo::me->pos)
	     <<"Teammate can score: "<<Tools::can_score(pass_potential_pos));


  if(Tools::can_score(pass_potential_pos) == true && potential2advance == false){ 
    // if I could score, than I would have already done it...
    LOGNEW_POL(NEWBASELEVEL, <<"Teammate can score, I not (otherwise would have done). pass quality cool!");
    pass_quality_cool = true;
  }

  if(Tools::compare_positions(pass_potential_pos, my_potential_pos,evaluation_delta) == FIRST){
    LOGNEW_POL(NEWBASELEVEL+1, <<"Teammate has greater potential than I: pass quality cool");
    pass_quality_cool = true;
  }

  if(attacked == true && overcome_offside_tm == true){
    LOGNEW_POL(NEWBASELEVEL+1, <<"I am attacked and teammate can overcome offside -> pass");
    pass_quality_cool = true;
  }

  if(Tools::close2_goalline(WSinfo::me->pos) && (attacked == true ||potential2advance == false) 
     && pass_resulting_pos.getX() > 25.){
    // I am already close to goalline, now play a pass if reasonable
    LOGNEW_POL(NEWBASELEVEL+1, <<"I am close2 goalline and am attacked -> pass");
    pass_quality_cool = true;    
  }


  bool do_priority_pass = false;

  // 1. check whether I should better pass than being selfish
  if(potential2advance == false){ // I cannot go further
    if(attacked == true){
      LOGNEW_POL(NEWBASELEVEL+1, <<"PRIORITY PASS: Cannot dribble and I am attacked -> pass");
      do_priority_pass = true;
    }
    if(passway_in_danger == true){
      LOGNEW_POL(NEWBASELEVEL+1, <<"PRIORITY PASS: Cannot dribble and passway in danger -> pass");
      do_priority_pass = true;
    }
    if(view_information_is_fresh == true && pass_quality_ok == true){
      LOGNEW_POL(NEWBASELEVEL+1, <<"PRIORITY PASS:  I cannot dribble and pass quality ok -> pass");
      do_priority_pass = true;
    }
  }

  if(I_am_heavily_attacked_since() > 1){ // I am heavily attacked for more than ... cycles
    if(view_information_is_fresh == true && pass_quality_ok == true){
      LOGNEW_POL(NEWBASELEVEL+1, <<"PRIORITY PASS:  I am heavily attacked since several cycles -> pass");
      do_priority_pass = true;
    }
  }

  /* this enables 'wild' passing
  if(pass_quality_ok == true){
    LOGNEW_POL(NEWBASELEVEL+1, <<"PRIORITY PASS: Pass quality_ok");
    do_priority_pass = true;
  }
  */
  // CIJAT_OSAKA
  if(pass_quality_ok == true && is_dribble_between_insecure && view_information_is_fresh){
    LOGNEW_POL(NEWBASELEVEL+1, <<"PRIORITY PASS: Pass quality_ok, dribbling insecure");
    do_priority_pass = true;
  }  

  if(potential2advance == false && pass_quality_ok == true){
    LOGNEW_POL(NEWBASELEVEL+1, <<"PRIORITY PASS: Pass quality_ok and I cannot advance");
    do_priority_pass = true;
  }

  if(pass_quality_cool == true){
    LOGNEW_POL(NEWBASELEVEL+1, <<"PRIORITY PASS: Pass quality_cool");
    do_priority_pass = true;
  }


  LOGNEW_POL(NEWBASELEVEL+0,"Test Priorty pass: Some Predicates: is dribble ok "<<is_dribble_ok
	     <<" is_dribble_insecure: "<<is_dribble_between_insecure);





  if(do_priority_pass == false){
    LOGNEW_POL(NEWBASELEVEL+1, <<"PRIORITY PASS: Scheduled pass not considered cool!");
    return false;
  }

  // I' d really like to play this pass!!!
  // 1. look into pass direction - no matter what happens
  my_blackboard.neckreq.set_request(NECK_REQ_LOOKINDIRECTION,pass_dir.get_value());

  // 1. check, if everything's ok
  if(pass_is_announced == true && view_information_is_fresh == true){
    //everythings'  allright, pass
    LOGNEW_POL(NEWBASELEVEL+1, <<"PRIORITY PASS: pass intention is Cool and I looked into direction and communicated -> Play Pass");
    intention = my_blackboard.pass_or_dribble_intention;
    return true;
  }
  
  //2. pass not accepted yet, maybe I did not look into pass direction
  if(view_information_is_fresh == false){  
    if(kicks2pass >= 2 && I_can_look_in_passdir){  // need at least 2 kicks and can look in pass direction
      LOGNEW_POL(NEWBASELEVEL+1, <<"PRIORITY PASS: view information old, but I need 2 kicks anyway, and can look in passdir -> DO PRIRORITY PASS");
      intention = my_blackboard.pass_or_dribble_intention;
      return true; // alternatively: immediately play killer pass then
    }
    // want to play pass, but need to look first
    if(onetwoholdturn->is_holdturn_safe() == true){
      LOGNEW_POL(NEWBASELEVEL+1, <<"PRIORITY PASS: Haven't looked in dir,  HoldTurn possible. Hold and look");
      intention.set_holdturn(ANGLE(pass_dir), WSinfo::ws->time);
      return true;
    }
    else{ // this is a very risky situation: didn't look, maybe not communicated, but cannot hold ball!!! So I have to play
      LOGNEW_POL(NEWBASELEVEL+1, <<"PRIORITY PASS: Haven't looked in dir, but HoldTurn NOT possible. Play immediately");
      intention = my_blackboard.pass_or_dribble_intention;
      return true;     
    }
  }
  
  if(pass_is_announced == false){
    if(onetwoholdturn->is_holdturn_safe() == true){
      LOGNEW_POL(NEWBASELEVEL+1, <<"PRIORITY PASS: Pass not communicated,  HoldTurn possible. Hold and look");
      intention.set_holdturn(ANGLE(pass_dir), WSinfo::ws->time);
      return true;
    }
    else{ // this is a very risky situation: didn't look, maybe not communicated, but cannot hold ball!!! So I have to play
      LOGNEW_POL(NEWBASELEVEL+1, <<"PRIORITY PASS: Pass not communicated, but HoldTurn NOT possible. Play immediately");
      intention = my_blackboard.pass_or_dribble_intention;
      return true;     
    }
  }

  LOGNEW_POL(NEWBASELEVEL+0, <<"PRIORITY PASS: OOOOOOOPs!  Something happened. Should not end here!!");
  return false;
}



bool Wball03::is_dribblestraight_possible2(){
  ANGLE opgoaldir = (Vector(52,0) - WSinfo::me->pos).ARG(); //go towards goal

  if(my_role == 0){// I'm a defender: 
    if(WSinfo::me->pos.getX() > -5)
      return false;
    int stamina = Stamina::get_state();
    //DBLOG_POL(0,<<"Check selfpass: check stamina state "<<staminaa);
    if((stamina == STAMINA_STATE_ECONOMY || stamina == STAMINA_STATE_RESERVE )
       && WSinfo::me->pos.getX() >-35){// stamina low
      DBLOG_POL(0,<<"DRIBBLE STRAIGHT: I'm a defender and stamina level not so good -> do not advance");
      return false;
    }
  }

  if((WSinfo::me->ang.diff(ANGLE(0))>20/180.*PI || WSinfo::me->pos.getX() > 45) &&
     (WSinfo::me->ang.diff(opgoaldir)>20/180.*PI)){
    return false;
  }

  //  DBLOG_POL(0,<<"DRIBBLE STRAIGHT: me->ang "<<RAD2DEG(WSinfo::me->ang.get_value())<<" OK for Dribble");

  if(dribblestraight->is_dribble_safe(0) == false){
    return false;
  }

  // from here on, dribblestraight is possible in principle. Check special cases.

  if(WSinfo::me->pos.getX()>25){// allready advanced enough, be more cautious now
    PlayerSet pset = WSinfo::valid_opponents;
    Vector endofregion;
    const double scanrange = 5;
    endofregion.init_polar(scanrange, WSinfo::me->ang);
    endofregion += WSinfo::me->pos;
    Quadrangle2d check_area = Quadrangle2d(WSinfo::me->pos, endofregion,scanrange/2., scanrange);
    //LOGNEW_DRAW(NEWBASELEVEL+1, check_area );
    pset.keep_players_in(check_area);
    if(pset.num > 0){
      if(pset.num > 1 || pset[0] != WSinfo::his_goalie){
	LOGNEW_POL(NEWBASELEVEL+1,<<"is DRIBBLESTRAIGHT possible?: in attack area: I CAN, but do NOT DRIBBLESTRAIGHT, crowded area ");
	return false;
      }
    }
  }


  // check area before me
  PlayerSet pset = WSinfo::valid_opponents;
  Vector endofregion;
  endofregion.init_polar(4, WSinfo::me->ang);
  endofregion += WSinfo::me->pos;
  Quadrangle2d check_area = Quadrangle2d(WSinfo::me->pos, endofregion, 2);
  DBLOG_DRAW(0, check_area );
  pset.keep_players_in(check_area);
  if(pset.num >0){
    LOGNEW_POL(BASELEVEL+1,<<"is DRIBBLESTRAIGHT possible: NO (player in front of me)");
    return false; 
  }
  LOGNEW_POL(BASELEVEL+1,<<"is DRIBBLESTRAIGHT possible: YES");
  return true;
}


bool Wball03::test_dribble_straight(Intention &intention){
  // a solo has preference before any other pass or holdturn
  //  double speed;
  //  Vector ipos;
  //int steps;
  //Vector op_pos; // position of attacking opponent -> view strategy


  if(is_dribblestraight_possible2() == false)
    return false;

  Vector target;
  target.init_polar(2.0,WSinfo::me->ang); 
  intention.set_dribblequickly(target, WSinfo::ws->time);
  return true;
}


bool Wball03::selfpass_dir_ok2(const ANGLE dir){

  const ANGLE opgoaldir = (Vector(47.,0) - WSinfo::me->pos).ARG(); //go towards goal

  if(dir.diff(ANGLE(0))<50/180.*PI )
    return true;
  if(WSinfo::me->pos.getX()>30 && dir.diff(opgoaldir)<20/180.*PI )
    return true;

  return false;
}

bool Wball03::selfpass_area_is_free(const ANGLE dir){

  PlayerSet pset= WSinfo::valid_opponents;
  Vector endofregion;
  double length = 35.;
  double startwidth = 1.;
  endofregion.init_polar(length, dir);
  endofregion += WSinfo::me->pos;
  //  Quadrangle2d check_area = Quadrangle2d(WSinfo::me->pos, endofregion, startwidth, 1.25*length);
  Quadrangle2d check_area = Quadrangle2d(WSinfo::me->pos, endofregion, startwidth, 1.5*length);
  //LOGNEW_DRAW(NEWBASELEVEL, check_area );
  pset.keep_players_in(check_area);
  if(pset.num >0){
    if(pset.num <= 1 && pset[0] == WSinfo::his_goalie){
      DBLOG_POL(0,<<"Selfpass area in dir "<<RAD2DEG(dir.get_value())<<" is free: Only goalie before me: free ");
    }
    else{
      DBLOG_POL(0,<<"Selfpass area in dir  "<<RAD2DEG(dir.get_value())<<" is NOT free");
      return false;
    }
  }
  return true;
}


bool Wball03::test_selfpasses(Intention &intention){ // was: advance_in_scoring_area
  Vector targetpos;
  double kickspeed;
  int steps2go;
  Vector oppos;
  int opnum;

#define MIN_IMPROVEMENT 2.0  // do selfpasses only, if it's worth it!!

  if(is_selfpass_possible()){

    int opage = -1;
    ANGLE opdir = scheduled_selfpass.targetdir;  // reasonable default (just in case...)
    // precompute attacker information:
    LOGNEW_POL(0, " Set Neck Selfpass: attacking op: "<<scheduled_selfpass.attacker_num);
    PlayerSet opset= WSinfo::valid_opponents;
    PPlayer attacker = opset.get_player_by_number(scheduled_selfpass.attacker_num);
    if(attacker != NULL){
      //LOGNEW_DRAW(0, C2D(attacker->pos.x, attacker->pos.y, 1.3, "orange"));
      Tools::display_direction(attacker->pos, attacker->ang, 3.0);
      opdir = (attacker->pos  - WSinfo::me->pos).ARG();
      opage = attacker->age;
    }

    double my_evaluation= Tools::evaluate_pos_analytically(WSinfo::me->pos);
    if(scheduled_selfpass.evaluation > my_evaluation + MIN_IMPROVEMENT){ // 
      // Hey, I found a reasonable selfpass. 
      if((WSmemory::last_seen_in_dir(scheduled_selfpass.targetdir) >1) ||
	 (WSinfo::me->ang.diff(scheduled_selfpass.targetdir) >90./180.*PI)){  
	// my world information about that region might be bad...
	int tmp_advantage;
	if(selfpass2->is_turn2dir_safe(tmp_advantage,scheduled_selfpass.targetdir, kickspeed,targetpos,steps2go, oppos, opnum)){
	  LOGNEW_POL(NEWBASELEVEL+0,"Havent' looked there too long. Turn first; it's possible");      
	  intention.set_selfpass(scheduled_selfpass.targetdir, targetpos,kickspeed, WSinfo::ws->time);
	  set_neck_selfpass2();
	  return true;
	}
	LOGNEW_POL(NEWBASELEVEL+0,"Selfpass scheduled, but information might be old. Not possible to turn");      	return false;
      }// dangerous targetdir
      if(attacker!= NULL && Tools::could_see_in_direction(opdir) == false 
	 && opdir.diff(scheduled_selfpass.targetdir) < 100/180.*PI 
	 && opage >= 1){
	// my world information about that region might be bad...
	int tmp_advantage;
	if(selfpass2->is_turn2dir_safe(tmp_advantage,scheduled_selfpass.targetdir, kickspeed,targetpos,steps2go, oppos, opnum)){
	  LOGNEW_POL(NEWBASELEVEL+0,"Havent' seen op too long. Turn first; it's possible");      
	  intention.set_selfpass(scheduled_selfpass.targetdir, targetpos,kickspeed, WSinfo::ws->time);
	  set_neck_selfpass2();
	  return true;
	}
	LOGNEW_POL(NEWBASELEVEL+0,"Selfpass scheduled, but information might be old. Not possible to turn");      	return false;
      }// dangerous targetdir
      intention.set_selfpass(scheduled_selfpass.targetdir, scheduled_selfpass.targetpos ,
			     scheduled_selfpass.kickspeed, WSinfo::ws->time);
      LOGNEW_POL(NEWBASELEVEL+0,"TEST SELFPASSES: scheduled and Improvement. DO it ");
      set_neck_selfpass2();
      return true;
    }
    else{
      LOGNEW_POL(NEWBASELEVEL+0,"TEST SELFPASSES: scheduled; but NO improvement. ");
      return false;
    }
  }
  LOGNEW_POL(NEWBASELEVEL+0,"TEST SELFPASSES: NO selfpass scheduled ");
  return false;
}

bool Wball03::test_selfpasses_in_scoring_area(Intention &intention){ // was: advance_in_scoring_area
  LOGNEW_POL(NEWBASELEVEL+0,"TEST SELFPASSES IN SCORING AREA: not yet implemented! ");
  return false;
}



bool Wball03::is_selfpass_possible(){
  if(scheduled_selfpass.valid_at == WSinfo::ws->time)
    return true;
  return false;
}

void Wball03::precompute_best_selfpass(){  // side effect: fill out scheduled selfpass formular
  precompute_best_selfpass(WSinfo::me->pos, WSinfo::me->vel, WSinfo::me->ang, WSinfo::me->stamina,
			   WSinfo::ball->pos, WSinfo::ball->vel);
}


bool Wball03::I_am_in_selfpass_goalarea(){

  if(WSinfo::me->pos.getX() >FIELD_BORDER_X -10.)
    return true;
  return false;
}

void Wball03::precompute_best_selfpass(const Vector mypos, const Vector myvel, const ANGLE myang,
				       const double mystamina,
					   const Vector ballpos, const Vector ballvel){  
  // side effect: fill out scheduled selfpass formular
  // can be also called with vitual positions.

#define NUM_DIRS 20

  if(I_am_in_selfpass_goalarea()){
    return precompute_best_selfpass_in_goalarea(mypos,myvel,myang, mystamina, ballpos,ballvel);
  }

  if(mypos.getX() <-30.){// do this only in an advance position
    LOGNEW_POL(0," precompute selfpass: pos < -30, false: "<<mypos.getX());

    scheduled_selfpass.valid_at = -1;
    return;
  }

  if(my_role == 0){ // defender
    LOGNEW_POL(0," precompute selfpass: Im a defender, false");
    scheduled_selfpass.valid_at = -1;
    return;
  }
  

  ANGLE targetdir;
  ANGLE testdir[NUM_DIRS];
  int num_dirs = 0;


  if(selfpass_dir_ok2(myang)== true)
    testdir[num_dirs ++] = myang;
  testdir[num_dirs ++] = ANGLE(0); // go straight
  //  if(mypos.y >-10 || mypos.x >FIELD_BORDER_X - 10){  // I am on the left side
  if(mypos.getY() >-10){  // I am on the left side
    testdir[num_dirs ++] = ANGLE(45/180.*PI);
    // tend to go to left (via wings)
  }
  //  if(mypos.y <+10||  mypos.x >FIELD_BORDER_X - 10  ){ // I am more on the right side
  if(mypos.getY() <+10){ // I am more on the right side
    testdir[num_dirs ++] = ANGLE(-45/180.*PI);
    // tend to go to right or straight (via wings)
  }

  ANGLE bestdir;
  double bestspeed = 0;
  Vector besttargetpos, bestattackerpos;
  double max_evaluation;
  int beststeps = 0;
  int bestattacker_num = 0;

  max_evaluation=-1000;
  int reduce_dashes = 0;

  for(int i=0; i<num_dirs; i++){
    targetdir = testdir[i];
    LOGNEW_POL(NEWBASELEVEL+1,"Check selfpass in dir"<<RAD2DEG(targetdir.get_value()));
    /*
      if(WSmemory::last_seen_in_dir(targetdir) >1){
      LOGNEW_POL(NEWBASELEVEL+1,"Havent' looked there too long. Do not consider");      
      continue;
    }
    */
    // if the following is activated, selfpasses are rather conservative ...
    if(selfpass_area_is_free(targetdir)== true){
      LOGNEW_POL(NEWBASELEVEL+1,"SELFPASSES area is free. consider");
    }
    else{ // selfpass_area_is_not_free 
      // test: if +x large, than play selfpassesmore often
      //      if(mypos.x + 10 > WSinfo::his_team_pos_of_offside_line()){
      //      if(mypos.x + 10.0 > WSinfo::his_team_pos_of_offside_line()){ // test only
      if(mypos.getX() + 3.0 > WSinfo::his_team_pos_of_offside_line()){ // test only
	// selfpass_area_is_not_free, but I am close to offsideline
	LOGNEW_POL(NEWBASELEVEL+1,"SELFPASSES area is occupied. But close 2 offside line, risk ");
	//	reduce_dashes = 2;
      }
      else{
	LOGNEW_POL(NEWBASELEVEL+1,"SELFPASSES area is occupied. not advanced. Do NOT consider");
	continue;
      }
    }
   

    Vector targetpos;
    double kickspeed;
    int steps2go;
    Vector oppos;
    int opnum;
    int tmp_advantage;
    if(selfpass2->is_selfpass_safe(tmp_advantage,targetdir, kickspeed,targetpos,steps2go, oppos, opnum,
				   mypos,myvel, myang,mystamina,ballpos, ballvel, false, reduce_dashes)){
      double evaluation= Tools::evaluate_pos_analytically(targetpos);
      LOGNEW_POL(NEWBASELEVEL+1,"SELFPASS in dir "<<RAD2DEG(targetdir.get_value())
		 <<"is safe.  Evaluation "<<evaluation<<" op.pos "<<oppos);
      if(evaluation > max_evaluation){
	bestdir = targetdir;
	bestspeed = kickspeed;
	besttargetpos = targetpos;
	max_evaluation=evaluation;
	beststeps = steps2go;
	bestattackerpos = oppos;
	bestattacker_num = opnum;
      }
    }// selfpass is safe
  } // for all dirs
  

  if(max_evaluation > -1000) {
    LOGNEW_POL(NEWBASELEVEL+0,"Precompute best selfpass "
	       <<" targetdir "<<RAD2DEG(bestdir.get_value()));
    scheduled_selfpass.valid_at = WSinfo::ws->time;
    scheduled_selfpass.targetpos = besttargetpos;
    scheduled_selfpass.kickspeed = bestspeed;
    scheduled_selfpass.targetdir = bestdir;
    scheduled_selfpass.steps2go = beststeps;
    scheduled_selfpass.evaluation = max_evaluation;
    scheduled_selfpass.attackerpos = bestattackerpos;
    scheduled_selfpass.attacker_num = bestattacker_num;
  }
  else{
    scheduled_selfpass.valid_at = -1;
    LOGNEW_POL(NEWBASELEVEL+0,"TEST SELFPASSES: no direction found! ");
  }

  return;
}

bool Wball03::I_am_close_to_offsideline(const double howclose){
  if(WSinfo::me->pos.getX() + howclose > WSinfo::his_team_pos_of_offside_line())
    return true;
  return false;
}


void Wball03::precompute_best_selfpass_in_goalarea(const Vector mypos, const Vector myvel, const ANGLE myang,
						   const double mystamina,
						   const Vector ballpos, const Vector ballvel){  
  // side effect: fill out scheduled selfpass formular
  // can be also called with vitual positions.

#define NUM_DIRS 20


  ANGLE targetdir;
  ANGLE testdir[NUM_DIRS];
  int num_dirs = 0;


  if(selfpass_dir_ok2(myang)== true)
    testdir[num_dirs ++] = myang;
  testdir[num_dirs ++] = ANGLE(0); // go straight
  testdir[num_dirs ++] = ANGLE(45/180.*PI);
  testdir[num_dirs ++] = ANGLE(-45/180.*PI);
  testdir[num_dirs ++] = ANGLE(90/180.*PI);
  testdir[num_dirs ++] = ANGLE(-90/180.*PI);

  ANGLE bestdir;
  double bestspeed = 0;
  Vector besttargetpos, bestattackerpos;
  double max_evaluation;
  int beststeps= 0;
  int bestattacker_num= 0;

  max_evaluation=-1000;
  int reduce_dashes = 0;

  for(int i=0; i<num_dirs; i++){
    targetdir = testdir[i];
    LOGNEW_POL(NEWBASELEVEL+1,"Check  GOALAREA selfpass in dir"<<RAD2DEG(targetdir.get_value()));
    // if the following is activated, selfpasses are rather conservative ...
    if(selfpass_area_is_free(targetdir)== true){
      LOGNEW_POL(NEWBASELEVEL+1,"SELFPASSES  GOALAREA area is free. consider");
    }
    else{ // selfpass_area_is_not_free 
      // test: if +x large, than play selfpassesmore often
      //      if(mypos.x + 10 > WSinfo::his_team_pos_of_offside_line()){
      if(mypos.getX() + 10.0 > WSinfo::his_team_pos_of_offside_line()){ // test only
	// selfpass_area_is_not_free, but I am close to offsideline
	LOGNEW_POL(NEWBASELEVEL+1,"SELFPASSES  GOALAREA area is occupied. But close 2 offside line, risk ");
	//	reduce_dashes = 2;
      }
      else{
	LOGNEW_POL(NEWBASELEVEL+1,"SELFPASSES  GOALAREA area is occupied. not advanced. Do NOT consider");
	continue;
      }
    }
   

    Vector targetpos;
    double kickspeed;
    int steps2go;
    Vector oppos;
    int opnum;
    int tmp_advantage;
    
    if(selfpass2->is_selfpass_safe(tmp_advantage, targetdir, kickspeed,targetpos,steps2go, oppos, opnum,
				   mypos,myvel, myang,mystamina,ballpos, ballvel, false, reduce_dashes)){
      double evaluation= Tools::evaluate_pos_analytically(targetpos);
      LOGNEW_POL(NEWBASELEVEL+1,"SELFPASS GOALAREA in dir "<<RAD2DEG(targetdir.get_value())
		 <<"is safe.  Evaluation "<<evaluation<<" op.pos "<<oppos);
      if(evaluation > max_evaluation){
	bestdir = targetdir;
	bestspeed = kickspeed;
	besttargetpos = targetpos;
	max_evaluation=evaluation;
	beststeps = steps2go;
	bestattackerpos = oppos;
	bestattacker_num = opnum;
      }
    }// selfpass is safe
  } // for all dirs
  

  if(max_evaluation > -1000) {
    LOGNEW_POL(NEWBASELEVEL+0,"Precompute best selfpass "
	       <<" targetdir "<<RAD2DEG(bestdir.get_value()));
    scheduled_selfpass.valid_at = WSinfo::ws->time;
    scheduled_selfpass.targetpos = besttargetpos;
    scheduled_selfpass.kickspeed = bestspeed;
    scheduled_selfpass.targetdir = bestdir;
    scheduled_selfpass.steps2go = beststeps;
    scheduled_selfpass.evaluation = max_evaluation;
    scheduled_selfpass.attackerpos = bestattackerpos;
    scheduled_selfpass.attacker_num = bestattacker_num;
  }
  else{
    scheduled_selfpass.valid_at = -1;
    LOGNEW_POL(NEWBASELEVEL+0,"TEST SELFPASSES: no direction found! ");
  }

  return;
}


