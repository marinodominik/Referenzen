#include "jk_wball03_bmp.h"
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

#include "../policy/planning.h"
#include "../policy/policy_tools.h"
#include "../policy/positioning.h"  // get_role()
#include "globaldef.h"
#include "ws_info.h"
//#include "pos_tools.h"
#include "blackboard.h"

#define LOG_INFO2(X)
#define LOG_INFO3(X)

#if 1
//#define DBLOG_POL(LLL,XXX) LOG_POL(LLL,XXX)
#define DBLOG_POL(LLL,XXX) LOG_POL(LLL,<<"JKWball03: " XXX)
#define DBLOG_DRAW(LLL,XXX) LOG_POL(LLL,<<_2D <<XXX)
#define DBLOG_ERR(LLL,XXX) LOG_ERR(LLL,XXX)
#define MYGETTIME (Tools::get_current_ms_time())
#define BASELEVEL 0 // level for logging; should be 3 for quasi non-logging
//#define LOG_DAN(YYY,XXX) LOG_DEB(YYY,XXX)
#define LOG_DAN(YYY,XXX)
#else
#define DBLOG_POL(LLL,XXX)
#define DBLOG_DRAW(LLL,XXX)
#define DBLOG_ERR(LLL,XXX)
#define MYGETTIME (0)
#define BASELEVEL 3 // level for logging; should be 3 for quasi non-logging
#define LOG_DAN(YYY,XXX)
#endif

#define DEBUG 1
#define LOG_LEVEL 0
#define DBLOG_POL_SCORE(LLL,XXX) LOG_DEB(LLL,<<"Score04: "<<XXX)
#define DBLOG_DRAW_SCORE(LLL,XXX) LOG_DEB(LLL,<<_2D<<XXX)


/* constructor method */
JKWball03::JKWball03() {
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

  ValueParser vp(CommandLineOptions::policy_conf,"Wball03_bmp");
  vp.get("at_ball_patience",at_ball_patience);
  vp.get("check_action_mode", check_action_mode);

  neurokick = new NeuroKick;
  dribblestraight = new DribbleStraight;
  selfpass = new Selfpass;
  basiccmd = new BasicCmd;
  onestepkick = new OneStepKick;
  oneortwo = new OneOrTwoStepKick;
  onetwoholdturn = new OneTwoHoldTurn;
  score = new Score;
  neuro_wball = new NeuroWball;

  my_blackboard.pass_or_dribble_intention.reset();
  my_blackboard.intention.reset();
}

JKWball03::~JKWball03() {
  delete neurokick;
  delete dribblestraight;
  delete selfpass;
  delete basiccmd;
  delete onestepkick;
  delete oneortwo;
  delete onetwoholdturn;
  delete score;
  delete neuro_wball;
}

void JKWball03::reset_intention() {
  my_blackboard.pass_or_dribble_intention.reset();
  my_blackboard.intention.reset();
  //ERROR_OUT << "  JKWball03 reset intention, cycle " << WSinfo::ws->time;
  DBLOG_POL(0, << "JKWball03 reset intention");
}

int dowhat;

bool JKWball03::get_cmd(Cmd & cmd,int dow) {
  dowhat = dow;
  return get_cmd(cmd);
}

bool JKWball03::get_cmd(Cmd & cmd) {
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

  LOG_POL(BASELEVEL, << "In JKWball03_BMC : ");
  if(!WSinfo::is_ball_kickable())
    return false;
  switch(WSinfo::ws->play_mode) {
  case PM_PlayOn:
    if(get_intention(intention)){
      intention2cmd(intention,cmd);
      if(cmd.cmd_body.get_type() == Cmd_Body::TYPE_KICK ||
	 cmd.cmd_body.get_type() == Cmd_Body::TYPE_TURN)
	last_waitandsee_at = WSinfo::ws->time;
      //DBLOG_POL(BASELEVEL, << "JKWball03: intention was set! ");
      return true;
    }
    else{
      DBLOG_POL(BASELEVEL, << "JKWball03: WARNING: NO CMD WAS SET");
      return false;
    }
    break;
  default:
    return false;  // behaviour is currently not responsible for that case
  }
  return false;  // behaviour is currently not responsible for that case
}


bool JKWball03::get_intention(Intention &intention){

  long ms_time= MYGETTIME;
  DBLOG_POL(BASELEVEL+2, << "Entering JKWball03");

  if(last_at_ball == WSinfo::ws->time -1)
    at_ball_for_cycles ++;
  else
    at_ball_for_cycles =1;
  last_at_ball = WSinfo::ws->time;

  if(last_waitandsee_at == WSinfo::ws->time -1) // waitandsee selected again
    cycles_in_waitandsee ++;
  else
    cycles_in_waitandsee = 0; // reset

  if(cycles_in_waitandsee >0){
    DBLOG_POL(BASELEVEL,<<"cylces in wait and see: "<<cycles_in_waitandsee);
  }


  my_role= DeltaPositioning::get_role(WSinfo::me->number);
  //my_role=2; // test only !!!!
  // determine closest opponent (frequently needed)
  PlayerSet pset = WSinfo::valid_opponents;
  pset.keep_and_sort_closest_players_to_point(1, WSinfo::me->pos);
  if(pset.num >0)
    closest_opponent = pset[0];
  else
    closest_opponent = NULL;

  //if(onestepkick->can_keep_ball_in_kickrange() == false)
  //  DBLOG_POL(0,<<"JKWball03: CANNOT keep ball in kickrange");

  Intention tmp, tmp_previous;

  tmp_previous = my_blackboard.pass_or_dribble_intention; // copy old intention

  if(my_blackboard.intention.get_type() == OPENING_SEQ){
    my_blackboard.pass_or_dribble_intention.reset(); //
    DBLOG_POL(0,<<"Current intention is opening_seq: invalidated original pass intention");
    return false;
  }
  else if(check_previous_intention(my_blackboard.pass_or_dribble_intention,
				   my_blackboard.pass_or_dribble_intention) == true){
    DBLOG_POL(0,<<"Pass or Dribble Intention was already set, still ok, just take it!");
  }

  if(my_blackboard.pass_or_dribble_intention.valid_at() != WSinfo::ws->time){ // not set yet
    if(get_pass_or_dribble_intention(tmp)){ // check for a new intention.
      if(tmp_previous.valid_at() == WSinfo::ws->time -1){// if previous intention was set in last cycle
	DBLOG_POL(0,"previous pass intention was set, dir "<<RAD2DEG(tmp_previous.kick_target.ARG().get_value()));
	if(tmp_previous.kick_target.ARG().diff(tmp.kick_target.ARG()) < 20./180. *PI){
	  tmp.set_valid_since(WSinfo::ws->time-1);  // indicate
	  DBLOG_POL(0,"current and previous pass direction have not changed too much, current: "
		    <<RAD2DEG(tmp.kick_target.ARG().get_value()));
	}
      } // previous intention was valid last cycle
      my_blackboard.pass_or_dribble_intention = tmp;
    }
    else{
      my_blackboard.pass_or_dribble_intention.reset();
      return false;
      }
  }
  DBLOG_POL(0,<<"Pass or Dribble Intention valid since "<<my_blackboard.pass_or_dribble_intention.valid_since());

  get_best_selfpass();
  is_planned_pass_a_killer = is_pass_a_killer(); // attention: depends on best selfpass!

  bool result = false;

  //if ((dowhat==ONLY_GOALKICK) && (result = score->test_shoot2goal(intention) ));

  if ( (result = check_previous_intention(my_blackboard.intention, intention)) );

  else if ( (result = test_two_teammates_control_ball(intention)));

  else if ( (result = test_in_trouble(intention)));

  else if ( in_penalty_mode == false && (result = test_priority_pass(intention)));

  else if ( (result = test_advance(intention)));

  else if ( (result = test_holdturn(intention)));

  else if ( in_penalty_mode == false && (result = test_pass_or_dribble(intention)));

  else result = test_default(intention); // modified version

  my_blackboard.intention = intention;
  check_write2blackboard();

  ms_time = MYGETTIME - ms_time;
  DBLOG_POL(BASELEVEL+1, << "JKWball03 policy needed " << ms_time << "millis to decide");
  return result;
}

bool JKWball03::selfpass_dir_ok(const ANGLE dir){

  const ANGLE opgoaldir = (Vector(47.,0) - WSinfo::me->pos).ARG(); //go towards goal

#if 0
  if(WSinfo::me->pos.y < -(FIELD_BORDER_Y - 10.) 
     && dir.get_value_mPI_pPI()<0.)
    return false;

  if(WSinfo::me->pos.y > (FIELD_BORDER_Y - 10.) 
     && dir.get_value_mPI_pPI()>0.)
    return false;

#endif

  if(WSinfo::me->pos.getX() < 45 && dir.diff(ANGLE(0))<25/180.*PI )
    return true;
  if (dir.diff(opgoaldir)<20/180.*PI)
    return true;
  return false;
}


void JKWball03::get_best_selfpass(){
#define NUM_DIRS 20

  const ANGLE opgoaldir = (Vector(47.,0) - WSinfo::me->pos).ARG(); //go towards goal
  ANGLE targetdir;
  ANGLE testdir[NUM_DIRS];
  double best_evaluation = -1;
  int num_dirs = 0;
  double speed;
  int steps, op_number;
  Vector ipos, op_pos;

  testdir[num_dirs ++] = ANGLE(0); // go straight
  testdir[num_dirs ++] = ANGLE(45/180.*PI);
  testdir[num_dirs ++] = ANGLE(-45/180.*PI);
  testdir[num_dirs ++] = ANGLE(30/180.*PI); 
  testdir[num_dirs ++] = ANGLE(-30/180.*PI);
  testdir[num_dirs ++] = WSinfo::me->ang;
  testdir[num_dirs ++] = opgoaldir; // close to goal: default: go towards goal


  int mode = 0;  //default: 0: evaluation by position; 1: 1vs1-Situation

  // chck to switch to 1vs1 mode:

  bool goalie_close = false;
  if(WSinfo::his_goalie){
    if(WSinfo::me->pos.distance(WSinfo::his_goalie->pos)<6.0)
      goalie_close = true;
  }

  if((WSinfo::me->pos.getX()  > FIELD_BORDER_X - PENALTY_AREA_LENGTH -5) &&
     (goalie_close == true)){
    Quadrangle2d check_area = Quadrangle2d(WSinfo::me->pos, Vector(52.0,0), 5., 14);
    DBLOG_DRAW(0, check_area );
    PlayerSet pset = WSinfo::valid_opponents;
    pset.keep_players_in(check_area);
    if(pset.num == 0){
      mode = 1;
      DBLOG_POL(0,"Get Best Selfpass:Nobody before me, switch to 1vs1 mode");
    }
    else if(pset.num == 1 && pset[0] == WSinfo::his_goalie){
      mode = 1;
      DBLOG_POL(0,"Get Best Selfpass:Only goalie before me, switch to 1vs1 mode");
    }
  }


  if(mode ==1){
    DBLOG_DRAW(0,C2D(50,30,.4,"red") );
  }

  if(mode ==1){
    testdir[num_dirs ++] = ANGLE(90/180.*PI);
    testdir[num_dirs ++] = ANGLE(-90/180.*PI);
    testdir[num_dirs ++] = ANGLE(120/180.*PI); 
    testdir[num_dirs ++] = ANGLE(-120/180.*PI);
    testdir[num_dirs ++] = ANGLE(179/180.*PI);
    testdir[num_dirs ++] = ANGLE(150/180.*PI); 
    testdir[num_dirs ++] = ANGLE(-150/180.*PI);
  }

  AState state;
  AbstractMDP::copy_mdp2astate(state);

  double V;


  for(int i=0; i<num_dirs; i++){
    targetdir = testdir[i];
    if(mode ==0 && selfpass_dir_ok(targetdir) == false) // in mode 0, check selfpass directions
      continue;
    //    DBLOG_POL(0,<<"selfpass DIRECTION ok : "<<RAD2DEG(targetdir.get_value()));

    if (check_selfpass(targetdir, speed, ipos, steps, op_pos, op_number) == false)
      continue;

    if(mode == 1){
      // 'model'
      state.ball.pos = ipos;
      state.my_team[state.my_idx].pos = ipos;
      //V = Planning::evaluate_byJnn(state,AACTION_TYPE_SELFPASS);
      V = Planning::evaluate_byJnn_1vs1(state);
    }
    else{
      V= 100; // simply avoid that it get <0
      V += ipos.getX() -WSinfo::me->pos.getX(); // relativ x gain

      if(WSinfo::me->pos.getX() > 42){
	if(WSinfo::me->pos.getY() >0)
	  V+=WSinfo::me->pos.getY() - ipos.getY();
	else
	  V+= - WSinfo::me->pos.getY() + ipos.getY();
      }

      if(targetdir.diff(ANGLE(0)) < 3/180.*PI && steps >0 && WSinfo::me->pos.getX() < 40.){
	//	V += 8;
	V += 2;
	DBLOG_POL(2,<<"selfpass in dir 0 is preferred: add +8");
      }
    }  // end mode 0

#if 0
    DBLOG_POL(0,<<"selfpass ok to "<<ipos<<" dir "<<RAD2DEG(targetdir.get_value())<<" w.speed "
	      <<speed<<" steps "<<steps<<" evaluation "<<V);    
#endif
    //    if(WSinfo::me->pos.distance(ipos) > best_evaluation){
    if(V > best_evaluation){
      //      best_evaluation = WSinfo::me->pos.distance(ipos);
      best_evaluation = V;
      best_selfpass.steps= steps;
      best_selfpass.ipos= ipos;
      best_selfpass.speed = speed;
      best_selfpass.targetdir = targetdir;
      best_selfpass.op_pos = op_pos;
      if(op_number >0)
	best_selfpass.op_number = op_number;
      else
	best_selfpass.op_number = -1;
    } 
  } // for all targetdirs
  if(best_evaluation >0){
    best_selfpass.valid_at = WSinfo::ws->time;
    DBLOG_POL(0,<<"best selfpass found to "<<best_selfpass.ipos
	      <<" dir "<<RAD2DEG(best_selfpass.targetdir.get_value())
	      <<" w.speed "<<best_selfpass.speed<<" steps "
	      <<best_selfpass.steps<<" evaluation: "<<best_evaluation);    
  }
  else
    best_selfpass.valid_at = -1;
}



void JKWball03::set_neck_selfpass(const ANGLE targetdir, const Vector &op_pos){
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
      DBLOG_DRAW(0, VC2D(op_pos, 1.3, "orange"));
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

bool JKWball03::check_selfpass(const ANGLE targetdir, double &ballspeed, Vector &target, int &steps,
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

  #ifdef  DEBUG
  DBLOG_POL_SCORE(LOG_LEVEL,"check_selfpass: starts");
  #endif

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
    DBLOG_POL(0,"Selfpass targetdir "<<RAD2DEG(targetdir.get_value())<<" I'm behind offside line;reducing max_dashes to "<<max_dashes);
  }

  #ifdef  DEBUG
  DBLOG_POL_SCORE(LOG_LEVEL,"check_selfpass: continues 1");
  #endif

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
  #ifdef  DEBUG
  if(result) {
    DBLOG_POL_SCORE(LOG_LEVEL,"check_selfpass: selfpass is safe");
  }
  if(!result) {
    DBLOG_POL_SCORE(LOG_LEVEL,"check_selfpass: selfpass is NOT safe");
  }
  #endif

  pset= WSinfo::valid_teammates_without_me;
  pset.keep_players_in_circle(target,2);
  if (pset.num >0){
    if (result == true) {
      DBLOG_POL_SCORE(LOG_LEVEL,"Selfpass ok, but target too close to teammate");
    }
    return false;
  }

  if(DeltaPositioning::get_role(WSinfo::me->number) == 0){
    DBLOG_POL_SCORE(LOG_LEVEL,"I am a defender: "<<DeltaPositioning::get_role(WSinfo::me->number)<<" "<<my_role);
    }

  if (result == false)
    return false;
  return false;
}


bool JKWball03::check_previous_intention(Intention prev_intention, Intention  &new_intention){
  if(prev_intention.valid_at() < WSinfo::ws->time -1)
    return false;
  if(is_planned_pass_a_killer){
    if(prev_intention.get_type() != PASS &&
       prev_intention.get_type() != LAUFPASS){
      DBLOG_POL(0,<<"Check previous intention: planned pass is a killer -> RESET intention");
      return false;
    }
  }
  // ridi04: check if I am attacked
  if(am_I_attacked() || is_my_passway_in_danger()){
    if(prev_intention.get_type() != PASS &&
       prev_intention.get_type() != LAUFPASS){
      DBLOG_POL(0,<<"Check previous intention: I am attacked or my passway is in danger!");
      return false;
    }
  }


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
  int steps; // return parameter for selfpass, not used
  Vector op_pos; // return parameter for selfpass, not used
  bool stillok = false;
  double ballspeed;
  int op_number;
  int risky_pass = prev_intention.risky_pass;

  switch(prev_intention.get_type()){
  case PASS:
    receiverpos = target; // default setting
    if(targetplayer_number >0){
      DBLOG_POL(BASELEVEL,
		<<"JKWball03: Check previous intention: PASS: correct target, old: "<<target);
      PPlayer p= WSinfo::get_teammate_by_number(targetplayer_number);
      if(p){
	receiverpos = WSinfo::get_teammate_by_number(targetplayer_number)->pos;
	if(receiverpos.getX() > WSinfo::his_team_pos_of_offside_line()){
	  DBLOG_POL(BASELEVEL,
		    <<"JKWball03: Check Pass considered harmful, offside of receiver -> RESET");
	  stillok = false;
	  break;
	}
	dir = (receiverpos - WSinfo::me->pos).arg(); // direction of the target
      }
      stillok = Planning::is_pass_successful_hetero(ballpos, speed, dir, p, interceptpos, advantage); // use player info
    }
    else{ // player number not known
      stillok = Planning::is_pass_successful(ballpos, speed, dir, receiverpos, interceptpos, advantage); //
    }
    if(stillok == false){
      DBLOG_POL(BASELEVEL,
		<<"JKWball03: Check previous intention: Pass considered harmful -> RESET");
    }
    else{
      DBLOG_POL(BASELEVEL,
		<<"JKWball03: Pass considered ok, speed "<<speed<<" dir "<<RAD2DEG(dir));
    }
    target = receiverpos;
    break;
  case LAUFPASS:
    stillok = Planning::is_laufpass_successful(ballpos,speed,
					       dir,interceptpos,advantage,
					       number,playerpos,risky_pass);
    if(playerpos.getX() > WSinfo::his_team_pos_of_offside_line()){
      DBLOG_POL(BASELEVEL,
		<<"JKWball03: Check Laufpass considered harmful, offside of receiver -> RESET");
      stillok = false;
      break;
    }

    if (advantage < 2)
      stillok = false;
    if(stillok == false) {
      DBLOG_POL(BASELEVEL,
		<<"JKWball03: Check previous intention: Pass considered harmful -> RESET");
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
  #ifdef  DEBUG
  DBLOG_POL_SCORE(LOG_LEVEL,"checking selfpass intention validity");
  #endif
    if (check_selfpass(targetdir, ballspeed, target, steps, op_pos, op_number) == true){
      if(steps > 2){ // I can go more than 2 steps -> reset pass intention
  #ifdef  DEBUG
  DBLOG_POL_SCORE(LOG_LEVEL,"checking selfpass: can go more than two steps -> reset pass intention");
  #endif
	DBLOG_POL(0,"Check Previous Intention Selfpass: can go more than two steps -> reset pass intention");
	//my_blackboard.pass_or_dribble_intention.reset();
      }
      speed = ballspeed; // probably correct speed
      set_neck_selfpass(targetdir, op_pos);  // sets internal blackboard neck intention
      DBLOG_DRAW(0, VC2D(op_pos, 1.5, "00ffff"));
      stillok = true;
  #ifdef  DEBUG
  DBLOG_POL_SCORE(LOG_LEVEL,"Selfpass Intention still ok, just take it!");
  #endif
      DBLOG_POL(BASELEVEL,
		<<"JKWball03: Reconsider Selfpass in dir "<<RAD2DEG(targetdir.get_value())<<" OK");
    }
    else
      stillok = false;
    break;
  case IMMEDIATE_SELFPASS:
    stillok = false;
    DBLOG_POL(BASELEVEL,
	      <<"JKWball03: Check previous intention: Selfpass ALWAYS reconsidered harmful -> RESET");
    break;
  case SCORE:
    stillok = true;
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
    DBLOG_DRAW(BASELEVEL, VL2D(WSinfo::ball->pos,target,"lightblue"));
    return true;
  }
  return false;
}


bool JKWball03::get_pass_or_dribble_intention(Intention &intention){
  AState current_state;
  AbstractMDP::copy_mdp2astate(current_state);

  return get_pass_or_dribble_intention(intention, current_state);
}

bool JKWball03::get_pass_or_dribble_intention(Intention &intention, AState &state){
  AAction best_aaction;

  intention.reset();

  if (neuro_wball->evaluate_passes_and_dribblings(best_aaction, state) == false) // nothing found
    return false;

  if(best_aaction.action_type == AACTION_TYPE_WAITANDSEE)
    last_waitandsee_at = WSinfo::ws->time;
  
  aaction2intention(best_aaction, intention);

  // found a pass or dribbling
  return true;
}


bool JKWball03::test_opening_seq(Intention &intention)
{
  if(onetwoholdturn->is_holdturn_safe() == false){
    DBLOG_POL(BASELEVEL, <<"WBALL 03: Opening not possible, since holdturn not safe");
    return false;
  }

  if (my_blackboard.pass_or_dribble_intention.valid_at() == WSinfo::ws->time){ // pass or dribble int. is set
    if(my_blackboard.pass_or_dribble_intention.kick_target.getX()>WSinfo::me->pos.getX()){
      DBLOG_POL(BASELEVEL+0,<<"JKWball03 No Kickrush, Pass is planned, and kick target before me!"
		<<my_blackboard.pass_or_dribble_intention.kick_target);
      return false;
    }
  }

#if 0
  if(WSinfo::me->pos.x <-10){
    DBLOG_POL(BASELEVEL+0,<<"JKWball03 No Opening, too far back");
    return false;
  }
#endif
  if(WSinfo::me->pos.getX() >ServerOptions::pitch_length - 20){
    DBLOG_POL(BASELEVEL+0,<<"JKWball03 No Opening, too far advanced");
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
    DBLOG_POL(BASELEVEL+0,<<"JKWball03 Opening sequence NOT possible, best ipos "<<ipos
	      <<" behind offsideline "<<WSinfo::his_team_pos_of_offside_line());
    return false;
  }
    
  target.init_polar(speed/(1-ServerOptions::ball_decay),dir);
  target += WSinfo::me->pos;
  
  //  speed = 3.0; // artifical speed to communicate opening seq.
  intention.set_opening_seq(target,speed, WSinfo::ws->time);
  DBLOG_POL(BASELEVEL+0,<<"JKWball03  Start Opening sequence to "<<target<<" dir "
	    <<RAD2DEG(dir)<<" w.speed "<<speed
	    <<" best ipos "<<ipos
	    <<" before offsideline "<<WSinfo::his_team_pos_of_offside_line());
  //DBLOG_POL(0, << _2D << C2D(target.x, target.y, 1, "#00ffff"));
  return true;
}





/** default move */
bool JKWball03::test_default(Intention &intention)
{

  if(onestepkick->can_keep_ball_in_kickrange() == false){
    PlayerSet pset= WSinfo::valid_opponents;
    Vector endofregion;
    endofregion.init_polar(5, WSinfo::ball->vel.ARG());
    endofregion += WSinfo::me->pos;
    Quadrangle2d check_area = Quadrangle2d(WSinfo::me->pos, endofregion, 3);
    DBLOG_DRAW(0, check_area );
    pset.keep_players_in(check_area);
    if(pset.num == 0){
      intention.set_holdturn(ANGLE(0.), WSinfo::ws->time); // this will maximally stop the ball
      return true;
    }
  }

  Vector opgoalpos = Vector (52.,0); // opponent goalpos

  if(cycles_in_waitandsee < at_ball_patience && onetwoholdturn->is_holdturn_safe() == true){
    ANGLE target_dir = (opgoalpos - WSinfo::me->pos).ARG();
    DBLOG_POL(BASELEVEL+0,<<"JKWball03 DEFAULT Move - Hold Turn is Safe");
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
    DBLOG_POL(BASELEVEL+0,<<"JKWball03 DEFAULT Clearance to "<<target<<" dir "<<RAD2DEG(dir)<<" w.speed "<<speed);
    return true;
  }

  DBLOG_POL(BASELEVEL+0,<<"JKWball03 DEFAULT Move - Hold Turn is NOT Safe, but no other alternative, TRY");
  intention.set_holdturn((opgoalpos - WSinfo::me->pos).ARG(), WSinfo::ws->time);
  return true;
}

/** selects one player who is responsible for the ball */
bool JKWball03::test_two_teammates_control_ball(Intention &intention){
  // test if Ball is in kickrange of me and a teammate!

  Vector ballpos = WSinfo::ball->pos;
  PPlayer p_tmp = WSinfo::valid_teammates_without_me.closest_player_to_point(ballpos);
  if (p_tmp == NULL) return false;
  int teammate = p_tmp->number;
  float dist = (p_tmp->pos - ballpos).norm();

  if((dist < .95*ServerOptions::catchable_area_l) && (teammate == WSinfo::ws->my_goalie_number)){
    LOG_ERR(0,<<"JKWball03: Our goalie has the ball, I could kick - Maybe I should back up?!");
    DBLOG_POL(BASELEVEL,<<"JKWball03: Our goalie has the ball, I could kick - Maybe I should back up?! (not yet implemented)");
  }

  if((dist < ServerOptions::kickable_area) && (WSinfo::me->number > teammate)){
    LOG_ERR(0,<<"JKWball03: Another teammate with lower number  has the ball, I could kick - Maybe I should back up?!");
    DBLOG_POL(BASELEVEL,<<"JKWball03: I could kick Another teammate with lower number  has the ball - Maybe I should back up?! (not yet implemented)");
  }

  intention.set_backup(WSinfo::ws->time);
  return false;
}

bool JKWball03::I_can_advance_behind_offside_line(){
  if(best_selfpass.valid_at != WSinfo::ws->time)
    return false;
  if (best_selfpass.ipos.getX() +1. >= WSinfo::his_team_pos_of_offside_line()||
      best_selfpass.op_number == WSinfo::ws->his_goalie_number){
    DBLOG_POL(0,<<"Test: I can advance_behind_offside_line or am only stopped by goalie!");
    return true;
  }
  return false;
}

bool JKWball03::is_pass_a_killer(){
  if(my_blackboard.pass_or_dribble_intention.valid_at() != WSinfo::ws->time)
    return false;

  bool result = false;

  // possibility 1: behind offside line
  Vector respos = my_blackboard.pass_or_dribble_intention.resultingpos;
  if(I_can_advance_behind_offside_line() == false &&
     respos.getX() > WSinfo::his_team_pos_of_offside_line()){
    DBLOG_POL(0,<<"Planned pass is a killer -> resulting pos is behind offside line and I can not advance behind offsideline");
    result = true;
  }

  PlayerSet pset;

  // possibility 2: teammate can advance
  if(WSinfo::me->pos.getX() > 30. &&
     fabs(WSinfo::me->pos.getY()) > fabs(respos.getY())){
    pset = WSinfo::valid_opponents;
    Vector endofregion;
    endofregion.init_polar(15, 0);
    endofregion += respos;
    Quadrangle2d check_area = Quadrangle2d(respos, endofregion, 5.);
    DBLOG_DRAW(0, check_area );
    pset.keep_players_in(check_area);
    if(pset.num == 0 &&
       respos.getX() >= WSinfo::ball->pos.getX() -2){
      DBLOG_POL(0,<<"Planned pass is a killer -> from resulting pos teammate can advance ");
      result = true;;
    }
  }

  // possibility 3: in scoring area

  if(fabs(my_blackboard.pass_or_dribble_intention.resultingpos.getY())<13 &&
     my_blackboard.pass_or_dribble_intention.resultingpos.getX()>35){ // in hot scoring area
    DBLOG_POL(BASELEVEL, <<"Teammate in hot scoring area -> pass");
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

bool JKWball03::test_priority_pass(Intention &intention){

  DBLOG_POL(0,<<"enter CHECK priority pass 1. planned pass is a killer: "<<is_planned_pass_a_killer);

  // first, check for killer passes
  if(is_planned_pass_a_killer){ // planned pass is a killer
    Vector respos = my_blackboard.pass_or_dribble_intention.resultingpos;
    if(fabs(respos.getY())<8 &&
       respos.getX()>42){ // in hot scoring area
      DBLOG_POL(BASELEVEL, <<"Teammate in hot scoring area -> pass IMMEDIATELY");
      Vector target;
      double speed;
      my_blackboard.pass_or_dribble_intention.get_kick_info(speed, target);
      ANGLE targetdir = (target-WSinfo::me->pos).ARG();
      my_blackboard.neckreq.set_request(NECK_REQ_LOOKINDIRECTION,targetdir.get_value());
      intention = my_blackboard.pass_or_dribble_intention;
      return true;
    } 

    if(my_blackboard.pass_or_dribble_intention.valid_since() < WSinfo::ws->time){ // it has been already communicated
      // this might be replaced by == -> play killer passes immediately, if I have looked in the direction
      Vector target;
      double speed;
      my_blackboard.pass_or_dribble_intention.get_kick_info(speed, target);
      ANGLE targetdir = (target-WSinfo::me->pos).ARG();
      if(WSmemory::last_seen_in_dir(targetdir) >1){ 
	my_blackboard.neckreq.set_request(NECK_REQ_LOOKINDIRECTION,targetdir.get_value());
	DBLOG_POL(0,<<"Planned pass is a killer, but I haven't seen in targetdir "<<RAD2DEG(targetdir.get_value())<<" since "
		  <<WSmemory::last_seen_in_dir(targetdir)<<" cycles -> wait and look ");
	if(onetwoholdturn->is_holdturn_safe() == false){
	  DBLOG_POL(BASELEVEL, <<"Priority Pass: HoldTurn NOT possible. Not Safe");
	  return false; // alternatively: immediately play killer pass then
	}
	intention.set_holdturn(ANGLE(0.), WSinfo::ws->time);
	return true;
      }
      DBLOG_POL(BASELEVEL, <<"pass intention is a KILLER and I looked into direction -> Priority pass");
      intention = my_blackboard.pass_or_dribble_intention;
      return true;
    } // end pass intention was valid for at least 1 cycle
    else{ // pass intention was just set -> wait
      DBLOG_POL(0,<<"Planned pass is a killer, wait one cycle");
      if(onetwoholdturn->is_holdturn_safe() == false){
	DBLOG_POL(BASELEVEL, <<"Priority Pass: HoldTurn NOT possible. Not Safe");
	return false; // alternatively: immediately play killer pass then
      }
      intention.set_holdturn(ANGLE(0.), WSinfo::ws->time);
      return true;
    }
  }


  // now check passing due to other situations
  const double scoring_area_width = 20.;
  const double scoring_area_depth = 15.;


  const XYRectangle2d scoring_area( Vector(FIELD_BORDER_X - scoring_area_depth, -scoring_area_width*0.5),
				    Vector(FIELD_BORDER_X, scoring_area_width * 0.5)); 

  DBLOG_DRAW(0,scoring_area);
  if(scoring_area.inside(WSinfo::me->pos)){
    DBLOG_POL(0,"I am within scoring area, do not do prioriy pass");
    return false;
  }

  // I am not in scoring area

  bool I_am_attacked= am_I_attacked();
  I_am_attacked =(I_am_attacked || is_my_passway_in_danger());

  if(I_am_attacked == false){
    DBLOG_POL(BASELEVEL, <<"Check prority pass: I am NOT attacked");
  }
  else{
    DBLOG_POL(BASELEVEL, <<"Check priority pass: I AM attacked or my passway is in danger");
  }

  PlayerSet pset= WSinfo::valid_opponents;
  Vector endofregion;
  endofregion.init_polar(5, 0);
  endofregion += WSinfo::me->pos;
  Quadrangle2d check_area = Quadrangle2d(WSinfo::me->pos, endofregion, 5);
  //  DBLOG_DRAW(1, check_area );
  pset.keep_players_in(check_area);
  bool area_infront_is_free = (pset.num == 0);
  
  int num_steps2pass = howmany_kicks2pass();
  DBLOG_POL(BASELEVEL, <<"Check priority pass: steps2pass: "<<num_steps2pass);
  

  if((I_am_attacked) && // opponent is close and pass intention communicated
     ((my_blackboard.pass_or_dribble_intention.valid_since() < WSinfo::ws->time) ||
      num_steps2pass >1)){ 

    if(DeltaPositioning::get_role(WSinfo::me->number) == 0 ||
       my_blackboard.pass_or_dribble_intention.kick_target.getX() > WSinfo::me->pos.getX() ||
       area_infront_is_free == false){
      DBLOG_POL(BASELEVEL, <<"I am attacked, pass intention is set -> Priority pass");
      intention = my_blackboard.pass_or_dribble_intention;
      return true;
    }
  }

  return false;

}



bool JKWball03::is_my_passway_in_danger(){
  if (my_blackboard.pass_or_dribble_intention.valid_at()  != WSinfo::ws->time){
    DBLOG_POL(BASELEVEL, <<"check passing corridor: pass intention currently not valid !");
    return false;
  }

  float width = 10.0;
  if(DeltaPositioning::get_role(WSinfo::me->number) == 0)
    width = 10.0;


  PlayerSet pset = WSinfo::valid_opponents;
  Vector respos = my_blackboard.pass_or_dribble_intention.resultingpos;
  Quadrangle2d check_area = Quadrangle2d(WSinfo::me->pos, respos,width);
  DBLOG_DRAW(0,check_area);
  pset.keep_players_in(check_area);
  if(pset.num>0){
     DBLOG_POL(BASELEVEL, <<"Someone enters passing corridor!");
    return true;
  }
  return false;
}

bool JKWball03::am_I_attacked(){
  PlayerSet pset= WSinfo::valid_opponents;
  double radius_of_attacked_circle =  2 * ServerOptions::kickable_area + 2 * ServerOptions::player_speed_max;
  DBLOG_DRAW(0,VC2D(WSinfo::me->pos,radius_of_attacked_circle,"red"));
  DBLOG_DRAW(0,VC2D(WSinfo::me->pos,radius_of_attacked_circle-+.5,"black"));
  pset.keep_players_in_circle(WSinfo::me->pos,radius_of_attacked_circle); 
  return  (pset.num >0);

}

bool JKWball03::test_holdturn(Intention &intention){
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

/** What to do if two opponents attack me */
bool JKWball03::test_in_trouble(Intention &intention){
  // test for situation 'a': Ball is in kickrange of me and opponent!

  Vector target;


  PlayerSet pset= WSinfo::valid_opponents;
  pset.keep_players_in_circle(WSinfo::ball->pos, ServerOptions::kickable_area); 
  // considr only close ops. for correct
  pset.keep_and_sort_closest_players_to_point(1,WSinfo::ball->pos);
  if ( pset.num == 0 )
    return false; // no op. has ball in kickrange



  DBLOG_POL(BASELEVEL, << "JKWball03: Ball is in kickrange of me and opponent "<<pset[0]->number);
  if(WSinfo::me->pos.getX() >-10){
    DBLOG_POL(BASELEVEL, << "JKWball03: I dont care for trouble. I'm too far advanced, just cont.");
    return false;
  }

  // Danger: Opponent can also kick -> kick ball straight away
  if(WSinfo::me->pos.getX() <10){ // I am far from opponents goal
    target = Vector(52.5,0);
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
  DBLOG_POL(BASELEVEL, << "JKWball03: panic: Kick Ball with maxpowert to target "<<target);
  intention.set_panic_kick(target, WSinfo::ws->time);
  return true;
}


bool JKWball03::get_best_panic_selfpass(const double testdir[],const int num_dirs,double &speed, double &dir){
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

int JKWball03::howmany_kicks2pass(){
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
    LOG_POL(0,"how many steps to pass to "<<target<<" speed "<<speed<<" : speed1 "<<speed1<<" speed2: "<<speed2);
    if(fabs(speed1 -speed)<.1){
      return 1;
    }
    else{
      return 2;
    }	
  }
  return 0;
}



void JKWball03::get_onestepkick_params(double &speed, double &dir){
  double tmp_speed;
  Vector final;
  Vector ballpos = WSinfo::ball->pos;
  const int max_targets = 360;
  double testdir[max_targets];
  double testspeed[max_targets];

  int num_dirs = 0;

  //  for(ANGLE angle=ANGLE(0);angle.get_value()<2*PI;angle+=ANGLE(5./180.*PI)){
  for(float ang=0.;ang<PI;ang+=5./180.*PI){
    for(int sign = -1; sign <= 1; sign +=2){
      ANGLE angle=ANGLE((float)(sign * ang));
      tmp_speed = onestepkick->get_max_vel_in_dir(angle);
      if(tmp_speed <0.1){
	final.init_polar(1.0, angle);
	final += ballpos;
	DBLOG_DRAW(0,VL2D(ballpos, final, "000000"));
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
	DBLOG_DRAW(0,VL2D(ballpos, final, "aaaaaa"));
      }
    }
  }


  int advantage;
  Vector ipos;
  int closest_teammate;
  Policy_Tools::get_best_kicknrush(WSinfo::me->pos,num_dirs,testdir,
				   testspeed,speed,dir,ipos,advantage, closest_teammate);
  if(ipos.getX() < WSinfo::me->pos.getX() -20){
    speed = 0;
    return;
  }

  if(WSinfo::me->pos.getX() >0){
    if(advantage<1 || closest_teammate == WSinfo::me->number || in_penalty_mode == true){
      DBLOG_POL(0,<<"JKWball03: Check Panic Selfpasses ");
      if(get_best_panic_selfpass(testdir,num_dirs,speed,dir)) {
	DBLOG_POL(0,<<"JKWball03: found a panic selfpass speed "<<speed<<" dir "<<RAD2DEG(dir));
      }
    }
  }

  if(speed >0){
    DBLOG_POL(0,<<"JKWball03: found onestepkick with advantage "<<advantage<<" resulting pos : "
	      <<ipos<<" closest teammate "<<closest_teammate);
    DBLOG_DRAW(0,VC2D(ipos,1.0,"00FFFF"));
    DBLOG_DRAW(0,VL2D(WSinfo::ball->pos,
		     ipos,"00FFFF"));
  }
  else{
    DBLOG_POL(0,<<"JKWball03: NO onestepkick found ");
  }
}

void JKWball03::get_clearance_params(double &speed, double &dir){
  int advantage, closest_teammate;
  Vector ipos;

  return get_opening_pass_params(speed,dir, ipos, advantage, closest_teammate);
}


void JKWball03::get_opening_pass_params(double &speed, double &dir, Vector &ipos, int &advantage,
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
    DBLOG_DRAW(0,VC2D(ipos,1.0,"00FFFF"));
    DBLOG_DRAW(0,VL2D(WSinfo::ball->pos,
		     ipos,"00FFFF"));
  }
  else{
    DBLOG_POL(0,<<"JKWball03: NO kicknrush found ");
  }


}


bool JKWball03::is_dribblestraight_possible(){
  ANGLE opgoaldir = (Vector(52,0) - WSinfo::me->pos).ARG(); //go towards goal


  if(DeltaPositioning::get_role(WSinfo::me->number) == 0){// I'm a defender:
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

  if((WSinfo::me->ang.diff(ANGLE(0))>20/180.*PI || WSinfo::me->pos.getX() > 45) &&
     (WSinfo::me->ang.diff(opgoaldir)>20/180.*PI)){
    return false;
  }

  DBLOG_POL(1,<<"me->ang "<<RAD2DEG(WSinfo::me->ang.get_value())<<" OK for Dribble");


  if(dribblestraight->is_dribble_safe(0)){
    if(my_blackboard.pass_or_dribble_intention.valid_since() < WSinfo::ws->time){
      // check area before me
      PlayerSet pset = WSinfo::valid_opponents;
      Vector endofregion;
      endofregion.init_polar(4, WSinfo::me->ang);
      endofregion += WSinfo::me->pos;
      Quadrangle2d check_area = Quadrangle2d(WSinfo::me->pos, endofregion, 2);
      DBLOG_DRAW(0, check_area );
      pset.keep_players_in(check_area);
      if(pset.num >0){
	DBLOG_POL(0,<<"Player directly before me and pass intention is set -> do not dribble");
	return false; // go on in any case
      }
    }
    return true;
  }
  return false;
}


bool JKWball03::test_advance(Intention &intention){
  // a solo has preference before any other pass or holdturn
  //  double speed;
  //  Vector ipos;
  //int steps;
  //Vector op_pos; // position of attacking opponent -> view strategy


  // first, check if area too crowded and dribble is prefered

  bool is_dribble_ok = is_dribblestraight_possible();

  if(is_dribble_ok && WSinfo::me->pos.getX() + 3< WSinfo::his_team_pos_of_offside_line()){
    PlayerSet pset = WSinfo::valid_opponents;
    Vector endofregion;
    const double scanrange = 10;
    endofregion.init_polar(scanrange, WSinfo::me->ang);
    endofregion += WSinfo::me->pos;
    Quadrangle2d check_area = Quadrangle2d(WSinfo::me->pos, endofregion,scanrange/2., scanrange);
    //    DBLOG_DRAW(0, check_area );
    pset.keep_players_in(check_area);
    if(pset.num > 0){
      if(pset.num > 1 || pset[0] != WSinfo::his_goalie){
	DBLOG_POL(0,<<"Test advance: Area before me is crowded -> rather dribble ");
	Vector target;
	target.init_polar(2.0,WSinfo::me->ang);
	intention.set_dribble(target, WSinfo::ws->time);
	if(closest_opponent)
	  set_neck_selfpass(WSinfo::me->ang, closest_opponent->pos);
	else
	  set_neck_selfpass(WSinfo::me->ang, Vector(52.0,0) );

	DBLOG_DRAW(0,VL2D(WSinfo::me->pos, target, "green"));
	return true;
      }
    }
  }

  if (best_selfpass.valid_at == WSinfo::ws->time){
    intention.set_selfpass(best_selfpass.targetdir, best_selfpass.ipos,best_selfpass.speed, WSinfo::ws->time);
    DBLOG_POL(0,<<"selfpass possible to "<<best_selfpass.ipos<<" w.speed "
	      <<best_selfpass.speed<<" steps "<<best_selfpass.steps);    
    DBLOG_DRAW(0, VC2D(best_selfpass.op_pos, 1.5, "00ffff"));
    if(best_selfpass.steps > 2){ // I can go more than 2 steps -> reset pass intention
      DBLOG_POL(1,"Advance w. Selfpass: can go more than two steps -> reset pass intention");
      my_blackboard.pass_or_dribble_intention.reset();
    } 
    DBLOG_POL(0,"Advance w. Selfpass");
    set_neck_selfpass(best_selfpass.targetdir, best_selfpass.op_pos);
    return true;
  }


  if(is_dribble_ok){
    DBLOG_POL(0,<<"Test advance: No chance for selfpasses, but dribble is possible ");
    Vector target;
    target.init_polar(2.0,WSinfo::me->ang);
    intention.set_dribble(target, WSinfo::ws->time);
    DBLOG_DRAW(0,
	       VL2D(WSinfo::me->pos, target,
		   "orange"));
    return true;
  }

  return false;
}


bool JKWball03::test_pass_or_dribble(Intention &intention){
  if(my_blackboard.pass_or_dribble_intention.valid_at() == WSinfo::ws->time){
    intention = my_blackboard.pass_or_dribble_intention;
    return true;
  }
  return false;
}


void JKWball03::check_write2blackboard(){
  int main_type = my_blackboard.intention.get_type();
  bool main_type_is_pass = false;

  if (main_type == PASS || main_type == LAUFPASS ||
      main_type == KICKNRUSH) 
    main_type_is_pass = true;
  
  // check for turn neck requests first
  if (my_blackboard.neckreq.is_set()){
    DBLOG_POL(0,<<"Write 2 Blackboard: Neck Request is SET to dir :"<<RAD2DEG(my_blackboard.neckreq.get_param()));
    
    Vector res;
    res.init_polar(50, my_blackboard.neckreq.get_param());
    DBLOG_DRAW(0, VL2D(WSinfo::me->pos,
		      res,
		      "red"));

    Blackboard::set_neck_request(my_blackboard.neckreq.get_type(),my_blackboard.neckreq.get_param());
  }
  
  // check communication request for main type
  if(main_type == OPENING_SEQ || main_type == KICKNRUSH){
    Vector target;
    double speed;
    my_blackboard.intention.get_kick_info(speed, target);
    Blackboard::pass_intention = my_blackboard.intention;
    DBLOG_POL(1,<<"JKWball03: Check write2blackboard: intention opening seq / kicknrush Set blackboard intention: valid at "
	      << Blackboard::pass_intention.valid_at());
    return;
  }

  if (my_blackboard.pass_or_dribble_intention.valid_at() != WSinfo::ws->time){
    DBLOG_POL(1,<<"JKWball03: Check write2blackboard: no pass or dribble intention is set");
    return;
  }

  // now, check for turn neck request
  int pass_type = my_blackboard.pass_or_dribble_intention.get_type();
  if(pass_type == PASS || pass_type == LAUFPASS || pass_type == KICKNRUSH){
    // turn neck to pass receiver
    double dummy;
    if(Blackboard::get_neck_request(dummy) == NECK_REQ_NONE){
      Vector target;
      double speed;
      my_blackboard.pass_or_dribble_intention.get_kick_info(speed, target);
      ANGLE ball2targetdir = (target - WSinfo::ball->pos).ARG(); // direction of the target
      if(WSmemory::last_seen_in_dir(ball2targetdir) >0 || main_type_is_pass == true){
	DBLOG_POL(1,<<"Neck intn. not set.  and haven't looked in pass dir. look in dir of pass "<<RAD2DEG(ball2targetdir.get_value()));
	Blackboard::set_neck_request(NECK_REQ_LOOKINDIRECTION, ball2targetdir.get_value());
      }
    }
    //    mdpInfo::set_my_neck_intention(NECK_INTENTION_LOOKINDIRECTION, ball2targetdir);
  }


  PlayerSet pset= WSinfo::valid_opponents;
  pset.keep_players_in_circle(WSinfo::me->pos, 2* ServerOptions::kickable_area + 2* ServerOptions::player_speed_max); 

  pset = WSinfo::valid_opponents;
  Vector endofregion;
  const double scanrange = 7;
  endofregion.init_polar(scanrange,(my_blackboard.intention.kick_target -  WSinfo::me->pos).ARG());
  endofregion += WSinfo::me->pos;
  Quadrangle2d check_area = Quadrangle2d(WSinfo::me->pos, endofregion, 5., scanrange);
  DBLOG_DRAW(0, check_area );
  pset.keep_players_in(check_area);

  // if opponent could get in my kickrange, then keep pass or dribble intention
  /*if((main_type == SELFPASS || main_type == DRIBBLE) && // I'm dribbling
     is_planned_pass_a_killer == false && // do not reset killer passes
     //(opps_around_me == true || opps_in_targetdir == true)){ // I have enough space around me
     (opps_around_me == false && opps_in_targetdir == false)){ // I have enough space around me
    DBLOG_POL(0,<<"I do selfpass or dribble and I'm not attacked -> do not communicate AND reset pass intent.");
    my_blackboard.pass_or_dribble_intention.reset();
    return;
  }*/


  // now, check for communication request

  if(pass_type == PASS || pass_type == LAUFPASS || pass_type == KICKNRUSH){
    DBLOG_POL(1,<<"JKWball03: Check write2blackboard: pass intention is set");
    // for now, communication is done indirectly; should be improved
    Blackboard::pass_intention = my_blackboard.pass_or_dribble_intention;
  }
}




void JKWball03::aaction2intention(const AAction &aaction, Intention &intention){

  double speed = aaction.kick_velocity;
  Vector target = aaction.target_position;
  double kickdir = aaction.kick_dir;
  int selfpass_steps = aaction.advantage;

  Vector op_pos = aaction.actual_resulting_position;

  switch(aaction.action_type){
  case  AACTION_TYPE_PASS:
    intention.set_pass(target,speed, WSinfo::ws->time, aaction.targetplayer_number, 0,
		       aaction.actual_resulting_position);
    break;
  case  AACTION_TYPE_LAUFPASS:
    intention.set_laufpass(target,speed, WSinfo::ws->time, aaction.targetplayer_number, 0,
			   aaction.actual_resulting_position, aaction.risky_pass);
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
    DBLOG_POL(0,<<"JKWball03 aaction2intention: AActionType not known");
    LOG_ERR(0,<<"JKWball03 aaction2intention: AActionType not known");
  }
}

bool JKWball03::get_turn_and_dash(Cmd &cmd){
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

bool JKWball03::get_opening_seq_cmd( const float  speed, const Vector target,Cmd &cmd){

  Vector ipos;
  bool myteam;
  int number;
  int advantage;
  Angle dir = (target-WSinfo::ball->pos).arg();
  DBLOG_POL(BASELEVEL+0,<<"Get Command for Opening sequence to "<<target<<" dir "
	    <<RAD2DEG(dir)<<" w.speed "<<speed);
  DBLOG_DRAW(0, VC2D(target, 1, "#00ffff"));

  Policy_Tools::earliest_intercept_pos(WSinfo::me->pos,speed,dir,
				       ipos, myteam, number, advantage);


  if((advantage >0) && (my_blackboard.intention.valid_since() < WSinfo::ws->time)){
    // pass possible and I decided it at least one cycle ago -> already communicated 
    my_blackboard.intention.set_kicknrush(target,speed, WSinfo::ws->time); // directly set new intent.
    DBLOG_POL(BASELEVEL+0,<<"JKWball03 Opening sequence DO LAUFPASS, advantage ok "<<advantage
	      <<target<<" dir "
	      <<RAD2DEG(dir)<<" w.speed "<<speed<<" receiver "<<number);
    DBLOG_DRAW(0, VC2D(target, 1, "#00ffff"));
    neurokick->kick_to_pos_with_initial_vel(speed,target);
    neurokick->get_cmd(cmd);
    return true;
  }

  DBLOG_POL(0,<<"JKWball03: get opening seq cmd: do holdturn");
  onetwoholdturn->get_cmd(cmd);
  return true;  
}

bool JKWball03::intention2cmd(Intention &intention, Cmd &cmd){
  double speed;
  Vector target;
  speed = intention.kick_speed;
  target = intention.kick_target;
  ANGLE targetdir = intention.target_body_dir; // for selfpasses
#if LOGGING && BASIC_LOGGING
  int targetplayer_number = intention.target_player;
#endif
  //  double ball2targetdir = (target - WSinfo::ball->pos).arg(); // direction of the target
  bool cankeepball = onestepkick->can_keep_ball_in_kickrange();
  //double speed1step = Move_1Step_Kick::get_vel_in_dir(speed, ball2targetdir);
  //bool need_only1kick = (fabs(speed -speed1step) < .2);
  double opposite_balldir;
  double kick_dir;

  switch(intention.get_type()){
  case  PASS:
  case  LAUFPASS:
    speed = intention.kick_speed;
    target = intention.kick_target;
    DBLOG_POL(BASELEVEL,<<"JKWball03: AAction2Cmd: passing to teammate "<<targetplayer_number
	      //<<" onestepkick is possible (0=false) "<<need_only1kick
	      <<" speed "<<speed<<" to target "<<target.getX()<<" "<<target.getY());
    if(intention.risky_pass == true){
      DBLOG_POL(BASELEVEL,<<"JKWball03: RISKY PASS!!!");
    }
    neurokick->kick_to_pos_with_initial_vel(speed,target);
    neurokick->get_cmd(cmd);
    return true;
    break;
  case  OPENING_SEQ:
    speed = intention.kick_speed;
    target = intention.kick_target;
    DBLOG_POL(BASELEVEL,<<"JKWball03: AAction2Cmd: Opening Seq for teammate "<<targetplayer_number
	      //<<" onestepkick is possible (0=false) "<<need_only1kick
	      <<" speed "<<speed<<" to target "<<target.getX()<<" "<<target.getY());
    return get_opening_seq_cmd(speed,target,cmd);
    break;
  case SELFPASS:
    DBLOG_POL(0,<<"JKWball03: Intention type SELFPASS in dir "<<RAD2DEG(targetdir.get_value()));
    //selfpass->set_params(speed,target);
    return selfpass->get_cmd(cmd, targetdir, speed, target);
    break;
  case IMMEDIATE_SELFPASS:
    speed = intention.kick_speed;
    target = intention.kick_target;
    kick_dir = (target - WSinfo::ball->pos).arg();
    mdpInfo::set_my_intention(DECISION_TYPE_IMMEDIATE_SELFPASS,
			      speed,target.getX(),target.getY(),0,
			      0);  
    if((Tools::get_abs_angle(WSinfo::ball->vel.arg() - kick_dir)
	<10/180. *PI) && fabs(WSinfo::ball->vel.norm() - speed) < 0.05){
      // Ball already has desired dir and speed
      return get_turn_and_dash(cmd);
    }
    DBLOG_POL(BASELEVEL,<<"JKWball03 AAction2Cmd: Immediate Selfpass with speed "
	      <<speed<<" to target "<<target.getX()<<" "<<target.getY());
    neurokick->kick_to_pos_with_initial_vel(speed,target);
    neurokick->get_cmd(cmd);
    return true;
    break;
  case  WAITANDSEE:
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
    return get_turn_and_dash(cmd);
    break;
  case  DRIBBLE:
    dribblestraight->get_cmd(cmd);
    return true;
    break;
  case SCORE:
    speed = intention.kick_speed;
    target = intention.kick_target;
    mdpInfo::set_my_intention(DECISION_TYPE_SCORE, speed, 0, target.getX(), target.getY(),0);
    LOG_POL(BASELEVEL,<<"JKWball03 aaction2cmd: try to score w speed "<<speed<<" to target "<<target);
    neurokick->kick_to_pos_with_initial_vel(speed,target);
    neurokick->get_cmd(cmd);
    return true;
    break;
  case KICKNRUSH:
    speed = intention.kick_speed;
    target = intention.kick_target;
    mdpInfo::set_my_intention(DECISION_TYPE_KICKNRUSH, speed,0,0,0, 0);
    LOG_POL(BASELEVEL,<<"JKWball03 aaction2cmd: kicknrush w speed "<<speed<<" to target "<<target);
    neurokick->kick_to_pos_with_initial_vel(speed,target);
    neurokick->get_cmd(cmd);
    return true;
    break;
  case PANIC_KICK:
    target = intention.kick_target;
    kick_dir = (target - WSinfo::me->pos).arg() - WSinfo::me->ang.get_value();
    basiccmd->set_kick(100,ANGLE(kick_dir));
    basiccmd->get_cmd(cmd);
    return true;
    break;
  case BACKUP:
    LOG_POL(BASELEVEL,<<"JKWball03 aaction2cmd: back up (two teammates at ball)  not yet implemented");
    LOG_ERR(BASELEVEL,<<"JKWball03 aaction2cmd: back up (two teammates at ball)  not yet implemented");
    //ridi03: todo
    return false;
    break;
  case HOLDTURN:
    DBLOG_POL(BASELEVEL,<<"JKWball03 Intention: holdturn in dir "<<RAD2DEG(intention.target_body_dir.get_value()));
    if(cankeepball){
      if(onetwoholdturn->is_holdturn_safe() == false){
	DBLOG_POL(BASELEVEL,<<"JKWball03 Intention: holdturn NOT safe, relaxed trial (should only occur in troubled sits)");
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
      LOG_ERR(0,<<"Can't keep the ball in my kickrange, and I do not plan to pass. Kick to "
	      <<RAD2DEG(opposite_balldir));
      basiccmd->set_kick(100,ANGLE(opposite_balldir));
      basiccmd->get_cmd(cmd);
      return true;
    }


    return true;
  default:
    DBLOG_POL(0,<<"JKWball03 aaction2cmd: AActionType not known");
    LOG_ERR(0,<<"JKWball03 aaction2cmd: AActionType not known");
    return false;
  }

  return false;
}
