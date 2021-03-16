#include "learn_wball_bmp.h"
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
#define DBLOG_POL(LLL,XXX) LOG_POL(LLL,<<"WBALL03: " XXX)
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


/* constructor method */
LearnWball::LearnWball() {
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

  ValueParser vp(CommandLineOptions::policy_conf,"wball03_bmp");
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

  cout<<"\nLearnWball status: Date: 7.02.03 / 1"<<endl;
}

LearnWball::~LearnWball() {
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

void LearnWball::reset_intention() {
  my_blackboard.pass_or_dribble_intention.reset();
  my_blackboard.intention.reset();
  //ERROR_OUT << "  wball03 reset intention, cycle " << WSinfo::ws->time;
  DBLOG_POL(0, << "wball03 reset intention");
}

bool LearnWball::get_cmd(Cmd & cmd) {
  Intention intention;

  LOG_POL(BASELEVEL, << "In LearnWBALL_BMC : ");

  if(!WSinfo::is_ball_kickable())
    return false;
  switch(WSinfo::ws->play_mode) {
  case PM_PlayOn:
    if(get_intention(intention)){
      intention2cmd(intention,cmd);
      if(cmd.cmd_body.get_type() == Cmd_Body::TYPE_KICK ||
	 cmd.cmd_body.get_type() == Cmd_Body::TYPE_TURN)
	last_waitandsee_at = WSinfo::ws->time;
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


bool LearnWball::get_intention(Intention &intention){

  long ms_time= MYGETTIME;
  DBLOG_POL(BASELEVEL+2, << "Entering LearnWball");

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
  // determine closest opponent (frequently needed)
  PlayerSet pset = WSinfo::valid_opponents;
  pset.keep_and_sort_closest_players_to_point(1, WSinfo::me->pos);
  if(pset.num >0)
    closest_opponent = pset[0];
  else
    closest_opponent = NULL;


  bool result = false;

  if ( (result = score->test_shoot2goal(intention)) );

  //else if ( (result = test_holdturn(intention)));

  else if ( (result = test_pass_or_dribble(intention)) );

  else result = test_default(intention); // modified version

  my_blackboard.intention = intention;

  check_write2blackboard();

  ms_time = MYGETTIME - ms_time;
  DBLOG_POL(BASELEVEL+1, << "WBALL03 policy needed " << ms_time << "millis to decide");
  return result;
}

bool LearnWball::selfpass_dir_ok(const ANGLE dir){

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


bool LearnWball::get_pass_or_dribble_intention(Intention &intention){
  AState current_state;
  AbstractMDP::copy_mdp2astate(current_state);

  return get_pass_or_dribble_intention(intention, current_state);
}

bool LearnWball::get_pass_or_dribble_intention(Intention &intention, AState &state){
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



void LearnWball::check_write2blackboard(){
  int main_type = my_blackboard.intention.get_type();
  bool main_type_is_pass = false;

  if (main_type == PASS || main_type == LAUFPASS || 
      main_type == KICKNRUSH) 
    main_type_is_pass = true;
  
  
  // now, check for communication request

  if ( main_type_is_pass ) {
    DBLOG_POL(0,<<"Wball03: Check write2blackboard: pass intention is set");
    // for now, communication is done indirectly; should be improved
    Blackboard::pass_intention = my_blackboard.intention;
  }
}



/** default move */
bool LearnWball::test_default(Intention &intention)
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
    DBLOG_POL(BASELEVEL+0,<<"WBALL03 DEFAULT Move - Hold Turn is Safe");
    intention.set_holdturn(target_dir, WSinfo::ws->time);
    return true;
  }

  DBLOG_POL(BASELEVEL+0,<<"WBALL03 DEFAULT Move - Hold Turn is NOT Safe, but no other alternative, TRY");
  intention.set_holdturn((opgoalpos - WSinfo::me->pos).ARG(), WSinfo::ws->time);
  return true;
}


bool LearnWball::test_pass_or_dribble(Intention &intention){
  return get_pass_or_dribble_intention(intention);
}


void LearnWball::aaction2intention(const AAction &aaction, Intention &intention){

  double speed = aaction.kick_velocity;
  Vector target = aaction.target_position;
  double kickdir = aaction.kick_dir;

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

    intention.set_dribble(target, WSinfo::ws->time);
    break;
  default:
    DBLOG_POL(0,<<"WBALL03 aaction2intention: AActionType not known");
    LOG_ERR(0,<<"WBALL03 aaction2intention: AActionType not known");
  }
}


bool LearnWball::intention2cmd(Intention &intention, Cmd &cmd){
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

  Vector tmp;

  switch(intention.get_type()){
  case  PASS:
  case  LAUFPASS:
    speed = intention.kick_speed;
    target = intention.kick_target;
    DBLOG_POL(BASELEVEL,<<"WBALL03: AAction2Cmd: passing to teammate "<<targetplayer_number
	      //<<" onestepkick is possible (0=false) "<<need_only1kick
	      <<" speed "<<speed<<" to target "<<target.getX()<<" "<<target.getY());
    if (intention.get_type() == PASS) {
      DBLOG_POL(0, << " PASS");
    } else {
      DBLOG_POL(0, << " LAUFPASS");
    }
    tmp = target - WSinfo::ball->pos;
    tmp.normalize(0.94 * speed);

    DBLOG_POL(0, << "using situation: ball_pos: " << WSinfo::ball->pos + tmp << " ball_vel: " << tmp);

    if(intention.risky_pass == true){
      DBLOG_POL(BASELEVEL,<<"WBALL03: RISKY PASS!!!");
    }
    neurokick->kick_to_pos_with_initial_vel(speed,target);
    neurokick->get_cmd(cmd);
    return true;
    break;
  case SELFPASS:
    DBLOG_POL(0,<<"WBALL03: Intention type SELFPASS in dir "<<RAD2DEG(targetdir.get_value()));
    //selfpass->set_params(speed,target);
    return selfpass->get_cmd(cmd, targetdir, speed, target);
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
  case  DRIBBLE:  
    dribblestraight->get_cmd(cmd);
    return true;
    break;
  case SCORE:
    speed = intention.kick_speed;
    target = intention.kick_target;
    mdpInfo::set_my_intention(DECISION_TYPE_SCORE, speed, 0, target.getX(), target.getY() ,0);
    LOG_POL(BASELEVEL,<<"WBALL03 aaction2cmd: try to score w speed "<<speed<<" to target "<<target);
    neurokick->kick_to_pos_with_initial_vel(speed,target);
    neurokick->get_cmd(cmd);
    return true;
    break;
  case KICKNRUSH:
    speed = intention.kick_speed;
    target = intention.kick_target;
    mdpInfo::set_my_intention(DECISION_TYPE_KICKNRUSH, speed,0,0,0, 0);
    LOG_POL(BASELEVEL,<<"WBALL03 aaction2cmd: kicknrush w speed "<<speed<<" to target "<<target);
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
    LOG_POL(BASELEVEL,<<"WBALL03 aaction2cmd: back up (two teammates at ball)  not yet implemented");
    LOG_ERR(BASELEVEL,<<"WBALL03 aaction2cmd: back up (two teammates at ball)  not yet implemented");
    //ridi03: todo
    return false;
    break;
  case HOLDTURN:
    DBLOG_POL(BASELEVEL,<<"WBALL03 Intention: holdturn in dir "<<RAD2DEG(intention.target_body_dir.get_value()));
    if(cankeepball){
      if(onetwoholdturn->is_holdturn_safe() == false){
	DBLOG_POL(BASELEVEL,<<"WBALL03 Intention: holdturn NOT safe, relaxed trial (should only occur in troubled sits)");
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
    DBLOG_POL(0,<<"WBALL03 aaction2cmd: AActionType not known");
    LOG_ERR(0,<<"WBALL03 aaction2cmd: AActionType not known");
    return false;
  }
  return false;
}
