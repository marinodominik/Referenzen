#include "abstract_mdp.h"
//#include "sort.h"
#include "mdp_info.h"
#include "log_macros.h"
#include "planning.h" //will be removed!
#include "tools.h"
#include "valueparser.h"

#define LOG_DEBUG3(X)
//#define LOG_DEBUG3(X) LOG_ERR(0,X)
#define BASELEVEL 0

int AbstractMDP::passreceiver_max_age;
int AbstractMDP::selfpass_mode;
int AbstractMDP::dribble_duration;

void AbstractMDP::init_params(){
  passreceiver_max_age = 100000;
  selfpass_mode = 1;  // 0 is the 2001 version, 1 the new
  dribble_duration=3;

  ValueParser vp(CommandLineOptions::policy_conf,"Action2Move");
  vp.get("passreceiver_max_age",passreceiver_max_age);
  vp.get("selfpass_mode",selfpass_mode);
  vp.get("dribble_duration",dribble_duration);
//  cout<<"|Abstract_mdp: passreceiver_max_age :"<<passreceiver_max_age<<"|";
}

void AbstractMDP::copy_mdp2astate(AState &astate){
  int my_index = -1;


  for(int i=0;i<11;i++){
    astate.my_team[i].pos = Vector(0);
    astate.my_team[i].vel = Vector(0);
    astate.my_team[i].body_angle = 0;
    astate.my_team[i].number = 0;
    astate.my_team[i].age = 0;
    astate.my_team[i].valid = false;
    astate.my_team[i].stamina = 0;
    astate.op_team[i].pos = Vector(0);
    astate.op_team[i].vel = Vector(0);
    astate.op_team[i].body_angle = 0;
    astate.op_team[i].number = 0;
    astate.op_team[i].age = 0;
    astate.op_team[i].valid = false;
    astate.op_team[i].stamina = 0;
  }

#if 0
  Sort sort = Sort(11);
  
  // sort teammates according to their x position
  for(int i=0;i<11;i++){
    if(mdpInfo::mdp->my_team[i].pos_x.p){  // >0: position valid
      sort.add(i,mdpInfo::mdp->my_team[i].pos().x);
    }
  }

  sort.do_sort(1);  // sort highest first
#else
  for(int i=0;i<11;i++){
    astate.my_team[i].valid = false;
    astate.op_team[i].valid = false;
  }
#endif

  astate.ball.pos= mdpInfo::mdp->ball.pos();
  astate.ball.vel= mdpInfo::mdp->ball.vel();
  astate.ball.valid= WSinfo::is_ball_pos_valid();

  for(int i=0;i<11;i++){
    int sortedidx = i;//sort.get_key(i);
    if(sortedidx >=0){// get player i from sorted player array
      if(mdpInfo::mdp->my_team[sortedidx].number==mdpInfo::mdp->me->number){
	my_index = i;
	//astate.ball.pos = mdpInfo::my_pos_abs(); 
      }
      if(mdpInfo::mdp->my_team[sortedidx].pos_x.p){  // >0: position valid
	astate.my_team[i].pos = mdpInfo::mdp->my_team[sortedidx].pos();
	astate.my_team[i].vel = mdpInfo::mdp->my_team[sortedidx].vel();
	astate.my_team[i].body_angle = mdpInfo::mdp->my_team[sortedidx].ang.v;
	astate.my_team[i].number = mdpInfo::mdp->my_team[sortedidx].number;
	astate.my_team[i].age = mdpInfo::age_playerpos(&(mdpInfo::mdp->my_team[sortedidx]));
	astate.my_team[i].valid = true;
	if(mdpInfo::mdp->my_team[sortedidx].stamina_valid())
	  astate.my_team[i].stamina = (int)mdpInfo::mdp->my_team[sortedidx].get_stamina();
	else // assume the best
	  astate.my_team[i].stamina = (int)ServerOptions::stamina_max;	
      }
      else{
	astate.my_team[i].valid = false;
      }
    }
    /*
      LOG_DEBUG2("planning mdp2state: stateplayer "<<i<<" pos "<<astate.my_team[i].position
	     <<" number "<<astate.my_team[i].number);
    */
    if(mdpInfo::mdp->his_team[i].pos_x.p){  // >0: position valid
      astate.op_team[i].pos = mdpInfo::mdp->his_team[i].pos();
      astate.op_team[i].vel = mdpInfo::mdp->his_team[i].vel();
      astate.op_team[i].body_angle = mdpInfo::mdp->his_team[i].ang.v;
      astate.op_team[i].number = mdpInfo::mdp->his_team[i].number;
      astate.op_team[i].age = mdpInfo::age_playerpos(&(mdpInfo::mdp->his_team[i]));
      astate.op_team[i].valid = true;
      if(mdpInfo::mdp->his_team[i].stamina_valid())
	astate.op_team[i].stamina = (int)mdpInfo::mdp->his_team[i].get_stamina();
      else // assume the best
	astate.op_team[i].stamina = (int)ServerOptions::stamina_max;	
    }
    else{
      astate.op_team[i].valid = false;
    }
  }
  astate.my_idx = my_index;
}


void AbstractMDP::set_action(AAction &action, int type, int actor, int recv, Vector target, float vel, float dir, Vector resulting_pos, 
			     Vector resulting_vel, int advantage,int time2intercept,int cycles2go, int targetplayer_number,const bool risky_pass ){
  action.subtype = SUBTYPE_NONE;
  if(type == AACTION_TYPE_PASS)
    action.subtype = SUBTYPE_PASS;
  action.V = 0; // init
  action.acting_player = actor;
  action.action_type = type;
  action.target_player = recv;
  action.target_position = target;
  action.kick_velocity = vel;
  action.kick_dir = dir;
  action.actual_resulting_position=resulting_pos;
  action.actual_resulting_ballvel = resulting_vel;
  action.advantage=advantage;
  action.time2intercept=time2intercept;
  action.cycles2go = cycles2go;
  action.targetplayer_number = targetplayer_number;
  action.risky_pass = risky_pass;
  action.potential_position = resulting_pos;
}

void AbstractMDP::set_action_go4pass(AAction &action, const Vector vballpos, const Vector vballvel){
  action.action_type = AACTION_TYPE_GO4PASS;
  action.virtual_ballpos = vballpos;
  action.virtual_ballvel = vballvel;  
}


void AbstractMDP::model(const AState &state, const AAction& action, AState &succ_state){

#ifdef TRAINING
  //Daniel: use V-function as a Q-function 
  if ( action.action_type == AACTION_TYPE_PASS || action.action_type == AACTION_TYPE_LAUFPASS || 
       action.action_type == AACTION_TYPE_SELFPASS && action.kick_velocity > 0 ) {
    succ_state = state;
    Vector tmp = action.target_position - state.ball.pos;
    tmp.normalize(0.94 * action.kick_velocity);
    succ_state.ball.pos = state.ball.pos + tmp;
    succ_state.ball.vel = tmp;
    succ_state.my_idx = action.target_player;
    return;
  } 
#endif

  succ_state = state;
  succ_state.ball.pos = action.actual_resulting_position;
  if (action.actual_resulting_ballvel.sqr_norm() > 0.01) {
    succ_state.ball.vel = action.actual_resulting_ballvel;
  }

  int recv_idx = action.target_player;

  if(recv_idx <0){
    recv_idx = 0; // default
    for(int i=0;i<11;i++){
      if(state.my_team[i].number==action.targetplayer_number){
	recv_idx = i;
	break;
      }
    }
  }

  succ_state.my_team[recv_idx].pos = 
    action.actual_resulting_position;
  succ_state.my_idx = action.target_player;
}

void AbstractMDP::target_model(const AState &state, const AAction& action, AState &succ_state){
  succ_state = state;
  succ_state.ball.pos = action.target_position;
  succ_state.my_team[action.target_player].pos = 
    action.target_position;
  succ_state.my_idx = action.target_player;
}

void AbstractMDP::model_fine(const AState &state, const AAction& action, AState &succ_state){
  Vector target_ball_vel;

  target_ball_vel = action.target_position - state.ball.pos;
  target_ball_vel.normalize(action.kick_velocity);
  succ_state = state;
  succ_state.ball.pos = state.ball.pos + target_ball_vel;
  succ_state.ball.vel = target_ball_vel;
}

#define PLAYERS 11
void AbstractMDP::model(const AState &state, const AAction action[], AState &succ_state){
  succ_state = state;
  for(int k=0;k<PLAYERS;k++){
    succ_state.my_team[k].pos = action[k].target_position;
  }
}

/* D
Main_Move *AbstractMDP::aaction2move(AAction action){
  // convert best action to a bs2k move an return it
  float ballspeed;
  Vector target;
  double targetdir,ball2targetdir;
  int pass_receiver;
  int pass_receiver_age;

  // this was used in Melbourne, but it may hinder fast dribbling,
  //   check.
  //   Intention: If the ball is too fast and I can not control,
  //   then try to stop it.
  //   But what if the Ball is going in the right direction?

  // ridi: reactivated it, because its useful for defense!
  
  bool cankeepball = Move_1Step_Kick::can_keep_ball_in_kickable_area();
  if(mdpInfo::is_ball_kickable_exact() &&
     cankeepball == false){
    if(action.action_type != AACTION_TYPE_PASS &&
       action.action_type != AACTION_TYPE_LAUFPASS &&
       action.action_type != AACTION_TYPE_SELFPASS &&
       action.action_type != AACTION_TYPE_IMMEDIATE_SELFPASS &&
       action.action_type != AACTION_TYPE_DEADLYPASS){
      double opposite_balldir = mdpInfo::ball_vel_abs().arg() + PI;
      opposite_balldir = opposite_balldir - mdpInfo::mdp->me->ang.v;
      LOG_ERR(0,<<"Can't keep the ball in my kickrange, and I do not plan to pass. Kick to "
	      <<RAD2DEG(opposite_balldir));
      return Move_Factory::get_Kick(100,opposite_balldir,1);
    }
  }

  Vector dummy1, dummy3;
  Vector mynewpos,ballnewpos;
  double dummy2;
  int dash;  
  Cmd_Main testcmd;
  double required_turn;
  bool dash_found;

  switch(action.action_type){
  case  AACTION_TYPE_PASS:
  case  AACTION_TYPE_LAUFPASS:
  case AACTION_TYPE_DEADLYPASS:
    target = action.target_position;
    ballspeed = action.kick_velocity;
    //targetdir = (target - mdpInfo::my_pos_abs()).arg();
    targetdir = (target - mdpInfo::ball_pos_abs()).arg();
    pass_receiver = mdpInfo::teammate_closest_to(target); // this is not elegant, but works
    pass_receiver_age = mdpInfo::teammate_age(pass_receiver);
    if(action.action_type == AACTION_TYPE_LAUFPASS)
      mdpInfo::set_my_intention(DECISION_TYPE_LAUFPASS,
				ballspeed,target.x,target.y,targetdir,
				pass_receiver);
    else
      mdpInfo::set_my_intention(DECISION_TYPE_PASS,
				ballspeed,target.x,target.y,targetdir,
				pass_receiver);
    if((pass_receiver_age >0) && (mdpInfo::could_see_in_direction(targetdir) == true)){
      // I can see in direction by turning neck without turning body
      mdpInfo::set_my_neck_intention(NECK_INTENTION_LOOKINDIRECTION, targetdir);
    }
    if(cankeepball && (pass_receiver_age >passreceiver_max_age) &&
       (mdpInfo::opponents_within_range(mdpInfo::my_pos_abs(),4.) <= 1)){
      //LOG_ERR(0,"Pass receiver too old");
      if(mdpInfo::could_see_in_direction(targetdir) == true){
	// I can see in direction by turning neck without turning body
	mdpInfo::set_my_neck_intention(NECK_INTENTION_LOOKINDIRECTION, targetdir);
	double speed1step = Move_1Step_Kick::get_vel_in_dir(ballspeed, targetdir);
	if(fabs(ballspeed -speed1step) >= .2){
	  // I will need at least two kicks, so I can turn neck and start to kick
	  LOG_POL(3,"AA2Move: Try 2 pass to "<<pass_receiver<<
		  " Teammate age  "<<pass_receiver_age<<" > "<<passreceiver_max_age
		  <<" START KICKING. Turn neck is enough to teammate pos "<<target
		  <<" turn neck 2 dir "<<RAD2DEG(targetdir)
		  <<" mypos "<<mdpInfo::my_pos_abs());
	  //return Move_Factory::get_Neuro_Kick2(1, ballspeed, target.x, target.y);
	  return Move_Factory::get_Neuro_Kick2(5, ballspeed, target.x, target.y);
	}
	LOG_POL(0,"AA2Move: Try 2 pass to "<<pass_receiver<<
		" Teammate age  "<<pass_receiver_age<<" > "<<passreceiver_max_age
		<<" Keep Ball. Turn neck is enough to teammate pos "<<target
		<<" turn neck 2 dir "<<RAD2DEG(targetdir)
		<<" mypos "<<mdpInfo::my_pos_abs());
	//return Move_Factory::get_Neuro_Hold_Ball(1); // just keep ball in kickr. for anoth. cyc.
	mdpInfo::set_my_intention(DECISION_TYPE_WAITANDSEE);
	return Move_Factory::get_Hold_Ball(1); // just keep ball in kickr. for anoth. cyc.
      }
      LOG_POL(0,"AA2Move: Try 2 pass to "<<pass_receiver<<
	      " Teammate age  "<<pass_receiver_age<<" > "<<passreceiver_max_age
	      <<" Cannot turn neck. Turn body to teammate pos "<<target
	      <<" turn 2 dir "<<RAD2DEG(targetdir)
	      <<" mypos "<<mdpInfo::my_pos_abs());
      mdpInfo::set_my_intention(DECISION_TYPE_WAITANDSEE);
      return Move_Factory::get_Hold_Ball(targetdir,1);      
    }
    if(Planning::is_multikick_possible()== false){
      // ridi: check for safety, this should not happen
      double speed1step = Move_1Step_Kick::get_vel_in_dir(ballspeed, targetdir);
      if(fabs(ballspeed -speed1step) >= .05 || ballspeed <0.3 ){
	LOG_ERR(0,<<"AAction2move ERROR: multi-kick not possible, desired speed "
		<<ballspeed<<" best 1 step can do "<<speed1step);
	LOG_POL(3,<<"AAction2move ERROR: multi-kick not possible, desired speed "
		<<ballspeed<<" best 1 step can do "<<speed1step);
	mdpInfo::set_my_intention(DECISION_TYPE_WAITANDSEE);
	return Move_Factory::get_Hold_Ball(1); 
      }
      //return Move_Factory::get_1Step_Kick(ballspeed,targetdir);
      return Move_Factory::get_Neuro_Kick2(5, ballspeed, target.x, target.y);
    }
    LOG_POL(3,<<"AAction2Move: passing to teammate "<<pass_receiver
	    <<" age "<<pass_receiver_age<<" with multikicks "
	    <<ballspeed<<" to target "<<target.x<<" "<<target.y);
    //return Move_Factory::get_Neuro_Kick2(1, ballspeed, target.x, target.y);
    return Move_Factory::get_Neuro_Kick2(5, ballspeed, target.x, target.y);
  case AACTION_TYPE_TURN_AND_DASH:
    // If I can get ball by dashing, dash
    dash_found=false;
    for(dash=10;dash<=100;dash+=10){
      testcmd.unset_lock();
      testcmd.unset_cmd();
      testcmd.set_dash(dash);
      Tools::model_cmd_main(mdpInfo::my_pos_abs(), mdpInfo::my_vel_abs(), mdpInfo::mdp->me->ang.v, 
			    mdpInfo::ball_pos_abs(),
			    mdpInfo::ball_vel_abs(),
			    testcmd, mynewpos, dummy1, dummy2, ballnewpos, dummy3);
      if((mynewpos-ballnewpos).norm()<0.8*ServerOptions::kickable_area){
	dash_found=true;
	break;
      }
    }
    if(dash_found ==true){
      LOG_POL(BASELEVEL,<<"TURN_AND_DASH: I found a dash "<<dash);
      return Move_Factory::get_Dash(dash,1);
    }
    required_turn = 
      Tools::get_angle_between_null_2PI(mdpInfo::ball_vel_abs().arg() - 
					mdpInfo::mdp->me->ang.v);
    if(Tools::get_abs_angle(required_turn)>10./180. *PI){
      LOG_POL(BASELEVEL,<<"AAction2Move: Turn and Dash: turn 2 balldir "
	      <<RAD2DEG(mdpInfo::ball_vel_abs().arg())<<" Have to turn by "
	      <<RAD2DEG(required_turn));
      return Move_Factory::get_Turn_Inertia(required_turn,1);
    }
    // dash forward
    return Move_Factory::get_Dash(100,1);
    break;
  case AACTION_TYPE_SELFPASS:
    if(selfpass_mode==1){ // new version !!!
      target = action.target_position;
      ballspeed = action.kick_velocity;
      targetdir = (target - mdpInfo::my_pos_abs()).arg();
      ball2targetdir = (target - mdpInfo::ball_pos_abs()).arg();
      pass_receiver = mdpInfo::mdp->me->number;
      mdpInfo::set_my_intention(DECISION_TYPE_PASS,
				ballspeed,target.x,target.y,targetdir,
				pass_receiver);

      // 1. check if ball has already desired speed and direction
      if((Tools::get_abs_angle(mdpInfo::ball_vel_abs().arg() - ball2targetdir) 
	  <10/180. *PI) && fabs(mdpInfo::ball_vel_abs().norm() - ballspeed) < 0.05){ // no kick required
	double required_turn =
	  Tools::get_angle_between_null_2PI(mdpInfo::ball_vel_abs().arg() - 
					    mdpInfo::mdp->me->ang.v);
	if(Tools::get_abs_angle(required_turn)>10./180. *PI){
	  LOG_POL(3,<<"AAction2MoveSelf Pass - Ball already has desired speed and dir, turn 2 balldir "
		  <<RAD2DEG(mdpInfo::ball_vel_abs().arg())<<" Have to turn by "
		  <<RAD2DEG(required_turn));
	  return Move_Factory::get_Turn_Inertia(required_turn,1);
	}
	else{
	  LOG_POL(3,<<"AAction2Move: Self Pass - Ball has desired speed and dir, Looking in dir, wait");
	  return Move_Factory::get_Turn_Inertia(required_turn,1);
	}      
      }
      // 2. keep ball and turn to desired kick direction
      if(cankeepball && (Tools::get_abs_angle(mdpInfo::mdp->me->ang.v - targetdir) >10/180. *PI)){
	// if((Tools::get_abs_angle(mdpInfo::mdp->me->ang.v - targetdir) >40/180. *PI)){
	LOG_POL(3,<<"AAction2MoveSelf Pass - target dir and body dir differ. Targetdir "
		<<RAD2DEG(targetdir));
	return Move_Factory::get_Hold_Ball(targetdir,1); 
      }
      
      // 3. I'm looking in target direction, now kick!
      LOG_POL(3,<<"AAction2Move: Selfpass with speed "
	      <<ballspeed<<" to target "<<target.x<<" "<<target.y);
      return Move_Factory::get_Neuro_Kick2(5, ballspeed, target.x, target.y);
    }
    // this it the selfpass version used in 2001 !!!
    else{ //original version
      target = action.target_position;
      ballspeed = action.kick_velocity;
      //targetdir = (target - mdpInfo::my_pos_abs()).arg();
      targetdir = action.kick_dir;
      pass_receiver = mdpInfo::mdp->me->number;
      mdpInfo::set_my_intention(DECISION_TYPE_PASS,
				ballspeed,target.x,target.y,targetdir,
				pass_receiver);
      if(cankeepball && (Tools::get_abs_angle(mdpInfo::mdp->me->ang.v - targetdir) >10/180. *PI)){
	// if((Tools::get_abs_angle(mdpInfo::mdp->me->ang.v - targetdir) >40/180. *PI)){
	LOG_POL(3,<<"AAction2MoveSelf Pass - target dir and body dir differ. Targetdir "
		<<RAD2DEG(targetdir));
	return Move_Factory::get_Hold_Ball(targetdir,1); 
      }
      if(Planning::is_multikick_possible()== false){
	// ridi: check for safety, this should not happen
	double speed1step = Move_1Step_Kick::get_vel_in_dir(ballspeed, targetdir);
	if (cankeepball && (fabs(ballspeed -speed1step) >= .05 || ballspeed <0.3)){
	  LOG_ERR(0,<<"AAction2move ERROR: multi-kick not possible, desired speed "
		  <<ballspeed<<" best 1 step can do "<<speed1step);
	  LOG_POL(3,<<"AAction2move ERROR: multi-kick not possible, desired speed "
		  <<ballspeed<<" best 1 step can do "<<speed1step);
	  mdpInfo::set_my_intention(DECISION_TYPE_WAITANDSEE);
	  return Move_Factory::get_Hold_Ball(1); 
	}
	LOG_POL(3,<<"AAction2move: multi-kick not possible. desired speed "
		<<ballspeed<<" but 1 step CAN do "<<speed1step);
	return Move_Factory::get_1Step_Kick(ballspeed,targetdir);
      }
      if((Tools::get_abs_angle(mdpInfo::ball_vel_abs().arg() - targetdir) 
	  <10/180. *PI) && fabs(mdpInfo::ball_vel_abs().norm() - ballspeed) < 0.05){
	double required_turn =
	  Tools::get_angle_between_null_2PI(mdpInfo::ball_vel_abs().arg() - 
					    mdpInfo::mdp->me->ang.v);
	if(Tools::get_abs_angle(required_turn)>10./180. *PI){
	  LOG_POL(3,<<"AAction2MoveSelf Pass - Ball already has desired speed and dir, turn 2 balldir "
		  <<RAD2DEG(mdpInfo::ball_vel_abs().arg())<<" Have to turn by "
		  <<RAD2DEG(required_turn));
	  return Move_Factory::get_Turn_Inertia(required_turn,1);
      }
	else{
	  LOG_POL(3,<<"AAction2Move: Self Pass - Ball has desired speed and dir, Looking in dir -> NO Dash, since I might loose the ball");
	  return Move_Factory::get_Turn_Inertia(required_turn,1);
	  LOG_POL(3,<<"AAction2Move: Self Pass - Ball has desired speed and dir, Looking in dir -> DASH!");
	  return Move_Factory::get_Dash(100,1);
	}      
	//return Move_Factory::get_Neuro_Intercept_Ball(1);
      }
      if(!(Tools::equal(Move_1Step_Kick::get_vel_in_dir(ballspeed,targetdir),
			ballspeed,0.1))){
	LOG_POL(3,<<"AAction2Move: Selfpass with multikicks "
		<<ballspeed<<" to target "<<target.x<<" "<<target.y);
	return Move_Factory::get_Neuro_Kick2(1, ballspeed, target.x, target.y);
      }
      // Self-Passing: single kick will do the job!;
      LOG_POL(3,<<"AAction2Move: Selfpass, 1 step kick will work with speed "
	      <<ballspeed<<" in dir "<<RAD2DEG(targetdir));
      return Move_Factory::get_1Step_Kick(ballspeed,targetdir);      
    }
    break;
    case AACTION_TYPE_IMMEDIATE_SELFPASS:
      //LOG_ERR(0,<<"AAction2move: Immediate Selfpass with speed "<<action.kick_velocity);
      target = action.target_position;
      ballspeed = action.kick_velocity;
      //targetdir = (target - mdpInfo::my_pos_abs()).arg();
      targetdir = action.kick_dir;
      pass_receiver = mdpInfo::mdp->me->number;
      mdpInfo::set_my_intention(DECISION_TYPE_IMMEDIATE_SELFPASS,
				ballspeed,target.x,target.y,targetdir,
				pass_receiver);  
    if(Planning::is_multikick_possible()== false){
      // ridi: check for safety, this should not happen
      double speed1step = Move_1Step_Kick::get_vel_in_dir(ballspeed, targetdir);
      if(fabs(ballspeed -speed1step) >= .05 || ballspeed <0.3 ){
	LOG_ERR(0,<<"AAction2move ERROR: multi-kick not possible, desired speed "
		<<ballspeed<<" best 1 step can do "<<speed1step);
	LOG_POL(3,<<"AAction2move ERROR: multi-kick not possible, desired speed "
		<<ballspeed<<" best 1 step can do "<<speed1step);
	mdpInfo::set_my_intention(DECISION_TYPE_WAITANDSEE);
	return Move_Factory::get_Hold_Ball(1); 
      }
      LOG_POL(3,<<"AAction2move: multi-kick not possible. desired speed "
	      <<ballspeed<<" but 1 step CAN do "<<speed1step);
      return Move_Factory::get_1Step_Kick(ballspeed,targetdir);
    }

    if((Tools::get_abs_angle(mdpInfo::ball_vel_abs().arg() - targetdir) 
	<10/180. *PI) && fabs(mdpInfo::ball_vel_abs().norm() - ballspeed) < 0.05){
      double required_turn =
	Tools::get_angle_between_null_2PI(mdpInfo::ball_vel_abs().arg() - 
					  mdpInfo::mdp->me->ang.v);
      if(Tools::get_abs_angle(required_turn)>10./180. *PI){
	LOG_POL(3,<<"AAction2Move Immediate Self Pass - Ball already has desired speed and dir, turn 2 balldir "
		<<RAD2DEG(mdpInfo::ball_vel_abs().arg())<<" Have to turn by "
		<<RAD2DEG(required_turn));
	return Move_Factory::get_Turn_Inertia(required_turn,1);
      }
      else{
	LOG_POL(3,<<"AAction2Move: Immediate Self Pass - Ball has desired speed and dir, Looking in dir -> DASH!");
	return Move_Factory::get_Dash(100,1);
      }      
      //return Move_Factory::get_Neuro_Intercept_Ball(1);
    }
    if(!(Tools::equal(Move_1Step_Kick::get_vel_in_dir(ballspeed,targetdir),
		      ballspeed,0.1))){
      LOG_POL(3,<<"AAction2Move: Immediate Selfpass with multikicks "
	      <<ballspeed<<" to target "<<target.x<<" "<<target.y);
      return Move_Factory::get_Neuro_Kick2(1, ballspeed, target.x, target.y);
    }
    // Immediate Self-Passing: single kick will do the job!;
    LOG_POL(3,<<"AAction2Move: Immediate Selfpass, 1 step kick will work with speed "
	    <<ballspeed<<" in dir "<<RAD2DEG(targetdir));
    return Move_Factory::get_1Step_Kick(ballspeed,targetdir);      
    break;
  case  AACTION_TYPE_DRIBBLE:
    mdpInfo::set_my_intention(DECISION_TYPE_DRIBBLE, action.target_position.x,action.target_position.y);
    LOG_POL(BASELEVEL,<<"AAction2Move: Dribble  to target "
	    <<action.target_position<<" CALLED FOR CYCLES: "<<dribble_duration);
    return Move_Factory::get_Move_Dribble(action.target_position,dribble_duration);
    //return Move_Factory::get_Neuro_Dribble(target.x, target.y,1);
  break;
  case  AACTION_TYPE_DRIBBLE_QUICKLY:
    mdpInfo::set_my_intention(DECISION_TYPE_DRIBBLE, action.target_position.x,action.target_position.y);
    LOG_POL(BASELEVEL,<<"AAction2Move: Dribble Quickly to target "
	    <<action.actual_resulting_position<<" CALLED FOR CYCLES: "<<action.cycles2go);
    //that nr_dash=action.cycles2go-1 works only because we dribble straight without turns
    return Move_Factory::get_Dribble_Quickly(action.target_position,action.cycles2go-1,action.cycles2go);
  break;
  case  AACTION_TYPE_WAITANDSEE:
    mdpInfo::set_my_intention(DECISION_TYPE_WAITANDSEE);
    //return Move_Factory::get_Neuro_Hold_Ball_Turn(1); 
    return Move_Factory::get_Hold_Ball(1); 
    // stop ball, hold ball and if posssible turn2goal
    break;
  case AACTION_TYPE_STAY:
    mdpInfo::set_my_intention(DECISION_TYPE_WAITANDSEE);
    return Move_Factory::get_Face_Ball();
    break;
  case AACTION_TYPE_GOTO: 
  case AACTION_TYPE_GOHOME:
    mdpInfo::set_my_intention(DECISION_TYPE_RUNTO);
    return Move_Factory::get_Neuro_Go2Pos_stamina(action.target_position.x, action.target_position.y, action.dash_power, 10, 5, 1);
    break;
  case AACTION_TYPE_GO2BALL:
    mdpInfo::set_my_intention(DECISION_TYPE_INTERCEPTBALL);
    return Move_Factory::get_Neuro_Intercept_Ball(2);
    break;
  }
  return NULL;
}
*/


/****************************************************************************/
/****************************************************************************/
/****************************************************************************/
/****************************************************************************/

void AbstractMDP::get_valid_teammates(const AState & state, AObjectArray & arr) {
  get_valid_objects(state.my_team,arr);
}

void AbstractMDP::get_valid_opponents(const AState & state, AObjectArray & arr) {
  get_valid_objects(state.op_team,arr);
}

void AbstractMDP::get_first_teammates_by_distance_to_pos(int num_of_first, const AState & state, const Vector & pos, AObjectArray & arr) {
  get_valid_objects(state.my_team,arr);
  get_first_objects_by_distance_to_pos(num_of_first,pos,arr);
} 

void AbstractMDP::get_first_opponents_by_distance_to_pos(int num_of_first, const AState & state, const Vector & pos, AObjectArray & arr) {
  get_valid_objects(state.op_team,arr);
  get_first_objects_by_distance_to_pos(num_of_first,pos,arr);
} 
 
void AbstractMDP::get_first_teammates_by_x_coord(int num_of_first, const AState & state, AObjectArray & arr) {
  double xpos[11];
  get_valid_objects(state.my_team, arr);
  for (int i=0 ; i < arr.num; i++)
    xpos[i] = arr.obj[i]->pos.getX();
  if (num_of_first > arr.num)
    num_of_first= arr.num;
  for (int i =0; i < num_of_first ; i++)
    for (int j=i+1; j< arr.num; j++) 
      if ( xpos[j] > xpos[i] ) {
	double int_tmp= xpos[i];
	xpos[i]= xpos[j];
	xpos[j]= int_tmp;
	AObject const* obj_tmp= arr.obj[i];
	arr.obj[j]= arr.obj[i];
	arr.obj[i]= obj_tmp;
      }
}

void AbstractMDP::get_last_teammates_by_x_coord(int num_of_last, const AState & state, AObjectArray & arr) {
  double xpos[11];
  get_valid_objects(state.my_team, arr);
  for (int i=0 ; i < arr.num; i++)
    xpos[i] = arr.obj[i]->pos.getX();
  if (num_of_last > arr.num)
    num_of_last= arr.num;
  for (int i =0; i < num_of_last ; i++)
    for (int j=i+1; j< arr.num; j++) 
      if ( xpos[j] < xpos[i] ) {
	double int_tmp= xpos[i];
	xpos[i]= xpos[j];
	xpos[j]= int_tmp;
	AObject const* obj_tmp= arr.obj[i];
	arr.obj[j]= arr.obj[i];
	arr.obj[i]= obj_tmp;
      }
}

/***************************************************************************/
void AbstractMDP::get_valid_objects(const AObject * obj, AObjectArray & arr) {
  arr.num= 0;
  AObject const* *tar= arr.obj;
  for (int i=0; i<11; i++, obj++)
    if ( obj->valid ) {
      *tar = obj;
      arr.num++;
      tar++;
    }
}

void AbstractMDP::get_first_objects_by_distance_to_pos(int num_of_first, const Vector & pos, AObjectArray & arr) {
  double dist[11];
  for (int i=0; i< arr.num; i++)
    dist[i]= arr.obj[i]->pos.distance(pos);

  if (num_of_first > arr.num)
    num_of_first= arr.num;

  for (int i=0; i< num_of_first; i++)
    for (int j=i+1; j< arr.num; j++) 
      if ( dist[j] < dist[i] ) {
	double int_tmp= dist[i];
	dist[i]= dist[j];
	dist[j]= int_tmp;
	AObject const* obj_tmp= arr.obj[i];
	arr.obj[j]= arr.obj[i];
	arr.obj[i]= obj_tmp;
      }
}


