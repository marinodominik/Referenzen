#include "onetwo_holdturn_bms.h"
#include "tools.h"
#include "options.h"
#include "ws_info.h"
#include "log_macros.h"

#define BASELEVEL 3

//#define DBLOG_MOV(LLL,XXX) LOG_MOV(LLL,XXX)
#define DBLOG_MOV(LLL,XXX) 

// CLASS CONSTANTS' INITIALIZATION
const double HoldTurnItrActions::kick_pwr_min = 10;
const double HoldTurnItrActions::kick_pwr_inc = 10;
const double HoldTurnItrActions::kick_pwr_steps = 10;
const double HoldTurnItrActions::kick_fine_pwr_inc = 1;
const double HoldTurnItrActions::kick_fine_pwr_steps = 10;
const double HoldTurnItrActions::kick_ang_min = 0;
const double HoldTurnItrActions::kick_ang_inc = 2*PI/72;
const double HoldTurnItrActions::kick_ang_steps = 72;
const double HoldTurnItrActions::kick_fine_ang_inc = 2*PI/360;
const double HoldTurnItrActions::kick_fine_ang_steps = 90;

bool OneTwoHoldTurn::initialized=false;

//#define PROT(XXX)(cout<<XXX)
#define PROT(XXX)
#define SAFETY_MARGIN 0.15

OneTwoHoldTurn::OneTwoHoldTurn(){
  //const double safety_margin = 0.15; //0.1 is too small
  //  success_sqrdist = SQUARE(WSinfo::me->kick_radius - SAFETY_MARGIN); // this doesnt work becaus player isn't initialized yet
  success_sqrdist = SQUARE(ServerOptions::kickable_area - SAFETY_MARGIN);
  current_cmd_valid_at = -1;
  holdturn_not_possible_at = -1;
}


bool OneTwoHoldTurn::init(char const * conf_file, int argc, char const* const* argv){
  if(initialized) return true;
  initialized = true;
  INFO_OUT << "\n12_HoldTurn behavior initialized.";
  return initialized;
}

bool OneTwoHoldTurn::is_holdturn_safe(){
  //#define HOLDTURN_SAFE_DIST .8

  DBLOG_MOV(0,"Holdturn: check safety");


  relaxed_checking = false;

  Cmd cmd;
  double dist;
  int steps;

  success_sqrdist = SQUARE(WSinfo::me->kick_radius - SAFETY_MARGIN);

  MyState state;
  
  state.my_pos = WSinfo::me->pos;
  state.my_vel = WSinfo::me->vel;
  state.ball_pos = WSinfo::ball->pos;
  state.ball_vel = WSinfo::ball->vel;
  state.my_angle = WSinfo::me->ang;

  PlayerSet opset;
  opset= WSinfo::valid_opponents;
  opset.keep_and_sort_closest_players_to_point(1, state.ball_pos);
  if ( opset.num ){
    state.op_pos= opset[0]->pos;
    state.op_bodydir =opset[0]->ang;
    state.op_bodydir_age = opset[0]->age_ang;
    state.op=opset[0];
    DBLOG_MOV(0,<< _2D << C2D(state.op_pos.x, state.op_pos.y ,1.4,"red")); 
    DBLOG_MOV(0,"get cmd. op number "<<state.op ->number);
  }
  else{
    state.op_pos= Vector(1000,1000); // outside pitch
    state.op_bodydir = ANGLE(0);
    state.op_bodydir_age = 100000;
    state.op=NULL;
    DBLOG_MOV(0,"Op.NOT  set ");
  }


  if (get_2step_cmd(cmd, dist,steps, state) == true)
    return true;
  else 
    return false;

}

bool OneTwoHoldTurn::ballpos_ok(const MyState &state){
  double sqrdist = state.my_pos.sqr_distance(state.ball_pos);
  if(sqrdist >success_sqrdist)
    return false;

  if(relaxed_checking == true){ // doing a relaxed check
    DBLOG_MOV(0,<<"Holdturn: Doing only a RELAXED CHECK ");
    if(state.op_pos.distance(state.ball_pos) > ServerOptions::kickable_area +.2)
      return true;
  }

  if(WSinfo::his_goalie){
    if(WSinfo::his_goalie->pos.distance(state.ball_pos) < 
       ServerOptions::catchable_area_l + ServerOptions::player_speed_max){
      //DBLOG_MOV(0,<<"Holdturn: goalie is close, check if he can catch Ball ");
      if(Tools::is_ball_safe_and_kickable(state.my_pos,WSinfo::his_goalie, state.ball_pos, false) == false){
	//DBLOG_MOV(0,<<"Holdturn: goalie can catch Ball, return false ");
	return false;
      }
    } // goalie is close
  } // goalie is known

  //bool result = Tools::is_ball_safe_and_kickable(state.my_pos, state.op_pos, state.op_bodydir, 
  //                  state.ball_pos, state.op_bodydir_age);
  /* Sput: new, improved, hopefully working hetero version... */
  bool result;
  //  if(state.op_bodydir_age < 100000) {
  if(state.op != 0) {
    //result = Tools::is_ball_safe_and_kickable(state.my_pos,state.op,state.ball_pos,false);
    //DBLOG_MOV(0,"HOLDTURN check ballpos, considering tackle");
    result = Tools::is_ball_safe_and_kickable(state.my_pos,state.op,state.ball_pos,true); // ridi04: consider tackle!
  }
  else{
    //DBLOG_MOV(0,"HOLDTURN check ballpos, OLD procedure");
    result = Tools::is_ball_safe_and_kickable(state.my_pos, state.op_pos, state.op_bodydir,
					      state.ball_pos, state.op_bodydir_age);		 
  }
  if(result == true){
    DBLOG_MOV(0,_2D<<C2D(state.ball_pos.x,state.ball_pos.y,.1,"#000000"));
  }

  return result;

}

bool OneTwoHoldTurn::ballpos_optimal(const MyState &state){
  //return false; // currently deactivated takes too long
  double sqrdist = state.my_pos.sqr_distance(state.ball_pos);
  if(sqrdist >success_sqrdist)
    return false;
  if (state.ball_pos.sqr_distance(state.my_pos)< SQUARE(0.4)) // keep it before you
    return false;

  Angle delta_dir = (state.ball_pos - state.my_pos).arg();
  delta_dir = Tools::get_angle_between_mPI_pPI(delta_dir);
  if(delta_dir <-90./180.*PI)
    return false;
  if(delta_dir >90./180.*PI)
    return false;
  return true;
}

bool OneTwoHoldTurn::check_1stepturn2goal(Cmd & cmd, const MyState &state){
  MyState next_state;
  Cmd_Body turn_action;

  Angle turn2goal_angle = (look2pos-state.my_pos).arg()-state.my_angle.get_value();
  turn2goal_angle = Tools::get_angle_between_mPI_pPI(turn2goal_angle);
  turn2goal_angle=turn2goal_angle*(1.0+(WSinfo::me->inertia_moment*
					(state.my_vel.norm())));
  if (turn2goal_angle > 3.14) turn2goal_angle = 3.14;
  if (turn2goal_angle < -3.14) turn2goal_angle = -3.14;
  turn2goal_angle = Tools::get_angle_between_null_2PI(turn2goal_angle);
  turn_action.unset_lock();
  turn_action.set_turn(turn2goal_angle);

  Tools::get_successor_state(state,turn_action,next_state);

  if(ballpos_ok(next_state)){
    cmd.cmd_body.unset_lock();
    cmd.cmd_body.clone( turn_action );
    return true;
  }
  
  return false;

}


bool OneTwoHoldTurn::get_2step_cmd(Cmd & cmd, double &sqrdist,int & steps, const MyState &state){


  if(holdturn_not_possible_at == WSinfo::ws->time){
    DBLOG_MOV(0,"HoldTurn Already computed: now NOT possible");
    return false;
  }

  if(current_cmd_valid_at == WSinfo::ws->time){
    if(check_1stepturn2goal(cmd, state) == true){
      DBLOG_MOV(0,"HoldTurn(1step): Cmd computed, and 1 step IS possible");
      return true;
    }
    DBLOG_MOV(0,<<"HoldTurn: 2 step cmd already computed, just take it");
    cmd.cmd_body.unset_lock();
    cmd.cmd_body.clone(current_cmd.cmd_body);
    return true;
  }
  DBLOG_MOV(0,<<"HoldTurn: No precomputed cmd available -> start computation");

  bool result = compute_2step_cmd(cmd, sqrdist, steps, state);
  if(result == true){
    cmd.cmd_body.unset_lock();
    current_cmd.cmd_body.clone(cmd.cmd_body );
    DBLOG_MOV(0, << " current_cmd_valid_at auf " << WSinfo::ws->time << " gesetzt");
    current_cmd_valid_at = WSinfo::ws->time;
  }
  else
    holdturn_not_possible_at = WSinfo::ws->time;
  return result;
}

bool OneTwoHoldTurn::get_emergency_cmd(Cmd & cmd, const MyState &state){

  MyState next_state;
  Cmd_Body steptwo_action;
  HoldTurnItrActions itr_actions;

  bool safe_action_found = false;

  itr_actions.reset();

  Vector me_to_opp = state.op_pos - state.my_pos;
  me_to_opp.normalize();
  Vector target = state.my_pos - 0.9 * me_to_opp;
  double min_sqr_dist_to_target = 1000.0;

  while ( Cmd_Body * action = itr_actions.next() ) {
    Tools::get_successor_state(state,*action,next_state);
    //DBLOG_MOV(0,"2HoldBall: Test action "<<*action<<" successor "<<next_state);    
    
    if ( (next_state.ball_pos - target).sqr_norm() < min_sqr_dist_to_target ) {
      min_sqr_dist_to_target = (next_state.ball_pos - target).sqr_norm();
      steptwo_action = *action;
      safe_action_found = true;
    }

  } // while ...

  if (safe_action_found == true){ // successful!
    cmd.cmd_body.unset_lock();
    cmd.cmd_body.clone( steptwo_action );
    return true;
  }
  return false;
}



bool OneTwoHoldTurn::compute_2step_cmd(Cmd & cmd, double &sqrdist,int & steps, const MyState &state){

  MyState next_state;
  Cmd_Body steptwo_action;
  HoldTurnItrActions itr_actions;
  double best2step_sqrdist = 10.*success_sqrdist;
  double sqrdist_tmp;
  Cmd cmd_tmp;

  if(check_1stepturn2goal(cmd, state) == true){
    DBLOG_MOV(0,"HoldTurn(1step): IS possible, current dist: "
	    <<state.op_pos.distance(state.ball_pos));
    return true;
  }

  Cmd dummy_cmd;
  bool safe_action_found = false;

  itr_actions.reset();

  Vector me_to_opp = state.op_pos - state.my_pos;
  me_to_opp.normalize();
  Vector target = state.my_pos - 0.9 * me_to_opp;
  double min_sqr_dist_to_target = 1000.0;

  while ( Cmd_Body * action = itr_actions.next() ) {
    Tools::get_successor_state(state,*action,next_state);
    //DBLOG_MOV(0,"2HoldBall: Test action "<<*action<<" successor "<<next_state);    

    if ( relaxed_checking == true ) {
      if ( (next_state.ball_pos - target).sqr_norm() < min_sqr_dist_to_target ) {
	min_sqr_dist_to_target = (next_state.ball_pos - target).sqr_norm();
	steptwo_action = *action;
	safe_action_found = true;
      }
      continue;
    }

    if (ballpos_ok(next_state)) {  // next ballpos is safe
      if (check_1stepturn2goal(dummy_cmd, next_state) == true) {
	//DBLOG_MOV(0,"HoldTurn(2step): turn in step 2 possible");
	sqrdist_tmp = next_state.my_pos.sqr_distance(next_state.ball_pos);
	if(ballpos_optimal(next_state)){
	  DBLOG_MOV(0,"HoldTurn(2step): Optimal Ballpos Found - 2 step successful");
	  sqrdist = sqrdist_tmp;
	  steps = 2;
	  cmd.cmd_body.unset_lock();
	  cmd.cmd_body.clone( *action );
	  return true;
	}
	
	if((sqrdist_tmp < success_sqrdist ) &&  // current action is ok
	   (best2step_sqrdist > success_sqrdist)){  // no action was found yet
	  //DBLOG_MOV(0,"2HoldTurn: 2Step SuccessTest action "<<*action<<" successor "<<next_state);    
	  best2step_sqrdist = sqrdist_tmp;
	  steptwo_action = *action;
	}
	if((sqrdist_tmp > SQUARE (WSinfo::me->radius +.2)) &&
	   (sqrdist_tmp < best2step_sqrdist)){// current action ok and improves best action
	  //DBLOG_MOV(0,"2HoldTurn: 2Step Success Test action "<<*action<<" successor "<<next_state);    
	  best2step_sqrdist = sqrdist_tmp;
	  steptwo_action = *action;
	}
      } // I found an action that allows me to turn!
      else{ // no action found yet that allows to turn
	if(best2step_sqrdist >= success_sqrdist){ // no yet successful for turn, but safe 
	  //DBLOG_MOV(0,"HoldTurn(2step): Found save action");
	  steptwo_action = *action;
	  safe_action_found = true;
	}
      }
    }//ballpos safe
  } // while ...

  if ((best2step_sqrdist <success_sqrdist) || (safe_action_found == true)){ // successful!
    cmd.cmd_body.unset_lock();
    cmd.cmd_body.clone( steptwo_action );
    sqrdist = best2step_sqrdist;
    steps = 2;
    DBLOG_MOV(0,"HoldTurn(2step): IS possible, current dist: "
	      <<state.op_pos.distance(state.ball_pos));
    return true;
  }
  DBLOG_MOV(0,"HoldTurn(2step): NOT possible");
  return false;
}





#define MIN(X,Y)(X<Y?X:Y)

bool OneTwoHoldTurn::get_cmd(Cmd & cmd, const Vector tmp_look2pos, const bool relaxed_check){
  success_sqrdist = SQUARE(WSinfo::me->kick_radius - SAFETY_MARGIN);

  look2pos = tmp_look2pos;

  if(relaxed_check != relaxed_checking){  // different type of checking; reset
    DBLOG_MOV(0, << " current_cmd_valid_at auf " << WSinfo::ws->time << " gesetzt (2)");
    current_cmd_valid_at = -1;
    holdturn_not_possible_at = -1;
  }

  relaxed_checking = relaxed_check; // set internal variable. default is false -> check safely

  MyState state;
  
  state.my_pos = WSinfo::me->pos;
  state.my_vel = WSinfo::me->vel;
  state.ball_pos = WSinfo::ball->pos;
  state.ball_vel = WSinfo::ball->vel;
  state.my_angle = WSinfo::me->ang;

  PlayerSet opset;
  opset= WSinfo::valid_opponents;
  opset.keep_and_sort_closest_players_to_point(1, state.ball_pos);
  if ( opset.num ){
    state.op_pos= opset[0]->pos;
    state.op_bodydir =opset[0]->ang;
    state.op_bodydir_age = opset[0]->age_ang;
    state.op = opset[0];
    DBLOG_MOV(0,"get cmd. op number "<<state.op ->number);
  }
  else{
    state.op_pos= Vector(1000,1000); // outside pitch
    state.op_bodydir =ANGLE(0);
    state.op_bodydir_age = 1000;
    state.op = 0;
    DBLOG_MOV(0,"get cmd: Op. NOT set ");
  }

  double dist;
  int steps;
  bool ret = (get_2step_cmd(cmd, dist,steps, state));

  if ( ret == false ) {
    DBLOG_MOV(0, << "ERROR!!!!!!!!! get_emergency_cmd had to be called");
    ret = get_emergency_cmd(cmd, state);
  }

  if ( ret == false ) {
    DBLOG_MOV(0, << "ERROR!!!!!!!!! get_emergency_cmd didn't work");
  }

  return ret;

}

bool OneTwoHoldTurn::get_cmd_relaxed(Cmd & cmd){
  //DBLOG_MOV(0, "Hold_turn: get_cmd_relaxed called" << " time " << WSinfo::ws->time);
  Vector const opponent_goalpos(52.5,0.);
  return get_cmd(cmd, opponent_goalpos, true);
}

bool OneTwoHoldTurn::get_cmd(Cmd & cmd){
  //DBLOG_MOV(0, "Hold_turn: get_cmd1 called" << " time " << WSinfo::ws->time);
  Vector const opponent_goalpos(52.5,0.);
  return get_cmd(cmd, opponent_goalpos,false);
}

bool OneTwoHoldTurn::get_cmd(Cmd & cmd, ANGLE target_dir){
  //DBLOG_MOV(0, "Hold_turn: get_cmd2 called" << " time " << WSinfo::ws->time);
  Vector look2target;
  look2target.init_polar(10., target_dir.get_value());
  look2target += WSinfo::me->pos;
  return get_cmd(cmd, look2target, false);
}

