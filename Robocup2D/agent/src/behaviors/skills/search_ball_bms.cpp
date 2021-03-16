#include "search_ball_bms.h"

#include "../../basics/Cmd.h"
#include "tools.h"
#include "options.h"
#include "ws_info.h"
#include "log_macros.h"
#include "intention.h"

/* Log Baselevel */
#define LB 1

bool SearchBall::initialized=false;

long SearchBall::time_of_turn;
long SearchBall::ms_time_of_last_sb;
bool SearchBall::turn_left;
int SearchBall::looked_to_ball;
int SearchBall::looked_left;
int SearchBall::looked_right;
ANGLE SearchBall::cur_vang_ball;
ANGLE SearchBall::cur_vang_left;
ANGLE SearchBall::cur_vang_right;
ANGLE SearchBall::next_vang_ball;
ANGLE SearchBall::next_vang_left;
ANGLE SearchBall::next_vang_right;
ANGLE SearchBall::last_updated_angle;
ANGLE SearchBall::my_angle_to_ball;
Vector SearchBall::new_ball_pos;

void SearchBall::start_search() {
  new_ball_pos=WSinfo::ball->pos+WSinfo::ball->vel;
  // ball pos is invalid (we wouldn't be searching the ball otherwise
  looked_to_ball=looked_left=looked_right=0;
  //LOG_MOV(LB+0,<<"SearchBall: Starting search!");
}

bool SearchBall::is_searching() {
  if(time_of_turn<WSinfo::ws->time-1 || WSinfo::is_ball_pos_valid()) {
    return false;
  }
  if(time_of_turn==WSinfo::ws->time) {
    if(  ms_time_of_last_sb
       < WSinfo::ws->ms_time_of_sb - (180*ServerOptions::slow_down_factor)) return false;
  }
  return true;
}

bool SearchBall::get_cmd(Cmd & cmd){
  if(WSinfo::is_ball_pos_valid()) {
    //LOG_MOV(LB+0,<<"SearchBall: Ball pos is valid, we won't do anything here!");
    return false;
  }
  got_update=(WSinfo::ws->view_quality==Cmd_View::VIEW_QUALITY_HIGH &&
	      WSinfo::ws->ms_time_of_see >= WSinfo::ws->ms_time_of_sb);
  if(!got_update) {
    //LOG_MOV(LB+0,<<"Face_Ball: Got no see update!");
  }
  if(!is_searching()) {
    start_search();
  }
  bool consider_turn_rand=false;
  ANGLE desired_turn= ANGLE(0);
  //LOG_MOV(LB+0,<<"SearchBall: I don't know where the ball is! time_of_turn: "<<time_of_turn
  //	  <<" last update: "<<WSinfo::ws->time_of_last_update);

  /* If we were in low quality view mode, turn to last ball pos first! */
  if(Tools::get_next_view_quality()==Cmd_View::VIEW_QUALITY_HIGH && 
     //Tools::get_last_lowq_cycle()==WSinfo::ws->time-1) {
     WSinfo::ws->view_quality==Cmd_View::VIEW_QUALITY_LOW) {
    desired_turn=Tools::my_angle_to(Tools::get_last_known_ball_pos());
    //LOG_MOV(LB+0,<<"SearchBall: Looking to last ball pos in highq view mode!");
    consider_turn_rand=false;
    //Tools::set_neck_request(NECK_REQ_LOOKINDIRECTION,
    //			    Tools::my_abs_angle_to(Tools::get_last_known_ball_pos()));
  } else {
    if(!got_update) {
      //LOG_MOV(LB+0,<<"SearchBall: No see update -> no turn!");
      consider_turn_rand=false;
    } else {
      if(looked_to_ball<2) {
	if(!looked_to_ball) {
	  if(got_update &&
	     fabs(Tools::my_neck_angle_to(new_ball_pos).get_value_mPI_pPI())
	     < .3*Tools::cur_view_angle_width().get_value()) {
	    //LOG_MOV(LB+0,<<"SearchBall: Ball disappeared from my view angle!");
	    cur_vang_ball=Tools::cur_view_angle_width();
	    my_angle_to_ball=WSinfo::me->neck_ang;
	    looked_to_ball=2;
	  } else {
	    //LOG_MOV(LB+0,<<"SearchBall: Looking to last known ball position...");
	    desired_turn=Tools::my_angle_to(Tools::get_last_known_ball_pos());
	    cur_vang_ball=Tools::cur_view_angle_width();
	    looked_to_ball=1;
	  }
	} else if(got_update && looked_to_ball==1
		  && fabs(Tools::my_neck_angle_to(Tools::get_last_known_ball_pos()).get_value_mPI_pPI())
		  < .5*Tools::cur_view_angle_width().get_value())  {
	  my_angle_to_ball=WSinfo::me->ang;
	  looked_to_ball=2;
	  cur_vang_ball=Tools::cur_view_angle_width();
	} else { // no see update, better turn again if necessary...
	  desired_turn=Tools::my_angle_to(Tools::get_last_known_ball_pos());
	}
      }
      if(looked_to_ball==2 && looked_left<2) {
	if(got_update && looked_left==1
	   && (Tools::cur_view_angle_width().get_value()>=next_vang_left.get_value())) {
	  // missed change_view?
	  looked_left=2;
	  cur_vang_left=Tools::cur_view_angle_width();
	} else {
	  next_vang_left=Tools::next_view_angle_width();
	  desired_turn=my_angle_to_ball-WSinfo::me->ang+ANGLE(.5*(cur_vang_ball + next_vang_left).get_value());
	  looked_left=1; 
	}
      }
      if(looked_left==2 && looked_right<2) {
	if(got_update && looked_right==1
	   && (Tools::cur_view_angle_width().get_value()>=next_vang_right.get_value())) {
	  // missed change_view?
	  looked_right=2;
	  cur_vang_right=Tools::cur_view_angle_width();
	  last_updated_angle=WSinfo::me->ang;
	} else {
	  next_vang_right=Tools::next_view_angle_width();
	  desired_turn=my_angle_to_ball-WSinfo::me->ang-ANGLE(.5*(cur_vang_ball + next_vang_right).get_value());
	  looked_right=1; 
	}
      }
      if(looked_right==2) {  // ball still not found, starting turning right...
	//LOG_MOV(LB+0,<<"SearchBall: Turning right to find ball!");
	if(got_update && (Tools::cur_view_angle_width().get_value()>=next_vang_right.get_value())) {
	  last_updated_angle=WSinfo::me->ang;
	  cur_vang_right=Tools::cur_view_angle_width();
	} else {
	  if(got_update) {
	    //LOG_MOV(LB+0,<<"SearchBall: Missed a change_view!?");
	  } else {
	    //LOG_MOV(LB+0,<<"SearchBall: Missed a see message!");
	  }
	}
	next_vang_right=Tools::next_view_angle_width();
	desired_turn=last_updated_angle-WSinfo::me->ang-ANGLE(.5*(next_vang_right+cur_vang_right).get_value());
	//Tools::set_neck_request(NECK_REQ_SCANFORBALL);  // neck lock
      }
      //LOG_MOV(LB+0,<<"SearchBall: desired_turn is "<<RAD2DEG(desired_turn.get_value()));
      consider_turn_rand=true;
      //Tools::set_neck_request(NECK_REQ_SCANFORBALL);
      //Tools::set_neck_request(NECK_REQ_LOOKINDIRECTION,desired_turn+WSinfo::me->ang);
    }
  }
  Tools::set_neck_request(NECK_REQ_SCANFORBALL);
  time_of_turn = WSinfo::ws->time;
  ms_time_of_last_sb=WSinfo::ws->ms_time_of_sb;
  double p_moment=desired_turn.get_value_mPI_pPI();
  if(p_moment != 0) {
    if(p_moment <PI)
      turn_left = true;
    else
      turn_left = false;
    // do a corrected turn, code taken from move_basics.c
    p_moment = p_moment * (1.0 + (WSinfo::me->inertia_moment * WSinfo::me->vel.norm()));
    if(consider_turn_rand) {
      p_moment*=(1-ServerOptions::player_rand);
    }
    if (p_moment > 3.14) p_moment = 3.14;
    if (p_moment < -3.14) p_moment = -3.14;
    p_moment = Tools::get_angle_between_null_2PI(p_moment);
    //LOG_MOV(0,<<"p_moment: "<<RAD2DEG(p_moment));
    cmd.cmd_body.set_turn(p_moment);
    //LOG_MOV(LB+0,<<"SearchBall - doing a corrected turn to " <<RAD2DEG(desired_turn.get_value()));
    //time_of_turn = WSinfo::ws->time;
    return true;
  } else {
    if(!got_update) {
      cmd.cmd_body.set_turn(0);
      return true;
    }
    //LOG_MOV(LB+0,<<"SearchBall: No turn needed, cmd not set");
    return false;
  }
}
