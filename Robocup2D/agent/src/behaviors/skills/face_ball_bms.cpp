#include "face_ball_bms.h"
#include "tools.h"
#include "options.h"
#include "ws_info.h"
#include "log_macros.h"
#include "intention.h"

#define LB 2

#define MAX_DIR_DIFF 20./180. * PI

bool FaceBall::initialized=false;

void FaceBall::turn_to_ball(bool lock) {
  LOG_MOV(0,<<"FaceBall: desired_dir is "<<RAD2DEG(desired_dir.get_value()));
  //  LOG_ERR(0,<<"FaceBall: desired_dir is "<<RAD2DEG(desired_dir.get_value()));

  Vector new_ball_pos = WSinfo::ball->pos + WSinfo::ball->vel;
  desired_dir = Tools::my_abs_angle_to(new_ball_pos);
  lock_neck = lock;
  turn_body = true;
}

void FaceBall::look_to_ball() {
  lock_neck = true;
  turn_body = false;
}

void FaceBall::turn_in_dir(ANGLE dir,bool lock) {
  lock_neck = lock;
  desired_dir = dir;
  turn_body = true;
}

void FaceBall::turn_to_point(Vector target,bool lock) {
  lock_neck = lock;
  desired_dir = Tools::my_abs_angle_to(target);
  turn_body = true;
}

bool FaceBall::get_cmd(Cmd & cmd){
  if(!WSinfo::is_ball_pos_valid()) {
    if(!search_ball->is_searching()) {
      LOG_MOV(LB+0,<<"FaceBall: Ball pos not valid, starting search...");
      search_ball->start_search();
    }
    bool res = search_ball->get_cmd(cmd);
    if(res) { // we need to turn to find the ball
      return false;
    }
  }
  if(cmd.cmd_body.is_cmd_set()) {
    // should never happen!
    ERROR_OUT << "\n cmd already set, FaceBall may not function properly!";
  }
  bool res = true;
  Vector new_ball_pos = WSinfo::ball->pos+WSinfo::ball->vel;
  ANGLE target_dir = desired_dir;
  ANGLE abs_to_ball = Tools::my_abs_angle_to(new_ball_pos);
  if(lock_neck) Tools::set_neck_request(NECK_REQ_FACEBALL);
  if(!turn_body) {
    LOG_MOV(LB+0,<<"FaceBall: Shall not turn body, only neck request is set.");
  } else {
    //ANGLE min_neck_ang = Tools::my_minimum_abs_angle_to_see();
    //ANGLE max_neck_ang = Tools::my_maximum_abs_angle_to_see();
    Angle max_turn_ang = PI/(1.0+WSinfo::me->inertia_moment*WSinfo::me->vel.norm());
    Angle min_turn_ang = PI/(1.0+WSinfo::me->inertia_moment*WSinfo::me->vel.norm());
    Angle turn_ang = (desired_dir - WSinfo::me->ang).get_value_mPI_pPI();
    if(turn_ang<0) {
      if(-turn_ang>min_turn_ang) turn_ang = -min_turn_ang;
    } else {
      if(turn_ang>max_turn_ang) turn_ang = max_turn_ang;
    }
    target_dir = WSinfo::me->ang + ANGLE(turn_ang);
    /* we always set a cmd here, since this is kind of an idle move... */
    //if(abs_turnangle(target_dir.get_value()) > MAX_DIR_DIFF) {
    //LOG_MOV(LB+0,<<"FaceBall: Turning body in desired direction!");
      double p_moment;
      p_moment = (target_dir - WSinfo::me->ang).get_value_mPI_pPI();
      p_moment = p_moment * (1.0 + (WSinfo::me->inertia_moment 
				    * (WSinfo::me->vel.norm())));
      if(fabs(p_moment)>3.14) {
	LOG_MOV(LB+0,<<"FaceBall: Cannot turn body in desired direction (inertia)!");
	res = false;
      }
      if (p_moment > 3.14) p_moment = 3.14;
      if (p_moment < -3.14) p_moment = -3.14;
      p_moment = Tools::get_angle_between_null_2PI(p_moment);
      cmd.cmd_body.set_turn(p_moment);
      LOG_MOV(LB+0,<<"FaceBall: Doing a corrected turn to " <<RAD2DEG(target_dir.get_value()));
      //} else {
      //LOG_MOV(LB+0,<<"FaceBall: Already facing in desired direction, no cmd will be set.");
      //}
  }
  if(!Tools::could_see_in_direction(abs_to_ball)) {
    LOG_MOV(LB+0,<<"FaceBall: WARNING: We won't be able to see the ball with this body angle!");
    return false;
  }
  return res;
}

bool FaceBall::get_cmd_turn_to_ball(Cmd &cmd, bool neck_req)
{
	this->turn_to_ball( neck_req );
	return this->get_cmd( cmd );
}

bool FaceBall::get_cmd_turn_to_point(Cmd &cmd, Vector &target, bool neck_req)
{
	this->turn_to_point( target, neck_req );
	return this->get_cmd( cmd );
}

float FaceBall::abs_turnangle(float angle){
  if(angle>PI)
    angle -= 2*PI;
  return(fabs(angle));
}
