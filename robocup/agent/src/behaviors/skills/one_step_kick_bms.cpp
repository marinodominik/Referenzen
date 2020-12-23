#include "one_step_kick_bms.h"
#define BASELEVEL 30

#define PLAYERSIZE_MARGIN .085  // prevent collisions

bool OneStepKick::initialized= false;
OneStepKick *OneStepKick::onestepkick;

//#define DBLOG_MOV(LLL,XXX) LOG_POL(LLL,<<"1StepKick: "<<XXX)
#define DBLOG_MOV(LLL,XXX) 
//#define DBLOG_DRAW(LLL,XXX) LOG_POL(LLL,<<_2D<<XXX)
#define DBLOG_DRAW(LLL,XXX) 

/********************************************************************/
/* Init functions                                                   */
/********************************************************************/

void OneStepKick::kick_in_dir_with_initial_vel(double vel,const ANGLE &dir) {
  target_dir = dir;
  target_vel = vel;
  kick_to_pos = false;
  set_in_cycle = WSinfo::ws->time;calc_done=false;
  DBLOG_MOV(2,"SET: kick in dir "<<RAD2DEG(dir.get_value())<<" w. init. speed "<<vel);
}

void OneStepKick::kick_in_dir_with_max_vel(const ANGLE &dir) {
  kick_in_dir_with_initial_vel(get_max_vel_in_dir(dir),dir);
  DBLOG_MOV(2,"SET: kick in dir "<<RAD2DEG(dir.get_value())<<" w. max speed ");
}

void OneStepKick::kick_to_pos_with_initial_vel(double vel,const Vector &pos) {
  MyState state = get_cur_state();
  target_dir = (pos - state.ball_pos).ARG();
  target_vel = vel;
  target_pos = pos;
  kick_to_pos = true;
  set_in_cycle = WSinfo::ws->time;calc_done=false;
  DBLOG_MOV(2,"SET: kick to pos "<<pos<<"(dir "<<RAD2DEG(target_dir.get_value())<<") w. init. speed "<<vel);
}

void OneStepKick::kick_to_pos_with_final_vel(double vel,const Vector &pos) {
  MyState state = get_cur_state();
  target_dir = (pos - state.ball_pos).ARG();
  target_vel = (1-ServerOptions::ball_decay)*((pos-state.ball_pos).norm()+vel*ServerOptions::ball_decay)
    + ServerOptions::ball_decay * vel;
  double max_vel = get_max_vel_to_pos(pos,false);
  if(target_vel>max_vel) {
    target_vel=max_vel;
    if(!do_not_log) {
      LOG_ERR(0,<<"OneStepKick: Point "<<pos<<" too far away, using max vel ("<<max_vel<<")!");
    }
  }
  target_pos = pos;
  kick_to_pos = true;
  set_in_cycle = WSinfo::ws->time;calc_done=false;
  DBLOG_MOV(2,"SET: kick to pos "<<pos<<"(dir "<<RAD2DEG(target_dir.get_value())<<") w. final speed "<<vel);

}

void OneStepKick::kick_to_pos_with_max_vel(const Vector &pos) {
  kick_to_pos_with_initial_vel(get_max_vel_to_pos(pos),pos);
  DBLOG_MOV(2,"SET: kick to pos "<<pos<<" w. max speed ");
}

/**********************************************************************/

void OneStepKick::get_ws_state(MyState &state) {
  state.my_pos = WSinfo::me->pos;
  state.my_vel = WSinfo::me->vel;
  state.my_angle = WSinfo::me->ang;
  state.ball_pos = WSinfo::ball->pos;
  state.ball_vel = WSinfo::ball->vel;

  PlayerSet pset= WSinfo::valid_opponents;
  pset.keep_and_sort_closest_players_to_point(1, state.ball_pos);
  if ( pset.num ){
    state.op_pos= pset[0]->pos;
    state.op_bodydir =pset[0]->ang;
    state.op_bodydir_age = pset[0]->age_ang;
    state.op = pset[0];
  }
  else{
    state.op_pos= Vector(1000,1000); // outside pitch
    state.op_bodydir = ANGLE(0);
    state.op_bodydir_age = 1000;
    state.op = 0;
  }

}  

void OneStepKick::set_state(const Vector &mypos,const Vector &myvel,const ANGLE &myang,
			    const Vector &ballpos,const Vector &ballvel,
			    const Vector &op_pos, 
			    const ANGLE &op_bodydir,
			    const int op_bodydir_age,
			    const PPlayer op){
  fake_state.my_pos = mypos;
  fake_state.my_vel = myvel;
  fake_state.my_angle = myang;
  fake_state.ball_pos = ballpos;
  fake_state.ball_vel = ballvel;
  fake_state_time = WSinfo::ws->time;
  fake_state.op_pos= op_pos;
  fake_state.op_bodydir =op_bodydir;
  fake_state.op_bodydir_age = op_bodydir_age;
  fake_state.op = op;

  set_in_cycle = -1;
}

void OneStepKick::reset_state() {
  fake_state_time = -1;
  set_in_cycle = -1;
}

MyState OneStepKick::get_cur_state() {
  MyState cur_state;
  if(fake_state_time == WSinfo::ws->time) {
    cur_state = fake_state;
  } else {
    get_ws_state(cur_state);
  }
  return cur_state;
}

/* static version */
bool OneStepKick::is_pos_ok(const Vector &mypos,const Vector &ballpos) {
  MyState state;
  state.my_pos=mypos;
  return onestepkick->is_pos_ok(state,ballpos);
}

bool OneStepKick::is_pos_ok(const Vector &pos) {
  MyState state = get_cur_state();
  return is_pos_ok(state,pos);
}

bool OneStepKick::is_pos_ok(const MyState &state,const Vector &pos) {
  Vector my_next_pos = state.my_pos+state.my_vel;

  if (WSinfo::ws->play_mode != PM_PlayOn) //TG08: be more cautious in stdsits
    if(my_next_pos.distance(pos) <= WSinfo::me->radius+2.5*PLAYERSIZE_MARGIN) return false;

  if(my_next_pos.distance(pos) <= WSinfo::me->radius+PLAYERSIZE_MARGIN) return false;

  compute_critical_opponents(state);
  for(int p=0;p<crit_opp_num;p++)
  {
    int minInact, maxInact;
    WSinfo::get_player_inactivity_interval( crit_opp[p].player, minInact, maxInact );
    if (   crit_opp[p].pos.distance(pos) <= crit_opp[p].radius
        && minInact < 1 )
      return false;
  }

  // ridi 06: test only: be risky
  //  return true;

  // too be very cautious: check where closest opponent might be in next step
  bool result;
  if(crit_opp_num>0)
    result = Tools::is_ballpos_safe(crit_opp[0].player,pos,false);
  else
  {
    int minInact = 0, maxInact = 0;
    if ( state.op ) WSinfo::get_player_inactivity_interval( state.op, minInact, maxInact );
    if (minInact == 0)
      result = Tools::is_ballpos_safe(state.op_pos,state.op_bodydir,pos, state.op_bodydir_age);
    else
      result = true;
  }
  return result;
}

void OneStepKick::compute_critical_opponents(const MyState &state) {
  if(crit_opp_time == WSinfo::ws->time) {
    if(state.my_pos.distance(crit_opp_basepos) < .1) return;
  }
  PlayerSet pset = WSinfo::valid_opponents;
  pset.keep_players_in_circle(state.my_pos,
			      WSinfo::me->kick_radius + ServerOptions::ball_speed_max
			      + ServerOptions::catchable_area_l + 2);
  for(int p=0;p<pset.num;p++) {
    crit_opp[p].pos = pset[p]->pos;
    crit_opp[p].player = pset[p];
    if(pset[p]->number == WSinfo::ws->his_goalie_number)
      crit_opp[p].radius = ServerOptions::catchable_area_l * 1.1;
    else
      crit_opp[p].radius = pset[p]->kick_radius * 1.2;
  }
  crit_opp_num = pset.num;
  crit_opp_time = WSinfo::ws->time;
  crit_opp_basepos = state.my_pos;
}

double OneStepKick::get_kick_decay(const MyState &state) {
  Vector tmp = state.ball_pos - state.my_pos;
  tmp.rotate(-state.my_angle.get_value()); //normalize to own dir
  tmp.setY( fabs(tmp.getY()) );
  double decay=1.0-tmp.arg()/(4.0*PI) -
    (tmp.norm()- ServerOptions::ball_size - WSinfo::me->radius)/
    (4.0* (WSinfo::me->kick_radius-WSinfo::me->radius-ServerOptions::ball_size));
  return decay;
}

double OneStepKick::get_max_acc_vel(const MyState &state) {
  double decay = get_kick_decay(state);
  return get_max_acc_vel(decay);
}

double OneStepKick::get_max_acc_vel(double decay) {
  double max_acc_vel=ServerOptions::maxpower*ServerOptions::kick_power_rate*decay;
  if (max_acc_vel > ServerOptions::ball_accel_max) {
    if (max_acc_vel > ServerOptions::ball_accel_max + 0.1 )
      if(!do_not_log) {
	LOG_ERR(0, << "OneStepKick: WARNING: reducing max_acc_vel= " << max_acc_vel
		<< " to " << ServerOptions::ball_accel_max);
      }
    max_acc_vel=ServerOptions::ball_accel_max;
  }
  return max_acc_vel;
}

double OneStepKick::get_max_vel_to_pos(const Vector &pos,bool check_pos) {
  return onestepkick->nonstatic_get_max_vel_to_pos(pos,check_pos);
}

double OneStepKick::get_max_vel_in_dir(const ANGLE &dir,bool check_pos) {
  return onestepkick->nonstatic_get_max_vel_in_dir(dir,check_pos);
}

double OneStepKick::nonstatic_get_max_vel_to_pos(const Vector &pos,bool check_pos) {
  MyState state=get_cur_state();
  Vector tmp=pos-state.ball_pos;
  ANGLE dir=tmp.ARG();
  return get_max_vel_in_dir(dir,check_pos);
}
  
double OneStepKick::nonstatic_get_max_vel_in_dir(const ANGLE &dir,bool check_pos) {
  MyState state=get_cur_state();
  double maxvel=get_max_vel_in_dir(state,dir);
  if(check_pos) {
    Vector intended_vel_normalized;
    intended_vel_normalized.init_polar(1.0,dir.get_value());
    const Vector new_ball_pos = state.ball_pos+maxvel*intended_vel_normalized;
    if(!is_pos_ok(state,new_ball_pos)) maxvel = 0;
  }
  return maxvel;
}

double OneStepKick::get_max_vel_in_dir(const MyState &state,const ANGLE &dir) {
  const Vector ball_pos_next = state.ball_pos+state.ball_vel;
  Vector intended_vel_normalized;
  intended_vel_normalized.init_polar(1.0,dir.get_value());
  const double max_acc_vel= get_max_acc_vel(state);

  double factor;
  { //intersect circle (with center in ball_pos_next and radius max_acc_vel) with 
    //ball_pos + factor * intended_vel_normalized.
    const Vector phi=state.ball_pos - ball_pos_next;
    double q=intended_vel_normalized.getX() * phi.getX() + intended_vel_normalized.getY() * phi.getY();
    double xxx=max_acc_vel*max_acc_vel - phi.getX()*phi.getX() - phi.getY()*phi.getY() + q*q;
    if (xxx<0.0) return 0;
    xxx=sqrt(xxx);factor=-q+ xxx;
    if (factor<0.0) return 0.0;
    
    if (factor > ServerOptions::ball_speed_max) {
      if(!do_not_log) {
	LOG_MOV(BASELEVEL+0, << "OneStepKick MAYBE UNSUCCESSFUL in dir= " << RAD2DEG(dir.get_value())
		<<" factor: "<<factor
		<<" q "<< q <<" phi "<< phi<<" xxx "<<xxx
		<<" intended "<<intended_vel_normalized);
      }
      factor = ServerOptions::ball_speed_max;
    }
  }
  return factor;
}

bool OneStepKick::nonstatic_can_keep_ball_in_kickrange() {
  const MyState state=get_cur_state();
  const double max_acc_vel = get_max_acc_vel(state);
  const Vector my_next_pos = state.my_pos + state.my_vel;
  /* consider own vel -> ball needs to be in kickrange in next cycle! */
  if(my_next_pos.distance(state.ball_pos+state.ball_vel)
     <= (max_acc_vel+WSinfo::me->kick_radius) * 0.95) {
    return true;
  }
  return false;
}

bool OneStepKick::can_keep_ball_in_kickrange() {
  return onestepkick->nonstatic_can_keep_ball_in_kickrange();
}

void OneStepKick::kick_vector_to_pwr_and_dir(const MyState &state,const Vector &kick_vec,
					     double &res_power, ANGLE &res_dir) {
  double kick_decay=get_kick_decay(state);
  res_power=kick_vec.norm()/(ServerOptions::kick_power_rate * kick_decay);

  if(res_power>ServerOptions::maxpower) {
    if(res_power>ServerOptions::maxpower+1.0) {
      if(!do_not_log) {
	LOG_ERR(0,<<"OneStepKick: WARNING: power = " << res_power << " > maxpower = " 
		<< ServerOptions::maxpower);
      }
    }
    res_power=ServerOptions::maxpower;
  }
  res_dir=kick_vec.ARG();
  res_dir-=state.my_angle;
}

/* Computes needed kick power and kick angle          */
/* Returns false if the resulting ball pos is not ok. */
/* (in spite of that, cmd is set, so you can use it!) */

bool OneStepKick::calculate(const MyState &state,double vel,const ANGLE &dir,
			    Cmd_Body &res_cmd,double &res_vel) {
  //if(!initialized) {
  //  ERROR_OUT << "\nOneStepKick_bms not initialized!";
  //  return false;
  //}
  const Vector ball_pos_next= state.ball_pos+state.ball_vel;
  Vector intended_vel;
  double max_vel_in_dir= get_max_vel_in_dir(state,dir);
  if(vel>max_vel_in_dir) {
#if 1
    if(vel>max_vel_in_dir+.1) {
      if(!do_not_log) {
	LOG_MOV(BASELEVEL+0,<<"OneStepKick: WARNING: Can't achieve desired vel "<<vel
		<<", max_vel_in_dir "<<max_vel_in_dir<<"!");
      }
    }
#endif
    vel=get_max_vel_in_dir(state,dir);
  }
  intended_vel.init_polar(vel,dir.get_value());
  const Vector kick_vec=state.ball_pos+intended_vel-ball_pos_next;
  double res_pwr;ANGLE res_ang;
  kick_vector_to_pwr_and_dir(state,kick_vec,res_pwr,res_ang);
  res_cmd.unset_lock();
  res_cmd.unset_cmd();  
  res_cmd.set_kick(res_pwr,res_ang.get_value_mPI_pPI());
  calc_done=true;
  Vector newpos=state.ball_pos+intended_vel;
  res_vel=vel;
  if(!is_pos_ok(state,newpos)) {
    return false;
  }
  return true;
}

bool OneStepKick::get_cmd(Cmd &cmd) {
  if(WSinfo::ws->time!=set_in_cycle) {
    ERROR_OUT << "\nOneStepKick::get_cmd() called without prior initialization!";
    return false;
  }
  MyState state=get_cur_state();
  if(!calc_done) {
    //MyState state=get_cur_state();
    result_status=calculate(state,target_vel,target_dir,result_cmd,result_vel);
  }
#if 1  
#if LOGGING && BASIC_LOGGING
  double x1=state.ball_pos.getX();
  double x2=state.ball_pos.getY();
#endif
  Vector target;
  target.init_polar(target_vel * 1./(1-ServerOptions::ball_decay),target_dir.get_value());
#if LOGGING && BASIC_LOGGING
  double x3 = x1 + target.getX();
  double x4 = x2 + target.getY();
#endif
  if(!do_not_log) {
    LOG_MOV(BASELEVEL+1,<< _2D << L2D(x1,x2,x3,x4,"blue"));
  }
#endif
  if(!do_not_log) {
    LOG_MOV(BASELEVEL+0,<< "OneStepKick: Kick with initial vel "<<target_vel
	    <<" in dir "<<RAD2DEG(target_dir.get_value()));
  }
  
  cmd.cmd_body.clone(result_cmd);
  return result_status;
}

bool OneStepKick::get_vel(double &vel) {
  if(WSinfo::ws->time!=set_in_cycle) {
    ERROR_OUT << "\nOneStepKick::get_vel() called without prior initialization!";
    return false;
  }
  if(!calc_done) {
    MyState state = get_cur_state();
    result_status=calculate(state,target_vel,target_dir,result_cmd,result_vel);
  }
  vel=result_vel;
  return result_status;
}

double OneStepKick::get_vel() {
  double vel;
  bool dum=get_vel(vel);
  if(!dum) vel=0;
  return vel;
}

void OneStepKick::set_log( bool do_log ) // set_log(false) disables all logging
{
    do_not_log = !do_log;
}

OneStepKick::OneStepKick()
{
    set_in_cycle = -1;
    crit_opp_time = -1;
    fake_state_time = -1;
    do_not_log = false;
}

OneStepKick::~OneStepKick()
{

}
bool OneStepKick::init( char const * conf_file, int argc, char const* const * argv )
{
    if( initialized )
        return initialized;
    initialized = true;
    onestepkick = new OneStepKick(); // init static dummy object
//    cout << "\nOneStepKick behavior initialized.";
    return initialized;
}
