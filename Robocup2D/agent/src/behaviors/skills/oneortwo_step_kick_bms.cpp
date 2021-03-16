#include "oneortwo_step_kick_bms.h"
#define BASELEVEL 3

#define KICKRANGE_MARGIN .15

bool OneOrTwoStepKick::initialized=false;

// JTS initialize the constants
const double OneOrTwoStepKickItrActions::kick_pwr_min = 20;
const double OneOrTwoStepKickItrActions::kick_pwr_inc = 10;
const double OneOrTwoStepKickItrActions::kick_pwr_max = 100;
const double OneOrTwoStepKickItrActions::kick_ang_min = 0;
const double OneOrTwoStepKickItrActions::kick_ang_inc = 2*PI/8.;
  //  static const double kick_ang_inc = 2*PI/36.; // ridi: I think it should be as much! too much
const double OneOrTwoStepKickItrActions::kick_ang_max = 2*PI-kick_ang_inc;
const double OneOrTwoStepKickItrActions::dash_pwr_min = 20;
const double OneOrTwoStepKickItrActions::dash_pwr_inc = 20;
const double OneOrTwoStepKickItrActions::dash_pwr_max = 100;
const double OneOrTwoStepKickItrActions::turn_ang_min = 0;
const double OneOrTwoStepKickItrActions::turn_ang_inc = 2*PI/18.;
  //  static const double turn_ang_max = 2*PI-turn_ang_inc;
const double OneOrTwoStepKickItrActions::turn_ang_max = 0;  // ridi: do not allow turns

const double OneOrTwoStepKickItrActions::kick_pwr_steps = (kick_pwr_max-kick_pwr_min)/kick_pwr_inc + 1;
const double OneOrTwoStepKickItrActions::dash_pwr_steps = (dash_pwr_max-dash_pwr_min)/dash_pwr_inc + 1;
const double OneOrTwoStepKickItrActions::turn_ang_steps = (turn_ang_max-turn_ang_min)/turn_ang_inc + 1;
const double OneOrTwoStepKickItrActions::kick_ang_steps = (kick_ang_max-kick_ang_min)/kick_ang_inc + 1;

//#define DBLOG_MOV(LLL,XXX) LOG_POL(LLL,<<"2StepKick: "<<XXX)
#define DBLOG_MOV(LLL,XXX) 
//#define DBLOG_DRAW(LLL,XXX) LOG_POL(LLL,<<_2D<<XXX)
#define DBLOG_DRAW(LLL,XXX) 

void OneOrTwoStepKickItrActions::reset()
{
    kick_pwr_done = 0;
    kick_ang_done = 0;
    dash_pwr_done = 0;
    turn_ang_done = 0;
    kick_pwr = kick_pwr_min;
    kick_ang = ANGLE( kick_ang_min );
    dash_pwr = dash_pwr_min;
    turn_ang = ANGLE( turn_ang_min );
}

Cmd_Body* OneOrTwoStepKickItrActions::next() {
if(kick_pwr_done<kick_pwr_steps && kick_ang_done<kick_ang_steps) {
  action.unset_lock();
  action.unset_cmd();
  action.set_kick(kick_pwr,kick_ang.get_value_mPI_pPI());
  kick_ang+= ANGLE(kick_ang_inc);
  if(++kick_ang_done>=kick_ang_steps) {
kick_ang=ANGLE(kick_ang_min);
kick_ang_done=0;
kick_pwr+=kick_pwr_inc;
kick_pwr_done++;
  }
  return &action;
}
if(dash_pwr_done<dash_pwr_steps) {
  action.unset_lock();
  action.unset_cmd();
  action.set_dash(dash_pwr);
  dash_pwr+=dash_pwr_inc;
  dash_pwr_done++;
  return &action;
}
if(turn_ang_done<turn_ang_steps) {
  action.unset_lock();
  action.unset_cmd();
  action.set_turn(turn_ang);
  turn_ang+= ANGLE(turn_ang_inc);
  turn_ang_done++;
  return &action;
}
return NULL;
}

bool OneOrTwoStepKick::can_keep_ball_in_kickrange(){
  return onestepkick->can_keep_ball_in_kickrange();
}


/********************************************************************/
/* Init functions                                                   */
/********************************************************************/

void OneOrTwoStepKick::kick_in_dir_with_initial_vel(double vel,const ANGLE &dir) {
  target_dir = dir;
  target_vel = vel;
  kick_to_pos = false;
  set_in_cycle = WSinfo::ws->time;calc_done=false;
  DBLOG_MOV(2,"SET: kick in dir "<<RAD2DEG(dir.get_value())<<" w. init. speed "<<vel);
}

void OneOrTwoStepKick::kick_in_dir_with_max_vel(const ANGLE &dir) {
  kick_in_dir_with_initial_vel(ServerOptions::ball_speed_max,dir);
  DBLOG_MOV(2,"SET: kick in dir "<<RAD2DEG(dir.get_value())<<" w. max speed ");

}

void OneOrTwoStepKick::kick_to_pos_with_initial_vel(double vel,const Vector &pos) {
  MyState state = get_cur_state();
  target_dir = (pos - state.ball_pos).ARG();
  target_vel = vel;
  target_pos = pos;
  kick_to_pos = true;
  set_in_cycle = WSinfo::ws->time;calc_done=false;
  DBLOG_MOV(2,"SET: kick to pos "<<pos<<"(dir "<<RAD2DEG(target_dir.get_value())<<") w. init. speed "<<vel);
}

void OneOrTwoStepKick::kick_to_pos_with_final_vel(double vel,const Vector &pos) {
  MyState state = get_cur_state();
  target_dir = (pos - state.ball_pos).ARG();
  target_vel = (1-ServerOptions::ball_decay)*((pos-state.ball_pos).norm()+vel*ServerOptions::ball_decay)
    + ServerOptions::ball_decay * vel;
  double max_vel = ServerOptions::ball_speed_max;
  if(target_vel>max_vel) {
    target_vel=max_vel;
    LOG_ERR(0,<<"OneOrTwoStepKick: Point "<<pos<<" too far away, using max vel ("<<max_vel<<")!");
  }
  target_pos = pos;
  kick_to_pos = true;
  set_in_cycle = WSinfo::ws->time;calc_done=false;
  DBLOG_MOV(2,"SET: kick to pos "<<pos<<"(dir "<<RAD2DEG(target_dir.get_value())<<") w. final speed "<<vel);
}

void OneOrTwoStepKick::kick_to_pos_with_max_vel(const Vector &pos) {
  kick_to_pos_with_initial_vel(ServerOptions::ball_speed_max,pos);
  DBLOG_MOV(2,"SET: kick to pos "<<pos<<" w. max speed ");
}

/**********************************************************************/

void OneOrTwoStepKick::get_ws_state(MyState &state) {
  state.my_pos = WSinfo::me->pos;
  state.my_vel = WSinfo::me->vel;
  state.my_angle = WSinfo::me->ang;
  state.ball_pos = WSinfo::ball->pos;
  state.ball_vel = WSinfo::ball->vel;

  PlayerSet pset= WSinfo::valid_opponents;
  pset.keep_and_sort_closest_players_to_point(1, state.ball_pos);
  if ( pset.num ){
    state.op_pos = pset[0]->pos;
    state.op_bodydir = pset[0]->ang;
    state.op_bodydir_age = pset[0]->age_ang;
  }
  else{
    state.op_pos = Vector(1000,1000); // outside pitch
    state.op_bodydir = ANGLE(0);
    state.op_bodydir_age = 1000;
  }

}  

/*
  void OneOrTwoStepKick::print_state() {
  MyState state =  get_cur_state();
  LOG_DEB(0, << " my_pos, my_vel, my_angle: [" << state.my_pos << "] [" << state.my_vel << "] [" << state.my_angle << "]");
  LOG_DEB(0, << " ball_pos, ball_vel: [" << state.ball_pos << "] [" << state.ball_vel << "]");
  LOG_DEB(0, << " op_pos, op_bodydir, op_bodydir_age: [" << state.op_pos << "] [" << state.op_bodydir << "] [" << state.op_bodydir_age << "]");
  }*/


void OneOrTwoStepKick::set_state( const AState &state )
{
    fake_state.my_pos = state.my_team[ state.my_idx ].pos;
    fake_state.my_vel = state.my_team[ state.my_idx ].vel;
    fake_state.my_angle = ANGLE( state.my_team[ state.my_idx ].body_angle );
    fake_state.ball_pos = state.ball.pos;
    fake_state.ball_vel = state.ball.vel;

    double minimum = 10000.0;
    int min_idx = -1;

    for( int i = 0; i < 11; i++ )
    {
        if( !state.op_team[ i ].valid )
            continue;
        if( ( state.op_team[ i ].pos - state.ball.pos ).sqr_norm() < minimum )
        {
            min_idx = i;
            minimum = ( state.op_team[ i ].pos - state.ball.pos ).sqr_norm();
        }
    }

    if( min_idx != -1 )
    {
        fake_state.op_pos = state.op_team[ min_idx ].pos;
        fake_state.op_bodydir = ANGLE( state.op_team[ min_idx ].body_angle );
        fake_state.op_bodydir_age = state.op_team[ min_idx ].age;
    }
    else
    {
        fake_state.op_pos = Vector( 1000, 1000 ); // outside pitch
        fake_state.op_bodydir = ANGLE( 0 );
        fake_state.op_bodydir_age = 1000;
    }

    fake_state_time = WSinfo::ws->time;
    set_in_cycle = -1;

}


void OneOrTwoStepKick::set_state(const Vector &mypos,const Vector &myvel,const ANGLE &myang,
				 const Vector &ballpos,const Vector &ballvel,
				 const Vector &op_pos, 
				 const ANGLE &op_bodydir,
				 const int op_bodydir_age
				 ){
  fake_state.my_pos = mypos;
  fake_state.my_vel = myvel;
  fake_state.my_angle = myang;
  fake_state.ball_pos = ballpos;
  fake_state.ball_vel = ballvel;
  fake_state.op_pos = op_pos;
  fake_state.op_bodydir = op_bodydir;
  fake_state.op_bodydir_age = op_bodydir_age ;

  fake_state_time = WSinfo::ws->time;
  set_in_cycle = -1;
}

void OneOrTwoStepKick::reset_state() {
  fake_state_time = -1;
  set_in_cycle = -1;
}

MyState OneOrTwoStepKick::get_cur_state() {
  MyState cur_state;
  if(fake_state_time == WSinfo::ws->time) {
    cur_state = fake_state;
  } else {
    get_ws_state(cur_state);
  }
  return cur_state;
}

#define CALCLOG
bool OneOrTwoStepKick::calculate(const MyState &state,double vel,const ANGLE &dir,const Vector &pos,
				 bool to_pos,Cmd_Body &res_cmd1,double &res_vel1,
				 Cmd_Body &res_cmd2,double &res_vel2,bool &need_2step) {

  calc_done = true;
  Cmd cmd1step;
  onestepkick->reset_state();
  if(to_pos) {
    onestepkick->kick_to_pos_with_initial_vel(vel,pos);
  } else {
    onestepkick->kick_in_dir_with_initial_vel(vel,dir);
  }
  if(onestepkick->get_cmd(cmd1step)) onestepkick->get_vel(res_vel1);
  else res_vel1=0; // pos not ok! (res_cmd1 is set nevertheless...)
  res_cmd1=cmd1step.cmd_body;
  
  if(fabs(res_vel1-vel) < 0.05) {
#ifdef CALCLOG
    LOG_MOV(BASELEVEL+0,<<"OneOrTwoStepKick: I can make it in one step to desired vel "
	    <<vel<<" in dir "<<RAD2DEG(dir.get_value())<<" (v: "<<res_vel1<<")");
#endif
    res_vel2 = res_vel1;
    res_cmd2 = res_cmd1;
    need_2step = false;
    return true;
  }
  // use second step

  double best_diff = fabs(res_vel1-vel);

  MyState next_state;
  Cmd_Body best_action;
  double best_vel = 0;
  

  itr_actions.reset();

Vector twoStepKSuccPos;

  while(Cmd_Body const *action = itr_actions.next()) {
    Tools::model_cmd_main(state.my_pos,state.my_vel,state.my_angle,state.ball_pos,state.ball_vel,
			  *action,next_state.my_pos,next_state.my_vel,next_state.my_angle,
			  next_state.ball_pos,next_state.ball_vel);
    onestepkick->set_state(next_state.my_pos - next_state.my_vel, //we subtract this here
                           next_state.my_vel,                     //since it will be added
			   next_state.my_angle,                   //within is_pos_ok
			   next_state.ball_pos,
			   next_state.ball_vel, 
			   state.op_pos, 
			   state.op_bodydir, 
			   state.op_bodydir_age);
    if(next_state.my_pos.distance(next_state.ball_pos) > WSinfo::me->kick_radius - KICKRANGE_MARGIN) {
      // ball out of kickrange
      continue;
    }
    if(!onestepkick->is_pos_ok(next_state.ball_pos)) {
      // ball gets too near to an opponent or to my body
      continue;
    }
    if(!Tools::is_ball_safe_and_kickable(next_state.my_pos,state.op_pos,state.op_bodydir,
					 next_state.ball_pos, state.op_bodydir_age)){
      // op. might get ball in next step in worst case
      continue;
    } else{ // Debug only -> plot safe positions
      ;
#if 0
      LOG_MOV(0,"Pos is safe and kickable opdir: "<<RAD2DEG(state.op_bodydir.get_value()));
      LOG_MOV(0,_2D<<C2D(next_state.ball_pos.x,next_state.ball_pos.y,.1,"#000000"));
#endif
    }

    Cmd tmp_cmd;
    double tmp_vel;
    if(to_pos) {
      onestepkick->kick_to_pos_with_initial_vel(vel,pos);
    } else {
      onestepkick->kick_in_dir_with_initial_vel(vel,dir);
    }
    bool pos_ok=onestepkick->get_cmd(tmp_cmd);
    onestepkick->get_vel(tmp_vel);
    
    double diff = fabs(tmp_vel - vel);
    if(pos_ok && (diff < best_diff)) {
      best_diff = diff;
      best_vel = tmp_vel;
      best_action  = *action;
twoStepKSuccPos=next_state.ball_pos;
    }
    /* we could use a feature that delivers best cmd even if resulting pos is not ok
       - contact sput for that!                                                      */
    //if(!pos_ok && (diff < best_diff_inv)) { // pos not ok
    //  best_diff_inv = diff;
    //  best_vel_inv = tmp_vel;
    //  best_action_inv = *action;
    //}
  }
  if(best_vel>0) {
    res_vel2 = best_vel;
    res_cmd2 = best_action;    
#ifdef CALCLOG
    double tgParK1, tgParK2, tgParW1, tgParW2;
    res_cmd1.get_kick(tgParK1,tgParW1);
    res_cmd2.get_kick(tgParK2,tgParW2);
    LOG_MOV(BASELEVEL+0,<<"OneOrTwoStepKick: Found better 2step action (1step vel "<<res_vel1<<", 2step vel "<<res_vel2
	    <<", desired "<<vel<<"), 1stK:("<<tgParK1<<"@"<<tgParW1<<"), 2ndK:("<<tgParK2<<"@"<<tgParW2<<"); myPos="<<WSinfo::me->pos<<" twoStepKSuccPos="<<twoStepKSuccPos);
#endif
    need_2step=true;
    return true;
  } else {
    res_vel2 = res_vel1;
    res_cmd2 = res_cmd1;
    need_2step = false;
#ifdef CALCLOG
    LOG_MOV(BASELEVEL+0,<<"OneOrTwoStepKick: DIDN'T FIND A BETTER 2STEP ACTION (1step vel "<<res_vel1
	    <<", 2step vel "<<best_vel<<", desired "<<vel<<")");
#endif
    return false;
  }
}

bool OneOrTwoStepKick::do_calc() {
  if(WSinfo::ws->time!=set_in_cycle) {
    ERROR_OUT << "\nOneOrTwoStepKick::do_calc() called without prior initialization!";
    return false;
  }
  if(!calc_done) {
    MyState state =  get_cur_state();
    result_status=calculate(state,target_vel,target_dir,target_pos,kick_to_pos,
			   result_cmd1,result_vel1,result_cmd2,result_vel2,need_2_steps);
  }
  return result_status;
}

bool OneOrTwoStepKick::get_vel(double &vel_1step,double &vel_2step) {
  do_calc();
  vel_1step = result_vel1;
  vel_2step = result_vel2;
  return result_status;
}

bool OneOrTwoStepKick::get_vel(double &bestvel) {
  double dum_vel;
  return get_vel(dum_vel,bestvel);
}

bool OneOrTwoStepKick::get_cmd(Cmd &cmd_1step,Cmd &cmd_2step) {
  do_calc();
  cmd_1step.cmd_body.clone(result_cmd1);
  cmd_2step.cmd_body.clone(result_cmd2);
  return result_status;
}

bool OneOrTwoStepKick::get_cmd(Cmd &bestcmd) {
  Cmd dum_cmd;
  return get_cmd(dum_cmd,bestcmd);
}

bool OneOrTwoStepKick::need_two_steps() {
  do_calc();
  return need_2_steps;
}

int OneOrTwoStepKick::is_kick_possible(double &speed,const ANGLE &dir){
  double speed1, speed2;
  speed1 = speed2 = 0.;

  kick_in_dir_with_initial_vel(speed,dir); // set parameters of kick
  get_vel(speed1, speed2);
  if(fabs(speed - speed1) <.1){
    speed = speed1;
    return 1;
  }
  if(fabs(speed - speed2) <.2){
    speed = speed2;
    return 2;
  }
  if(speed1 > speed2){
    speed = speed1;
    return 1;
  }
  speed = speed2;
  if(speed >0)
    return 2;
  else
    return 0;
}

bool OneOrTwoStepKick::init( char const *conf_file, int argc, char const* const* argv )
{
    if( initialized ) return initialized;

    initialized = OneStepKick::init( conf_file, argc, argv );

    if( !initialized )
    {
        cout << "\nOneOrTwoStepKick behavior NOT initialized!!!";
        exit( 1 );
    }

    return initialized;
}

OneOrTwoStepKick::OneOrTwoStepKick()
{
    result_vel1     = 0;
    result_vel2     = 0;
    result_status   = false;
    need_2_steps    = false;
    set_in_cycle    = -1;
    target_vel      = 0;
    kick_to_pos     = false;
    calc_done       = false;
    fake_state_time = 0;

    onestepkick     = new OneStepKick();
    onestepkick->set_log( false ); // we don't want OneStepKick-Info in our logs!
}
OneOrTwoStepKick::~OneOrTwoStepKick()
{
    delete onestepkick;
}
