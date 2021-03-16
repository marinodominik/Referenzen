#include "dribble_straight_bms.h"
#include "tools.h"
#include "options.h"
#include "ws_info.h"
#include "log_macros.h"

#define BASELEVEL 1

//#define DBLOG_MOV(LLL,XXX) LOG_MOV(LLL,<<"DribbleStraight: "<<XXX)
#define DBLOG_MOV_URGENT(LLL,XXX) LOG_MOV(LLL,<<"DribbleStraight: "<<XXX)
#define DBLOG_MOV(LLL,XXX) 
//#define DBLOG_DRAW(LLL,XXX) LOG_POL(LLL,<<_2D<<XXX)
#define DBLOG_DRAW(LLL,XXX)

// CLASS VARIABLES
const double DribbleStraightItrActions::kick_pwr_min = 5;
const double DribbleStraightItrActions::kick_pwr_inc = 5;
const double DribbleStraightItrActions::kick_pwr_steps = 20;
const double DribbleStraightItrActions::kick_fine_pwr_inc = 1;
const double DribbleStraightItrActions::kick_fine_pwr_steps = 10;
const double DribbleStraightItrActions::kick_ang_min = 0;
const double DribbleStraightItrActions::kick_ang_inc = 2*PI/144;
const double DribbleStraightItrActions::kick_ang_steps = 144;
const double DribbleStraightItrActions::kick_fine_ang_inc = 2*PI/360;
const double DribbleStraightItrActions::kick_fine_ang_steps = 90;

bool DribbleStraight::initialized=false;

#define PROT(XXX)(std::cout<<XXX)
//#define PROT(XXX)
#define SAFETY_MARGIN 0.2

DribbleStraight::DribbleStraight(){
  //const double safety_margin = 0.15; //0.1 is too small

  //success_sqrdist = SQUARE(WSinfo::me->kick_radius - SAFETY_MARGIN);

  onetwokick = new OneOrTwoStepKick;
  onestepkick = new OneStepKick;
}

//double DribbleStraight::success_sqrdist;

bool DribbleStraight::init(char const * conf_file, int argc, char const* const* argv){
  if(initialized) return true;

  bool res = OneOrTwoStepKick::init(conf_file,argc,argv) &&
    OneStepKick::init(conf_file,argc,argv);
  if(!res) exit(1);

  //success_sqrdist = SQUARE(WSinfo::me->kick_radius - SAFETY_MARGIN);
  
  initialized = true;
  INFO_OUT << "\nDribbleStraight behavior initialized.";
  return initialized;
}

bool DribbleStraight::is_dribble_safe_old() {
  //#define HOLDTURN_SAFE_DIST .8

  Cmd cmd;
  //double dist;
  //int steps;

  MyState state;
  
  state.my_pos = WSinfo::me->pos;
  state.my_vel = WSinfo::me->vel;
  state.ball_pos = WSinfo::ball->pos;
  state.ball_vel = WSinfo::ball->vel;
  state.my_angle = WSinfo::me->ang;

  PlayerSet pset= WSinfo::valid_opponents;
  pset.keep_and_sort_closest_players_to_point(1, state.ball_pos);
  if ( pset.num ){
    state.op_pos= pset[0]->pos;
    state.op_bodydir =pset[0]->ang;
    state.op_bodydir_age = pset[0]->age_ang;
    DBLOG_MOV(0,"HoldTurn: Age of Angle: "<<pset[0]->age_ang);
  }
  else{
    state.op_pos= Vector(1000,1000); // outside pitch
    state.op_bodydir = ANGLE(0);
    state.op_bodydir_age = 1000;
  }


  return false;
}

bool DribbleStraight::ballpos_ok(const MyState &state){
  if(!Tools::is_position_in_pitch(state.ball_pos)) return false;
  double sqrdist = state.my_pos.sqr_distance(state.ball_pos);
  if(sqrdist > SQUARE(WSinfo::me->kick_radius - SAFETY_MARGIN)) {
    //DBLOG_MOV(0,"ballpos not ok: out of kickrange");
    return false;
  }
  if(sqrdist < SQUARE(WSinfo::me->radius+.2)) {
    //DBLOG_MOV(0,"ballpos not ok: collision");
    return false; // collision ?
  }
  return true;
  //return Tools::is_ball_safe_and_kickable(state.my_pos, state.op_pos, state.op_bodydir, state.ball_pos, state.op_bodydir_age);
}

bool DribbleStraight::ballpos_optimal(const MyState &state){
  //return false; // currently deactivated takes too long
  double sqrdist = state.my_pos.sqr_distance(state.ball_pos);
  if(sqrdist > SQUARE(WSinfo::me->kick_radius - SAFETY_MARGIN))
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

bool DribbleStraight::get_cmd(Cmd & cmd){
  
  //return get_cmd(cmd, opponent_goalpos);

  MyState state;
  state.get_from_WS();
#if 0
  PlayerSet pset= WSinfo::valid_opponents;
  pset.keep_and_sort_closest_players_to_point(1, state.ball_pos);
  if ( pset.num ){
    state.op_pos= pset[0]->pos;
    state.op_bodydir =pset[0]->ang;
    state.op_bodydir_age = pset[0]->age_ang;
    DBLOG_MOV(0,"DribbleStraight: Age of Angle: "<<pset[0]->age_ang);
  }
  else{
    state.op_pos= Vector(1000,1000); // outside pitch
    state.op_bodydir = ANGLE(0);
    state.op_bodydir_age = 1000;
  }
#endif  
  return calc_next_cmd(cmd,state);
}

bool DribbleStraight::calc_next_cmd(Cmd &next_cmd, const MyState &cur_state) {
#define SAFETY 0.15
#define MAX_STATES 20
#define STAMINA_THRESHOLD 2550
                          //TG09: alt: 1400

  if(!WSinfo::is_ball_kickable()) return false;
  if(WSinfo::ws->play_mode==PM_PlayOn && last_calc==WSinfo::ws->time) {
    next_cmd.cmd_body.clone(cached_cmd);
    return cached_res;
  }
  last_calc=WSinfo::ws->time;
  cached_cmd.unset_lock();cached_cmd.unset_cmd();
  cached_res=false;
  
  Cmd dashcmd,dumcmd;
  MyState state[MAX_STATES];MyState tmpstate;
  Vector vec2target;

  double maxdash=100;
  if(WSinfo::me->stamina<STAMINA_THRESHOLD) {
    DBLOG_MOV(0,"Activating stamina saving mode...");
    maxdash=WSinfo::me->stamina_inc_max;
  }
  
  dashcmd.cmd_body.set_dash(maxdash);
  dumcmd.cmd_body.set_turn(0);

  ANGLE target_dir=WSinfo::me->ang;

  Tools::get_successor_state(cur_state,dumcmd.cmd_body,state[0]);
  DBLOG_DRAW(0,C2D(state[0].my_pos.x,state[0].my_pos.y,0.3,"red"));
  DBLOG_DRAW(0,C2D(state[0].my_pos.x,state[0].my_pos.y,WSinfo::me->kick_radius,"red"));
  for(int i=1;i<MAX_STATES;i++) {
    Tools::get_successor_state(state[i-1],dashcmd.cmd_body,state[i]);
    DBLOG_DRAW(0,C2D(state[i].my_pos.x,state[i].my_pos.y,0.3,"blue"));
    DBLOG_DRAW(0,C2D(state[i].my_pos.x,state[i].my_pos.y,WSinfo::me->kick_radius,"blue"));
  }
  
  Vector vec2ball = (state[0].ball_pos - state[0].my_pos);
  vec2ball.rotate(-state[0].my_angle.get_value());
  double ydist = vec2ball.getY();
  ydist = fabs(ydist);
  DBLOG_MOV(0,"ydist: "<<ydist<<", kick_radius: "<<WSinfo::me->kick_radius);

  //double max_op_radius = ServerOptions::tackle_dist+ServerOptions::player_speed_max+SAFETY;
  double max_op_radius = 4; // just to be sure...
  bool must_kick=false;
  
  // what happens if we do not kick the ball (and dash instead)?

  PlayerSet opnearball=WSinfo::valid_opponents;
  opnearball.keep_players_in_circle(state[0].ball_pos,max_op_radius);
  if(opnearball.num>0) { // check if ball is still safe if we do not kick...
    DBLOG_MOV(0,"Opponents may be in range... checking!");
    for(int p=0;p<opnearball.num;p++) {
      DBLOG_MOV(0,<< _2D << C2D(opnearball[p]->pos.x, opnearball[p]->pos.y ,1.4,"red")); 
      //      if(!Tools::is_ballpos_safe(opnearball[p],state[0].ball_pos)) {
      if(!Tools::is_ballpos_safe(opnearball[p],state[0].ball_pos,true)) { // check w. tackle possibility
	DBLOG_MOV(0,"Opp #"<<opnearball[p]->number<<" may get the ball if we don't kick!");
	must_kick=true;
	break;
      }
    }
  }

  if(!must_kick) {  // ok, so we try to dash first!
    Tools::get_successor_state(cur_state,dashcmd.cmd_body,tmpstate);
    if(ballpos_ok(tmpstate)) {
      DBLOG_MOV(0,"Ballpos ok, dashing...");
      next_cmd.cmd_body.clone(dashcmd.cmd_body);
      cached_cmd.clone(dashcmd.cmd_body);
      cached_res=true;
      return true;
    }
  }
  // hmm, so we must kick...
  DBLOG_MOV(0,"We need to kick now! Searching best kick move...");
  
  Cmd_Body bestcmd;
  int maxsteps=-1;
  Vector cur_pos,cur_vel;
  Vector bestpos,bestvel;
  double bestdist=0;
  DribbleStraightItrActions itr_actions;

  itr_actions.reset();
  while(Cmd_Body *action = itr_actions.next()) {
    bool all_ok = true;
    Tools::get_successor_state(cur_state,*action,state[0]);
    ///* always kick more or less parallel to body ang... */
    // INSERT APPROPRIATE CODE HERE //
    if(!ballpos_ok(state[0])) continue;
    for(int p=0;p<opnearball.num;p++) {
      DBLOG_MOV(0,<< _2D << C2D(opnearball[p]->pos.x, opnearball[p]->pos.y ,1.4,"red")); 
      //      if(!Tools::is_ballpos_safe(opnearball[p],state[0].ball_pos)) {
      if(!Tools::is_ballpos_safe(opnearball[p],state[0].ball_pos,true)) { // check w. tackle possibily
	all_ok=false;break;
      }
    }
    if(!all_ok) continue;
    // we have found a kick move that keeps the ball in kickrange AND leads to a safe position!
    int steps;
    cur_pos=state[0].ball_pos;
    cur_vel=state[0].ball_vel;
    for(steps=1;steps<MAX_STATES;steps++) {
      cur_pos+=cur_vel;
      state[steps].ball_pos = cur_pos;
      cur_vel*=ServerOptions::ball_decay;
      state[steps].ball_vel = cur_vel;
      if(!ballpos_ok(state[steps])) {
	//steps--;
	break;
      }
    }
#if 0   // step based
    if(steps>=maxsteps) {
      double dist=(state[steps-1].ball_pos-cur_state.ball_pos).norm();
      if(steps>maxsteps || dist>bestdist) {
	maxsteps=steps;
	bestcmd.unset_lock();bestcmd.unset_cmd();
	bestcmd.clone(*action);
	bestpos=state[0].ball_pos;
	bestvel=state[0].ball_vel;
	bestdist=dist;
      }
    }
#else
    // dist based
    Vector ballvec=state[steps-1].ball_pos-cur_state.ball_pos;
    ballvec.rotate(-WSinfo::me->ang.get_value());
    double dist=ballvec.getX();
    if(dist>=bestdist) {
      bestdist=dist;
      maxsteps=steps;
      bestcmd.unset_lock();bestcmd.unset_cmd();
      bestcmd.clone(*action);
      bestpos=state[0].ball_pos;
      bestvel=state[0].ball_vel;
    }
#endif
  }
  if(bestdist>0) {
    DBLOG_MOV(0,"Found a suitable kick move ("<<maxsteps<<" steps, dist="<<bestdist<<")");
    Vector cur_vel,cur_pos;
    cur_vel=bestvel;
    cur_pos=bestpos;
    for(int i=0;i<maxsteps;i++) {
      DBLOG_DRAW(0,C2D(cur_pos.x,cur_pos.y,ServerOptions::ball_size,"black"));
      cur_pos+=cur_vel;
      cur_vel*=ServerOptions::ball_decay;
    }
    next_cmd.cmd_body.clone(bestcmd);
    cached_cmd.clone(bestcmd);
    cached_res=true;
    return true;
  } else {
    DBLOG_MOV_URGENT(0,"Could not find any suitable kick action! DribbleStraight not possible!");
    //std::cerr << "\nCould not find a dribble action!";
    return false;
  }
}
  
bool DribbleStraight::is_dribble_safe(int opp_time2react) {
  //MyState state;
  //state.get_from_WS();
  Cmd dumcmd;
  
  return get_cmd(dumcmd);
}



bool DribbleStraight::is_dribble_safe( const AState & state, int opp_time2react ) {

  Cmd dumcmd;
  
  MyState my_state;

  my_state.my_vel = state.my_team[state.my_idx].vel;
  my_state.my_pos = state.my_team[state.my_idx].pos;
  my_state.ball_vel = state.ball.vel;
  my_state.ball_pos = state.ball.pos;
  my_state.my_angle = ANGLE(state.my_team[state.my_idx].body_angle);

  return calc_next_cmd(dumcmd,my_state);
}

























#if 0
bool DribbleStraight::calc_next_cmd(Cmd &next_cmd, const MyState &cur_state) {
  //#define SAFETY 0.15
  //#define MAX_STATES 20

  if(last_calc==WSinfo::ws->time) {
    next_cmd.cmd_body.clone(cached_cmd);
    return cached_res;
  }
  last_calc=WSinfo::ws->time;
  cached_cmd.unset_lock();cached_cmd.unset_cmd();
  
  Cmd tmpcmd,tmpcmd2;
  MyState state[MAX_STATES];
  MyState tmpstate;
  Vector vec2target;

  tmpcmd.cmd_body.set_dash(100);
  tmpcmd2.cmd_body.set_turn(0);

  ANGLE target_dir=WSinfo::me->ang;

  bool place_ball = false;

  Tools::get_successor_state(cur_state,tmpcmd2.cmd_body,state[0]);
  DBLOG_DRAW(0,C2D(state[0].my_pos.x,state[0].my_pos.y,0.3,"red"));
  DBLOG_DRAW(0,C2D(state[0].my_pos.x,state[0].my_pos.y,WSinfo::me->kick_radius,"red"));
  for(int i=1;i<MAX_STATES;i++) {
    Tools::get_successor_state(state[i-1],tmpcmd.cmd_body,state[i]);
    DBLOG_DRAW(0,C2D(state[i].my_pos.x,state[i].my_pos.y,0.3,"blue"));
    DBLOG_DRAW(0,C2D(state[i].my_pos.x,state[i].my_pos.y,WSinfo::me->kick_radius,"blue"));
  }
  
  Vector vec2ball = (state[0].ball_pos - state[0].my_pos);
  vec2ball.rotate(-state[0].my_angle.get_value());
  double ydist = vec2ball.y;
  bool dribble_left = (ydist>=0 ? true : false);
  ydist = fabs(ydist);
  DBLOG_MOV(0,"ydist: "<<ydist<<", kick_radius: "<<WSinfo::me->kick_radius);

  // do we need to place the ball into its corridor?
 
  if(ydist<WSinfo::me->radius+SAFETY || ydist > .9 || !ballpos_ok(cur_state)) {
    // already in corridor?
    Vector dum=(WSinfo::ball->pos - WSinfo::me->pos);
    dum.rotate(-WSinfo::me->ang.get_value());
    dum.y=fabs(dum.y);
    
    vec2target.y = WSinfo::me->radius+ServerOptions::ball_size+SAFETY; 
    vec2target.x = -.5*sqrt(SQUARE(WSinfo::me->kick_radius-SAFETY)-SQUARE(vec2target.y));
    //vec2target.x=0;
    //vec2target.y = WSinfo::me->radius+ServerOptions::ball_size+SAFETY;
    if(!dribble_left) vec2target.y*=-1;
    vec2target.ROTATE(state[0].my_angle);
    if(dum.y<WSinfo::me->radius+SAFETY || dum.y > .9 /*|| !ballpos_ok(cur_state)*/) {
      DBLOG_MOV(0,"Putting ball into corridor...");
      place_ball = true;
      vec2target+=state[0].my_pos;
      DBLOG_DRAW(0,C2D(vec2target.x,vec2target.y,.5,"gray"));
      target_dir=(vec2target-WSinfo::ball->pos).ARG();
    } else {
      //DBLOG_MOV(0,"Correcting target_dir...");
      //vec2target+=state[5].my_pos;
      //target_dir=(vec2target-WSinfo::ball->pos).ARG();
    }
  } else {
  
    // dash possible?
    
    Tools::get_successor_state(cur_state,tmpcmd.cmd_body,tmpstate);
    if(/*!place_ball &&*/ ballpos_ok(tmpstate)) {
      DBLOG_MOV(0,"Ballpos ok, dashing...");
      next_cmd.cmd_body.clone(tmpcmd.cmd_body);
      cached_cmd.clone(tmpcmd.cmd_body);
      cached_res=true;
      return true;
    }
  }

  // kick!
 
  DBLOG_MOV(0,"Kicking...");

  // find target dir

  double bestvel=0;
  int maxsteps=-1;
  double maxvel = onestepkick->get_max_vel_in_dir(target_dir,false);
  if(place_ball) {
    bestvel = (vec2target-WSinfo::ball->pos).norm();
    maxsteps = 1;
  } else {
    for(double vel=0.1;vel<=maxvel;vel+=.1) {
      Vector cur_vel,cur_pos;
      cur_vel.init_polar(vel,target_dir);
      cur_pos=WSinfo::ball->pos;
      int steps;
      for(steps=0;steps<MAX_STATES;steps++) {
	cur_pos+=cur_vel;
	state[steps].ball_pos = cur_pos;
	cur_vel*=ServerOptions::ball_decay;
	state[steps].ball_vel = cur_vel;
	if(!ballpos_ok(state[steps])) {
	  steps--;
	  break;
	}
      }
      if(steps>=maxsteps) {
	maxsteps=steps; bestvel=vel;
      }
    }
  }
  DBLOG_MOV(0,"maxsteps: "<<maxsteps<<", bestvel: "<<bestvel<< ", maxvel: "<<maxvel);
  if(bestvel>0 && maxsteps>=0) {
    Vector cur_vel,cur_pos;
    cur_vel.init_polar(bestvel,target_dir);
    cur_pos=WSinfo::ball->pos;
    for(int i=0;i<=maxsteps;i++) {
      DBLOG_DRAW(0,C2D(cur_pos.x,cur_pos.y,ServerOptions::ball_size,"black"));
      cur_pos+=cur_vel;
      cur_vel*=ServerOptions::ball_decay;
    }
    if(place_ball) {
      onestepkick->kick_to_pos_with_initial_vel(bestvel,vec2target);
    } else {
      onestepkick->kick_in_dir_with_initial_vel(bestvel,target_dir);
    }
    onestepkick->get_cmd(next_cmd);
    cached_cmd.clone(next_cmd.cmd_body);
    cached_res=true;
    return true;
  } else {
    DBLOG_MOV(0,"Could not find a dribble action!");
    std::cerr << "\nCould not find a dribble action!";
  }
  cached_cmd.set_turn(0);
  cached_res=false;
  return false;
}
#endif
