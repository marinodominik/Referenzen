#include "neuro_kick_bms.h"
#define BASELEVEL 0

Net *NeuroKick::nets[2];
bool NeuroKick::initialized=false;
bool NeuroKick::use_12step_def;
bool NeuroKick::do_finetuning;

const double NeuroKickItrActions::kick_pwr_min = 10;
const double NeuroKickItrActions::kick_pwr_inc = 10;
const double NeuroKickItrActions::kick_pwr_steps = 10;
const double NeuroKickItrActions::kick_fine_pwr_inc = 1;
const double NeuroKickItrActions::kick_fine_pwr_steps = 10;
const double NeuroKickItrActions::kick_ang_min = 0;
const double NeuroKickItrActions::kick_ang_inc = 2*PI/45;
const double NeuroKickItrActions::kick_ang_steps = 45;
const double NeuroKickItrActions::kick_fine_ang_inc = 2*PI/360;
const double NeuroKickItrActions::kick_fine_ang_steps = 90;

const double NeuroKick::kick_finetune = 2*PI/360.;
const double NeuroKick::kick_finetune_power = 1;
const double NeuroKick::turn_finetune = 0;

const double NeuroKick::tolerance_velocity = 0.2;
const double NeuroKick::tolerance_direction = 0.05;
const double NeuroKick::min_ball_dist = 0.15;
const double NeuroKick::kickable_tolerance = 0.15;


void NeuroKickItrActions::reset(bool finetune,double orig_pwr,ANGLE orig_ang) {
  if(!finetune) {
    ang_min = ANGLE(kick_ang_min);
    ang_inc = ANGLE(kick_ang_inc);
    ang_steps = kick_ang_steps;
    pwr_min = kick_pwr_min;
    pwr_inc = kick_pwr_inc;
    pwr_steps = kick_pwr_steps;
  } else {
    ang_min = orig_ang-ANGLE(.5*kick_fine_ang_steps*kick_fine_ang_inc);
    ang_inc = ANGLE(kick_fine_ang_inc);
    ang_steps = kick_fine_ang_steps + 1;
    pwr_min = orig_pwr-(.5*kick_fine_pwr_steps*kick_fine_pwr_inc);
    pwr_inc = kick_fine_pwr_inc;
    pwr_steps = kick_fine_pwr_steps + 1;
  }
  ang_done = 0;pwr_done=0;
  ang = ang_min; pwr = pwr_min;
}

Cmd_Body* NeuroKickItrActions::next() {
  if(pwr_done<pwr_steps && ang_done<ang_steps) {
    action.unset_lock();
    action.unset_cmd();
    action.set_kick(pwr,ang.get_value_mPI_pPI());
    ang+=ang_inc;
    if(++ang_done>=ang_steps) {
  ang=ang_min;ang_done=0;
  pwr+=pwr_inc;
  pwr_done++;
    }
    return &action;
  }
  return NULL;
}

NeuroKick::NeuroKick() {
  init_in_cycle = -1;
  fake_state_time = -1;
  twostepkick = new OneOrTwoStepKick();
}

NeuroKick::~NeuroKick() {
  for(int i=0;i<2;i++) {  // 2 is the max. number of nets
    delete nets[i];
  }
  delete twostepkick;
}

bool NeuroKick::init(char const * conf_file, int argc, char const* const* argv) {
  if(initialized) return true;
  if(!OneOrTwoStepKick::init(conf_file,argc,argv)) {
    ERROR_OUT << "\nCould not initialize OneOrTwoStepKick behavior - stop loading.";
    exit(1);
  }

  do_finetuning = true;
  char NN0_name[500];
  char NN1_name[500];
  
  //currently at most 2 nets are used
  nets[0] = new Net();
  nets[1] = new Net();
  
  use_12step_def = true;  // do use 1 or 2 step kick
  sprintf(NN0_name,"%s","../data/nets_neuro_kick2/fastspeed.net\0");
  sprintf(NN1_name,"%s","../data/nets_neuro_kick2/slowspeed.net\0");
  
  ValueParser vp1(CommandLineOptions::moves_conf,"Neuro_Kick");
  //vp.set_verbose();
  vp1.get("use_12step", use_12step_def);

  ValueParser vp(conf_file,"Neuro_Kick2");
  //vp.set_verbose();
  vp.get("do_finetuning", do_finetuning);
  vp.get("use_12step", use_12step_def);
  vp.get("NN_name", NN0_name, 500);
  vp.get("NN2_name", NN1_name, 500);
  
  if(nets[0]->load_net(NN0_name) == FILE_ERROR) {
    ERROR_OUT<<"NeuroKick_bms: No net "<<NN0_name<<" found - stop loading\n";
    exit(0);
  }
  if(nets[1]->load_net(NN1_name) == FILE_ERROR) {
    ERROR_OUT<<"NeuroKick_bms: No net2 "<<NN1_name<<" found - stop loading\n";
    exit(0);
  }
  
//  cout<<"\nNeuroKick behavior initialized (using 1or2step per default: "
//	<<use_12step_def<<").";
  initialized=true;
  return true;
}
  
/*********************************************************************/
/* init functions, use one of them to init kick request...           */


bool NeuroKick::get_cmd(Cmd & cmd) { 
  if(!initialized) {
    ERROR_OUT << "\nNeuroKick_bms not initialized!";
    return false;
  }
  if(WSinfo::ws->time!=init_in_cycle) {
    ERROR_OUT << "\nNeuroKick::get_cmd() called without prior initialization!";
    return false;
  }
  LOG_MOV(0,<<"Starting NeuroKick behavior (dir: "<< target_dir.get_value()
	  <<", vel: "<<target_vel<<").");
  return decide(cmd);
}


void NeuroKick::kick_to_pos_with_initial_vel(double vel,const Vector &pos,bool oneortwo) {
  MyState state = get_cur_state();
  target_dir = (pos - state.ball_pos).ARG();
  target_vel = vel;
  target_pos = pos;
  do_target_tracking=true;
  use_12step = oneortwo;
  init_in_cycle = WSinfo::ws->time;
}

void NeuroKick::kick_to_pos_with_final_vel(double vel,const Vector &pos,bool oneortwo) {
  MyState state = get_cur_state();
  target_dir = (pos - state.ball_pos).ARG();
  target_vel = (1-ServerOptions::ball_decay)*((pos-state.ball_pos).norm()+vel*ServerOptions::ball_decay)
    + ServerOptions::ball_decay * vel;
  /* Does not work if target_vel > ball_speed_max... use max speed instead! */
  if(target_vel>ServerOptions::ball_speed_max) {
    target_vel=ServerOptions::ball_speed_max;
    LOG_ERR(0,<<"NeuroKick: Point "<<pos<<" too far away, using max vel!");
  }
  target_pos = pos;
  do_target_tracking=true;
  use_12step = oneortwo;
  init_in_cycle = WSinfo::ws->time;
}

void NeuroKick::kick_to_pos_with_max_vel(const Vector &pos,bool oneortwo) {
  kick_to_pos_with_initial_vel(ServerOptions::ball_speed_max,pos,oneortwo);
}

void NeuroKick::kick_in_dir_with_max_vel(const ANGLE &dir,bool oneortwo) {
  kick_in_dir_with_initial_vel(ServerOptions::ball_speed_max,dir,oneortwo);
}

void NeuroKick::kick_in_dir_with_initial_vel(double vel,const ANGLE &dir,bool oneortwo) {
  target_dir = dir;
  target_vel = vel;
  do_target_tracking=false;
  use_12step = oneortwo;
  init_in_cycle = WSinfo::ws->time;
}
/**********************************************************************/

void NeuroKick::get_ws_state(MyState &state) {
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

void NeuroKick::set_state(const Vector &mypos,const Vector &myvel,const ANGLE &myang,
			  const Vector &ballpos,const Vector &ballvel) {
  fake_state.my_pos = mypos;
  fake_state.my_vel = myvel;
  fake_state.my_angle = myang;
  fake_state.ball_pos = ballpos;
  fake_state.ball_vel = ballvel;
  fake_state_time = WSinfo::ws->time;
  init_in_cycle = -1;
}

void NeuroKick::reset_state() {
  fake_state_time = -1;
  init_in_cycle = -1;
}

MyState NeuroKick::get_cur_state() {
  MyState cur_state;
  if(fake_state_time == WSinfo::ws->time) {
    cur_state = fake_state;
  } else {
    get_ws_state(cur_state);
  }
  return cur_state;
}

bool NeuroKick::decide(Cmd &cmd) {
  MyState state = get_cur_state();
  MyState next_state;
  Cmd_Body best_action;
  double best_val,best_pwr;
  Angle best_ang;
  bool best_val_ok= false;

  /* use OneOrTwoStepKick if it makes sense... */
  if(use_12step) {
    double res_vel1,res_vel2;
    Cmd res_cmd1,res_cmd2;

    twostepkick->set_state(state.my_pos,state.my_vel,state.my_angle,
			   state.ball_pos,state.ball_vel,state.op_pos, state.op_bodydir, state.op_bodydir_age);
    
    if(do_target_tracking) {
      //LOG_DEB(0, << " neurokick: targetvel: " << target_vel << " targetpos: " << target_pos);
      twostepkick->kick_to_pos_with_initial_vel(target_vel,target_pos);
    } else {
      //LOG_DEB(0, << " neurokick: targetvel: " << target_vel << " targetdir: " << target_dir);
      twostepkick->kick_in_dir_with_initial_vel(target_vel,target_dir);
    }
    
    twostepkick->get_cmd(res_cmd1,res_cmd2);
    twostepkick->get_vel(res_vel1,res_vel2);

    if(fabs(res_vel1 - target_vel)<tolerance_velocity) {
      // I can do it with 1 single kick
      cmd.cmd_body.clone(res_cmd1.cmd_body);
      LOG_MOV(BASELEVEL+0,<< "NeuroKick: 1-Step-Kick IS possible. desired vel: " << target_vel
	      << " desired dir: " << RAD2DEG(target_dir.get_value())
	      << " Answer from 12Step: "
	      << " 1step vel "<<res_vel1
	      << " 2step vel "<<res_vel2);
      return true;
    }
    if(fabs(res_vel2 - target_vel) < 0.7 * tolerance_velocity ) {
      // I can (safely) do it with 2 kicks
      cmd.cmd_body.clone(res_cmd2.cmd_body);
      LOG_MOV(BASELEVEL+0,<< "NeuroKick: 2-Step-Kick IS possible. desired vel: " << target_vel
	      << " desired dir: " << RAD2DEG(target_dir.get_value())
	      << " Answer from 12Step: "
	      << " 1step vel "<<res_vel1
	      << " 2step vel "<<res_vel2);
      return true;
    }
    LOG_MOV(BASELEVEL+0,<< "NeuroKick: 1or2-Step-Kick is NOT possible. desired vel: " << target_vel
	    << " desired dir: " << RAD2DEG(target_dir.get_value())
	    << " Answer from 12Step: "
	    << " 1step vel "<<res_vel1
	    << " 2step vel "<<res_vel2);
  }
  net=choose_net();
  
  itr_actions.reset(false);
  target_vel *= ServerOptions::ball_decay;
  // target vel. is adjusted, since neuro kick tries to achieve vel. AFTER first step

  for(int ft=0;ft<2;ft++) {
    ft==0 ? itr_actions.reset(false) : itr_actions.reset(true,best_pwr, ANGLE(best_ang));
    while(Cmd_Body const *action = itr_actions.next()) {
      Tools::model_cmd_main(state.my_pos,state.my_vel,state.my_angle,state.ball_pos,state.ball_vel,*action,
			    next_state.my_pos,next_state.my_vel,next_state.my_angle,next_state.ball_pos,
			    next_state.ball_vel);
      double val= evaluate( next_state );
      
      if(!best_val_ok || val < best_val) {
	best_val= val;
	best_action= *action;
	best_val_ok= true;
	best_action.get_kick(best_pwr,best_ang);
      }
    }
    if(!do_finetuning || !best_val_ok) break;
  }
  if(best_val_ok) return cmd.cmd_body.clone(best_action);
  return false;
}

double NeuroKick::evaluate(MyState const& state) {
  // ridi03 :  if(is_failure(state)) return 0.0;
  // ridi03: if(is_success(state)) return 1.0;

  if(is_failure(state)) return 1.0;
  if(is_success(state)) return 0.0;

  get_features(state,target_dir,target_vel,net->in_vec);
  net->forward_pass(net->in_vec,net->out_vec);
  return(net->out_vec[0]);
}

void NeuroKick::get_features(const MyState& state,const ANGLE tdir, const double tvel,float *net_in) {
  /* relative features version */
  Vector pos = state.ball_pos-state.my_pos;
  Vector vel = state.ball_vel;
  double view = ((state.my_angle-target_dir)-ANGLE(PI)).get_value_mPI_pPI();
    
  /* rotate whole system (pos.y should be zero*/
  pos.ROTATE((ANGLE(2*PI) - target_dir));
  vel.ROTATE((ANGLE(2*PI) - target_dir));
  //view = Tools::get_angle_between_mPI_pPI((view - target_direction)-PI);
  
  if(view<0) {
    vel.setY( -vel.getY() );
    pos.setY( -pos.getY() );
  }
  
  net_in[0] = pos.getX(); //distance to target
  net_in[1] = pos.getY(); //velocity in x direction
  net_in[2] = vel.getX(); //distance to target
  net_in[3] = vel.getY(); //velocity in x direction
  net_in[4] = fabs(view); //abs. relative view angle to target
}

bool NeuroKick::is_failure(const MyState& state) {
  double ball_dist = (state.ball_pos - state.my_pos).norm();
  if(ball_dist > WSinfo::me->kick_radius - kickable_tolerance) {
    //left the kickrange - so its the negation of is_success()
    return !(is_success(state));
  }
  if(ball_dist < WSinfo::me->radius+min_ball_dist){
    return true;
  }
  return false;
}

bool NeuroKick::is_success(const MyState& state) {
  // only 1 2 step kick can be a successful finish of neuro-kick!
  return false;

  double ball_dist = (state.ball_pos - state.my_pos).norm();
  if(ball_dist > WSinfo::me->kick_radius - kickable_tolerance) { 
    // ball has left my Kickrange

    // compute difference between real and target direction
    double diff_direction = fabs((state.ball_vel.ARG() - target_dir).get_value_mPI_pPI());
    //diff_direction = fabs(Tools::get_angle_between_mPI_pPI(diff_direction));
    
    // compute velocity of the ball
    double diff_velocity = fabs(state.ball_vel.norm() - target_vel);

    //check velocity and direction
    if((diff_direction < tolerance_direction) 
       && (diff_velocity < tolerance_velocity)){ 
      // successful!
      return true;
    }
  }
  //no successful state
  return false;
}

Net* NeuroKick::choose_net() {
  if(target_vel > 2.1) return nets[0];  // fastkick
  else return nets[1];                  // slowkick	
}

