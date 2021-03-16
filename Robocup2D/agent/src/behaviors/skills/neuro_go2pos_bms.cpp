#include "neuro_go2pos_bms.h"

#include "log_macros.h"
#include "ws_info.h"
#include "tools.h"
#include "valueparser.h"
#include "intercept2.h"  // new go2pos


/* JTS: initialization of constants */
/* neurogo2pos uses dashs */
const double NeuroGo2PosItrActions::dash_power_min = 40;
const double NeuroGo2PosItrActions::dash_power_inc = 20;
const double NeuroGo2PosItrActions::dash_power_max= 100;
const double NeuroGo2PosItrActions::dash_power_steps = (int)((dash_power_max - dash_power_min)/dash_power_inc) +1;
/* neurogo2pos uses turns */
const double NeuroGo2PosItrActions::turn_min= -M_PI;  // important: start with strongest turn to the right
const double NeuroGo2PosItrActions::turn_max= M_PI;  // and end with strongest turn to the left!
const double NeuroGo2PosItrActions::turn_inc= 2*M_PI/72.0;
const int NeuroGo2PosItrActions::turn_steps=(int)((turn_max - turn_min)/turn_inc) +1;

const double NeuroGo2Pos::MAX_GO2POS_DISTANCE= 5.0;
const double NeuroGo2Pos::costs_failure = 1.0;
const double NeuroGo2Pos::costs_success = 0.0;

Net * NeuroGo2Pos::net;
bool NeuroGo2Pos::initialized= false;
int NeuroGo2Pos::op_mode;
int NeuroGo2Pos::num_stored;
int NeuroGo2Pos::num_epochs;
int NeuroGo2Pos::store_per_cycle;
int NeuroGo2Pos::repeat_mainlearningloop;
int NeuroGo2Pos::state_memory_ctr;
double NeuroGo2Pos::target_tolerance;
int NeuroGo2Pos::consider_obstacles;
int NeuroGo2Pos::use_old_go2pos;
double NeuroGo2Pos::costs_per_action;
bool NeuroGo2Pos::use_regular_states;

NeuroGo2Pos::NeuroGo2Pos(){ 
  op_mode = 0; 
  num_stored = 0;
  num_epochs= 50;
  store_per_cycle = 100;
  repeat_mainlearningloop = 10;
  state_memory_ctr = 0;
  target_tolerance = 1;
  consider_obstacles = 1;
  //  costs_per_action = 0.01;
  costs_per_action = 0.05;
  use_regular_states = true;

  uparams[0] = 0.0;
  uparams[1] = 0.0;
  uparams[2] = 0.0;
  uparams[3] = 0.0;


  ValueParser vp(CommandLineOptions::policy_conf,"NeuroGo2Pos_bms");
  //vp.set_verbose(true);
  vp.get("op_mode", op_mode);
#if 1
  if(initialized && (op_mode == 1)){// do learning
    // LOG_MOV(0,"NeuroGo2Pos: Operation Mode LEARNING");
    net->init_weights(0,.5);
    net->set_update_f(1,uparams); // 1 is Rprop
    net->save_net("test.net");
  }
#endif
#if 0
  cout<<"\nNeuroGo2Pos_bms: Read Parameters.";
#endif
  training_set.ctr = 0;

  basic_cmd = new BasicCmd;
  obstacle_found = false;
}

NeuroGo2Pos::~NeuroGo2Pos() {
  delete basic_cmd;
}

bool NeuroGo2Pos::init(char const * conf_file, int argc, char const* const* argv) {
  if(initialized) return true; // only initialize once...
  initialized= true;

  net= new Net();

  /* load neural network */
  char netname[] = "../data/nets_neuro_go2pos/go2pos.net";

  if(net->load_net(netname) == FILE_ERROR)
  {
    ERROR_OUT << "Neuro_Go2Pos: No net-file found in directory ./nets - stop loading\n";
    initialized = false;
  }
  else
  {
	  initialized = initialized && BasicCmd::init(conf_file, argc, argv);
  }
  if(!initialized) { cout << "\nNeuroGo2Pos behavior NOT initialized!!!"; }

  return initialized;
}

bool NeuroGo2Pos::get_cmd(Cmd & cmd) { 

  // ridi 2003: use new go2pos to compute command
  if(1 && !use_old_go2pos){
    Intercept2 inter;
    int time;
    inter.time_to_target( target_pos, target_tolerance, WSinfo::me, time, cmd.cmd_body); // this corresponds to go2pos
    LOG_MOV(0,<<"NeuroGo2Pos: Using ANALYTICAL go2pos. estimated steps 2 go "<<time<< " cmd_set: "<< (cmd.cmd_body.is_cmd_set() == true)
    		<<" CMD IS: "<< cmd);
    if(cmd.cmd_body.is_cmd_set() == true)
      return true;
    else 
      return false;
  }


  if ( ! initialized ) {
    ERROR_OUT << "NeuroGo2Pos not intialized";
    return false;
  }


  if(op_mode == 1){
    if(learn(cmd) == true) //learning procedure has set a command
      return true;
  }
  if(WSinfo::me->pos.sqr_distance( target_pos ) <=  SQUARE( target_tolerance )){
     LOG_MOV(0,<<"NeuroGo2Pos: target reached");
     //LOG_ERR(0,<<"Yeah I reached the target!!");
     cmd.cmd_body.set_turn(0);  // Ok, I'm fine, do nothing 
     //GO2003
     return true;
  }

  neuro_decide(cmd); 

  if(op_mode == 1){
    check_cmd(cmd);
  }
  return true;
}

void NeuroGo2Pos::get_cur_state( State & state) {
  state.my_pos= WSinfo::me->pos;
  state.my_vel= WSinfo::me->vel;
  state.my_angle=  WSinfo::me->ang;
  PlayerSet pset= WSinfo::valid_opponents;
  
  Vector tmp= target_pos - state.my_pos;
  tmp.normalize(0.9);
  tmp+= state.my_pos;

  pset.keep_players_in_quadrangle( state.my_pos, tmp, 2.0*WSinfo::me->radius);
				   
  pset.keep_and_sort_closest_players_to_point(1, state.my_pos);

  if ( pset.num ) {
    state.opnt1_pos= pset[0]->pos;
  }
  else
    state.opnt1_pos= Vector(60,60);
}

void NeuroGo2Pos::get_features(State const& state, Vector target, float * net_in) {
  Vector temp_target = target;
  if ( state.my_pos.sqr_distance( target ) > SQUARE(MAX_GO2POS_DISTANCE) ) {
    Vector temp = target - state.my_pos;
    temp.normalize(MAX_GO2POS_DISTANCE);
    temp_target = state.my_pos + temp;
  }
    
  const Vector my_op_dist = state.opnt1_pos - state.my_pos;
  const Vector my_target_dist = temp_target - state.my_pos;
  Vector my_vel = state.my_vel;

  double my_target_dist_arg= my_target_dist.arg(); //precomputation [ arg() is quite expensive ]
  ANGLE my_target_angle= ANGLE(my_target_dist_arg);
  my_target_angle -= state.my_angle;
  my_vel.rotate(2*PI - my_target_dist_arg);
    
  net_in[0] = my_target_dist.norm();
  net_in[1] = my_target_angle.get_value_mPI_pPI();
  net_in[2] = my_vel.getX();
  net_in[3] = my_vel.getY();
  
  
  
  /* compute distance of obstacle to the line me/target */
  //andi_kommentar: projeziere Gegner auf die Linie zwischen mir und Ziel und bestimme, ob der Gegner zwischen mir und dem Ziel liegt.
  float dist = my_target_dist.sqr_norm(); //here really square norm is meant
  float alpha = ((my_target_dist.getX()*my_op_dist.getX())+(my_target_dist.getY()*my_op_dist.getY()))/dist;

  /* default: ignore obstacles */
  net_in[4] = 3.0;
  net_in[5] = 3.0;
  net_in[6] = 0;
  net_in[7] = PI/2;    
  
  obstacle_found = false;

  if (consider_obstacles) {
    if ((alpha>=0.0)&&(alpha<1.0)) {
      Vector cdist;
      cdist.setX( alpha*my_target_dist.getX() - my_op_dist.getX() );
      cdist.setY( alpha*my_target_dist.getY() - my_op_dist.getY() );
      if(cdist.norm() < 2.0){
	ANGLE angle_diff1 = ANGLE(my_target_dist_arg);
	angle_diff1 -= my_op_dist.ARG();
	
	net_in[4] = my_op_dist.norm();
	net_in[5] = cdist.norm();  
	net_in[6] = 0; //not used 
	net_in[7] = angle_diff1.get_value_mPI_pPI();
	obstacle_found = true;
      }
    }
  }
  if(op_mode ==1 ){
    net_in[4] = 0.0;
    net_in[5] = 0.0;
    net_in[6] = 0;
    net_in[7] = 0;    
    obstacle_found = false;
  }
}

bool NeuroGo2Pos::neuro_decide(Cmd & cmd) {
  //cmd.cmd_main.set_dash(20); return true;
  State state;
  State next_state;
  Cmd_Body best_action;
  double best_val;
  bool best_val_ok= false;

  get_cur_state( state );

  if(op_mode == 1){
    LOG_MOV(0,"Estimated cycles 2 target: "<<evaluate(state,target_pos)/ costs_per_action);
  }

  itr_actions.reset();
  Angle a= state.my_angle.get_value_0_p2PI(); 
  Angle na;
  Vector tmp= Vector(60,60);
  while ( Cmd_Body const* action = itr_actions.next() ) {
    Tools::model_cmd_main( state.my_pos, 
			   state.my_vel, 
			   a,
			   tmp,
			   tmp,
			   *action,
			   next_state.my_pos,
			   next_state.my_vel,
			   na,
			   tmp,
			   tmp );
    next_state.my_angle= ANGLE(na);
    double val= evaluate( next_state, target_pos );

    if ( !best_val_ok || val < best_val ) {
      best_val= val;
      best_action= *action;
      best_val_ok= true;
    }
  }

  if ( best_val_ok ) {
    double power;
    Angle angle;
    if ((best_action.get_type(power, angle) == Cmd_Body::TYPE_TURN) && !obstacle_found) {
      best_action.unset_lock();
      best_action.unset_cmd();
      //LOG_DEB(0, << "target pos is " << target_pos.x << " " << target_pos.y);
      double new_ang = Tools::my_angle_to(target_pos).get_value();
      new_ang = Tools::get_angle_between_mPI_pPI(new_ang);
      //LOG_DEB(0, << "old turn angle in go2pos: " << angle << " new angle " << new_ang);
      basic_cmd->set_turn_inertia(new_ang);
      return basic_cmd->get_cmd(cmd);
      }
    return cmd.cmd_body.clone( best_action );
  }

  return false;
}

bool NeuroGo2Pos::is_failure( State const& state,  Vector const& target) {
  if( state.my_pos.sqr_distance( state.opnt1_pos) < SQUARE(0.3) )
    return true;
  return false;
}


bool NeuroGo2Pos::is_success( State const& state,  Vector const& target) {
  if( state.my_pos.sqr_distance( target ) <=  SQUARE( target_tolerance ) ) 
    return true;
  return false;
}

double NeuroGo2Pos::evaluate( State const& state, Vector const& target ) {
  if(is_failure(state, target))
    return costs_failure;

  if(is_success(state,target))  
    return costs_success;

  get_features(state, target, net->in_vec);
  net->forward_pass(net->in_vec,net->out_vec);
  return(net->out_vec[0]);
}

void NeuroGo2Pos::set_target(Vector target, double tolerance, int cons_obstacles, int use_old) {
#if 0
  if(target->sqr_distance(target_pos) >= SQUARE(0.1)){ // new sequence has started
    // something useful could be done here
  }
#endif
  consider_obstacles = cons_obstacles;
  use_old_go2pos = use_old;
  target_tolerance = tolerance;
  target_pos= target; 
}


/***********************************************************************************/

/* LEARNING STUFF */

/***********************************************************************************/

bool NeuroGo2Pos::learn(Cmd &cmd){
  if(num_stored < store_per_cycle){
    store_state();
    num_stored ++;
  }
  else{ // time for learning !!
    num_stored = 0; // reset
    store_state();
    num_stored ++;
    LOG_MOV(0,<<"NeuroGo2Pos: Time 2 learn");
    for(int i=0;i<repeat_mainlearningloop; i++){
      generate_training_patterns();
      print_memory();
      train_nn();
    }
    LOG_ERR(0,<<"NeuroGo2Pos: Finished training cylce");
  }
  return false; // learning set no cmd 
}

void NeuroGo2Pos::check_cmd(Cmd &cmd) {
  State next_state,state;
  get_cur_state(state);
  Angle a= state.my_angle.get_value_0_p2PI(); 
  Angle na;
  Vector tmp= Vector(60,60);

  Tools::model_cmd_main( state.my_pos, state.my_vel, a,tmp,tmp, cmd.cmd_body,next_state.my_pos,
			 next_state.my_vel,na,tmp, tmp );
  if((next_state.my_pos.sqr_distance( Vector(0,0)) >=  SQUARE( 30. )) &&
     state.my_pos.sqr_distance(Vector(0,0)) + 1. < next_state.my_pos.sqr_distance(Vector(0,0))){
    if(cmd.cmd_body.get_type() == Cmd_Body::TYPE_TURN){
      LOG_MOV(0,<<"check turn cmd ");
    }
    LOG_MOV(0,<<"NeuroGo2Pos: Cmd would bring me out of area; Reset"
	    <<" my sqr dist "<<state.my_pos.sqr_distance(Vector(0,0))
	    <<" my next sqr dist "<<next_state.my_pos.sqr_distance(Vector(0,0))
	    );

    cmd.cmd_body.unset_lock();
    cmd.cmd_body.unset_cmd();
    cmd.cmd_body.set_turn(PI/2.); 
  }
}


void NeuroGo2Pos::store_state(){
  //State state;

  if(state_memory_ctr >= STATE_MEMORY_SIZE){
    LOG_MOV(0,<<"NeuroGo2Pos: Warning state memory full");
    return;
  }
  //  get_cur_state( state );
  //  state_memory[state_memory_ctr].state.my_pos = state.my_pos;
  get_cur_state(state_memory[state_memory_ctr].state);
  state_memory[state_memory_ctr].target_pos = target_pos;
  state_memory_ctr ++;
  LOG_MOV(0,<<"NeuroGo2Pos: Store state "<<state_memory_ctr);
}

void NeuroGo2Pos::print_memory(){
#if 1
  for(int i=0;i<state_memory_ctr; i++){
    LOG_MOV(0,<<"NeuroGo2Pos: Stored state "<<i<<" : "
	    << state_memory[i].state.my_pos
	    << state_memory[i].state.my_vel
	    << RAD2DEG(state_memory[i].state.my_angle.get_value())
	    << state_memory[i].state.opnt1_pos
	    <<"   target pos :"<< state_memory[i].target_pos
	    <<" Dist: "<<state_memory[i].state.my_pos.distance(state_memory[i].target_pos)
	    );
  }
#endif
  for(int i=0;i<training_set.ctr; i++){
    LOG_MOV(0,<<"NeuroGo2Pos: Training pattern "<<i<<" : "
	    <<training_set.input[i][0]<<" "
	    <<training_set.input[i][1]<<" "
	    <<training_set.input[i][2]<<" "
	    <<training_set.input[i][3]<<" "
	    <<training_set.input[i][4]<<" "
	    <<training_set.input[i][5]<<" "
	    <<training_set.input[i][6]<<" "
	    <<training_set.input[i][7]<<" "
	    <<" ---> "<<training_set.target[i]<<" ");
  }
}

void NeuroGo2Pos::generate_training_patterns(){
  float target_val;

#define RANGE 3
#define DX 1
#define DY 1
#define VRANGE 1
#define DV 1
#define DA PI/2.

  training_set.ctr =0;

  if(use_regular_states){
    state_memory_ctr= 0;  // simply overwrite memory
    for(double x = -RANGE; x<= RANGE; x += DX){
      for(double y = -RANGE; y<= RANGE; y += DY){
	for(double vx = -VRANGE; vx<= VRANGE; vx += DV){
	  for(double vy = -VRANGE; vy<= VRANGE; vy += DV){
	    for(double a = -PI; a< PI; a += DA){
	      state_memory[state_memory_ctr].state.my_pos = Vector(x,y);
	      state_memory[state_memory_ctr].state.my_vel = Vector(vx,vy);
	      state_memory[state_memory_ctr].state.my_angle = ANGLE(a);
	      state_memory[state_memory_ctr].target_pos = Vector (0,0);
	      state_memory_ctr ++;
	    }
	  }
	}
      }
    }
  } // if regular states

  for(int i=0;i<state_memory_ctr; i++){
    if(is_success(state_memory[i].state, state_memory[i].target_pos)){
      target_val = costs_success;
    }
    else{
      target_val = costs_per_action + 
	get_value_of_best_successor(state_memory[i].state, state_memory[i].target_pos);
    }

    get_features(state_memory[i].state, state_memory[i].target_pos, training_set.input[training_set.ctr]);
    training_set.target[training_set.ctr] = target_val;
    training_set.ctr ++;
  }
  //LOG_ERR(0,<<"Generated n training pats: "<<training_set.ctr);
}


double NeuroGo2Pos::get_value_of_best_successor(State const &state, Vector const& target) {
  State next_state;
  double best_val;
  bool best_val_ok= false;

  itr_actions.reset();
  Angle a= state.my_angle.get_value_0_p2PI(); 
  Angle na;
  Vector tmp= Vector(60,60);
  while ( Cmd_Body const* action = itr_actions.next() ) {
    Tools::model_cmd_main( state.my_pos, state.my_vel, a,tmp,tmp, *action,next_state.my_pos,
			   next_state.my_vel,na,tmp, tmp );
    next_state.my_angle= ANGLE(na);
    double val= evaluate( next_state, target );
    if ( !best_val_ok || val < best_val ) {
      best_val= val;
      best_val_ok= true;
    }
  }

  if ( best_val_ok ) 
    return best_val;

  LOG_ERR(0,<<"NeuroGo2Pos: Error: did not find best successor state");
  LOG_MOV(0,<<"NeuroGo2Pos: Error: did not find best successor state");
  return -1; // error
}



void NeuroGo2Pos::train_nn(){
  float error,tss; 

  net->init_weights(0,.5);
  net->set_update_f(1,uparams); // 1 is Rprop
  for(int n=0;n<num_epochs; n++){
    tss = 0.0;
    for(int p=0;p<training_set.ctr;p++){
      net->forward_pass(training_set.input[p],net->out_vec);
      for(int i=0;i<net->topo_data.out_count;i++){
	error = net->out_vec[i] = net->out_vec[i] - training_set.target[p]; 
	/* out_vec := dE/do = (o-t) */
	tss += error * error;
      }
      net->backward_pass(net->out_vec,net->in_vec);
    }
    net->update_weights();  /* learn by epoch */
    LOG_MOV(0,<<"NeuroGo2pos: TSS: "<<tss);
  }
}

