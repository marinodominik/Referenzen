#include "neuro_intercept_bms.h"

#include "log_macros.h"
#include "ws_info.h"
#include "tools.h"
#include "valueparser.h"
#include "intercept2.h"  // new go2pos

/* JTS: initialize constants */
 /* neuro_intercept uses dashs */
const double NeuroInterceptItrActions::dash_power_min = 40;
const double NeuroInterceptItrActions::dash_power_inc = 20;
const double NeuroInterceptItrActions::dash_power_max= 100;
const double NeuroInterceptItrActions::dash_power_steps = (int)((dash_power_max - dash_power_min)/dash_power_inc) +1;

  /* neuro_intercept uses turns */
const double NeuroInterceptItrActions::turn_min= -M_PI;  // important: start with strongest turn to the right
const double NeuroInterceptItrActions::turn_max= M_PI;  // and end with strongest turn to the left!
const double NeuroInterceptItrActions::turn_inc= 2*M_PI/72.0;
const int NeuroInterceptItrActions::turn_steps=(int)((turn_max - turn_min)/turn_inc) +1;

const double NeuroIntercept::MAX_INTERCEPT_DIST= 5.0;
const double NeuroIntercept::costs_failure = 1.0;
const double NeuroIntercept::costs_success = 0.0;

Net * NeuroIntercept::net;
bool NeuroIntercept::initialized= false;
int NeuroIntercept::op_mode;
bool NeuroIntercept::do_stochastic;
int NeuroIntercept::learn12step;
int NeuroIntercept::init_mode;
int NeuroIntercept::max_sequence_len;
long NeuroIntercept::num_trainstates;
long NeuroIntercept::num_teststates;
double NeuroIntercept::init_param;
double NeuroIntercept::learn_param;
double NeuroIntercept::prob_easy;
double NeuroIntercept::vball_angle_min;
double NeuroIntercept::vball_angle_max;
int NeuroIntercept::num_stored;
int NeuroIntercept::num_epochs;
int NeuroIntercept::store_per_cycle;
int NeuroIntercept::repeat_mainlearningloop;
int NeuroIntercept::state_memory_ctr;
int NeuroIntercept::test_memory_ctr;
int NeuroIntercept::num_testrepeats;
int NeuroIntercept::neuro_maxcycles;
double NeuroIntercept::safety_margin;
double NeuroIntercept::costs_per_action;
double NeuroIntercept::stress;
bool NeuroIntercept::use_regular_states;
bool NeuroIntercept::do_net_init;
bool NeuroIntercept::do_pi;
bool NeuroIntercept::do_reference;
bool NeuroIntercept::adjust_targets;

#define TRAIN_PROT(XXX)(cout<<XXX)
#define TRAIN_RESULT(XXX)(resultfile<<XXX)
#define GET_RANDOM(X,Y)((X)+drand48() *((Y)-(X)))


namespace regular{
    float x_range, dx, v_range, dv, vball_range, dvball;
    int angle_steps;
    float prob;
}

namespace test{
    float x_range, dx, v_range, dv, vball_range, dvball;
    int angle_steps;
    float prob;
}

namespace statistics{
  long total_wins, total_draws;
}



NeuroIntercept::~NeuroIntercept(){ 
  if(op_mode == 1){// do learning
    delete [] state_memory;
    delete [] test_memory;
    for(int i= 0;i<STATE_MEMORY_SIZE_ICPT;i++){
      delete [] training_set.input[i];
    }
    delete [] training_set.input;
    delete [] training_set.target;
  }

  cout<<"NeuroIntercept: DeInit "<<endl;
}

NeuroIntercept::NeuroIntercept(){ 

  intercept = new InterceptBall;
  onetwostep_intercept = new OneTwoStep_Intercept;

  op_mode = 0;
  init_mode = 0;
  num_stored = 0;
  state_memory_ctr = 0;
  test_memory_ctr = 0;
  do_net_init = false;
  stress = 1.0;
  do_pi = false;
  adjust_targets = true;

  srand48(16514);  // always draw the same samples

  statistics::total_wins = 0;
  statistics::total_draws = 0;

  sprintf(save_name_suffix,"%s","");

#define MY_MAX_SPEED 1.0 // this is the maximum speed for a homogeneous player!

#if 0
#define RANGE 10
#define DX .5
#define VRANGE .4
#define DV .2
#define VRANGE_ball 1.0
#define DV_ball .5
#define DA PI/8.
#endif

  regular::x_range = 10;
  regular::dx = .5;
  regular::v_range = .4;
  regular::dv = .2;
  regular::vball_range = 0.;
  regular::dvball = .5;
  regular::angle_steps= 8;
  regular::prob=1.0;

  test::x_range = 8;
  test::dx = 4;
  test::v_range = .4;
  test::dv = .4;
  test::vball_range = .5;
  test::dvball = .5;
  test::angle_steps= 3;
  test::prob=1.0;

  num_epochs= 50;
  store_per_cycle = 100;
  repeat_mainlearningloop = 10000;
  safety_margin = 0.15;
  costs_per_action = 0.02;
  use_regular_states = true;
  learn_params[0] = .1;
  learn_params[1] = 1.0;
  learn_params[2] = 0.0;
  learn_params[3] = 0.0;
  init_param = 0.5;
  learn_param = 0.0001;
  train_loops_ctr = 0;
  num_trainstates = 10000;
  num_teststates = 30;
  prob_easy = 0.1;
  learn12step = 0;
  vball_angle_min = 0.0;
  vball_angle_max = 0.0;
  do_stochastic = false;
  num_testrepeats = 1;
  do_reference = true;
  neuro_maxcycles = 100;

  ValueParser vp(CommandLineOptions::policy_conf,"NeuroIntercept_bms");
  //vp.set_verbose(true);
  vp.get("do_pi", do_pi);
  vp.get("adjust_targets", adjust_targets);
  vp.get("learn12step", learn12step);
  vp.get("num_trainstates", num_trainstates);
  vp.get("num_teststates", num_teststates);
  vp.get("neuro_maxcycles", neuro_maxcycles);
  vp.get("num_testrepeats", num_testrepeats);
  vp.get("max_sequence_len", max_sequence_len);
  vp.get("prob_easy", prob_easy);
  vp.get("vball_angle_min", vball_angle_min);
  vp.get("vball_angle_max", vball_angle_max);
  vp.get("init_mode", init_mode);
  vp.get("init_param", init_param);
  vp.get("learn_param", learn_param);
  vp.get("stress", stress);
  vp.get("op_mode", op_mode);
  vp.get("do_stochastic", do_stochastic);
  vp.get("do_net_init", do_net_init);
  vp.get("num_epochs", num_epochs);
  vp.get("safety_margin", safety_margin);
  vp.get("costs_per_action", costs_per_action);
  vp.get("repeat_mainlearningloop", repeat_mainlearningloop);
  vp.get("train_loops_ctr", train_loops_ctr);
  vp.get("xrange", regular::x_range);
  vp.get("dx", regular::dx);
  vp.get("vrange", regular::v_range);
  vp.get("dv", regular::dv);
  vp.get("vballrange", regular::vball_range);
  vp.get("dvball", regular::dvball);
  vp.get("angle_steps", regular::angle_steps);
  vp.get("test::xrange", test::x_range);
  vp.get("test::dx", test::dx);
  vp.get("test::vrange", test::v_range);
  vp.get("test::dv", test::dv);
  vp.get("test::vballrange", test::vball_range);
  vp.get("test::dvball", test::dvball);
  vp.get("test::angle_steps", test::angle_steps);
  vp.get("prob", regular::prob);
  vp.get("test::prob", test::prob);
  vp.get("save_name_suffix", save_name_suffix,500);

  if(op_mode == 1){// do learning
    if(init_mode == 1)
      learn_params[0] = .0001;

    if(initialized){// do learning
      if(do_net_init)
	net->init_weights(0,.5);
      net->set_update_f(1,learn_params); // 1 is Rprop
    }
    state_memory = new MyStateMemoryEntry[STATE_MEMORY_SIZE_ICPT];
    test_memory = new MyStateMemoryEntry[TEST_MEMORY_SIZE_ICPT];
    training_set.input = new float* [STATE_MEMORY_SIZE_ICPT];
    training_set.target = new float[STATE_MEMORY_SIZE_ICPT];
    for(int i= 0;i<STATE_MEMORY_SIZE_ICPT;i++){
      training_set.input[i] = new float[NUM_ICPT_FEATURES];
    }
    // generate test memory
#if 0
    generate_test_state(Vector(3,0), Vector(0,0), Vector(0,0), Vector(0,0), 0.8 *PI);
    generate_test_state(Vector(5,0), Vector(0,0), Vector(0,0), Vector(0,0), 0.7 *PI);
    generate_test_state(Vector(5,0), Vector(.4,0), Vector(0,0), Vector(0,0), 0.7 *PI);
    generate_test_state(Vector(5,0), Vector(-.4,0), Vector(0,0), Vector(0,0), 0.7 *PI);
    generate_test_state(Vector(5,0), Vector(.2,2), Vector(0,0), Vector(0,0), 0.7 *PI);
    generate_test_state(Vector(5,0), Vector(-.2,2), Vector(0,0), Vector(0,0), 0.7 *PI);
    generate_test_state(Vector(5,0), Vector(.2,-2), Vector(0,0), Vector(0,0), 0.7 *PI);
    generate_test_state(Vector(10,0), Vector(0,0), Vector(0,0), Vector(0,0), 0.6 *PI);
#endif
    //    generate_test_state(Vector(5,0), Vector(-.4,0), Vector(0,0), Vector(0,0), 0.);
    //generate_test_state(Vector(5,0), Vector(.4,0), Vector(0,0), Vector(0,0), 0.);
    //generate_test_state(Vector(0,0), Vector(0,0), Vector(15.,4.), Vector(-2.7,0), 0.25 *PI);
    //generate_test_state(Vector(0,0), Vector(0,0), Vector(12.,-1.), Vector(-1.9,1.1), 0.25 *PI);
    //generate_test_state(Vector(0,0), Vector(0,0), Vector(18.,0.), Vector(0.263312,0.3761128), 0.125 *PI);
    generate_test();
  }

#if 0
  TRAIN_PROT("NeuroIntercept_bms: Read Parameters. "<<endl
	     <<"op_mode "<<op_mode
	     <<" do_net_init "<<do_net_init
	     <<" num_epochs "<<num_epochs
	     <<" safety_margin "<<safety_margin
	     <<" costs_per_action "<<costs_per_action<<endl
	     <<"save name suffix "<<save_name_suffix<<endl
	     <<"x_range "<<regular::x_range
	     <<" dx "<<regular::dx 
	     <<" v_range "<<regular::v_range
	     <<" dv "<<regular::dv <<endl
	     <<"vball_range "<<regular::vball_range
	     <<" dvball "<<regular::dvball 
	     <<" angle_steps "<<regular::angle_steps<<endl
	     <<endl);
#endif
}

void NeuroIntercept::generate_test(){
  Vector my_vel;
  Vector ball_vel;
  //  double y= 0;
  double a;
  Angle ballvel_angle;

  const int test_mode = 1; // earlier version was 0
  //const int num_teststates = 30;

  if(test_mode == 1){
    while(test_memory_ctr < num_teststates){
      double x=GET_RANDOM(-test::x_range,test::x_range);
      double y=GET_RANDOM(-test::x_range,test::x_range);
      a = GET_RANDOM(0,PI);
      my_vel.setX( GET_RANDOM(-test::v_range,test::v_range) );
      my_vel.setY( GET_RANDOM(-test::v_range,test::v_range) );
      if(vball_angle_min == 0 && vball_angle_max == 0){ // standard case
	ball_vel.setX( GET_RANDOM(0,test::vball_range) );
	ball_vel.setY( 0 );
      }
      else{
	do{
	  ball_vel.setX( GET_RANDOM(-test::vball_range,test::vball_range) );
	  ball_vel.setY( GET_RANDOM(-test::vball_range,test::vball_range) );
	  ballvel_angle = Tools::get_angle_between_null_2PI(ball_vel.arg());
	} 
	while(ballvel_angle<vball_angle_min || ballvel_angle > vball_angle_max);
      }

      if(SQUARE(ball_vel.getX()) + SQUARE(ball_vel.getY()) >= SQUARE(ServerOptions::ball_speed_max)){
	ball_vel.normalize(ServerOptions::ball_speed_max);
      }
      if(SQUARE(my_vel.getX())+SQUARE(my_vel.getY())>=SQUARE(MY_MAX_SPEED *ServerOptions::player_decay)){
	my_vel.normalize(MY_MAX_SPEED *ServerOptions::player_decay);
      }
      
      if((my_vel.norm() > 0.16) && // 1.0 * 0.4 * 0.4
	 (Tools::get_abs_angle(my_vel.arg() - a) > 2/180. *PI)){
	continue;
      }
      
      Vector mypos = Vector(x,y);
      if(mypos.norm() < 1.2)
	mypos.normalize(1.2);
      generate_test_state(mypos,my_vel, Vector(0,0),ball_vel,a);
    }
    return;
  } //test_mode == 1



  for(double x = -test::x_range; x<= test::x_range; x += test::dx){
  for(double y = -test::x_range; y<= test::x_range; y += test::dx){
    for(double vx = -test::v_range; vx<= test::v_range; vx += test::dv){
      for(double vy = -test::v_range; vy<= test::v_range; vy += test::dv){
#if 0 
	for(double vx_ball = -test::vball_range; vx_ball<= test::vball_range; vx_ball += test::dvball){
	  for(double vy_ball = -test::vball_range; vy_ball<= test::vball_range; vy_ball += test::dvball){
#endif
	for(double vx_ball = 0; vx_ball<= test::vball_range; vx_ball += test::dvball){
	  //for(double vy_ball =0; vy_ball<= test::vball_range; vy_ball += test::dvball){
	  for(double vy_ball =0; vy_ball<=0; vy_ball += test::dvball){
	    for(int asteps = 0; asteps<= test::angle_steps; asteps ++ ){
	      a = asteps * PI/(float)(test::angle_steps);
	      if(asteps ==test::angle_steps)
		a = PI;
	      ball_vel = Vector(vx_ball,vy_ball);
	      if(SQUARE(ball_vel.getX()) + SQUARE(ball_vel.getY()) >= SQUARE(ServerOptions::ball_speed_max)){
		ball_vel.normalize(ServerOptions::ball_speed_max);
	      }
	      my_vel = Vector(vx,vy);
	      if(SQUARE(my_vel.getX())+SQUARE(my_vel.getY())>=SQUARE(MY_MAX_SPEED *ServerOptions::player_decay)){
		my_vel.normalize(MY_MAX_SPEED *ServerOptions::player_decay);
	      }
	      if((my_vel.norm() >= 0.15) &&
		 (Tools::get_abs_angle(my_vel.arg() - a) > 2/180. *PI)){
#if 0
		TRAIN_PROT("max speed "<<my_vel.norm()<<" and vel angle "
			   <<RAD2DEG(my_vel.arg())<<" differs from my angle "
			   <<RAD2DEG(a)<<endl);
#endif
		continue;
	      }
	      if(test::prob >= drand48()){
		if(x==0)
		  generate_test_state(Vector(1.2,y),my_vel, Vector(0,0),ball_vel,a);
		else
		  generate_test_state(Vector(x,y),my_vel, Vector(0,0),ball_vel,a);
	      } // take test state
	    }
	  }
	}
      }
	}
    }
  }
}

bool NeuroIntercept::init(char const * conf_file, int argc, char const* const* argv) {
  if(initialized) return true; // only initialize once...
  initialized= true;


  InterceptBall::init(conf_file,argc,argv);

  net= new Net();

  /* load neural network */
  //  char netname[] = "./data/nets_neuro_intercept/intercept_5_10.net";

  char netname[500];
  sprintf(netname,"train.net");

  ValueParser vp(CommandLineOptions::policy_conf,"NeuroIntercept_bms");
  //vp.set_verbose(true);
  vp.get("load_net", netname,500);

  if(net->load_net(netname) == FILE_ERROR){
    ERROR_OUT << "NeuroIntercept_bms: No net-file found "<<netname<<" - stop loading\n";
    initialized = false;
    exit(0);
    return false;
  }
//  cout<<"\nNeuroIntercept_bms successfully initialized. Net used: "<<netname<<endl;
  return true;
}

int NeuroIntercept::get_steps2intercept(){
  double dummyResultingBallDist;
  return get_steps2intercept(dummyResultingBallDist);
}
 
int NeuroIntercept::get_steps2intercept(double & resultingBallDist){
  int num_cycles;
  Cmd intercept_cmd, cmd;


  if(op_mode == 0 || op_mode == 2){
    if(onetwostep_intercept->get_cmd(cmd,num_cycles,resultingBallDist) == true)
      return num_cycles;
    
    intercept->get_cmd(intercept_cmd, num_cycles, resultingBallDist); // determine the number of cycles2go
    return num_cycles;
  }

  if(op_mode == 3){
    Intercept2 inter;
    inter.intercept( WSinfo::ball->pos, WSinfo::ball->vel, WSinfo::me, num_cycles, cmd.cmd_body);
    return num_cycles;
  }

  return -1;
  
}

bool NeuroIntercept::get_cmd(Cmd & cmd) { 

  if ( ! initialized ) {
    ERROR_OUT << "NeuroIntercept not intialized";
    return false;
  }

  MyState state;
  get_cur_state( state );


  if(op_mode == 3){
    Intercept2 inter;
    int time;
    //    inter.intercept( WSinfo::ball->pos, WSinfo::ball->vel, WSinfo::me, time, cmd.cmd_main);
    inter.intercept( state.ball_pos, state.ball_vel, WSinfo::me, time, cmd.cmd_body);
    LOG_MOV(0,<<"NeuroIntercept: Using ANALYTICAL intercept. estimated steps 2 go "<<time);
    return true;
  }

  int num_cycles;
  Cmd intercept_cmd;

  intercept->get_cmd(intercept_cmd,state.my_pos,state.my_vel,
		     state.my_angle,state.ball_pos,state.ball_vel,num_cycles, true);//allow riskiness
  //intercept->get_cmd(intercept_cmd, num_cycles); // determine the number of cycles2go
  if((op_mode == 2) || (num_cycles > neuro_maxcycles)){// Sungs intercept
    int steps;
    if(onetwostep_intercept->get_cmd(cmd,state,steps)){ // check onetwostep first
      //TRAIN_PROT("NIcpt: onetwostep takes "<<steps<<" steps"<<endl);
      LOG_MOV(0,<<"NeuroIntercept: Onetwostep takes control");
      return true;
    }
    LOG_MOV(0,<<"NeuroIntercept: op mode 2 or estimated cycles "<<num_cycles<<" >neuro_maxcycles: "
	    <<neuro_maxcycles<<" : Calling Sung's intercept");

    cmd.cmd_body.clone(intercept_cmd.cmd_body);

    /*
    intercept->get_cmd(cmd,state.my_pos,state.my_vel,
		       state.my_angle,state.ball_pos,state.ball_vel);
    */
    return true;
  }

  if(op_mode == 1){ // store and learn
    if(use_regular_states){
      learn(cmd); 
    } 
    else{ // use store states mechanism
      if(num_stored <= store_per_cycle){
	store_state();
	num_stored ++;
      }
      if(num_stored == store_per_cycle){
	learn(cmd); 
	num_stored = 0;
      }
    } // use store states
  } // op_mode == learn
  if(WSinfo::is_ball_kickable()){
     LOG_MOV(0,<<"NeuroIntercept: ball reached");
     LOG_ERR(0,<<"Yeah I reached the ball!!");
     cmd.cmd_body.set_turn(0);  // Ok, I'm fine, do nothing 
     return false;
  }

  neuro_decide(state, cmd); 

  if(op_mode == 1){
    check_cmd(cmd);
  }
  LOG_MOV(0,<<"NeuroIntercept: Selected CMD: "<<cmd.cmd_body);
  return true;
}

void NeuroIntercept::set_virtual_state(Vector const ballpos, Vector const ballvel){
  virtual_state.ballpos = ballpos;
  virtual_state.ballvel = ballvel;
  virtual_state.my_pos.setX( 60.0 );
  virtual_state.valid_at = WSinfo::ws->time;
}

void NeuroIntercept::set_virtual_state(Vector const mypos, Vector const myvel, ANGLE myang, Vector const ballpos, Vector const ballvel){
  virtual_state.my_pos=mypos;
  virtual_state.my_vel=myvel;
  virtual_state.my_angle=myang;
  virtual_state.ballpos = ballpos;
  virtual_state.ballvel = ballvel;
  virtual_state.valid_at = WSinfo::ws->time;
}

void NeuroIntercept::get_cur_state( MyState & state) {
  if(virtual_state.valid_at == WSinfo::ws->time){ // currently valid
    state.ball_pos = virtual_state.ballpos;
    state.ball_vel = virtual_state.ballvel;
    if(virtual_state.my_pos.getX()!=60.0){
      state.my_pos=virtual_state.my_pos;
      state.my_vel=virtual_state.my_vel;
      state.my_angle=virtual_state.my_angle;}
    else{
      state.my_pos= WSinfo::me->pos;
      state.my_vel= WSinfo::me->vel;
      state.my_angle=  WSinfo::me->ang;
      }
    LOG_MOV(0,<<"Neuro Intercept: Virtual ballpos is set: pos "<<state.ball_pos
	    <<" vel "<<state.ball_vel);
  }
  else{ // take actual values
    state.my_pos= WSinfo::me->pos;
    state.my_vel= WSinfo::me->vel;
    state.my_angle=  WSinfo::me->ang;
    state.ball_pos = WSinfo::ball->pos;
    state.ball_vel = WSinfo::ball->vel;
  }
}

void NeuroIntercept::get_features(MyState const& state, float * net_in) {
  /** features of neuro intercept ball:
      (0) distance between player and ball
      (1) angle between player`s view direction and ball (-PI ~ PI)
      (2) player velocity x-direction (rotated)
      (3) player velocity y-direction (rotated)
      (4) ball velocity x-direction (rotated)
      (5) ball velocity y-direction (rotated)
  */ 
  Vector ball_vel = state.ball_vel;
  Vector my_pos = state.my_pos - state.ball_pos;
  Vector my_vel = state.my_vel;
  double my_angle = state.my_angle.get_value();
  
  float angle = my_pos.arg();
  /* rotate whole system (pos.y should be zero*/
  my_pos.rotate(2*PI - angle);
  my_vel.rotate(2*PI - angle);
  ball_vel.rotate(2*PI - angle);
  my_angle = Tools::get_angle_between_mPI_pPI((my_angle - angle)-PI);
  
  if(my_angle<0) {
    my_vel.setY( -my_vel.getY() );
    ball_vel.setY( -ball_vel.getY() );
  }
  
#if 1
  net_in[0] = my_pos.norm(); //distance to target
  net_in[1] = fabs(my_angle); //abs. relative view angle to target
  net_in[2] = my_vel.getX(); //velocity in x direction
  net_in[3] = my_vel.getY();
  net_in[4] = 1.0*ball_vel.getX();
  net_in[5] = 1.0*ball_vel.getY();
#endif
}

bool NeuroIntercept::check_onestep(Cmd & cmd, const MyState &state){
#define onestep_safetymargin .1

  Vector my_pos = state.my_pos;
  Vector my_vel = state.my_vel;
  Vector ball_pos = state.ball_pos;
  Vector ball_vel = state.ball_vel;
  ANGLE my_ang = state.my_angle;

  int bestdash = 200;
  double balldist,closest = 200;
  int dash;
  Vector dummy1,dummy2;
  Vector my_new_pos;
  Vector ball_new_pos;
  ANGLE my_new_angle;
  Cmd_Body command;

  for(int dashabs=0;dashabs<=100;dashabs+=50){
    for(int sign=1;sign>=-1;sign-=2){
      dash=sign*dashabs;
      command.unset_lock();
      command.unset_cmd();
      command.set_dash(dash);
      Tools::model_cmd_main(my_pos,my_vel,my_ang, ball_pos,ball_vel,
			    command, my_new_pos, dummy1, my_new_angle, ball_new_pos, dummy2);
        
      balldist = (my_new_pos - ball_new_pos).norm();
      if(balldist <= ServerOptions::kickable_area - onestep_safetymargin ){
	if((closest>0.7) && (balldist<closest)){
	  bestdash=dash;
	  closest=balldist;
	}	
      }
    }
  }
  
  if(bestdash < 200){
    if(bestdash==0){
      // Hey, I can get the Ball without dashing, so let's turn toward opponent goal
      Angle p_moment = ((Tools::opponent_goalpos()-my_pos).ARG() - my_ang).get_value_mPI_pPI();
      p_moment=p_moment*(1.0+(ServerOptions::inertia_moment*(my_vel.norm())));
      if (p_moment > 3.14) p_moment = 3.14;
      if (p_moment < -3.14) p_moment = -3.14;
      p_moment = Tools::get_angle_between_null_2PI(p_moment);
      cmd.cmd_body.set_turn(p_moment);
      return true;
    }
    cmd.cmd_body.set_dash(bestdash);
    return true;
  }
  return false;
}

bool NeuroIntercept::neuro_decide(const MyState &state, Cmd & cmd) {
  //cmd.cmd_main.set_dash(20); return true;
  MyState next_state;
  Cmd_Body best_action;
  double best_val;
  bool best_val_ok= false;

  /*
  if(check_onestep(cmd, state))
    return true;
  */

  int steps;

  if(onetwostep_intercept->get_cmd(cmd,state,steps)){
    //TRAIN_PROT("NIcpt: onetwostep takes "<<steps<<" steps"<<endl);
    LOG_MOV(0,<<"NeuroIntercept: Onetwostep takes control");
    return true;
  }

  //if(op_mode == 1){
#if 1
  double jnn = evaluate(state);
  jnn -= costs_success;
  LOG_MOV(0,"NeuroIntercept: Estimated cycles2intercept: "<<(jnn/ costs_per_action)
	  <<" Jnn: "<<jnn
	  <<" dist2ball "<<state.my_pos.distance(state.ball_pos));
#endif
    //}

  itr_actions.reset();
  while ( Cmd_Body const* action = itr_actions.next() ) {
    Tools::get_successor_state(state,*action,next_state);
    double val= evaluate( next_state );
    //LOG_MOV(1,"NeuroIntercept: Test Action; "<<*action<<" J:: "<<val);
    //TRAIN_PROT("Select action "<<info<<" successor "<<next_state<<" J: "<<val<<endl);
    if ( !best_val_ok || val < best_val ) {
      best_val= val;
      best_action= *action;
      best_val_ok= true;
    }
  }
  //  TRAIN_PROT(endl);

  if ( best_val_ok ) 
    return cmd.cmd_body.clone( best_action );

  return false;
}

bool NeuroIntercept::is_failure( MyState const& state) {
  return false;
}


bool NeuroIntercept::is_success( MyState const& state) {
  if(state.my_pos.sqr_distance(state.ball_pos)<=SQUARE(ServerOptions::kickable_area - safety_margin)) 
    return true;
  return false;
}

double NeuroIntercept::evaluate( MyState const& state ) {
  if(is_failure(state))
    return costs_failure;

  if(is_success(state))  
    return costs_success;

  if(learn12step == 2){
    Cmd dummy;
    int steps;
    if(onetwostep_intercept->get_cmd(dummy,state,steps)){ // I can do it in 1 or two steps!
      //TRAIN_PROT("Evaluate: I can do it in "<<steps<<" from state "<<state<<endl);
      return (costs_success + costs_per_action * steps);
    }
  }

  get_features(state, net->in_vec);
  net->forward_pass(net->in_vec,net->out_vec);
  return(net->out_vec[0]);
}

/***********************************************************************************/

/* LEARNING STUFF */

/***********************************************************************************/

bool NeuroIntercept::learn(Cmd &cmd){
  char savename[50];

  TRAIN_PROT("\nStart Training "<<endl);
  LOG_ERR(0,<<"NeuroIntercept: It's time 2 learn");
  do_test();
  for(int i=0;i<repeat_mainlearningloop; i++){
    TRAIN_PROT("Starting Training loop "<<i+1<<endl);
    generate_training_patterns();
    //print_memory();
    train_nn();
    train_loops_ctr ++;
    sprintf(savename,"%strained.%d.net",save_name_suffix,train_loops_ctr);
    net->save_net(savename);
    sprintf(savename,"%strained.current.net",save_name_suffix);
    net->save_net(savename); // save additionally
    do_test();
  }

  LOG_ERR(0,<<"NeuroIntercept: Finished training cylce");
  return false; // learning set no cmd 
}

void NeuroIntercept::check_cmd(Cmd &cmd) {
  MyState next_state,state;
  get_cur_state(state);
  Angle a= state.my_angle.get_value_0_p2PI(); 
  Angle na;
  Vector tmp= Vector(60,60);

  Tools::model_cmd_main( state.my_pos, state.my_vel, a,tmp,tmp, cmd.cmd_body,next_state.my_pos,
			 next_state.my_vel,na,tmp, tmp );

  if((next_state.my_pos.getX() >= ServerOptions::pitch_length/2.) ||
     (next_state.my_pos.getX() <= - ServerOptions::pitch_length/2.) ||
     (next_state.my_pos.getY() <= - ServerOptions::pitch_width/2.) ||
     (next_state.my_pos.getY() >=  ServerOptions::pitch_width/2.)){
    if(cmd.cmd_body.get_type() == Cmd_Body::TYPE_TURN){
      LOG_MOV(0,<<"check turn cmd ");
    }
    LOG_MOV(0,<<"NeuroIntercept: Cmd would bring me out of pitch; Reset"
	    <<" my next pos "<<next_state.my_pos
	    );

    cmd.cmd_body.unset_lock();
    cmd.cmd_body.unset_cmd();
    cmd.cmd_body.set_turn(PI/2.);
  }
}

void NeuroIntercept::store_state(){
  //MyState state;

  if(state_memory_ctr >= STATE_MEMORY_SIZE_ICPT){
    LOG_MOV(0,<<"NeuroIntercept: Warning state memory full");
    return;
  }
  //  get_cur_state( state );
  //  state_memory[state_memory_ctr].state.my_pos = state.my_pos;
  get_cur_state(state_memory[state_memory_ctr].state);
  state_memory_ctr ++;
  LOG_MOV(0,<<"NeuroIntercept: Store state "<<state_memory_ctr);
}

void NeuroIntercept::print_memory(){
#if 1
  for(int i=0;i<state_memory_ctr; i++){
    TRAIN_PROT("NeuroIntercept: Stored state "<<i<<" : "
	       << state_memory[i].state.my_pos
	       << state_memory[i].state.my_vel
	       << RAD2DEG(state_memory[i].state.my_angle.get_value())
	       << state_memory[i].state.ball_pos
	       << state_memory[i].state.ball_vel
	       << RAD2DEG(state_memory[i].state.ball_vel.arg()) <<endl
	       );
  }
#endif
#if 0
  for(int i=0;i<training_set.ctr; i++){
    LOG_MOV(1,<<"NeuroIntercept: Training pattern "<<i<<" : "
	    <<training_set.input[i][0]<<" "
	    <<training_set.input[i][1]<<" "
	    <<training_set.input[i][2]<<" "
	    <<training_set.input[i][3]<<" "
	    <<training_set.input[i][4]<<" "
	    <<training_set.input[i][5]<<" "
	    <<" ---> "<<training_set.target[i]<<" ");
  }
#endif
}


void NeuroIntercept::generate_training_patterns(){
  float target_val;

  training_set.ctr =0;
  Vector my_vel;
  Vector ball_vel;
  //  double y= 0;
  double a;
  Angle ballvel_angle;


  //srand48(16514);  // always draw the same samples
  if(use_regular_states){
    state_memory_ctr = 0;
    while(state_memory_ctr < num_trainstates){
      double x=GET_RANDOM(-regular::x_range,regular::x_range);
      double y=GET_RANDOM(-regular::x_range,regular::x_range);
      a = GET_RANDOM(0,PI);
      my_vel.setX( GET_RANDOM(-regular::v_range,regular::v_range) );
      my_vel.setY( GET_RANDOM(-regular::v_range,regular::v_range) );
      if(vball_angle_min == 0 && vball_angle_max == 0){ // standard case
	ball_vel.setX( GET_RANDOM(0,regular::vball_range) );
	ball_vel.setY( 0 );
      }
      else{
	do{
	  ball_vel.setX( GET_RANDOM(-regular::vball_range,regular::vball_range) );
	  ball_vel.setY( GET_RANDOM(-regular::vball_range,regular::vball_range) );
	  ballvel_angle = Tools::get_angle_between_null_2PI(ball_vel.arg());
	} 
	while(ballvel_angle<vball_angle_min || ballvel_angle > vball_angle_max);
      }


      if(SQUARE(ball_vel.getX()) + SQUARE(ball_vel.getY()) >= SQUARE(ServerOptions::ball_speed_max)){
	ball_vel.normalize(ServerOptions::ball_speed_max);
      }
      if(SQUARE(my_vel.getX())+SQUARE(my_vel.getY())>=SQUARE(MY_MAX_SPEED *ServerOptions::player_decay)){
	my_vel.normalize(MY_MAX_SPEED *ServerOptions::player_decay);
      }
      
      if((my_vel.norm() > 0.16) && // 1.0 * 0.4 * 0.4
	 (Tools::get_abs_angle(my_vel.arg() - a) > 2/180. *PI)){
	continue;
      }

      
      Vector mypos = Vector(x,y);
      mypos.normalize(1.2);
      state_memory[state_memory_ctr].state.my_pos = mypos;
      state_memory[state_memory_ctr].state.my_vel = my_vel;
      state_memory[state_memory_ctr].state.my_angle = ANGLE(a);
      state_memory[state_memory_ctr].state.ball_pos = Vector (0,0);
      state_memory[state_memory_ctr].state.ball_vel = ball_vel;
      if(prob_easy >=drand48()){
	//TRAIN_PROT("generated easy state: "<<state_memory[state_memory_ctr].state<<endl);
	state_memory_ctr ++;
      }
      
      state_memory[state_memory_ctr].state.my_pos = Vector(x,y);
      state_memory[state_memory_ctr].state.my_vel = my_vel;
      state_memory[state_memory_ctr].state.my_angle = ANGLE(a);
      state_memory[state_memory_ctr].state.ball_pos = Vector (0,0);
      state_memory[state_memory_ctr].state.ball_vel = ball_vel;
      //TRAIN_PROT("generated state: "<<state_memory[state_memory_ctr].state<<endl);
      state_memory_ctr ++; // else: do not consider !
    } // while statememorycounter < 10000

#if 0
    state_memory_ctr= 0;  // simply overwrite memory
    for(double x = -regular::x_range; x<= regular::x_range; x += regular::dx){
    for(double y = -regular::x_range; y<= regular::x_range; y += regular::dx){
      for(double vx = -regular::v_range; vx<= regular::v_range; vx += regular::dv){
	for(double vy = -regular::v_range; vy<= regular::v_range; vy += regular::dv){
#if 0
	  for(double vx_ball = -regular::vball_range; vx_ball<= regular::vball_range; vx_ball += regular::dvball){
	    for(double vy_ball = -regular::vball_range; vy_ball<= regular::vball_range; vy_ball += regular::dvball){
#endif
	  for(double vx_ball = 0; vx_ball<= regular::vball_range; vx_ball += regular::dvball){
	    //	    for(double vy_ball = 0; vy_ball<= regular::vball_range; vy_ball += regular::dvball){
	    for(double vy_ball = 0; vy_ball<= 0; vy_ball += regular::dvball){
	      for(int asteps = 0; asteps<= regular::angle_steps; asteps ++ ){
		if(regular::prob < drand48())
		  continue;
		a = asteps * PI/(float)(regular::angle_steps);
		if(asteps ==regular::angle_steps)
		  a = PI;
		ball_vel = Vector(vx_ball,vy_ball);
		if(SQUARE(ball_vel.x) + SQUARE(ball_vel.y) >= SQUARE(ServerOptions::ball_speed_max)){
		  ball_vel.normalize(ServerOptions::ball_speed_max);
		}
		my_vel = Vector(vx,vy);
		if(SQUARE(my_vel.x)+SQUARE(my_vel.y)>=SQUARE(MY_MAX_SPEED *ServerOptions::player_decay)){
		  my_vel.normalize(MY_MAX_SPEED *ServerOptions::player_decay);
		}
		     
		if((my_vel.norm() > 0.16) && // 1.0 * 0.4 * 0.4
		   (Tools::get_abs_angle(my_vel.arg() - a) > 5/180. *PI)){
#if 0
		  TRAIN_PROT("max speed "<<my_vel.norm()<<" and vel angle "
			     <<RAD2DEG(my_vel.arg())<<" differs from my angle "
			     <<RAD2DEG(a)<<endl);
#endif
		  continue;
		}
#if 0

		if(Vector(x,y).norm()> train_loops_ctr)
		  continue;
#endif



#if 0
		if(x==0){
		  state_memory[state_memory_ctr].state.my_pos = Vector(1.2,0);
		  state_memory[state_memory_ctr].state.my_vel = my_vel;
		  state_memory[state_memory_ctr].state.my_angle = a;
		  state_memory[state_memory_ctr].state.ball_pos = Vector (0,0);
		  state_memory[state_memory_ctr].state.ball_vel = ball_vel;
		  state_memory_ctr ++; // else: do not consider !
		}
		if(y==0){
		  state_memory[state_memory_ctr].state.my_pos = Vector(0,1.2);
		  state_memory[state_memory_ctr].state.my_vel = my_vel;
		  state_memory[state_memory_ctr].state.my_angle = a;
		  state_memory[state_memory_ctr].state.ball_pos = Vector (0,0);
		  state_memory[state_memory_ctr].state.ball_vel = ball_vel;
		  state_memory_ctr ++; // else: do not consider !
		}
#endif
		Vector mypos = Vector(x,y);
		mypos.normalize(1.2);
		state_memory[state_memory_ctr].state.my_pos = mypos;
		state_memory[state_memory_ctr].state.my_vel = my_vel;
		state_memory[state_memory_ctr].state.my_angle = a;
		state_memory[state_memory_ctr].state.ball_pos = Vector (0,0);
		state_memory[state_memory_ctr].state.ball_vel = ball_vel;
		if(0.1 >=drand48())
		  state_memory_ctr ++;


		state_memory[state_memory_ctr].state.my_pos = Vector(x,y);
		state_memory[state_memory_ctr].state.my_vel = my_vel;
		state_memory[state_memory_ctr].state.my_angle = a;
		state_memory[state_memory_ctr].state.ball_pos = Vector (0,0);
		state_memory[state_memory_ctr].state.ball_vel = ball_vel;
		state_memory_ctr ++; // else: do not consider !
	      }
	    }
	  }
	}
      }      
    }
      }
#endif

  } // if regular states

  for(int i=0;i<state_memory_ctr; i++){
    if(is_success(state_memory[i].state)){
      target_val = costs_success;
    }
    else{
      MyState successor_state;
      target_val = costs_per_action + 
	get_value_of_best_successor(state_memory[i].state,successor_state);
#if 0 // Residual Gradient
      get_features(successor_state, training_set.input[training_set.ctr]);
      float tmp_target = evaluate(state_memory[i].state) - costs_per_action;;
      training_set.target[training_set.ctr] = tmp_target;
      training_set.ctr ++;
#endif
    }

    get_features(state_memory[i].state, training_set.input[training_set.ctr]);
    if((adjust_targets == true) && (target_val > (train_loops_ctr+1) * costs_per_action))
      target_val = (train_loops_ctr+1) * costs_per_action;
    if(target_val > 0.9)
      target_val = 0.9;
      
    training_set.target[training_set.ctr] = target_val;
    //    if(target_val >= .99* costs_per_action)
    training_set.ctr ++;
  }
  TRAIN_PROT("Generated n training pats: "<<training_set.ctr
	     <<" No. of regular states "<<state_memory_ctr<<endl);
}

#if 0
void NeuroIntercept::Tools::get_successor_state(MyState const &state, Cmd_Body const &cmd, MyState &next_state) {
  Angle na;
  Angle a= state.my_angle.get_value_0_p2PI(); 

  Tools::model_cmd_main( state.my_pos, state.my_vel, a,state.ball_pos,state.ball_vel, 
			 cmd ,next_state.my_pos,
			 next_state.my_vel,na,next_state.ball_pos, next_state.ball_vel );
  next_state.my_angle= na;
  if((next_state.my_vel.norm()/ServerOptions::player_decay) > 1.08 * MY_MAX_SPEED){
    LOG_ERR(0,<<"Max speed too high "<<next_state.my_vel.norm()/ServerOptions::player_decay);
    LOG_ERR(0,<<"Position change: "<<(next_state.my_pos - state.my_pos).norm());
  }
}
#endif

double NeuroIntercept::get_value_of_best_successor(MyState const &state,
						  MyState &successor_state) {
  MyState next_state;
  double best_val;
  bool best_val_ok= false;

  if(learn12step == 1){
    Cmd dummy;
    int steps;
    if(onetwostep_intercept->get_cmd(dummy,state,steps)){ // I can do it in 1 or two steps!
      //TRAIN_PROT("Get Value: I can do it in "<<steps<<" from here "<<state<<endl);
      return(costs_success + costs_per_action * (steps-1));
    }
  }

  if(do_pi){ // policy iteration: just take neuro intercept
    Cmd cmd;
    intercept->get_cmd(cmd,state.my_pos,state.my_vel,
		       state.my_angle,state.ball_pos,state.ball_vel);
    Tools::get_successor_state(state,cmd.cmd_body ,next_state);
    successor_state = next_state;
    return (evaluate(next_state));
  }

  itr_actions.reset();
  double val;
  while ( Cmd_Body const* action = itr_actions.next() ) {
    Tools::get_successor_state(state,*action ,next_state);
    val= evaluate( next_state );
    if ( !best_val_ok || val < best_val ) {
      best_val= val;
      successor_state = next_state;
      best_val_ok= true;
    }
  }

  if ( best_val_ok ) 
    return best_val;

  LOG_ERR(0,<<"NeuroIntercept: Error: did not find best successor state");
  LOG_MOV(0,<<"NeuroIntercept: Error: did not find best successor state");
  return -1; // error
}



void NeuroIntercept::train_nn(){
  float error,tss; 
  float pattern_stress;
  int step1_ctr = 0;
  int step2_ctr = 0;

  if(init_mode == 0){
    net->init_weights(0,init_param);
    net->set_update_f(1,learn_params); // 1 is Rprop
  }
  else if(init_mode == 1){
    net->init_weights(2,init_param);
  }
  else if(init_mode == 2){
    //    net->init_weights(2,init_param);
    net->init_weights(2,init_param);
    learn_params[0]= learn_param;
    net->set_update_f(1,learn_params); // 1 is Rprop
  }
  net->save_net("init.net");
  for(int n=0;n<num_epochs; n++){
    tss = 0.0;
    step1_ctr = 0;
    step2_ctr = 0;
    for(int p=0;p<training_set.ctr;p++){
      net->forward_pass(training_set.input[p],net->out_vec);
      pattern_stress = 1.0;
      for(int i=0;i<net->topo_data.out_count;i++){
	if(training_set.target[p] == costs_success){
#if 0
	  TRAIN_PROT("Special pattern: "<<p<<" feature 0: "
		     <<training_set.input[p][0]<<" target "
		     <<training_set.target[p]<<endl);
#endif
	}
	if(fabs(training_set.target[p]-costs_per_action)<0.0001){
#if 0
	  TRAIN_PROT("Special pattern: "<<p<<" feature 0: "
		     <<training_set.input[p][0]<<" target "
		     <<training_set.target[p]<<endl);
#endif
	  step1_ctr ++;
	  pattern_stress = stress;
	}
	if(fabs(training_set.target[p]-2*costs_per_action)<0.0001){
#if 0
	  TRAIN_PROT("Special pattern: "<<p<<" feature 0: "
		     <<training_set.input[p][0]<<" target "
		     <<training_set.target[p]<<endl);
#endif
	  step2_ctr ++;
	  pattern_stress = stress;
	}
#if 0
	if(n==num_epochs-1){
	  TRAIN_PROT("pattern: "<<p<<" "
		     <<training_set.input[p][0]<<" "
		     <<training_set.input[p][1]<<" "
		     <<training_set.input[p][2]<<" "
		     <<training_set.input[p][3]<<" "
		     <<training_set.input[p][4]<<" "
		     <<training_set.input[p][5]<<" "
		     <<" output "<<net->out_vec[0]
		     <<" target "<<training_set.target[p]<<endl);
	}
#endif
	error = net->out_vec[i] = net->out_vec[i] - training_set.target[p]; 
	net->out_vec[i] *= pattern_stress;
	/* out_vec := dE/do = (o-t) */
	tss += error * error;
      }
       net->backward_pass(net->out_vec,net->in_vec);
    }
    //   TRAIN_PROT(endl);
    net->update_weights();  /* learn by epoch */
    LOG_MOV(0,<<"NeuroIntercept: epoch "<<n<<" TSS: "<<tss);
    LOG_ERR(0,<<"NeuroIntercept: epoch "<<n<<" TSS: "<<tss<<" per pattern "<<tss/float(training_set.ctr));
    if(n==0 || (n%20) == 0 || n==num_epochs -1)
      TRAIN_PROT("Train: Epoch "<<n<<" TSS: "<<tss<<" per pattern "<<tss/float(training_set.ctr)<<endl);
  }
  TRAIN_PROT("Counted "<<step1_ctr<<" 1 step patterns and "<<step2_ctr<<" two step pats"<<endl);
}

void NeuroIntercept::do_sequence(const MyState & initial_state, double *result, const int N){
  MyState cur_state, next_state;
  Cmd cmd;
  int n;

#define SEQ_VERBOSE 0
  int imax;

  if(do_reference)
    imax = 2;
  else
    imax = 1;

  for(int i=0;i<imax;i++){
    result[i] = 0;
    for(int k=0;k<num_testrepeats;k++){
      n=0;
      cur_state = initial_state;
      while(n<N &&(cur_state.my_pos.sqr_distance(cur_state.ball_pos)
		   >SQUARE(ServerOptions::kickable_area))){
	cmd.cmd_body.unset_lock();
	if(i==0)
	  neuro_decide(cur_state, cmd);
	else{
	  if(cur_state.my_pos.distance(cur_state.ball_pos) > test::x_range){ // stop sequence
	    //TRAIN_PROT("Do Sequence: ball too far: "<<cur_state<<endl);
	    result[i] = N;
	    return;
	  }
	  intercept->get_cmd(cmd,cur_state.my_pos,cur_state.my_vel,
			     cur_state.my_angle,cur_state.ball_pos,cur_state.ball_vel);
	}
	
	Tools::get_successor_state(cur_state, cmd.cmd_body, next_state, do_stochastic);
#if SEQ_VERBOSE
	TRAIN_PROT("t:"<<n<<" controller "<<i<<" state(t) "<<cur_state<<" cmd: "<<cmd.cmd_body<<endl);
	//<<" state(t+1) "<<next_state<<endl);
#endif
	cur_state = next_state;
	n++;
      }
#if SEQ_VERBOSE
      TRAIN_PROT(endl);
#endif
      result[i] += n;
    } // for k
    result[i] /= (float)num_testrepeats;
  }
}


void NeuroIntercept::do_test(){
  double average[2];
  int won[2];

  TRAIN_PROT("\nDo Test:"<<endl);
#if 0
  for(int i=0;i<test_memory_ctr; i++){
    double J = evaluate( test_memory[i].state );
    TRAIN_PROT("Eval: "<<test_memory[i].state.my_pos<<test_memory[i].state.my_vel
	       <<test_memory[i].state.ball_pos
	       <<test_memory[i].state.ball_vel<<RAD2DEG(test_memory[i].state.my_angle.get_value())
	       <<" J: "<<J<<" estimated cycles2go "<<J/costs_per_action<<endl);
  }
#endif

  for(int i=0;i<test_memory_ctr; i++){
    //TRAIN_PROT("do sequence "<<i<<" "<<test_memory[i].state<<endl);
    if(test_memory_result[i][1] < max_sequence_len)
      do_sequence(test_memory[i].state, test_memory_result[i], max_sequence_len);
    else {
      //      TRAIN_PROT("Test Seq "<<i<<" not valid, sungs intercept needs max. "<<endl);
    }
  }
  do_reference = false; // do it only once

  average[0]=   average[1]=0.0;
  won[0]= 0;
  won[1]=0;

  int i;

  for(i=0;i<test_memory_ctr; i++){
    if(test_memory_result[i][1] == max_sequence_len)
      continue;
    //TRAIN_PROT("i: "<<i<<endl);
    average[0] += test_memory_result[i][0];
    average[1] += test_memory_result[i][1];
    double J = evaluate( test_memory[i].state );
    if(test_memory_result[i][0]<test_memory_result[i][1]){
      won[0] ++;
      //cout<<"\nWon 0 incremented: "<<won[0]<<endl;
      TRAIN_PROT("+ Sequence "<<i<<" "<<test_memory[i].state
		 <<" J: "<<J<<" ("<<(J/costs_per_action)<<") "
		 <<" neuro: " <<test_memory_result[i][0]
		 <<" classic Sung "<<test_memory_result[i][1]<<endl);
    }
    else if(test_memory_result[i][0]>test_memory_result[i][1]){
      won[1] ++;
      TRAIN_PROT("- Sequence "<<i<<" "<<test_memory[i].state
		 <<" J: "<<J<<" ("<<(J/costs_per_action)<<") "
		 <<" neuro: " <<test_memory_result[i][0]
		 <<" classic Sung "<<test_memory_result[i][1]<<endl);
    }
    else{
      TRAIN_PROT("  Sequence "<<i<<" "<<test_memory[i].state
		 <<" J: "<<J<<" ("<<(J/costs_per_action)<<") "
		 <<" neuro: " <<test_memory_result[i][0]
		 <<" classic Sung "<<test_memory_result[i][1]<<endl);
    }
  }

  TRAIN_PROT(endl);
  if(average[0]==average[1]){
    TRAIN_PROT("JUHUUU, Neuro zieht gleich"<<endl);
    statistics::total_draws ++;
  }
  else if(average[0]<average[1]){
    TRAIN_PROT("JUHUU, SUPRRRR, Neuro gewinnt!"<<endl);
    statistics::total_wins ++;
  }
  TRAIN_PROT("Average: neuro: "<<average[0]<<" won "<<won[0]
	     <<" classic Sung "<<average[1]<<" won "<<won[1]
	     <<" total wins "<<statistics::total_wins
	     <<" total draws "<<statistics::total_draws
	     <<" total sum "<<statistics::total_draws + statistics::total_wins
	     <<endl);
  TRAIN_RESULT(train_loops_ctr<<" "<<average[0]<<" "<<won[0]
	     <<" "<<average[1]<<" "<<won[1]
	     <<" "<<statistics::total_wins
	     <<" "<<statistics::total_draws
	     <<" "<<statistics::total_draws + statistics::total_wins
	     <<endl);
  resultfile.flush();
}


void NeuroIntercept::generate_test_state(const Vector mypos, const Vector myvel,const Vector ballpos,const Vector ballvel,const Angle myangle ){
  if(test_memory_ctr >= TEST_MEMORY_SIZE_ICPT){
    LOG_MOV(0,<<"NeuroIntercept: Warning Test memory full");
    return;
  }
  test_memory[test_memory_ctr].state.my_pos = mypos;
  test_memory[test_memory_ctr].state.my_vel = myvel;
  test_memory[test_memory_ctr].state.ball_pos = ballpos;
  test_memory[test_memory_ctr].state.ball_vel = ballvel;
  test_memory[test_memory_ctr].state.my_angle = ANGLE(myangle);
  test_memory_ctr ++;
}

