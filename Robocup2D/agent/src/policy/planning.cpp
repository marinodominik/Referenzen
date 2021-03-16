#include "planning.h"

//#define SQUARE(x)((x)*(x))

//#define DBLOG_INFO2 DBLOG_INFO
#define DBLOG_INFO2(X)
// ridi: added Debug Logs which are deactivated for competition
#if 0
#define DBLOG_DRAW(LLL,XXX) LOG_POL(LLL,<<_2D <<XXX)
#define DBLOG_POL(LLL,XXX) LOG_POL(LLL,XXX)
#define DBLOG_ERR(LLL,XXX) LOG_ERR(LLL,XXX)
#else
#define DBLOG_DRAW(LLL,XXX)
#define DBLOG_POL(LLL,XXX)
#define DBLOG_ERR(LLL,XXX) 
#endif

//#define MIN_POSITION_NET -20
#define BASELEVEL 0
#define MAX_STR_LEN 500

/*****************************************************************************/
// paramters for planning
#define CONSIDER_AGE_MAXINCREASE 2.0

double Planning::pass_arrival_speed;
double Planning::pass_inrun_arrival_speed;

double Planning::selfpass_consider_age;
double Planning::consider_age;
double Planning::consider_freedom;
double Planning::selfpass_ok_zone;

double Planning::pass_opponent_bonusstep;
double Planning::pass_goalie_bonusstep;
double Planning::pass_ownhalf_factor;
double Planning::pass_attack_factor;
double Planning::selfpass_myhandicap;
int    Planning::valuefunction;

int    Planning::risky_mytime2react;
int    Planning::risky_optime2react;


bool   Planning::dribblings_allowed;


double Planning::multikick_opponent_speed;


double Planning::dribble_velocity_loss; // slower than without ball
double Planning::dribble_opponent_bonusstep;
double Planning::dribble_goalie_bonusstep;
float  Planning::Jnn1_active_thresh;
float  Planning::J1vs1_active_thresh;
float  Planning::Jnn2_active_thresh;
float  Planning::Jnn2_deactive_thresh;
int    Planning::activeJnn;

Netdescription Planning::Qnet;
Netdescription Planning::Jnn1;
Netdescription Planning::J1vs1;
Netdescription Planning::Jnn2;
Netdescription Planning::P1net;
Netdescription Planning::P2net;

OneOrTwoStepKick *Planning::twostepkick;


/*****************************************************************************/

// global variables:
bool Planning::pass_receiver_array[11];
/*****************************************************************************/

Netdescription::Netdescription(){
    loaded = false;
    opponents_num = 0;
    teammates_num = 0;
    sort_type = 1;
}

void Planning::init_params()
{
#if 0
    if(!OneOrTwoStepKick::init(conf_file,argc,argv))
    {
        ERROR_OUT << "\nPlanning::Could not initialize OneOrTwoStepKick behavior - stop";
        exit(1);
    }
#endif

    twostepkick = new OneOrTwoStepKick();

    // default settings
    pass_opponent_bonusstep    =  1;   // opponent is 1 step closer to ball
    pass_goalie_bonusstep      =  2;   // goalie can catch the ball
    pass_ownhalf_factor        =  1.5; // if receiver x <0 then multiply risk by 1.5
    pass_attack_factor         =   .1; // if receiver x > 25 then multiply risk by .2
    selfpass_myhandicap        =  1;   // if selfpassing, consider me to miss 1 cycle
    selfpass_ok_zone           = 10.;
    valuefunction              =  1;
    //immediate_selfpass_min_advantage = 1;
    dribble_velocity_loss      =  0.6; // slower than without ball
    dribble_opponent_bonusstep =  0;
    dribble_goalie_bonusstep   =  0;

    dribblings_allowed         = true;

    Jnn1_active_thresh         =  0; // in ops. half only
    J1vs1_active_thresh        =  0; // in ops. half only
    Jnn2_active_thresh         = ServerOptions::pitch_length; // default: do not use
    Jnn2_deactive_thresh       = 30;
    activeJnn                  =  1;

    pass_arrival_speed         = PASS_ARRIVAL_SPEED;
    pass_inrun_arrival_speed   = PASS_ARRIVAL_SPEED;
    consider_age               =  0.2;
    selfpass_consider_age      =  0.2;
    consider_freedom           =  0.;

    char net_name[   MAX_STR_LEN ];
    char Qnet_name[  MAX_STR_LEN ];
    char P1net_name[ MAX_STR_LEN ];

    int risky_mytime2react = 2;
    int risky_optime2react = 2;


    sprintf( Qnet_name, "../data/posJ.net" ); // Default
    //sprintf(P1net_name,"../data/p1.net"); // Default // Auskommentriert, weil net nicht mehr existiert und das nur Warnings schmeißt ...

    ValueParser vp( CommandLineOptions::policy_conf, "Planning" );
    //vp.set_verbose(true);
    vp.get( "risky_mytime2react", risky_mytime2react );
    vp.get( "risky_optime2react", risky_optime2react );

    vp.get( "pass_opponent_bonusstep",    pass_opponent_bonusstep    );
    vp.get( "pass_goalie_bonusstep",      pass_goalie_bonusstep      );
    vp.get( "pass_ownhalf_factor",        pass_ownhalf_factor        );
    vp.get( "pass_attack_factor",         pass_attack_factor         );
    vp.get( "selfpass_myhandicap",        selfpass_myhandicap        );
    vp.get( "valuefunction",              valuefunction              );
    vp.get( "dribble_velocity_loss",      dribble_velocity_loss      );
    vp.get( "dribblings_allowed",         dribblings_allowed         );
    vp.get( "dribble_opponent_bonusstep", dribble_opponent_bonusstep );
    vp.get( "dribble_goalie_bonusstep",   dribble_goalie_bonusstep   );
    vp.get( "pass_arrival_speed",         pass_arrival_speed         );
    vp.get( "pass_inrun_arrival_speed",   pass_inrun_arrival_speed   );
    vp.get( "consider_age",               consider_age               );
    vp.get( "selfpass_consider_age",      selfpass_consider_age      );
    vp.get( "consider_freedom",           consider_freedom           );
    vp.get( "selfpass_ok_zone",           selfpass_ok_zone           );
    vp.get( "multikick_opponent_speed",   multikick_opponent_speed   );
    vp.get( "Jnn1_active_thresh",         Jnn1_active_thresh         );
    vp.get( "Jnn1_opponents_num",         Jnn1.opponents_num         );
    vp.get( "Jnn1_teammates_num",         Jnn1.teammates_num         );
    vp.get( "Jnn1_sort_type",             Jnn1.sort_type             );
    vp.get( "Jnn2_active_thresh",         Jnn2_active_thresh         );
    vp.get( "Jnn2_deactive_thresh",       Jnn2_deactive_thresh       );
    vp.get( "Jnn2_opponents_num",         Jnn2.opponents_num         );
    vp.get( "Jnn2_teammates_num",         Jnn2.teammates_num         );
    vp.get( "Jnn2_sort_type",             Jnn2.sort_type             );
    vp.get( "J1vs1_active_thresh",        J1vs1_active_thresh        );
    vp.get( "J1vs1_opponents_num",        J1vs1.opponents_num        );
    vp.get( "J1vs1_teammates_num",        J1vs1.teammates_num        );
    vp.get( "J1vs1_sort_type",            J1vs1.sort_type            );

    if( vp.get( "Jnn1", net_name, MAX_STR_LEN ) > -1 )
    {

        if( Jnn1.net.load_net( net_name ) == 0 )
            Jnn1.loaded = true;
        else
        {
            cerr << "\nError in Planning.c: Can not load Jnn1! " << net_name << endl;
        }
    } // Jnn1 loaded

    if( vp.get( "Jnn2", net_name, MAX_STR_LEN ) > -1 )
    {

        if( Jnn2.net.load_net( net_name ) == 0 )
            Jnn2.loaded = true;
        else
        {
            cerr << "\nError in Planning.c: Can not load Jnn2! " << net_name << endl;
        }
    } // Jnn2 loaded

    if( vp.get( "J1vs1", net_name, MAX_STR_LEN ) > -1 )
    {
        if( J1vs1.net.load_net( net_name ) == 0 )
        {
            J1vs1.loaded = true;
        }
        else
        {
            cerr << "\nError in Planning.c: Can not load J1vs1! " << net_name << endl;
        }
    } // J1vs1 loaded

    vp.get( "Qnet", Qnet_name, MAX_STR_LEN );
    if( Qnet.net.load_net( Qnet_name ) == 0 )
        Qnet.loaded = true;

    vp.get( "Qnet_opponents_num", Qnet.opponents_num );
    vp.get( "Qnet_teammates_num", Qnet.teammates_num );
    vp.get( "Qnet_sort_type", Qnet.sort_type );

    vp.get( "P1net", P1net_name, 500 );
    if( P1net_name[ 0 ] != '\0' )
    {
        if( P1net.net.load_net( P1net_name ) == 0 )
            P1net.loaded = true;
    }

    vp.get( "P1net_opponents_num", P1net.opponents_num );
    vp.get( "P1net_teammates_num", P1net.teammates_num );
    vp.get( "P1net_sort_type", P1net.sort_type );

    unmark_all_pass_receiver();
}
 
float Planning::goalshot_chance_vs_optigoalie(const Vector ballpos){
  /* 
     returns a value between 0 (nochance) and 2 (bestchance)
     procedure assumes that goalie is exactly in the middle between a shot to the
     left and right corner.
     if result >1, than theoretically the ball can reach the goal corner faster than the
     goalie */
  const double v_ballmax = ServerOptions::ball_speed_max;
  const double v_playermax = 0.8;

  Vector goalpos_right = mdpInfo::opponent_goalpos() + Vector(0,-ServerOptions::goal_width/2.);
  Vector goalpos_left = mdpInfo::opponent_goalpos() + Vector(0,ServerOptions::goal_width/2.);

  // Determine angle between shot to left and right corner
  double alpha = 0.5 *
    Tools::get_abs_angle((ballpos-goalpos_left).arg() - (ballpos-goalpos_right).arg());

  /*  DBLOG_POLICY<<"Goalshot_chance vs opti goalie "<<(v_ballmax/v_playermax * sin(alpha))
      <<" alpha: "<<alpha;     */
  
  return(Tools::min(2.,v_ballmax/v_playermax * sin(alpha)));
}


/******************************************************************************
   Action Planning
*******************************************************************************/

#define SEARCH_DEPTH 0

float  Planning::evaluate(const AState &state,const int last_action_type ){

  if(last_action_type != AACTION_TYPE_PASS){
    DBLOG_POL(BASELEVEL+1,<<"Planning: action was NOT a pass: Just Evaluate Position");
    return evaluate_state(state);
  }

  DBLOG_POL(BASELEVEL+1,<<"Planning: last action was a pass: DO Planning");

  AAction solution_path[SEARCH_DEPTH+1];
  AAction actions[MAX_AACTIONS];

  // do not reset receiver array here; this may cause runtime problems! 
  //Planning::unmark_all_pass_receiver(); // reset pass receiver array


  int pactions_size = generate_pactions(state, actions);
  float Vbest = -1;
  DBLOG_POL(BASELEVEL+1,<<"In evaluate(state): paction size = "<<pactions_size);

  for(int a=0; a<pactions_size;a++){

    for(int i=0;i<SEARCH_DEPTH+1;i++){ // reset solution path before testing each action
      solution_path[i].V = 0;
      solution_path[i].action_type = 0;
      solution_path[i].target_player=0;
      solution_path[i].acting_player=0;
    }

    float V = evaluate_action(state, actions[a], SEARCH_DEPTH, solution_path);
    actions[a].V = V;
#if 1
  char solutionstr[100];
  sprintf(solutionstr," Solution Path %d-%d-",state.my_team[actions[a].acting_player].number,
	  state.my_team[actions[a].target_player].number);
  for(int i=SEARCH_DEPTH;i>0;i--){
    sprintf(solutionstr,"%s%d-%d-",solutionstr,
	    state.my_team[solution_path[i].acting_player].number,state.my_team[actions[a].target_player].number);
  }
  sprintf(solutionstr,"%s%d ",solutionstr,
	  state.my_team[solution_path[0].target_player].number);
  DBLOG_POL(BASELEVEL+1, <<"Planning: Evaluating branch "<<solutionstr<< "  Eval: "<< V);
#endif
    if(V > Vbest){
      Vbest = V;
    }
  }
  return Vbest * .99;
}


//#define DBLOG_RECURSIVE 

float  Planning::evaluate_action(const AState &state, AAction &action, int steps2go, 
				 AAction solution_path[]){

  // ridi: assuming that action-array contains only succesful actions - 
  // this was already tested in generate actions

  //    DBLOG_POL(0, "Enter Evaluate RECURSIVE  steps2go : "<<steps2go);


  if(action.action_type == AACTION_TYPE_SCORE){
    //    return evaluate_score_action(state);  // ridi04: no special scoring evaluation currently
    return evaluate_state(state); // just evaluate the current state.
  }

  AState next_state;
  AbstractMDP::model(state, action,next_state); // compute the state resulting form action 

  if(action.action_type == AACTION_TYPE_DEADLYPASS ||
     action.action_type == AACTION_TYPE_LAUFPASS ||
     action.action_type == AACTION_TYPE_SOLO ||
     action.action_type == AACTION_TYPE_WAITANDSEE ||
     action.action_type == AACTION_TYPE_DRIBBLE || // this forbids to pass after dribbling
     action.action_type == AACTION_TYPE_SELFPASS || // as well for selfpasses
     steps2go == 0){ // the search ends here!

    float score = 0.0;

#if  0 // ridi04  should be reactivated, but currently score evaluation is missing 
    float score =  evaluate_score_action(next_state);
    if(score>0){ // scoring seems to be successful
      DBLOG_POL(0, "Evaluate_action - scoring situation: "<<score);
      return score;
    }
#endif


    score = evaluate_state(next_state);
    
    PlayerSet pset_tmp = WSinfo::valid_opponents;
    PPlayer p_tmp = pset_tmp.closest_player_to_point(next_state.ball.pos);
    float freedom;
    if (p_tmp != NULL) {
      freedom = (p_tmp->pos - next_state.ball.pos).norm();
    } else {
      freedom = 0.0;
    }
    //int closest = mdpInfo::opponent_closest_to(next_state.ball.pos);
    //float freedom = mdpInfo::opponent_distance_to(closest,next_state.ball.pos);
    freedom = Tools::min(freedom,7.);      
    score += consider_freedom * freedom;

    //    DBLOG_POL(0, "Evaluate_action - Result : "<<score);
    return score;
  }

// if I made a pass to a teammate, I never want the ball back!
  if((action.action_type == AACTION_TYPE_PASS) &&
     (action.acting_player != action.target_player))
    next_state.my_team[action.acting_player].valid = false;

  AAction actions[MAX_AACTIONS];
  int pactions_size = generate_pactions(next_state, actions);
  DBLOG_POL(BASELEVEL+1,<<"In evaluate(state,action): paction size = "<<pactions_size);
  float Vbest = -1;
  for(int a=0; a<pactions_size;a++){
    for(int b=0;b<pactions_size;b++){
      // invalidate current level pass candidates in succeeding levels
      if((actions[b].action_type == AACTION_TYPE_PASS) &&
	 (actions[b].acting_player != actions[b].target_player))
	next_state.my_team[actions[b].target_player].valid = false;
    }

    float V = evaluate_action(next_state, actions[a], steps2go-1,solution_path);
    actions[a].V = V;
    if(V > Vbest){
      Vbest = V;
      if (V>solution_path[steps2go].V) // if action is best action on this level, store
	solution_path[steps2go] = actions[a];
    }
  }
  return 0.999 * Vbest;
}


bool Planning::is_laufpass_successful(const Vector ballpos,
				      float & speed, const float dir, const bool risky_pass){
  
  Vector interceptpos,playerpos; 
  int advantage, number;
  
  return is_laufpass_successful(ballpos,speed,dir,interceptpos,advantage,number,playerpos,risky_pass);
}


bool Planning::is_laufpass_successful(const AState & state, const Vector ballpos,
				      float & speed, const float dir,
				      Vector &interceptpos, int &advantage, int &number,
				      Vector &playerpos, const bool risky_pass){
#define MIN_DIST_FOR_LAUFPASS 7
  
  if(speed <= 0) 
    return false;
  advantage = -1; // init
  interceptpos = Vector(0); // init
  number = 0;

  Vector targetpos;
  targetpos.init_polar(1.0/(1.0-0.94) *speed,dir);
  targetpos += ballpos;

  //if(is_kick_possible(speed,dir) == false){
  if(is_kick_possible(state, speed, targetpos) == false){
    //DBLOG_POL(BASELEVEL,<<"Planning: LAUFPASS in dir  "<<RAD2DEG(dir)<<" NOT possible. Kick fails");
    return false;
  }

  bool result;
  if(risky_pass == false){
     result = Policy_Tools::myteam_intercepts_ball_hetero(ballpos,speed, 
							      dir,interceptpos,advantage,
							      number,playerpos);
  }
  else{
#if 0
    const int risky_mytime2react = 2;
    const int risky_optime2react = 2;
#endif

    result = Policy_Tools::myteam_intercepts_ball_hetero(ballpos,speed, 
							 dir,interceptpos,advantage,
							 number,playerpos, 
							 risky_mytime2react,
							 risky_optime2react);
  }


  if(result == true){
    if(mdpInfo::is_position_in_pitch(interceptpos,5.0) == false)
      // laufpasses: check if position is safe within pitch
      return false;
    if(mdpInfo::is_object_in_my_penalty_area(interceptpos) == true){
#if 0
      DBLOG_ERR(BASELEVEL,"LAUFPASS to dir "<<RAD2DEG(dir)<<" with speed "<<speed
		<<" seems ok, but resulting position in my penalty area: "<<interceptpos);
      DBLOG_POL(BASELEVEL,"LAUFPASS to dir "<<RAD2DEG(dir)<<" with speed "<<speed
		<<" seems ok, but resulting position in my penalty area: "<<interceptpos);
#endif
      return false;
    }
#if 0
    if(ballpos.sqr_distance(interceptpos) <SQUARE(MIN_DIST_FOR_LAUFPASS)){
      DBLOG_ERR(BASELEVEL,"LAUFPASS to dir "<<RAD2DEG(dir)<<" with speed "<<speed
		<<" seems ok, but resulting position is too close: "<<interceptpos);
      DBLOG_POL(BASELEVEL,"LAUFPASS to dir "<<RAD2DEG(dir)<<" with speed "<<speed
		<<" seems ok, but resulting position is too close: "<<interceptpos);
      return false;
    }
#endif
    return true;  // result was true, no exceptions found!
  }
  return false;
}


bool Planning::is_laufpass_successful(const Vector ballpos,
				      float & speed, const float dir,
				      Vector &interceptpos, int &advantage, int &number,
				      Vector &playerpos, const bool risky_pass){
#define MIN_DIST_FOR_LAUFPASS 7
  
  if(speed <= 0) 
    return false;
  advantage = -1; // init
  interceptpos = Vector(0); // init
  number = 0;

  Vector targetpos;
  targetpos.init_polar(1.0/(1.0-0.94) *speed,dir);
  targetpos += ballpos;

  //if(is_kick_possible(speed,dir) == false){
  if(is_kick_possible(speed, targetpos) == false){
    //DBLOG_POL(BASELEVEL,<<"Planning: LAUFPASS in dir  "<<RAD2DEG(dir)<<" NOT possible. Kick fails");
    return false;
  }

  bool result;
  if(risky_pass == false){
     result = Policy_Tools::myteam_intercepts_ball_hetero(ballpos,speed, 
							      dir,interceptpos,advantage,
							      number,playerpos);
  }
  else{
    const int risky_mytime2react = 2;
    const int risky_optime2react = 2;

    result = Policy_Tools::myteam_intercepts_ball_hetero(ballpos,speed, 
							 dir,interceptpos,advantage,
							 number,playerpos, 
							 risky_mytime2react,
							 risky_optime2react);
  }


  if(result == true){
    if(mdpInfo::is_position_in_pitch(interceptpos,5.0) == false)
      // laufpasses: check if position is safe within pitch
      return false;
    if(mdpInfo::is_object_in_my_penalty_area(interceptpos) == true){
#if 0
      DBLOG_ERR(BASELEVEL,"LAUFPASS to dir "<<RAD2DEG(dir)<<" with speed "<<speed
		<<" seems ok, but resulting position in my penalty area: "<<interceptpos);
      DBLOG_POL(BASELEVEL,"LAUFPASS to dir "<<RAD2DEG(dir)<<" with speed "<<speed
		<<" seems ok, but resulting position in my penalty area: "<<interceptpos);
#endif
      return false;
    }
#if 0
    if(ballpos.sqr_distance(interceptpos) <SQUARE(MIN_DIST_FOR_LAUFPASS)){
      DBLOG_ERR(BASELEVEL,"LAUFPASS to dir "<<RAD2DEG(dir)<<" with speed "<<speed
		<<" seems ok, but resulting position is too close: "<<interceptpos);
      DBLOG_POL(BASELEVEL,"LAUFPASS to dir "<<RAD2DEG(dir)<<" with speed "<<speed
		<<" seems ok, but resulting position is too close: "<<interceptpos);
      return false;
    }
#endif
    return true;  // result was true, no exceptions found!
  }
  return false;
}


bool Planning::check_action_laufpass(const AState &state,AAction &candidate_action, 
				     float speed, const float dir, const bool risky_pass){

  Vector interceptpos;
  int advantage, number;
  Vector playerpos; // current position of the player intercepting the ball

  float desired_speed = speed;

  bool is_successful= is_laufpass_successful(state, state.ball.pos,speed, 
					     dir,interceptpos,advantage,
					     number,playerpos, risky_pass);

  //Daniel: for learning
  /*
  if (fabs(interceptpos.y) > PENALTY_AREA_WIDTH/2.0 - 3.0)
    return false;
  */

  if (number == state.my_team[state.my_idx].number) {
    //DBLOG_POL(BASELEVEL+3,<<"Planning: Check action LAUFPASS in dir  "<<RAD2DEG(dir)<<" speed "<<speed<<" I get it myself -> forget it ");
    return false;
  }


  int myteam_fastest = 2000;
  Vector player_intercept_pos;
  Policy_Tools::get_time2intercept(player_intercept_pos,state.ball.pos,speed,dir,playerpos,Policy_Tools::pass_mytime2react, myteam_fastest);

  Vector arrival_vel;
  double tmp_speed= 0.0;
  if ((myteam_fastest >= 0) && (myteam_fastest != 2000)) {
    tmp_speed = speed * pow(ServerOptions::ball_decay, myteam_fastest);
  }
  arrival_vel.init_polar(tmp_speed, dir);

  if(desired_speed != speed){
    DBLOG_POL(BASELEVEL+3,<<"Planning: Check action LAUFPASS in dir  "<<RAD2DEG(dir)<<" Desired speed "
	      <<desired_speed<<" possible speed "<<speed );
  }


  if(playerpos.getX() > WSinfo::his_team_pos_of_offside_line()){
    DBLOG_POL(BASELEVEL+3,<<"Planning: check action lauf pass receiver pos "
	      <<playerpos
	      <<" is offside -> DO NOT PASS"
	      <<" his offsideline "<<WSinfo::his_team_pos_of_offside_line()
	      <<" my offsideline "<<WSinfo::my_team_pos_of_offside_line()
	      );
    return false;
  }


#if 0
  if(is_successful == true){
    DBLOG_POL(1,<<"planning: laufpass "<<"dir "<<RAD2DEG(dir)<<" speed "<<speed
	    <<" WE GET BALL number: "<<number
	    <<" at position "<<interceptpos
	    <<" advantage "<<advantage);
  }
  else{
    DBLOG_POL(1,<<"planning: laufpass "<<"dir "<<RAD2DEG(dir)<<" speed "<<speed
	    <<" we dont get ball (in pitch) number: "<<number
	    <<" at position "<<interceptpos
	    <<" advantage "<<advantage);
  } 
#endif

  if(is_successful==false)
    return false;
  
  if((Tools::get_abs_angle(Tools::get_angle_between_mPI_pPI(dir)) > 45/180.*PI) &&
     Tools::get_abs_angle(Tools::get_angle_between_mPI_pPI(dir)) >
     5./180. *PI + 
     Tools::get_abs_angle(Tools::get_angle_between_mPI_pPI((playerpos-
							    state.my_team[state.my_idx].pos).arg()))){
    DBLOG_POL(BASELEVEL + 0,<<"Laufpass in dir "<<RAD2DEG(dir)
	    <<" successful, but too far back");
    return false;
  }


  Vector targetpos;
  targetpos.init_polar(1.0/(1.0-0.94) *speed,dir);
  targetpos += state.ball.pos;

  int recv_idx = 0;
  for(int i=0;i<11;i++){
    if(state.my_team[i].number==number){
      recv_idx = i;
      break;
    }
  }
  AbstractMDP::set_action(candidate_action, AACTION_TYPE_LAUFPASS, 
			  state.my_idx, recv_idx, targetpos, speed,dir,
			  interceptpos,arrival_vel,advantage,0,0,
			  state.my_team[recv_idx].number,
			  risky_pass);

  mark_pass_receiver(recv_idx);
  return true;
}
/* Daniel
bool Planning::reconsider_pass_successful(const Vector ballpos,
					  float & speed, const float dir,
					  const Vector playerpos,
					  Vector &interceptpos, int &advantage){
  
  // currently, no special test is implemented
  return is_pass_successful(ballpos, speed, dir, playerpos, interceptpos, advantage);
}
*/

bool Planning::is_pass_successful(const AState & state, const Vector ballpos,
				  float & speed, const float dir,
				  const Vector playerpos,
				  Vector &interceptpos, 
				  int &advantage){
  
  if(speed <= 0) 
    return false;
  advantage = -1; // init
  interceptpos = Vector(0); // init


  if(is_kick_possible(state,speed,dir) == false){
    DBLOG_POL(BASELEVEL,<<"Planning: PASS in dir  "<<RAD2DEG(dir)<<" NOT possible. Kick fails");
    return false;
  }

  return Policy_Tools::myplayer_intercepts_ball(ballpos,speed, 
						dir,playerpos,interceptpos,advantage);
}


bool Planning::is_pass_successful_hetero(const AState & state, const Vector ballpos,
					 float & speed, const float dir,
					 const PPlayer player,
					 Vector &interceptpos, 
					 int &advantage){
  
  if(speed <= 0) 
    return false;
  advantage = -1; // init
  interceptpos = Vector(0); // init


  if(is_kick_possible(state,speed,dir) == false){
    DBLOG_POL(BASELEVEL,<<"Planning: PASS in dir  "<<RAD2DEG(dir)<<" NOT possible. Kick fails");
    return false;
  }
#if 0
  LOG_POL(0,"Planning: Is pass2player successful? "<<player->number);
#endif

  return Policy_Tools::myplayer_intercepts_ball_hetero(ballpos,speed, 
						       dir,player,interceptpos,advantage);
}


bool Planning::is_pass_successful(const Vector ballpos,
				  float & speed, const float dir,
				  const Vector playerpos,
				  Vector &interceptpos, 
				  int &advantage){
  
  if(speed <= 0) 
    return false;
  advantage = -1; // init
  interceptpos = Vector(0); // init


  if(is_kick_possible(speed,dir) == false){
    DBLOG_POL(BASELEVEL,<<"Planning: PASS in dir  "<<RAD2DEG(dir)<<" NOT possible. Kick fails");
    return false;
  }

  return Policy_Tools::myplayer_intercepts_ball(ballpos,speed, 
						dir,playerpos,interceptpos,advantage);
}


bool Planning::is_pass_successful_hetero(const Vector ballpos,
					 float & speed, const float dir,
					 const PPlayer player,
					 Vector &interceptpos, 
					 int &advantage){
  
  if(speed <= 0) 
    return false;
  advantage = -1; // init
  interceptpos = Vector(0); // init


  if(is_kick_possible(speed,dir) == false){
    DBLOG_POL(BASELEVEL,<<"Planning: PASS in dir  "<<RAD2DEG(dir)<<" NOT possible. Kick fails");
    return false;
  }
#if 0
  LOG_POL(0,"Planning: Is pass2player successful? "<<player->number);
#endif

  return Policy_Tools::myplayer_intercepts_ball_hetero(ballpos,speed, 
						       dir,player,interceptpos,advantage);
}


bool Planning::check_action_pass(const AState &state,AAction &candidate_action, 
				 int recv_idx, Vector rel_target,
				 double min_pass_dist){

  //static float short_pass_distance=8;
  //  static float short_pass_distance=8;

  Vector receiverpos = state.my_team[recv_idx].pos;
  if(receiverpos.getX() > WSinfo::his_team_pos_of_offside_line()){
    DBLOG_POL(BASELEVEL,<<"Planning: check action pass receiver pos "
	      <<receiverpos
	      <<" is offside -> DO NOT PASS "
	      <<" his offsideline "<<WSinfo::his_team_pos_of_offside_line()
	      <<" my offsideline "<<WSinfo::my_team_pos_of_offside_line()
	      );
    return false;
  }

  if(is_pass_receiver_marked(recv_idx)){ // this player already gets a pass from another player
    //DBLOG_POL(BASELEVEL,<<"Planning: pass to  "<<state.my_team[recv_idx].number<<" NOT possible. MARKED");
    return false; 
  }

  float dist = (receiverpos-state.ball.pos).norm();

  if(dist < min_pass_dist){
    //DBLOG_POL(BASELEVEL,<<"Planning: pass to  "<<state.my_team[recv_idx].number<<" NOT possible. Too close");
    // very short passes are not allowed at all
    return false; 
  }
  
  double arrival_speed;
  Vector arrival_vel;

  //Daniel: changed for learning
#if 0
  PPlayer p_tmp = WSinfo::get_teammate_by_number(state.my_team[recv_idx].number);
  Vector targetpos;
  if (p_tmp != NULL) {
    targetpos = receiverpos + rel_target + 1.0 / ServerOptions::player_decay * p_tmp->vel;
  } else {
    targetpos = receiverpos + rel_target;
  }
#endif
  // ridi04: changed back:
  Vector targetpos = receiverpos + rel_target;

  float speed;
  Angle dir = (targetpos-state.ball.pos).arg();
  speed = compute_pass_speed_with_arrival_vel(state.ball.pos,receiverpos,targetpos, MAX_PASS_SPEED, arrival_speed); // this is desired
  arrival_vel.init_polar(arrival_speed, dir);

  
  if (state.my_team[state.my_idx].number == WSinfo::me->number){
    if(is_kick_possible(state, speed,targetpos) == false){
      //DBLOG_POL(BASELEVEL,<<"Planning: pass to  "<<state.my_team[recv_idx].number<<" speed "<<speed<<" NOT possible. Kick fails");
      return false;
    }
  } 

  if(speed <= 0){
    //DBLOG_POL(BASELEVEL+2,<<"Planning: pass to  "<<state.my_team[recv_idx].number<<" NOT possible. speed: "<<speed);
    return false;
  }

  Vector interceptpos;
  int advantage;

  bool is_successful;
  int number =  state.my_team[recv_idx].number;
  PPlayer player = WSinfo::get_teammate_by_number(number);
  if(player){
    //LOG_POL(0,"Planning: Check pass2player "<<number<<" speed "<<speed<<" dir: "<<RAD2DEG(dir));
    is_successful= is_pass_successful_hetero(state, state.ball.pos,speed, 
					     dir,player,interceptpos,advantage);
  }
  else{ // this was used before Padua
    LOG_POL(0,"Planning: Warning: cannot determine player number of pass candidate -> taking standard is_pass_succesful!");
    is_successful= is_pass_successful(state, state.ball.pos,speed, 
					   dir,receiverpos,interceptpos,advantage);
  }

  if(is_successful==false){
    //DBLOG_POL(BASELEVEL +0,<<"Planning: pass to  "<<state.my_team[recv_idx].number<<" speed "<<speed<<" dir "<<RAD2DEG(dir)<<" NOT possible. Is pass successful says NO");
    return false;
  }

  //DBLOG_POL(BASELEVEL +0,<<"Planning: pass to "<<state.my_team[recv_idx].number<<" speed "<<speed<<" dir "<<RAD2DEG(dir)<<" POSSIBLE. Is pass successful says YES");

  
  AbstractMDP::set_action(candidate_action, AACTION_TYPE_PASS, 
			  state.my_idx, recv_idx, targetpos, speed,dir,
			  interceptpos, arrival_vel, advantage,0,0,state.my_team[recv_idx].number);
  mark_pass_receiver(recv_idx);

  return true;
}



bool Planning::check_action_solo(const AState &state, AAction &candidate_action){
  // this action is always possible, only the resulting position depends on opponents behaviour

  Vector resulting_position;

  if(state.my_team[state.my_idx].number == mdpInfo::mdp->me->number){
    // I am the acting player
    if(mdpInfo::stamina4meters_if_dashing(100) <7.)
      resulting_position = state.my_team[state.my_idx].pos;
  }
  else{
    if(state.my_team[state.my_idx].stamina < ServerOptions::stamina_max/3.)
      resulting_position = state.my_team[state.my_idx].pos;
  }

  // easy way: stay where you are - better: compute several directions and compute how far to go
  resulting_position = state.my_team[state.my_idx].pos; 

  AbstractMDP::set_action(candidate_action, AACTION_TYPE_SOLO, 
	     state.my_idx, state.my_idx, resulting_position, 0);

  candidate_action.actual_resulting_position = resulting_position;
  return true;
}


void Planning::unmark_all_pass_receiver(){
  for(int i=0;i<11;i++)
    pass_receiver_array[i] = false;
}

int Planning::generate_pactions(const AState &state, AAction *actions){
  int num_actions = 0;
  AAction candidate_action; 
  Vector test_pass[30];
#ifdef DBLOG_GENERATE
  //DBLOG_POLICY_D << "Generating Pactions for player "
  //       << state.my_team[state.my_idx].number;
#endif

  for(int recv_idx=0;recv_idx<11;recv_idx++){
    /* passes to other players */
    if(state.my_team[state.my_idx].number == mdpInfo::our_goalie_number() )
      break; // the goalie can not play passes
    if(state.my_team[recv_idx].valid == false)
      continue;
    if(state.my_idx == recv_idx)
      continue;
    if(state.my_team[recv_idx].number == mdpInfo::our_goalie_number())
      continue;

    // passes to other players 
    // the order determines the priority
    int i = 0;
    test_pass[i++] = Pass_Forward;
    test_pass[i++] = Pass_Direct;
    test_pass[i++] = Pass_Right;
    test_pass[i++] = Pass_Left;
    // test_pass[i++] = Pass_Backward;
    int number_passpos = i;

    for(int i=0;i<number_passpos;i++){
      if(Planning::check_action_pass(state,candidate_action,recv_idx,
					 test_pass[i])){
	actions[num_actions++] = candidate_action; // insert action
	break;  // stop testing  
      }
    }
  }

  if(Planning::check_action_solo(state,candidate_action))
    actions[num_actions++] = candidate_action; // insert action
  
  // try to score: 
  AbstractMDP::set_action(actions[num_actions++], AACTION_TYPE_SCORE, state.my_idx, 
	     0, Vector(0,0), 0);
#ifdef DBLOG_GENERATE
  //DBLOG_POLICY_D << "Generating Pactions: inserting score";
#endif
  return num_actions;
}


float  Planning::compute_pass_speed(Vector ballpos,Vector receiverpos,
				    Vector targetpos, const double maxspeed){
  double suggested_speed;
  double arrival_speed;
  //  const double shortpass_dist = 10;
  //const double shortpass_arrival_speed = .8;
  const double min_pass_speed = 0.7;
  //const double min_pass_arrival_speed = .2;  // .5 is is about 33 m, .25 is about 37.75m
  double pass_dist = (targetpos - ballpos).norm();

#if 0

  if(pass_dist > shortpass_dist)
    arrival_speed = pass_arrival_speed;
  else
    arrival_speed = shortpass_arrival_speed;

  if((targetpos - receiverpos).norm() > 0) //not an exact pass
    arrival_speed = pass_inrun_arrival_speed;

#endif

  arrival_speed = 2.0;


  suggested_speed =  (1-ServerOptions::ball_decay)*pass_dist + 
    ServerOptions::ball_decay * arrival_speed;

  suggested_speed= Tools::max(min_pass_speed,suggested_speed);
  suggested_speed= Tools::min(maxspeed,suggested_speed);

  //  double v_arrival = (suggested_speed - (1-ServerOptions::ball_decay) * pass_dist)/     ServerOptions::ball_decay;
#if 0
  if(v_arrival < min_pass_arrival_speed)
    return -1; // pass not possible
#endif
  return suggested_speed;
}


float  Planning::compute_pass_speed_with_arrival_vel(Vector ballpos,Vector receiverpos,
				    Vector targetpos, const double maxspeed, double & arrival_vel){
  double suggested_speed;
  double arrival_speed;
  //  const double shortpass_dist = 10;
  //const double shortpass_arrival_speed = .8;
  const double min_pass_speed = 0.7;
  //const double min_pass_arrival_speed = .2;  // .5 is is about 33 m, .25 is about 37.75m
  double pass_dist = (targetpos - ballpos).norm();

#if 0

  if(pass_dist > shortpass_dist)
    arrival_speed = pass_arrival_speed;
  else
    arrival_speed = shortpass_arrival_speed;

  if((targetpos - receiverpos).norm() > 0) //not an exact pass
    arrival_speed = pass_inrun_arrival_speed;

#endif

  arrival_speed = 2.0;


  suggested_speed =  (1-ServerOptions::ball_decay)*pass_dist + 
    ServerOptions::ball_decay * arrival_speed;

  suggested_speed= Tools::max(min_pass_speed,suggested_speed);
  suggested_speed= Tools::min(maxspeed,suggested_speed);

  double v_arrival = (suggested_speed - (1-ServerOptions::ball_decay) * pass_dist)/
    ServerOptions::ball_decay;
#if 0
  if(v_arrival < min_pass_arrival_speed)
    return -1; // pass not possible
#endif
  arrival_vel = v_arrival;
  return suggested_speed;
}

 
float Planning::pass_success_level(const Vector ballpos, Vector receiverpos, 
				   const double speed, const double dir,
				   Vector &intercept_pos){
  //call getball_chance with standard values
  return getball_chance(ballpos,receiverpos,speed,dir,intercept_pos); 
}

float Planning::receive_pass_success_level(const Vector ballpos, Vector receiverpos, 
				   const double speed, const double dir,
				   Vector &intercept_pos){
  //call getball_chance without considering too close players
  return getball_chance(ballpos,receiverpos,speed,dir,intercept_pos,0,false); 
}


bool Planning::is_player_a_passcandidate(Vector playerpos){
  /** checks if the player at position playerpos can receive a pass from the current ballpos */
  Vector test_pass[30];
  Vector intercept_pos;
  Vector ballpos = WSinfo::ball->pos;
  double maxpass_speed = MAX_PASS_SPEED;

  if((WSinfo::ball->pos-playerpos).norm() < MIN_PASS_DISTANCE)
    // very short passes are not allowed at all
    return false; 


  int i = 0;
  test_pass[i++] = Pass_Forward;
  test_pass[i++] = Pass_LeftForward;
  test_pass[i++] = Pass_RightForward;
  test_pass[i++] = 1.5 * Pass_Forward;
  test_pass[i++] = 1.5 * Pass_LeftForward;
  test_pass[i++] = 1.5 * Pass_RightForward;
  test_pass[i++] = Pass_Direct;
  test_pass[i++] = Pass_Right;
  test_pass[i++] = Pass_Left;
  test_pass[i++] = 1.5 * Pass_Right;
  test_pass[i++] = 1.5 * Pass_Left;
  // test_pass[i++] = Pass_Backward;
  int number_passpos = i;

  PlayerSet pset_tmp = WSinfo::valid_teammates;
  PPlayer teammate = pset_tmp.closest_player_to_point(WSinfo::ball->pos);
  if(teammate != NULL){
    double dist;
    //double dir;
    Vector teammate_pos = teammate->pos;
    //int opponent = mdpInfo::opponent_closest_to(teammate_pos,dist,dir);
    pset_tmp = WSinfo::valid_opponents;
    PPlayer opponent = pset_tmp.closest_player_to_point(teammate_pos);
    if (opponent != NULL) {
      dist = (opponent->pos - teammate_pos).norm();
      if(dist <= 2.0*ServerOptions::kickable_area){
	maxpass_speed = 1.5;
	/*
	  DBLOG_INFO("Ballholder "<<teammate<<" is attacked by op "<<opponent<<" maxpassspeed "
	  <<maxpass_speed);
	*/
      }
    }
  }

  for(int i=0;i<number_passpos;i++){
    double dir =  (playerpos + test_pass[i] - ballpos).arg();
    double speed = Planning::compute_pass_speed(ballpos, playerpos,
					       (playerpos+test_pass[i]),maxpass_speed);
    if(speed<0)  // pass distance too far
      continue;
    if(Planning::receive_pass_success_level(ballpos,playerpos,speed,dir,intercept_pos)>0.0)
      return true;
  }
  return false;
}

float Planning::getball_chance(const Vector ballpos, Vector receiverpos, 
			       const double speed, const double dir,
			       Vector &intercept_pos,const double receiverbonus,
			       bool consider_close_players ,
			       double consider_age_factor ,
			       bool selfpass ){
  float fastest_op_steps = 2000;

  double consider_age_maxincrease = CONSIDER_AGE_MAXINCREASE;
  Vector velocity;
  velocity.init_polar(speed,dir);

  Vector addpos = velocity;
  if(selfpass == true){
    addpos.normalize(1.2);
    consider_age_maxincrease = 2;
  }
  else
    addpos = Vector(0);

  Vector my_intercept_pos;
  int tmp_steps;

  if(consider_age_factor <0) // this is the default value
    consider_age_factor = consider_age;

  /*
    BUG: This avoids selfpasses behind the current offside_line
  if(receiverpos.x > mdpInfo::his_team_pos_of_offside_line() && receiverpos.x > ballpos.x){
    //DBLOG_INFO2("Would like to pass to "<<receiverpos<<" but he is offside!!");
    return -1;
  }
  */

  // I assume that my teammmate needs an additional turn: player_size = 0
  Policy_Tools::intercept_min_time_and_pos_hetero(tmp_steps, my_intercept_pos, ballpos + addpos, 
			     velocity, receiverpos, -1, true, -1,receiverbonus); 
  float teammate_steps = (float) tmp_steps;

  if(mdpInfo::is_position_in_pitch(my_intercept_pos, PASS_SAFETY_MARGIN)==false)
    return -1;


  /*
  DBLOG_POLICY_D<<"Checking pass success level: fastest opponent "<<fastest_op_steps
	      <<" my steps "<<teammate_steps<<" getting ball at "<<my_intercept_pos.x<<" "
	      <<my_intercept_pos.y;
  */

  int goalie = WSinfo::ws->his_goalie_number;
  Vector op_pos;
  int op_number = 0;
  double op_age;

  for(int op_idx = 0;op_idx<12;op_idx++){
    if(op_idx<11){// this is a regular player from the player array
      if(mdpInfo::mdp->his_team[op_idx].pos_x.p == 0)  // >0: position valid
	continue;
      op_pos = mdpInfo::mdp->his_team[op_idx].pos();
      op_number = mdpInfo::mdp->his_team[op_idx].number;
      op_age = mdpInfo::age_playerpos(&(mdpInfo::mdp->his_team[op_idx]));
    }
    else{ //idx 11: check for closest player
      //op_number = mdpInfo::opponent_closest_to_me(op_pos);
      PlayerSet pset_tmp = WSinfo::valid_opponents;
      PPlayer p_tmp = pset_tmp.closest_player_to_point(WSinfo::me->pos);
      if(p_tmp != NULL) { // the closest player is already in the player array
	op_number = p_tmp->number;
	op_pos = p_tmp->pos;
	continue;
      }
      // ok, I found a surprising player!
      op_age = 0;
    }

    double angle2op = (op_pos-ballpos).arg();
    double not_considered_angle = 90./180. * PI; // standard
    if(speed < 1.0)
      not_considered_angle = 175/180.*PI; // if ballspeed too slow, an op. from behind may get it
    if(selfpass == true)
      not_considered_angle = 150./180. * PI;
    if(Tools::get_abs_angle(angle2op - velocity.arg()) > not_considered_angle){ // op. not in balldir
      continue; // do not consider
    }

    double dist2op = (op_pos-ballpos).norm();
    if ((dist2op < 2.5 && consider_close_players == false) ||
	((dist2op < 2.*ServerOptions::kickable_area) && (selfpass == false))){
      // do not consider players that are in my kickrange - important to single kick
      DBLOG_INFO2("Op "<<op_number<<" too close -> not considered");
      continue;
    }

    int tmp_steps;

    Vector player_intercept_pos;
    double opponent_bonus;

    if(op_number == goalie){
      opponent_bonus = pass_goalie_bonusstep + consider_age_factor * op_age;
      if(opponent_bonus > consider_age_maxincrease + pass_goalie_bonusstep){
	opponent_bonus =  consider_age_maxincrease +pass_goalie_bonusstep;
	//DBLOG_POLICY<<"Planning.c: Warning: age increases goalie pass bonus too much. Restricting";
      }
    }
    else{ // player is not the goalie
      opponent_bonus = pass_opponent_bonusstep + consider_age_factor * op_age;
      if(opponent_bonus > consider_age_maxincrease + pass_opponent_bonusstep){
	opponent_bonus = consider_age_maxincrease + pass_opponent_bonusstep;
	//DBLOG_POLICY<<"Planning.c: Warning: age increases pass bonus too much. Restricting";
      }
      //if(receiverpos.x > ServerOptions::pitch_length/2 - 25.0)
      if(receiverpos.getX() > 0) // receiver is opponent's half
	// the receiver is in attack zone, so probably risk more:
	opponent_bonus *= pass_attack_factor;
      if(receiverpos.getX() < 0)
	// the receiver is in our half, so probably be more cautious:
	opponent_bonus *= pass_ownhalf_factor;
    }

    Policy_Tools::intercept_min_time_and_pos_hetero(tmp_steps, player_intercept_pos, ballpos, velocity, op_pos, 
					     op_number, false, -1, opponent_bonus); 

    float op_steps = (float) tmp_steps;    
    if(op_steps < teammate_steps){ // opponent is faster !!
      intercept_pos = player_intercept_pos;
      return -1;
    }
    if(op_steps < fastest_op_steps){ // the fastest opponent
      fastest_op_steps = op_steps;
    }
  }

  // I was succesful!
  intercept_pos = my_intercept_pos;
  return (fastest_op_steps - teammate_steps);
}

bool Planning::is_kick_possible( const AState & state, float &speed, Vector target ) {

  double speed1 = 0.;
  double speed2 = 0.;

  twostepkick->set_state(state);

  twostepkick->kick_to_pos_with_initial_vel(speed,target);
  twostepkick->get_vel(speed1,speed2);

  twostepkick->reset_state(); // use current ws-state

  if(speed2<0.1){ // probably could not hold ball in kickrange
    speed = speed1;
  }
  else
    speed = speed2;
  if(speed < 0.1)
    return false;

  return true;

}


bool Planning::is_kick_possible( const AState & state, float &speed, Angle dir ) {

  double speed1 = 0.;
  double speed2 = 0.;


  twostepkick->set_state(state);

  twostepkick->kick_in_dir_with_initial_vel(speed,ANGLE(dir));
  twostepkick->get_vel(speed1,speed2);

  twostepkick->reset_state(); // use current ws-state

  if(speed2<0.1){ // probably could not hold ball in kickrange
    speed = speed1;
  }
  else
    speed = speed2;
  if(speed < 0.1)
    return false;

  return true;

}


bool Planning::is_kick_possible(float &speed, Vector target) {

  double speed1 = 0.;
  double speed2 = 0.;


  twostepkick->reset_state(); // use current ws-state

  twostepkick->kick_to_pos_with_initial_vel(speed,target);
  twostepkick->get_vel(speed1,speed2);

  if(speed2<0.1){ // probably could not hold ball in kickrange
    speed = speed1;
  }
  else
    speed = speed2;
  if(speed < 0.1)
    return false;

  return true;

}


bool Planning::is_kick_possible(float &speed, Angle dir) {

  double speed1 = 0.;
  double speed2 = 0.;


  twostepkick->reset_state(); // use current ws-state

  twostepkick->kick_in_dir_with_initial_vel(speed,ANGLE(dir));
  twostepkick->get_vel(speed1,speed2);

  if(speed2<0.1){ // probably could not hold ball in kickrange
    speed = speed1;
  }
  else
    speed = speed2;
  if(speed < 0.1)
    return false;

  return true;

}


/********************************************************************************/

/*                   EVALUATION FUNCTIONS                                       */

/********************************************************************************/

void Planning::code_pos(const AState& state, const Vector& orig, Vector & feat,
			const int sort_type, const float min_x ,
			const bool use_x_symmetry){
  feat = orig;
  if((use_x_symmetry == true) && (state.ball.pos.getY() < 0))
    feat.mulYby( -1 );  // put on upper half
  if(orig.getX() <min_x){
    feat.setX( min_x );
    //DBLOG_POL(1,"x-coordinate of pos "<<orig<<" < 0, setting netinput to zero ");
  }
  if(sort_type ==2)
    feat-=state.ball.pos;
}

void Planning::code_ball(const AState& state, const Vector& orig, Vector & feat,
			 const int sort_type, const float min_x,
			const bool use_x_symmetry){
  feat = orig;
  if((use_x_symmetry == true) && (state.ball.pos.getY() < 0))
    feat.mulYby( -1 );  // put on upper half
  if(orig.getX() <min_x){
    feat.setX( min_x );
    //DBLOG_POL(1,"x-coordinate of pos "<<orig<<" < 0, setting netinput to zero ");
  }
}

void Planning::code_ball_vel(const AState& state, const Vector& orig, Vector & feat,
			     const int sort_type,
			     const bool use_x_symmetry){
  feat = orig;
  if((use_x_symmetry == true) && (state.ball.pos.getY() < 0))
    feat.mulYby( -1 );  // put on upper half
}

float Planning::Jnn(const AState& state, const bool testoutput, 
		    const int index_of_relevant_player){

#if 0
  if(state.ball.pos.x > Jnn2_active_thresh ||
     state.ball.pos.x > 40 && fabs(state.ball.pos.y)<16.0 ||  // this is the training goal
     state.ball.pos.x > 48){  // this is the training goal 
  }
#endif
  if(state.ball.pos.getX() > Jnn2_active_thresh ||
     (Jnn2_active_thresh < ServerOptions::pitch_length/2. &&
     ((state.ball.pos.getX() > 40 && fabs(state.ball.pos.getY())<16.0) ||  // this is the training goal
     state.ball.pos.getX() > 48))){  // this is the training goal
    // enter Jnn2 zone
    activeJnn = 2;
    DBLOG_POL(BASELEVEL+1,"Planning: Jnn 2 used for evaluation ");
    //DBLOG_ERR(BASELEVEL+1,"Planning: Jnn 2 used for evaluation ");
    return(compute_Jnn(Jnn2,state,testoutput));
  } // conditions for Jnn2 are fulfilled
  
  if(activeJnn == 2 && state.ball.pos.getX() > Jnn2_deactive_thresh){
    activeJnn = 2;
    DBLOG_POL(BASELEVEL+1,"Planning: Jnn 2 used for evaluation ");
    //DBLOG_ERR(BASELEVEL+1,"Planning: Jnn 2 used for evaluation ");
    return(compute_Jnn(Jnn2,state,testoutput));
  } 

  activeJnn = 1;
  DBLOG_POL(BASELEVEL+1,"Planning: Jnn 1 used for evaluation ");
  //DBLOG_ERR(BASELEVEL+1,"Planning: Jnn 1 used for evaluation ");
  return(compute_Jnn(Jnn1,state,testoutput));

}

#define PLAYERS 11 // max. number of players

float Planning::compute_Jnn(Netdescription &netinfo,const AState& state, 
			    const bool testoutput , 
			    const int index_of_relevant_player){
  float result;
  Vector feat;
  int in_count;

  if(netinfo.loaded == false){
    LOG_ERR(0,"ERROR: Planning.c: Requested Jnn NOT loaded !");
    return -1.;
  }

  if(netinfo.net.topo_data.in_count != 4+2*(netinfo.teammates_num+netinfo.opponents_num)){
    cout<<"Error: Jnet input != "<<4+2*(netinfo.teammates_num+netinfo.opponents_num)<<endl;
    DBLOG_POL(0,<<"Error: Jnet input != "<<4+2*(netinfo.teammates_num+netinfo.opponents_num));
  }
  if(netinfo.sort_type < 1){ // sort players according to their distance, not by number!
    cout<<"ERROR: PJ: netinfo.sort_type "<<netinfo.sort_type<<" NOT SUPPORTED"<<endl;
    return 0.0;
  }

  // determine net input (straightforward, to be improved!)
  code_ball(state,state.ball.pos,feat,netinfo.sort_type);
  netinfo.net.in_vec[0] = feat.getX();
  netinfo.net.in_vec[1] = feat.getY();
  code_ball_vel(state,state.ball.vel,feat,netinfo.sort_type);
  netinfo.net.in_vec[2] = feat.getX();
  netinfo.net.in_vec[3] = feat.getY();
  in_count = 4;

  Vector sort_reference;
  if(netinfo.sort_type ==1)
    sort_reference=mdpInfo::opponent_goalpos();
  else
    sort_reference = state.ball.pos;
  // sort teammates
  Sort teamsort = Sort(PLAYERS);
  for(int i=0;i<PLAYERS;i++){
    if(!state.my_team[i].valid)
      continue;
    Vector ppos = state.my_team[i].pos;
    teamsort.add(i,ppos.sqr_distance(sort_reference));
  }
  teamsort.do_sort();
  bool I_am_a_netinput = false;
  for(int k=0;(in_count <4+2*netinfo.teammates_num&&k<PLAYERS);k++){ 
    int idx = teamsort.get_key(k);
    if(idx == index_of_relevant_player)
      I_am_a_netinput = true;
    if(idx<0){ // there is no player left to fill the inputs -> set him very far back!
      //LOG_DEB(0, << "Error with net: player number " << k << " not found");
      LOG_ERR(0, << "Error with net: player number " << k << " not found");
      code_pos(state, Vector(MIN_POSITION_NET,
			     (drand48()-0.5)*ServerOptions::pitch_width),feat,netinfo.sort_type);
      //code_pos(state, state.ball.pos - Vector(10,0),feat);
    }
    else
      code_pos(state, state.my_team[idx].pos,feat,netinfo.sort_type);
    netinfo.net.in_vec[in_count++] = feat.getX();
    netinfo.net.in_vec[in_count++] = feat.getY();
  }
  if(I_am_a_netinput== false && index_of_relevant_player >=0){
    DBLOG_POL(BASELEVEL +1,<<"OOPS: Relevant Player, idx "<<
	    index_of_relevant_player<<" not considered as a net input, CORRECT this!. Pos: "
	    <<state.my_team[index_of_relevant_player].pos);
    in_count -= 2;  // go back
    code_pos(state, state.my_team[index_of_relevant_player].pos,feat,netinfo.sort_type);
    netinfo.net.in_vec[in_count++] = feat.getX();
    netinfo.net.in_vec[in_count++] = feat.getY();
  }
  

  // sort opponents
  teamsort.reset();
  bool opgoalie_found = false;
  for(int i=0;i<PLAYERS;i++){
    if(!state.op_team[i].valid)
      continue;
    if(state.op_team[i].number == WSinfo::ws->his_goalie_number){
      // opponents goalie always should be the first opponent
      code_pos(state, state.op_team[i].pos,feat,netinfo.sort_type);
      netinfo.net.in_vec[in_count++] = feat.getX();
      netinfo.net.in_vec[in_count++] = feat.getY();
      opgoalie_found = true;
    }
    else{ // regular player
      Vector ppos = state.op_team[i].pos;
      teamsort.add(i,ppos.sqr_distance(sort_reference));
    }
  }

  if(opgoalie_found==false){
    // didnt find goalie; use dummy pos instead
    code_pos(state, Vector(52.0,0),feat,netinfo.sort_type);
    netinfo.net.in_vec[in_count++] = feat.getX();
    netinfo.net.in_vec[in_count++] = feat.getY();
  }

  teamsort.do_sort();

  for(int k=0;k<netinfo.opponents_num-1;k++){ // consider op_num -1 ops, goalie is already inserted.
    int idx = teamsort.get_key(k);
    if(idx<0){ // there is no player left to fill the inputs -> set him very far back!
      //LOG_DEB(0, << "Error with net: player number " << k << " not found");
      //LOG_ERR(0, << "Error with net: player number " << k << " not found");
      code_pos(state, Vector(MIN_POSITION_NET,
			     (drand48()-0.5)*ServerOptions::pitch_width),feat,netinfo.sort_type);
      //code_pos(state, state.ball.pos - Vector(10,0),feat);
    }
    else
      code_pos(state, state.op_team[idx].pos,feat,netinfo.sort_type);
    netinfo.net.in_vec[in_count++] = feat.getX();
    netinfo.net.in_vec[in_count++] = feat.getY();
  }


  if(netinfo.net.topo_data.in_count != in_count){
    cout<<"Error: PJ: Jnet input "<<in_count<<" != Netinput size "<<netinfo.net.topo_data.in_count
	<<endl;
    return 0.0;
  }

  netinfo.net.forward_pass(netinfo.net.in_vec,netinfo.net.out_vec);
  result = netinfo.net.out_vec[0];

  //debug only:
#if 0 // be careful in activating - takes a lot of time
  if(testoutput==true){
    char buffer[500];
    sprintf(buffer," ");
    for(int k=0;k<in_count;k++){ 
      sprintf(buffer,"%s %.3f ",buffer,netinfo.net.in_vec[k]);
    }
    sprintf(buffer,"%s ---> %.3f ",buffer,netinfo.net.out_vec[0]);
    DBLOG_POL(1,"Planning: Net in and out "<<buffer);
  }
#endif
  
  return result;
}


float Planning::evaluate_byQnn(const AState &state, const AAction &action){
  AState successor_state;

  // compute velocity of leaving ball -> 'action information'
  AbstractMDP::model_fine(state, action,successor_state);
  
  //evaluate successor state:
  //return Qnn(successor_state,true);  // true means with 
  return Qnn(successor_state);
}

float Planning::evaluate_byP1nn(const AState &state, const AAction &action){
  AState successor_state;

  // compute velocity of leaving ball -> 'action information'
  AbstractMDP::model_fine(state, action,successor_state);
  
  //return P1nn(successor_state,true);
  return P1nn(successor_state);
}

float Planning::evaluate_byJnn(const AState &state, const int last_action_type ){
  float ballpos = state.ball.pos.getX();
  // ridi: new code starts here  
  float result;
  if(ballpos > 0 && ballpos <30){
    double Vnn=0;
    Vnn= compute_Jnn(Jnn1,state,false);
    result = 52.5 * (1.-Vnn) + 52.5;
    DBLOG_POL(BASELEVEL,"BALLPOS >0 >30 by NEURAL value function; ballpos "<<ballpos<<" V "<<result);
    return 1 - result/1000.;  // ridi04: transforms in: the lower, the better    
  }
  else{
    result = evaluate(state, last_action_type);
    DBLOG_POL(BASELEVEL,"Do evaluation by HANDCODED value function; ballpos "<<ballpos<<" V "<<result);
    return 1 - result/1000.;  // ridi04: transforms in: the lower, the better
    // return 1 - (result - 52.5) / 52.5; // daniel's version 
  }
  // ridi: new code ends here

#if 0
  if(ballpos > Jnn1_active_thresh){
    double Vnn=0;
    Vnn=Jnn(state,true);
    //return 52.5 * (1.-Vnn) + 52.5;
    return Vnn;
  }
  else{
    float result = evaluate(state, last_action_type);
    DBLOG_POL(BASELEVEL,"Do evaluation by HANDCODED value function; ballpos "<<ballpos<<" V "<<result);
    return 1 - result/1000.;  // ridi04: transforms in: the lower, the better

    // return 1 - (result - 52.5) / 52.5; // daniel's version 
  }
#endif
}


float Planning::evaluate_byJnn_1vs1(const AState &state){
  float ballpos = state.ball.pos.getX();

  if(ballpos > J1vs1_active_thresh){
    double Vnn=0;
    Vnn=compute_Jnn(J1vs1,state,true);
    LOG_POL(0,"Do 1vs1 evaluation by NEURAL value function; ballpos "<<ballpos);
    //    return 52.5 * (1.-Vnn) + 52.5;
    return Vnn;   // ridi 05: return direct NN value
  }
  else{
    LOG_POL(0,"Do 1vs1 evaluation by HANDCODED value function; ballpos "<<ballpos);
    //    return evaluate(state, AACTION_TYPE_SELFPASS);
    return -1;  // ridi 05: return -1 to indicate, that NN is not used.
  }
}


float Planning::evaluate_state(const AState &state){
#define SCORE_ZONE 30
  Vector goalpos = mdpInfo::opponent_goalpos();
  float ballpos = state.ball.pos.getX();
  float amplifier = 1.;

  if(valuefunction ==2){
    if(ballpos > Jnn1_active_thresh){
      double Vnn=0;
      Vnn=Jnn(state,true);
      return 52.5 * (1.-Vnn) + 52.5;
    }
    else
      return evaluate_state1(state);
  }
  if(valuefunction ==1)
    return evaluate_state1(state);

  PlayerSet pset_tmp = WSinfo::valid_opponents;
  Halfplane2d plane(Vector(WSinfo::me->pos.getX() - 3.0, 0.0), Vector(1.0, 0.0));
  pset_tmp.keep_players_in(plane);

  if(ballpos > (goalpos.getX() - SCORE_ZONE)){
    amplifier = 1.+ goalshot_chance_vs_optigoalie(state.ball.pos);
    if(state.my_team[state.my_idx].number == mdpInfo::mdp->me->number
       && WSinfo::me->pos.getX()<47
       && pset_tmp.num >=2){
      // I'm the one that has the ball at the end of playing (e.g. SOLO)
      // and I'm not yet standing at the end of the pitch
      amplifier = 3.0;
    }
  }
  ballpos = state.ball.pos.getX() * amplifier; // go as far as you can
  return ballpos+52.5;
}

float Planning::evaluate_state1(const AState &state){
#define SCORE_ZONE_X 22.5
#define SCORE_ZONE_Y 20.0
#define Vmax 100.

  if(Tools::is_a_scoring_position(state.ball.pos)){
    DBLOG_POL(0,"EVAL 1 POSITION IS IN FINAL SCORING POS!");
    PlayerSet pset_tmp = WSinfo::valid_opponents;
    pset_tmp.keep_players_in_circle(state.ball.pos, 5.0);
    if(pset_tmp.num == 0)
      return 1000;
    pset_tmp.keep_players_in_circle(state.ball.pos, 4.0);
    if(pset_tmp.num == 0)
      return 999;
    pset_tmp.keep_players_in_circle(state.ball.pos, 3.0);
    if(pset_tmp.num == 0)
      return 998;
    pset_tmp.keep_players_in_circle(state.ball.pos, 2.0);
    if(pset_tmp.num == 0)
      return 997;
    pset_tmp.keep_players_in_circle(state.ball.pos, 1.0);
    if(pset_tmp.num == 0)
      return 996;
  }
  
  double alpha, gamma, Vattack;

  Vector goalpos = mdpInfo::opponent_goalpos();
  Vector ballpos = state.ball.pos;

  if(ballpos.getX() < SCORE_ZONE_X){ // ballpos not in attack zone
    return ballpos.getX() + 52.5;
  }
  PlayerSet pset_tmp = WSinfo::valid_opponents;
  pset_tmp.keep_players_in_circle(ballpos, 5.0);
  // final ballpos is in region F (for Flanke) or region S (for score)
  if(fabs(ballpos.getY())>SCORE_ZONE_Y){ // in region F
    if(ballpos.getX() < 47 && pset_tmp.num > 0){
      Vattack = (ballpos.getX() - SCORE_ZONE_X) / (ServerOptions::pitch_length - SCORE_ZONE_X) * Vmax;
    }
    else{
      double maxdist = sqrt(ServerOptions::pitch_width * ServerOptions::pitch_width +
			   (ServerOptions::pitch_length - SCORE_ZONE_X) *
			   (ServerOptions::pitch_length - SCORE_ZONE_X));
      Vattack = (maxdist - ballpos.distance(goalpos)) / (maxdist - SCORE_ZONE_Y) * Vmax;
    }
  }
  else{ // final ballpos is in score region S
    if(WSinfo::me->pos.getX() > SCORE_ZONE_X && fabs(WSinfo::me->pos.getY())>SCORE_ZONE_Y){
      // I (the planner himself) am standing in the F region.
      gamma = (ballpos.getX() - SCORE_ZONE_X)/(ServerOptions::pitch_length - SCORE_ZONE_X);
      gamma = gamma*gamma;
      if (gamma > .8)
	gamma = 1.;
    }
    else{ // I am currently in S or in the rest of the field
      gamma = 1;
    }

    alpha = goalshot_chance_vs_optigoalie(state.ball.pos);
    Vattack = (alpha + gamma) * Vmax;
  }
  return (Vattack + SCORE_ZONE_X + 52.5);
}


/********************************************************************************/

/*                   Neural Functions */

/********************************************************************************/

#define PLAYERS 11

void Planning::generate_input(const AState& state, const Netdescription &netinfo){
  Vector feat;
  int in_count;

  if(netinfo.sort_type == 0) { // sort players according to number
    DBLOG_ERR(0,"Planning. Warning: net_sort_type 0 no longer supported");
  }

  // determine net input (straightforward, to be improved!)
  code_ball(state,state.ball.pos,feat);
  netinfo.net.in_vec[0] = feat.getX();
  netinfo.net.in_vec[1] = feat.getY();
  netinfo.net.in_vec[2] = state.ball.vel.getX();
  netinfo.net.in_vec[3] = state.ball.vel.getY();
  in_count = 4;

  Vector sort_reference;
  if(netinfo.sort_type ==1)
    sort_reference=mdpInfo::opponent_goalpos();
  else
    sort_reference = state.ball.pos;
  Sort teamsort = Sort(PLAYERS);
  for(int i=0;i<PLAYERS;i++){
    if(!state.my_team[i].valid)
      continue;
    Vector ppos = state.my_team[i].pos;
    teamsort.add(i,ppos.distance(sort_reference));
  }
  teamsort.do_sort();
  for(int k=0;(in_count <4+2*netinfo.teammates_num && k<PLAYERS);k++){ 
    int idx = teamsort.get_key(k);
    if((k==0) && (netinfo.sort_type ==3))
      netinfo.net.in_vec[in_count++] = state.my_team[idx].body_angle;
    code_pos(state, state.my_team[idx].pos,feat,netinfo.sort_type);
    netinfo.net.in_vec[in_count++] = feat.getX();
    netinfo.net.in_vec[in_count++] = feat.getY();
  }
  
  teamsort.reset();
  
  for(int i=0;i<PLAYERS;i++){
    if(!state.op_team[i].valid)
      continue;
    Vector ppos = state.op_team[i].pos;
    teamsort.add(i,ppos.distance(sort_reference));
  }
  teamsort.do_sort();
  for(int k=0;k<netinfo.opponents_num;k++){ // opponents
    int idx = teamsort.get_key(k);
    code_pos(state, state.op_team[idx].pos,feat,netinfo.sort_type);
    netinfo.net.in_vec[in_count++] = feat.getX();
    netinfo.net.in_vec[in_count++] = feat.getY();
  }
}


float Planning::Qnn(const AState& state, const bool testoutput ){
  float result;

  if(Qnet.loaded == false)
    return 0.0;
  if(Qnet.net.topo_data.in_count != 4+2*(Qnet.teammates_num+Qnet.opponents_num)){
    cout<<"Error: Qnet.Net input != "<<4+2*(Qnet.teammates_num+Qnet.opponents_num)<<endl;
  }
    
  generate_input(state,Qnet);

  Qnet.net.forward_pass(Qnet.net.in_vec,Qnet.net.out_vec);
  result = Qnet.net.out_vec[0];

  //debug only:
#if 1 // be careful in activating - takes a lot of time
  if(testoutput==true){
    char buffer[500];
    sprintf(buffer," ");
    for(int k=0;k<Qnet.net.topo_data.in_count;k++){ 
      sprintf(buffer,"%s %.1f ",buffer,Qnet.net.in_vec[k]);
    }
    sprintf(buffer,"%s ---> %.3f ",buffer,Qnet.net.out_vec[0]);
    DBLOG_POL(1,"Planning: Q-Net in and out "<<buffer);
  }
#endif

  return result;
}

float Planning::P1nn(const AState& state, const bool testoutput ){
  float result;
  int extra_input = 0;

  if(P1net.loaded == false)
    return 0.0;
  if(P1net.sort_type == 3)
    extra_input = 1;
  if(P1net.net.topo_data.in_count != 4+extra_input+2*(P1net.teammates_num+P1net.opponents_num)){
    cout<<"Error: P1net.Net input != "<<4+2*(P1net.teammates_num+P1net.opponents_num)<<endl;
  }
    
  generate_input(state,P1net);

  P1net.net.forward_pass(P1net.net.in_vec,P1net.net.out_vec);
  result = P1net.net.out_vec[0];

#if 1 // be careful in activating - takes a lot of time
  if(testoutput==true){
    char buffer[500];
    sprintf(buffer," ");
    for(int k=0;k<P1net.net.topo_data.in_count;k++){ 
      sprintf(buffer,"%s %.1f ",buffer,P1net.net.in_vec[k]);
    }
    sprintf(buffer,"%s ---> %.3f ",buffer,P1net.net.out_vec[0]);
    DBLOG_POL(1,"Planning: P1-net in and out "<<buffer);
  }
#endif


  return result;
}
 
bool Planning::is_laufpass_successful2(const Vector ballpos, const float speed, const float dir, Vector & interceptpos,
				       int & number, int & advantage, Vector & playerpos){
  // quick check of success or failure:
  PlayerSet pset= WSinfo::valid_opponents;
  pset.append( WSinfo::valid_teammates_without_me);
  Vector endofregion;
  double length = 30.;
  double startwidth = 2.;
  endofregion.init_polar(length, dir);
  endofregion += WSinfo::me->pos;
  Quadrangle2d check_area = Quadrangle2d(WSinfo::me->pos, endofregion, startwidth, 0.4 * length);
  //  LOG_POL(0,<<_2D<< check_area);
  pset.keep_players_in(check_area);

  if(pset.num >0){ 
    PPlayer player=pset.closest_player_to_point(WSinfo::me->pos);
    if(player->team == HIS_TEAM)
      return false;
    /* ridi05: this is too dangerous; if passway is free, check regularly by intercept.
    if(is_offside(player->pos) == true)
      return false;
    // found a player in direct passway!
    playerpos = player->pos;
    interceptpos = player->pos;
    number = player->number;
    pset = WSinfo::valid_opponents;
    pset.keep_and_sort_closest_players_to_point(1, playerpos);
    if(pset.num >0){
      if(playerpos.distance(pset[0]->pos)<1.)
	advantage = 0;
      else if(playerpos.distance(pset[0]->pos)<2.)
	advantage = 1;
      else if(playerpos.distance(pset[0]->pos)<3.)
	advantage = 2;
      else
	advantage = 5;
    }
    else // no op. found
      advantage = 5;  // default
    return true;
    */
  }  // found a player in passway


  PlayerSet oppPlayers = WSinfo::valid_opponents;
  PlayerSet myPlayers =WSinfo::valid_teammates_without_me;
  PPlayer myGoalie = WSinfo::get_teammate_by_number(WSinfo::ws->my_goalie_number);
  if (myGoalie) myPlayers.remove(myGoalie);
  
  length = 40.;
  startwidth = 5.;
  endofregion.init_polar(length, dir);
  endofregion += WSinfo::me->pos;
  check_area = Quadrangle2d(WSinfo::me->pos, endofregion, startwidth, 1. * length);

  //  LOG_POL(0,<<_2D<< check_area);
  myPlayers.keep_players_in(check_area);
  oppPlayers.keep_players_in(check_area);
 
  Vector passVelocity;
  passVelocity.init_polar( speed, dir );
  int numberOfBestInterceptorsToBeConsidered = myPlayers.num;

  InterceptResult *myintercept_res = new InterceptResult [numberOfBestInterceptorsToBeConsidered];
  myPlayers.keep_and_sort_best_interceptors_with_intercept_behavior
                                       ( numberOfBestInterceptorsToBeConsidered,
                                         WSinfo::ball->pos,
                                         passVelocity, 
                                         myintercept_res);

  numberOfBestInterceptorsToBeConsidered = oppPlayers.num;
  InterceptResult *oppintercept_res = new InterceptResult [numberOfBestInterceptorsToBeConsidered];
  oppPlayers.keep_and_sort_best_interceptors_with_intercept_behavior( numberOfBestInterceptorsToBeConsidered,
								      WSinfo::ball->pos,
								      passVelocity,
								      oppintercept_res);
/*
  while (myPlayers.num > 0)
  {
    number       = myPlayers[0]->number;
    interceptpos = myintercept_res[0].pos;
    playerpos    = myPlayers[0]->pos;
    double requiredStaminaForInterception
      =   myintercept_res[0].time 
        * (100.0 - myPlayers[0]->stamina_inc_max);
    double bestCaseTeammateStaminaLevel = ServerOptions::stamina_max;
    if ( myPlayers[0]->pointto_flag == true )
      bestCaseTeammateStaminaLevel
        =   Tools::getStaminaFromPointto( myPlayers[0]->pointto_dir )
          + myPlayers[0]->pointto_age * myPlayers[0]->stamina_inc_max 
          +   (ServerOptions::point_to_ban - 1)* myPlayers[0]->stamina_inc_max;
    if ( bestCaseTeammateStaminaLevel > ServerOptions::stamina_max )
      bestCaseTeammateStaminaLevel = ServerOptions::stamina_max;
    if (   bestCaseTeammateStaminaLevel
         < 1250 + requiredStaminaForInterception )
    {
      DBLOG_POL(0,"TG@Planning: I removed one player ("
        <<myPlayers[0]->num<<") from myPlayers.");
      myPlayer.remove( myPlayers[0] );
      DBLOG_POL(0,"TG@Planning: That player requires "
        <<requiredStaminaForInterception<<", but, to my information,"
        <<" has only "<<bestCaseTeammateStaminaLevel<<" (best case).");
      for ( int p=0; p<myPlayers.num; p++)
      {
        myintercept_res[p].pos  = myintercept_res[p+1].pos;
        myintercept_res[p].time = myintercept_res[p+1].pos;
      }
      continue;
    }
  }
*/


    //FIXME: is that save here???
  InterceptResult myinterceptFinalResult = myintercept_res [0];
  delete [] myintercept_res;
  myintercept_res = NULL;
  InterceptResult oppinterceptFinalResult = oppintercept_res [0];
  delete [] oppintercept_res;
  oppintercept_res = NULL;
  if (myPlayers.num > 0)
  {
    number = myPlayers[0]->number;
    interceptpos = myinterceptFinalResult.pos;
    playerpos = myPlayers[0]->pos;
    if (is_offside(playerpos) ==true)
      return false;
    if (Tools::is_position_in_pitch(interceptpos) == false)
      return false;
    if (oppPlayers.num >0)
    {
      if(myinterceptFinalResult.time < oppinterceptFinalResult.time)
      {
        //my team is at least faster at ball as opponent
        advantage = oppinterceptFinalResult.time - myinterceptFinalResult.time;
        return true;
      }
      else
      { // my team is not faster than op team
        return false;
      }
    }// opponent found
    else
    {// no opponent found
      advantage = 10;
      return true;
    }
  }
  else
  { // did not find a teammate that intercepts
    return false;
  }
  return false;
}

bool Planning::is_laufpass_successful2(const Vector ballpos, float speed, const float dir){

  Vector interceptpos;
  int advantage, number;
  Vector playerpos; // current position of the player intercepting the ball

  return is_laufpass_successful2(ballpos,speed,dir,interceptpos,number, advantage, playerpos);
}

bool Planning::check_action_laufpass2(AAction &candidate_action, float speed, const float dir){

  Vector interceptpos;
  int advantage, number;
  Vector playerpos; // current position of the player intercepting the ball
  bool is_risky;
  
  if(is_laufpass_successful2(WSinfo::ball->pos,speed,dir,interceptpos,number, advantage, playerpos) == false)
    return false;

  ANGLE line2player = (playerpos - WSinfo::me->pos).ARG();
  ANGLE line2goal = (Vector(52.0,0) - WSinfo::me->pos).ARG();

  bool dir_ok=false;

  if(line2goal.diff(line2player) > 1.1 * line2goal.diff(ANGLE(dir))){ // pass direction lies within line to player and line to goal
    dir_ok = true;
  }
  if(line2goal.diff(ANGLE(dir)) < 10./180.*PI){
    dir_ok = true;
  }

  if(dir_ok == false){
    //  Tools::display_direction(WSinfo::me->pos, ANGLE(dir), 20.);
        DBLOG_DRAW(0,C2D(playerpos.x,playerpos.y,2.0,"brown"));
        DBLOG_POL(0,"Check Laufpass: Laufpass is posible, but gets behind targetplayer -> return false");
    return false;
  }

  if(advantage >= 1)
    is_risky = false;
  else
    is_risky = true;


  if (number == WSinfo::me->number) {
    return false;
  }

  Vector targetpos;
  targetpos.init_polar(1.0/(1.0-0.94) *speed,dir);
  targetpos += WSinfo::ball->pos;

  AbstractMDP::set_action(candidate_action, AACTION_TYPE_LAUFPASS, 
			  0, 0, targetpos, speed,dir,
			  interceptpos,Vector(0),advantage,0,0,
			  number,
			  is_risky);


  return true;
}

bool Planning::is_penaltyareapass_successful(const Vector ballpos,float speed, const float dir, int & advantage, int &number,
Vector & playerpos){

  PlayerSet pset= WSinfo::valid_opponents;
  Vector endofregion;
  double length = 20.;
  double startwidth = 2.;
  endofregion.init_polar(length, dir);
  endofregion += WSinfo::me->pos;
  Quadrangle2d check_area = Quadrangle2d(WSinfo::me->pos, endofregion, startwidth, 0.25 * length);
  //  LOG_POL(0,<<_2D<< check_area);
  pset.keep_players_in(check_area);
  if(pset.num >0){
    return false;  // pass not possible at all
  }

  // found a quadrangle with no opponent. Now check myself
  pset= WSinfo::valid_teammates_without_me;
  pset.keep_players_in(check_area);
  if(pset.num >0){ 
    PPlayer player=pset.closest_player_to_point(WSinfo::me->pos);
    playerpos = player->pos;
    number = player->number;
    if(is_offside(playerpos) == true)
      return false;
    pset = WSinfo::valid_opponents;
    pset.keep_and_sort_closest_players_to_point(1, playerpos);
    if(pset.num >0){
      if(playerpos.distance(pset[0]->pos)<1.)
	advantage = 0;
      else if(playerpos.distance(pset[0]->pos)<2.)
	advantage = 1;
      else if(playerpos.distance(pset[0]->pos)<3.)
	advantage = 2;
      else
	advantage = 5;
    }
    else // no op. found
      advantage = 5;  // default
  }
  return false;
}

bool Planning::is_offside( const Vector pos){
  if(pos.getX() > WSinfo::his_team_pos_of_offside_line())
    return true;
  return false;
}

bool Planning::is_penaltyareapass_successful(const Vector ballpos,float speed, const float dir){

  int advantage, number;
  Vector playerpos; // current position of the player intercepting the ball

  return is_penaltyareapass_successful(ballpos,speed,dir,advantage, number, playerpos);
}

bool Planning::check_action_penaltyareapass(AAction &candidate_action, float speed, const float dir){

  int advantage, number;
  Vector playerpos; // current position of the player intercepting the ball
  bool risky_pass;


  if(is_penaltyareapass_successful(WSinfo::ball->pos,speed,dir,advantage, number, playerpos) == false)
    return false;

  if(advantage >0)
    risky_pass = false;
  else
    risky_pass = true;


  AbstractMDP::set_action(candidate_action, AACTION_TYPE_LAUFPASS, 
			  0, 0, playerpos, speed,dir,
			  playerpos,Vector(0),advantage,0,0,
			  number,
			  risky_pass);
  
  return true;
}


/***************************************************************************************/

/* new pass check routines */

/***************************************************************************************/

bool Planning::check_kick(double &speed, const double dir ) {

  double speed1 = 0.;
  double speed2 = 0.;

  if(WSinfo::is_ball_kickable() == false)
    return true; // might seem strange, but is correct: this enables foreseeing with an optimistic view ;-)

  ANGLE passdir = ANGLE(dir);
  if(speed > ServerOptions::ball_speed_max - 0.1)
    twostepkick->kick_in_dir_with_max_vel(passdir);
  else
    twostepkick->kick_in_dir_with_initial_vel(speed,passdir);
  twostepkick->get_vel(speed1,speed2);
  twostepkick->reset_state(); // use current ws-state
  if(speed2<0.1){ // probably could not hold ball in kickrange
    speed = speed1;
  }
  else
    speed = speed2;  // that's possible in two steps

  if(speed < 0.1)
    return false;

  return true;
}



bool Planning::check_pass_with_known_ipos(AAction &candidate_pass, const Vector ballpos){
  // basically computes advantage

  double opteam_time2react = 0; // see is pass safe
  
  if(mdpInfo::is_position_in_pitch(candidate_pass.actual_resulting_position,5.0) == false)
    return false;


  Policy_Tools::opteam_intercepts_ball_faster06(ballpos,candidate_pass.kick_velocity,
							      candidate_pass.kick_dir,
							      candidate_pass.time2intercept,
							      candidate_pass.actual_resulting_position,
							      candidate_pass.advantage,
							      opteam_time2react);
  if(candidate_pass.advantage >0)
    return true;
  else 
    return false;
}


bool Planning::compute_ipos_of_candidate_pass(AAction &candidate_pass, const Vector ballpos, const double speed, const double dir){

#define INFINITE_STEPS 1000

  double pass_speed = speed;

  if(check_kick(pass_speed,dir) == false){
    //LOG_POL(0,"compute ipos checking kick returns FALSE: dir "<<RAD2DEG(dir)<<" pass_speed "<<pass_speed);
    return false;
  }


  int myteam_fastest = INFINITE_STEPS;
  Vector intercept_pos = Vector(0);
  int my_time2react = 1; // see is_pass_safe!
  int fastest_idx = -1;

  PlayerSet pset = WSinfo::valid_teammates_without_me;
  for (int idx = 0; idx < pset.num; idx++) {
    Vector player_intercept_pos;
    if(pset[idx]->number == WSinfo::me->number){ 
      continue;  // do not consider me!
    }
    int steps = Policy_Tools::get_time2intercept_hetero(player_intercept_pos, ballpos, 
							pass_speed,dir, pset[idx],
							my_time2react,myteam_fastest); 
    if(steps <0) 
      steps =INFINITE_STEPS; // player doesnt get ball in reasonable time
    if(steps < myteam_fastest){ // the fastest own player yet
      myteam_fastest = steps;
      fastest_idx = idx;
      intercept_pos = player_intercept_pos;
    }
  }



  if(myteam_fastest>=INFINITE_STEPS)
    return false;
  if(fastest_idx<0)
    return false;
  
  // now determine Ipos exactly:

  int subtype = SUBTYPE_LAUFPASS;

  Vector ipos;  // result of compute_ipos...
  int num_steps; // result of compute_ipos...

  if(Policy_Tools::compute_ipos_for_teammate06(ballpos, pass_speed, dir, pset[fastest_idx]->number, 
					       num_steps, ipos, my_time2react) == true){
    intercept_pos = ipos;
    myteam_fastest = num_steps;
  }



  if(mdpInfo::is_position_in_pitch(intercept_pos,5.0) == false)
    return false;


  Vector ballvel;
  ballvel.init_polar(pass_speed,dir);

  double distance = Tools::get_dist2_line(ballpos, (ballpos + 20 * ballvel), pset[fastest_idx]->pos);
  if(distance < 1.0 *pset[fastest_idx]->kick_radius)
    subtype = SUBTYPE_PASS;  // default
  else{
    subtype = SUBTYPE_LAUFPASS;  // default
    if(check_if_pass_in_back(ballpos, ipos, pset[fastest_idx]->pos, dir) == true){
      subtype = SUBTYPE_PASS_IN_BACK;  // default
    }
  }

  Vector targetpos = intercept_pos;
  int advantage = 0; // not known yet
  AbstractMDP::set_action(candidate_pass, AACTION_TYPE_PASS, 
			  0, -1, targetpos, pass_speed,dir,
			  intercept_pos, Vector(0), advantage,myteam_fastest,0,pset[fastest_idx]->number);
  candidate_pass.subtype = subtype;  // default
  return true;
}



bool Planning::is_pass_safe(const Vector ballpos,  double & speed, const double dir, Vector & interceptpos, int & advantage, int & receiver,
const int desired_receiver){
  // speed might be adapted to be quick!!!

  int my_time2react = 1;  // these are the values used in 2005. risky: 2
  //  int my_time2react = 1;  // these are the values used in 2005. risky: 2
  int optime2react = 0; // these are the values used in 2005. risky: 2

  Vector playerpos; // not used yet

  // todo:   check_if_kick_is_possible(ballpos, dir, speed)
  if(check_kick(speed,dir) == false){
    //LOG_POL(0,"Checking kick returns FALSE: dir "<<RAD2DEG(dir)<<" speed "<<speed);
    return false;
  }

  bool result = false;
  if(desired_receiver < 0){ // no receiver predetermined
    result = Policy_Tools::myteam_intercepts_ball_hetero06(ballpos,speed,  dir,interceptpos,advantage,
							   receiver,playerpos, my_time2react, optime2react);
  }
  else{
    result = Policy_Tools::myteam_intercepts_ball_hetero06(ballpos,speed,  dir,interceptpos,advantage,
							   receiver,playerpos, my_time2react, optime2react, 
							   desired_receiver);
  }

  if(result == false)
    return false;

  if(WSinfo::ws->play_mode == PM_PlayOn 
  && mdpInfo::is_position_in_pitch(interceptpos,5.0) == false)
    return false;

  return true;
}

bool Planning::check_if_pass_in_back(const Vector ballpos, const Vector ipos, const Vector receiverpos, const double dir ){
  if(fabs(Tools::get_angle_between_mPI_pPI(dir))<45./180. *PI){ // consider only sideward passes
    return false;
  }
#if 0
  if(ballpos.x > BORDER_X - 15.0){
    return false;
  }
#endif
  if(ipos.getX() > receiverpos.getX() - 2.){
    return false;
  }

  return true;    
}


bool Planning::check_candidate_pass(AAction &candidate_action, const Vector ballpos, const double speed, const double dir, const int desired_receiver){

  Vector interceptpos;
  int advantage;
  int receiver_number;
  int subtype = SUBTYPE_PASS;  // default

  Vector ballvel;
  ballvel.init_polar(speed,dir);
  Vector targetpos = ballpos + 30 * ballvel;  // default computation, normally replaced by computed interceptpos

  // todo:   check_if_kick_is_possible(ballpos, dir, speed)

  double pass_speed = speed;

  if(is_pass_safe(ballpos, pass_speed,dir, interceptpos, advantage, receiver_number, desired_receiver) == false){
    return false;
  }
  else //TG17: i.e. pass considered safe
  {
    if (   WSinfo::ws->play_mode == PM_my_GoalKick
        && advantage < 4 )
    {
      LOG_POL(0,"My goal kick: advantage of this pass is too low: "<<advantage);
      return false;
    }
    else
    {
      if (WSinfo::ws->play_mode == PM_my_GoalKick)
      {LOG_POL(0,"My goal kick: advantage of this pass is good: "<<advantage);}
    }
  }
  if(receiver_number == WSinfo::me->number){ // selfpass -> do not consider
    return false;
  }

  PPlayer teammate = WSinfo::valid_teammates.get_player_by_number(receiver_number);
  if(teammate != NULL){
    double distance = Tools::get_dist2_line(ballpos, targetpos, teammate->pos);
    if(distance < teammate->kick_radius)
      subtype = SUBTYPE_PASS;  // default
    else{
      subtype = SUBTYPE_LAUFPASS;  // default
      if(check_if_pass_in_back(ballpos, interceptpos, teammate->pos, dir) == true){
	subtype = SUBTYPE_PASS_IN_BACK;  // default
	return false;
      }
    }
  }

  targetpos = interceptpos;

  AbstractMDP::set_action(candidate_action, AACTION_TYPE_PASS, 
			  0, -1, targetpos, pass_speed,dir,
			  interceptpos, Vector(0), advantage,0,0,receiver_number);
  candidate_action.subtype = subtype;  // default
  return true;
}

bool Planning::is_penaltyareapass_safe(const Vector ballpos,float speed, const float dir, int & advantage, int &number,
Vector & playerpos){

  PlayerSet pset= WSinfo::valid_opponents;
  Vector endofregion;
  double length = 20.;
  double startwidth = 2.;
  endofregion.init_polar(length, dir);
  endofregion += WSinfo::me->pos;
  Quadrangle2d check_area = Quadrangle2d(WSinfo::me->pos, endofregion, startwidth, 0.25 * length);
  //  LOG_POL(0,<<_2D<< check_area);
  pset.keep_players_in(check_area);
  if(pset.num >0){
    return false;  // pass not possible at all
  }

  // found a quadrangle with no opponent. Now check myself
  pset= WSinfo::valid_teammates_without_me;
  pset.keep_players_in(check_area);
  if(pset.num >0){ 
    PPlayer player=pset.closest_player_to_point(WSinfo::me->pos);
    playerpos = player->pos;
    number = player->number;
    if(is_offside(playerpos) == true)
      return false;
    pset = WSinfo::valid_opponents;
    pset.keep_and_sort_closest_players_to_point(1, playerpos);
    if(pset.num >0){
      if(playerpos.distance(pset[0]->pos)<1.)
	advantage = 0;
      else if(playerpos.distance(pset[0]->pos)<2.)
	advantage = 1;
      else if(playerpos.distance(pset[0]->pos)<3.)
	advantage = 2;
      else
	advantage = 5;
    }
    else // no op. found
      advantage = 5;  // default
  }
  return false;
}



bool Planning::check_action_penaltyareapass06(const Vector ballpos, AAction &candidate_action, float speed, const float dir){

  int advantage, number;
  Vector playerpos; // current position of the player intercepting the ball
  bool risky_pass;


  if(is_penaltyareapass_safe(ballpos,speed,dir,advantage, number, playerpos) == false)
    return false;

  if(advantage >0)
    risky_pass = false;
  else
    risky_pass = true;


  AbstractMDP::set_action(candidate_action, AACTION_TYPE_PASS, 
			  0, -1, playerpos, speed,dir,
			  playerpos,Vector(0),advantage,0,0,
			  number,
			  risky_pass);
  
  candidate_action.subtype = SUBTYPE_PENALTYAREAPASS;  // default
  return true;
}


