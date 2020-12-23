#include "neuro_wball.h"

#define MAX(X,Y) ((X>Y)?X:Y)

#if 0
#define LOGNEW_POL(LLL,XXX) LOG_POL(LLL,<<"NEUROWBALL03: "XXX)
#define LOGNEW_DRAW(LLL,XXX) LOG_POL(LLL,<<_2D <<XXX)
#define LOGNEW_ERR(LLL,XXX) LOG_ERR(LLL,XXX)
#define MYGETTIME (Tools::get_current_ms_time())
#define NEWBASELEVEL 0 // level for logging; should be 3 for quasi non-logging
#else
#define LOGNEW_POL(LLL,XXX)
#define LOGNEW_DRAW(LLL,XXX)
#define LOGNEW_ERR(LLL,XXX) 
#define MYGETTIME (0)
#define NEWBASELEVEL 3 // level for logging; should be 3 for quasi non-logging
#define LOG_DAN(YYY,XXX)
#endif


#if 0
#define DBLOG_POL(LLL,XXX) LOG_POL(LLL,<<"NEUROWBALL03: "XXX)
#define DBLOG_DRAW(LLL,XXX) LOG_POL(LLL,<<_2D <<XXX)
#define DBLOG_ERR(LLL,XXX) LOG_ERR(LLL,XXX)
//#define MYGETTIME (Tools::get_current_ms_time())
#define BASELEVEL 0 // level for logging; should be 3 for quasi non-logging
//#define LOG_DAN(YYY,XXX) LOG_DEB(YYY,XXX)
#define LOG_DAN(YYY,XXX)
#else
#define DBLOG_POL(LLL,XXX)
#define DBLOG_DRAW(LLL,XXX)
#define DBLOG_ERR(LLL,XXX) 
//#define MYGETTIME (0)
#define BASELEVEL 3 // level for logging; should be 3 for quasi non-logging
#define LOG_DAN(YYY,XXX)
#endif

//#define TRAINING


NeuroWball::NeuroWball() {
  exploration_mode = 0;
  exploration = 0.0;
  evaluation_mode = 3;

  /*
  char P1net_name[500];
  P1net.opponents_num = 1;
  P1net.teammates_num = 1;
  P1net.sort_type = 1;

  sprintf(P1net_name,"./data/pass.net"); // Default
  */

  ValueParser vp(CommandLineOptions::policy_conf,"NeuroWball");
  /*
  vp.get("P1net",P1net_name,500);
  if(P1net.net.load_net(P1net_name) == 0)
    P1net.loaded = true;
  else
    P1net.loaded = false;
  vp.get("P1net_opponents_num",P1net.opponents_num);
  vp.get("P1net_teammates_num",P1net.teammates_num);
  */
  vp.get("exploration_mode", exploration_mode);
  vp.get("exploration", exploration);
  vp.get("evaluation_mode", evaluation_mode);

  selfpass = new Selfpass;
  dribblestraight = new DribbleStraight;
}


NeuroWball::~NeuroWball() {
  delete selfpass;
  delete dribblestraight;
}

bool NeuroWball::evaluate_passes_and_dribblings(AAction & best_aaction_P, AState &current_state){
  /* ridi 05: this is still called by standard situations. To avoid inconsistencies and crashes, 
     calls are momentarily redirected to evaluate_passes. Should be changed in standard situations.
     After Osaka....

  */
  return evaluate_passes(best_aaction_P, current_state);

  // original code starts here

  AAction action_set[MAX_AACTIONS];
  AAction best_aaction;
  int action_set_size;
  float Vbest;

  action_set_size = generate_action_set(current_state, action_set);
  LOGNEW_POL(0, << "Action set generation needed " << MYGETTIME- ms_time << " millis");

  // 3. select best action

  if((Vbest = select_best_aaction(current_state,action_set,action_set_size,best_aaction)) <0.0)
    return false;

 
  LOGNEW_POL(0, << "select best needed " <<MYGETTIME- ms_time << " millis");

  if(Vbest <0.0)
    return false;

  best_aaction_P = best_aaction;

  return true;

  //return AbstractMDP::aaction2move(bestaction);
}




/*------------------------------------------------------------------------------*/
/*PLANNING----------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/


bool NeuroWball::is_better(const float V,const float Vbest,
				     const AState &state, const AAction *actions,
				     const int a, const int abest){

  //DANIEL: DONT USE,AS VALUEFUNCTION HAS CHANGED: THE LOWER,THE BETTER!!!!


  // this routine depends on the order of generate_pactions: 
  // 1. Passes, 2. Dribblings, 3. WaitandSee

  if(Vbest >= 2.0)
    return true;  // currently no valid move found
  if(V == Vbest){ 
    double posa_dist, posabest_dist;
    
    PPlayer p_tmp = WSinfo::valid_opponents.closest_player_to_point(actions[abest].actual_resulting_position);
    if (p_tmp == NULL) return true;
    posabest_dist = (p_tmp->pos - actions[abest].actual_resulting_position).sqr_norm();

    p_tmp = WSinfo::valid_opponents.closest_player_to_point(actions[a].actual_resulting_position);
    if (p_tmp == NULL) return true;
    posa_dist = (p_tmp->pos - actions[a].actual_resulting_position).sqr_norm();


    //mdpInfo::opponent_closest_to(actions[abest].actual_resulting_position,posabest_dist,dir);
    //mdpInfo::opponent_closest_to(actions[a].actual_resulting_position,posa_dist,dir);
    
    /*
      Vector Goal_pos = Vector(ServerOptions::pitch_length/2.,0);
    if (Goal_pos - actions[a].target_position).norm() < 
      (Goal_pos - actions[abest].target_position).norm()){
    */
    if(posa_dist<posabest_dist){
      DBLOG_POL(BASELEVEL+2,<<"Action "<<a<<" is has same V like current best action "<<abest
		<<" but has more freedom -> select");
     return true;
    }
  }


  if((V < Vbest) && 
     (actions[abest].action_type == AACTION_TYPE_PASS)){
    // the current winner is a pass, but LAUFPASS has a better value
    if(actions[a].action_type == AACTION_TYPE_LAUFPASS){
      if(actions[abest].target_player == actions[a].target_player){
	if(actions[a].actual_resulting_position.getX() <0){
	  DBLOG_POL(1,"Planning prefers Laufpass to "<<
		    state.my_team[actions[abest].target_player].number<<" but I can also Pass and in own half ->pass ");
	  return false;
	}
	if(actions[a].actual_resulting_position.getX() <WSinfo::his_team_pos_of_offside_line()+3){
	  DBLOG_POL(1,"Planning prefers Laufpass to "<<
		    state.my_team[actions[abest].target_player].number<<" and resulting pos is behind offside ->pass ");
	  return false;
	}
	// in opponents half
	if(actions[abest].actual_resulting_position.distance(actions[a].actual_resulting_position) <5.){
	  DBLOG_POL(1,"Planning prefers Laufpass to "<<
		    state.my_team[actions[abest].target_player].number<<" but I can also Pass and positions close ->pass ");
	  return false;
	}
	else{
	  DBLOG_POL(1,"Planning prefers Laufpass to "<<
		    state.my_team[actions[abest].target_player].number
		    <<" and Pass is possible but laufpass brings position advantage -> laufpass");
	  return true;
	}
      }
    }
  }

  if((V < Vbest) && 
     ((actions[abest].action_type == AACTION_TYPE_PASS) ||
      (actions[abest].action_type == AACTION_TYPE_LAUFPASS) ||
      (actions[abest].action_type == AACTION_TYPE_DEADLYPASS))){
    // the current winner is a pass, but WAITANDSEE has a better value!
    /* Daniel: not needed because WAITANDSEE is not considered as an action!
    if(actions[a].action_type == AACTION_TYPE_WAITANDSEE){
      //the candidate action is to wait and see
      if(onetwoholdturn->is_holdturn_safe() == false){
	DBLOG_POL(1,"Planning prefers WaitandSee  V: "<<V<<", Vbest: "<<Vbest
		  <<", BUT I'm attacked, so pass ");
	return false;
      }
      Vector pass_result = actions[abest].actual_resulting_position;
      if((pass_result.x > 32.0) &&
	 ((fabs(pass_result.y) < fabs(WSinfo::me->pos.y)) ||
	  (fabs(pass_result.y) < ServerOptions::goal_width/2.))
	 ){
	  DBLOG_POL(1,"Planning prefers WaitandSee  V: "<<V<<", Vbest: "<<Vbest
		    <<", BUT teammate has attractive position so pass ");
	  return false;
      }

    } // end candidate Wait and See
    */
  }
  return (V<Vbest); //standard
}


void NeuroWball::refine_laufpass(const AState state,
			      AAction &action, const float Vaction){
  float Vbest = 2.0;
  AAction best_action;
  //  const float refine_min_advantage = 2; // should be higher than policy_tools::selfpass_my_min_advantage

  //  float speed_min = 1.3;
  float speed_max = 2.5;//ServerOptions::ball_speed_max; /TG08: NeuroKick cannot kick as hard, yet !!!
  float speed_inc = .3;
  float dangle_range = 2.5;  // check +- angle_range degrees
  float dangle_inc = 2.5;

  long ms_time= MYGETTIME;

  AAction candidate_action;
  float refdir = action.kick_dir;
  float speed_min = action.kick_velocity;
  const float refine_min_advantage = action.advantage;

  for(float dangle=-dangle_range;dangle<dangle_range+dangle_inc/2.;dangle+=dangle_inc){
    for(float speed=speed_min;speed<=speed_max;speed+=speed_inc){
      float dir = refdir + dangle/180. *PI; 
      if(speed > ServerOptions::ball_speed_max || speed < 0.4)
	continue;
#if 0
      DBLOG_POL(BASELEVEL,<<"refine laufpass check speed "<<speed<<" dir "<<RAD2DEG(dir)<<" delta dir "
	      <<dangle);
#endif
      if(Planning::check_action_laufpass(state,candidate_action,speed,dir)){
	if(candidate_action.advantage < refine_min_advantage)
	  continue;
	float V = do_evaluation(state,candidate_action);
	if ( V < Vbest ) {
	  best_action=candidate_action;
	  Vbest = V;
	}
      }
    }
  }

  ms_time = MYGETTIME - ms_time;
  if ( Vbest < Vaction) {
    action = best_action;
    LOGNEW_POL(NEWBASELEVEL+1,<<"Refine SUCCESSFUL V improved by " << Vbest-Vaction
	    <<"Time needed: "<<ms_time<<" ms. new action: ");
    print_action_data(state,action,0,Vbest,0,1);
  }
  else{
    DBLOG_POL(BASELEVEL+1,<<"Refine NO ALTERNATIVE FOUND: Time needed: "<<ms_time<<" ms. new action: ");
  }
}


void NeuroWball::print_action_data(const AState current_state, const AAction action,const int idx, const float V,const float display_value, const int log_level){


    if (action.action_type != AACTION_TYPE_WAITANDSEE) {
      LOGNEW_DRAW(NEWBASELEVEL+log_level, L2D(WSinfo::ball->pos.x, WSinfo::ball->pos.y, 
			       action.target_position.x,
			       action.target_position.y, 
			       "lightblue"));
      LOGNEW_DRAW(NEWBASELEVEL+log_level, C2D(action.actual_resulting_position.x,
					      action.actual_resulting_position.y, 1.0,"lightblue"));
      LOGNEW_DRAW(NEWBASELEVEL+log_level, C2D(action.potential_position.x,
					      action.potential_position.y, 1.0,"magenta"));
		  }

    if (action.action_type != AACTION_TYPE_WAITANDSEE
	&& action.risky_pass == true) {
      LOGNEW_DRAW(NEWBASELEVEL+2, L2D(WSinfo::ball->pos.x, WSinfo::ball->pos.y, 
				  action.target_position.x,
				  action.target_position.y, 
				  "lightred"));
    }




  if(action.action_type == AACTION_TYPE_SELFPASS){
    LOGNEW_POL(NEWBASELEVEL+log_level, << "Action: "<<idx<<") SELFPASS"
	    << " dir "<<RAD2DEG((action.target_position - WSinfo::ball->pos).arg())
	    <<" ballspeed "<<action.kick_velocity
	    << " Resulting Position: "<<action.actual_resulting_position
	    <<"  V: "<< V << " net: "<<display_value);
  }
  else if(action.action_type == AACTION_TYPE_IMMEDIATE_SELFPASS){
    LOGNEW_POL(NEWBASELEVEL+log_level, << "Action: "<<idx<<") IMMEDIATE SELFPASS"
	    << " dir "<<RAD2DEG((action.target_position - WSinfo::ball->pos).arg())
	    <<" ballspeed "<<action.kick_velocity
	    << " Resulting Position: "<<action.actual_resulting_position
	    << " advantage: "<<action.advantage
	    << " time2intercept: "<<action.time2intercept
	    << " cycles2go: "<<action.cycles2go
	    <<"  V: "<< V<< " net: "<<display_value);
  }
  else if(action.action_type == AACTION_TYPE_DRIBBLE){
    LOGNEW_POL(NEWBASELEVEL+log_level, << "Action: "<<idx<<") DRIBBLE"
	    << " dir "<<RAD2DEG((action.target_position - WSinfo::me->pos).arg())
	      <<" ballspeed "<<action.kick_velocity
	    << " Resulting Position: "<<action.actual_resulting_position
	    <<"  V: "<< V<< " net: "<<display_value);
  }
  else if(action.action_type == AACTION_TYPE_DRIBBLE_QUICKLY){
    LOGNEW_POL(NEWBASELEVEL+log_level, << "Action: "<<idx<<") DRIBBLE QUICKLY "
	    << " dir "<<RAD2DEG((action.target_position - WSinfo::me->pos).arg())
	      <<" ballspeed "<<action.kick_velocity
	    << " Resulting Position: "<<action.actual_resulting_position
	    <<"  V: "<< V<< " net: "<<display_value);
  }
  else if(action.action_type == AACTION_TYPE_WAITANDSEE){
    LOGNEW_POL(NEWBASELEVEL+log_level, << "Action: "<<idx<<") WAIT_AND_SEE " 
	    << " Resulting Position: "<<action.actual_resulting_position
	    <<"  V: "<< V<< " net: "<<display_value);
  }
  else if(action.action_type == AACTION_TYPE_TURN_AND_DASH){
    LOGNEW_POL(NEWBASELEVEL+log_level, << "Action: "<<idx<<") TURN_AND_DASH " 
	    << " Resulting Position: "<<action.actual_resulting_position
	    <<"  V: "<< V<< " net: "<<display_value);
  }
  else if(action.action_type == AACTION_TYPE_PASS){
    LOGNEW_POL(NEWBASELEVEL+log_level, << "Action: "<<idx<<") PASS " 
	    <<action.action_type<<" Receiver: "
	       //<<current_state.my_team[action.target_player].number
	       <<action.targetplayer_number
	       //<<" age "
	      //<<mdpInfo::teammate_age(current_state.my_team[action.target_player].number)
	    << " Target: "<<action.target_position
	      << " speed: "<<action.kick_velocity
	      << " kickdir: "<<RAD2DEG(action.kick_dir)
	    //<<action.target_position.x<<" "<<action.target_position.y
	    << " Resulting Position: "<<action.actual_resulting_position
	    <<"  V: "<< V<< " net: "<<display_value);
  }
  else if(action.action_type == AACTION_TYPE_LAUFPASS){
    LOGNEW_POL(NEWBASELEVEL+log_level, << "Action: "<<idx<<") LAUFPASS " 
	    <<action.action_type<<" Receiver: "
	       //<<current_state.my_team[action.target_player].number
	       <<action.targetplayer_number
	      //<<" age "
	      //<<mdpInfo::teammate_age(current_state.my_team[action.target_player].number)
	      <<" RISKY PASS "<<action.risky_pass
	    << " Direction: "<<RAD2DEG(action.kick_dir)
	      << " speed: "<<action.kick_velocity
	      << " kickdir: "<<RAD2DEG(action.kick_dir)
	    //<<action.target_position.x<<" "<<action.target_position.y
	    << " Resulting Position: "<<action.actual_resulting_position
	    << " Advantage: "<<action.advantage
	    <<"  V: "<< V<< " net: "<<display_value);
  }
  else if(action.action_type == AACTION_TYPE_DEADLYPASS){
    LOGNEW_POL(NEWBASELEVEL+log_level, << "Action: "<<idx<<") DEADLY PASS " 
	    <<action.action_type<<" Receiver: "
	    <<current_state.my_team[action.target_player].number
	      //<<" age "
	      //<<mdpInfo::teammate_age(current_state.my_team[action.target_player].number)
	    << " Target: "<<action.target_position
	    //<<action.target_position.x<<" "<<action.target_position.y
	    << " Resulting Position: "<<action.actual_resulting_position
	    <<"  V: "<< V<< " net: "<<display_value);
  }
  else{
    LOGNEW_ERR(BASELEVEL, <<"Strange Action type "<<action.action_type);
    LOGNEW_POL(NEWBASELEVEL+log_level, << "Action: "<<idx<<") Type "<<action.action_type);
  }

   LOGNEW_POL(NEWBASELEVEL+log_level,<<" res pos : "<<action.actual_resulting_position<<" eval: "
	     <<Tools::evaluate_pos(action.actual_resulting_position)
	     <<" potential pos "<<action.potential_position<<" eval: "
	      <<Tools::evaluate_pos(action.potential_position));
}




float NeuroWball::do_evaluation(const AState current_state,const AAction action){
  //do the 2 steps: 1. compute successor state:
  //                2. evaluate successor state:

  float V = 0;
  AState successor_state;


  if(evaluation_mode == 1){
    float Q= Planning::evaluate_byQnn(current_state, action);
    V= 52.5 * (1.-Q) + 52.5;  // make it comparable to original evaluation
  }
  else if(evaluation_mode == 2){
    AbstractMDP::target_model(current_state, action,successor_state);
    V = Planning::evaluate_byJnn(successor_state);
  }
  else if(evaluation_mode == 3){
    AbstractMDP::model(current_state, action,successor_state);
    V = Planning::evaluate_byJnn(successor_state,action.action_type);
  }
  /* D
  else{
    AbstractMDP::model(current_state, action,successor_state);
    LOG_POL(BASELEVEL+1,"Evaluate by Planning");
    V = Planning::evaluate(successor_state,action.action_type);
  }
  */
  return V;
}



float NeuroWball::select_best_aaction(const AState current_state,
				   const AAction *action_set,
				   int action_set_size,
				   AAction &best_action){
  long ms_time= MYGETTIME;

  /* search for Abstract action with best evaluation */
  float Vbest = 2.0;
  int action_idx_best=-1;
  float V = 1.0;
  //  float pass_success_probability;
  float display_value = 0;
  char color[10];
  //float action_success_threshold;
  
  //1. check all actions in set:
  for(int a=0;a<action_set_size;a++){
    V = do_evaluation(current_state, action_set[a]);
    //    DBLOG_POL(0,"action "<<a<<" V: "<<V);
   if (evaluation_mode == 3)
     display_value= V/100.;

    if (display_value < 0.2)
      sprintf(color,"000000");
    else if (display_value < 0.4)
      sprintf(color,"007F00");
    else if (display_value < 0.6)
      sprintf(color,"00FF00");
    else if (display_value < 0.8)
      sprintf(color,"7FFF00");
    else
      sprintf(color,"FFFF00");

    if (action_set[a].action_type != AACTION_TYPE_WAITANDSEE) {
      LOGNEW_DRAW(NEWBASELEVEL+2, L2D(WSinfo::ball->pos.x, WSinfo::ball->pos.y, 
			       //action_set[a].actual_resulting_position.x,
			       //action_set[a].actual_resulting_position.y, 
			       action_set[a].target_position.x,
			       action_set[a].target_position.y, 
			       color));
    }

    if (action_set[a].action_type != AACTION_TYPE_WAITANDSEE
       && action_set[a].risky_pass == true) {
      LOGNEW_DRAW(NEWBASELEVEL+2, L2D(WSinfo::ball->pos.x, WSinfo::ball->pos.y, 
				  //action_set[a].actual_resulting_position.x,
				  //action_set[a].actual_resulting_position.y, 
				  action_set[a].target_position.x,
				  action_set[a].target_position.y, 
				  "red"));
    }


#if 1
    print_action_data(current_state,action_set[a],a,V,display_value);
#endif

    //4. check if evaluation is actually better than current best
    //  could be simply if(V>Vbest) if value function is perfect: 
    if (is_better(V, Vbest, current_state, action_set, a, action_idx_best)) { // ridi04: reactivated. CHECK
    //if ( V < Vbest ) {
      Vbest = V;
      action_idx_best = a;
    }
  }

  if(action_idx_best <0){ // no action found!
    LOGNEW_POL(NEWBASELEVEL+1,<<"No PASS found");
    return -1;
  }

  best_action = action_set[action_idx_best];
  best_action.V=Vbest;

  LOGNEW_POL(NEWBASELEVEL+1, << "Select PASS NO: " <<action_idx_best<<" V: "<<Vbest<<" as follows: ");
  print_action_data(current_state,best_action,action_idx_best,Vbest,0,1);
  //  print_action_data(current_state,action_set[action_idx_best],action_idx_best,Vbest,0);


  if(action_set[action_idx_best].action_type == AACTION_TYPE_LAUFPASS){
    bool changed = false;
    Vector resultingpos = action_set[action_idx_best].actual_resulting_position;
    Vector receiverpos = current_state.my_team[action_set[action_idx_best].target_player].pos;
    Angle dir = ( receiverpos-current_state.ball.pos).arg();
    double arrival_speed = 1.5; // play sharp
    float speed = Planning::compute_pass_speed_with_arrival_vel(current_state.ball.pos,receiverpos,receiverpos, 
								2.5, arrival_speed);
    bool risky_pass= action_set[action_idx_best].risky_pass;
    AAction candidate_action;
    Vector opgoal = Vector(FIELD_BORDER_X,0.0);
    double minadvantage= 10.0; // a real laufpass must be closer 2 goal by this at least
    LOGNEW_POL(NEWBASELEVEL+1,<<"Checking DIRECT LAUFPASS in dir "<<RAD2DEG(dir)<<" speed "<<speed);
    if (Planning::check_action_laufpass(current_state,candidate_action,
					speed,dir,
					risky_pass)){
      if(candidate_action.actual_resulting_position.distance(opgoal) < minadvantage + resultingpos.distance(opgoal)){
	LOGNEW_POL(NEWBASELEVEL+1,
		   <<"Checking DIRECT Laufpass with ORIGINAL SPEED . risky pass: "<<risky_pass<<
		  " SUCCESS:  CHANGE TO receiverpos: "<<receiverpos);
	best_action = candidate_action;
	best_action.action_type = AACTION_TYPE_PASS; // do this to enforce target tracking
	changed = true;
      }
      else{
	LOGNEW_POL(NEWBASELEVEL+1,<<"Checking DIRECT Laufpass. Could change, but loose too much");
      }
    } // check with original speed
    if (changed == false){
      LOGNEW_POL(NEWBASELEVEL+1,<<"Checking DIRECT LAUFPASS in dir "<<RAD2DEG(dir)<<" speed "<<speed);
    }

    if (changed == false && Planning::check_action_laufpass(current_state,candidate_action,
					     action_set[action_idx_best].kick_velocity,dir,
					     risky_pass)){
      if(candidate_action.actual_resulting_position.distance(opgoal) < minadvantage + resultingpos.distance(opgoal)){
	LOGNEW_POL(NEWBASELEVEL+1,<<"Checking DIRECT Laufpass with ORIGINAL SPEED . risky pass: "<<risky_pass<<
		  " SUCCESS:  CHANGE TO receiverpos: "<<receiverpos);
	best_action = candidate_action;
	best_action.action_type = AACTION_TYPE_PASS; // do this to enforce target tracking
	changed = true;
      }
      else{
	LOGNEW_POL(NEWBASELEVEL+1,<<"Checking DIRECT Laufpass. Could change, but loose too much");
      }
    }
    if(changed== false){
      LOGNEW_POL(NEWBASELEVEL+1,<<"Checking DIRECT Laufpass instead. FAILURE. DO NOT CHANGE. risky pass: "
		<< risky_pass);
      refine_laufpass(current_state,best_action,Vbest);
    }

    LOGNEW_POL(NEWBASELEVEL+1, << "FINAL SELECTION: " <<action_idx_best<<" V: "<<Vbest<<" as follows: ");
    print_action_data(current_state,best_action,action_idx_best,Vbest,0,1);
  }


  if(action_set[action_idx_best].action_type != AACTION_TYPE_WAITANDSEE){
    LOGNEW_DRAW(NEWBASELEVEL+2, L2D(WSinfo::ball->pos.x, WSinfo::ball->pos.y, 
			      action_set[action_idx_best].actual_resulting_position.x,
			      action_set[action_idx_best].actual_resulting_position.y, 
			      "blue"));
  }
    
  if(action_set[action_idx_best].action_type != AACTION_TYPE_WAITANDSEE
     && action_set[action_idx_best].risky_pass == true){
    LOGNEW_DRAW(NEWBASELEVEL+2, L2D(WSinfo::ball->pos.x, WSinfo::ball->pos.y, 
			      action_set[action_idx_best].actual_resulting_position.x,
			      action_set[action_idx_best].actual_resulting_position.y, 
			      "red"));
  }

  
  ms_time= MYGETTIME - ms_time;
  DBLOG_POL(1,<< "WBALL03: Action selection  needed " << ms_time  << " ms");
  if (ms_time > (40*ServerOptions::slow_down_factor)) {
    LOGNEW_ERR(0,<< "WBALL03: Action selection needed " << ms_time  << " ms");
  }
  return Vbest;
}



void NeuroWball::generate_safe_passes(const AState &state, AAction *actions, 
				   int & num_actions){
  AAction candidate_action;
  double inner_defense_x = 16.;
  double inner_defense_y = 32.;

  Vector my_pos = state.my_team[state.my_idx].pos;



  for(int recv_idx=0;recv_idx<11;recv_idx++){
    // dont play backpasses, if I'm not advanced enough
    if((WSinfo::me->pos.getX() < -20)
       && (state.my_team[recv_idx].pos.getX() < my_pos.getX()))
      continue;

    if(state.my_team[recv_idx].valid == false)
      continue;
    if(state.my_idx == recv_idx) // passes to myself are handled below
      continue;
    if(state.my_team[recv_idx].number == WSinfo::ws->my_goalie_number)
      //never pass to own goalie
      continue;

    //    DBLOG_POL(0,<<"generate safe passes. check receiver "<<state.my_team[recv_idx].number);


    if ((my_pos.getX() < - ServerOptions::pitch_length/2. + inner_defense_x)
       && (fabs(my_pos.getY()) <inner_defense_y/2.0)){
      // I am in my penalty area
      if( (state.my_team[recv_idx].pos.getX() < - ServerOptions::pitch_length/2.+inner_defense_x) &&
	  (fabs(state.my_team[recv_idx].pos.getY()) < inner_defense_y/2.0)){
	// receiver is within penalty area
	DBLOG_POL(2,<<"Pass receiver "<<state.my_team[recv_idx].number
		<<" also within penalty area -> DO NOT PASS");
	continue;
      }
      if ((my_pos.getY() <0)
	 && (my_pos.getY() < state.my_team[recv_idx].pos.getY())) {
      	DBLOG_POL(2,<<"Pass receiver "<<state.my_team[recv_idx].number
		<<" too far left -> DO NOT PASS");
	continue;
      }
      if ((my_pos.getY() >0)
	 && (my_pos.getY() > state.my_team[recv_idx].pos.getY())) {
      	DBLOG_POL(2,<<"Pass receiver "<<state.my_team[recv_idx].number
		<<" too far right -> DO NOT PASS");
	continue;
      }
    }
    else{ // I am not in my penalty area, but I do want to  pass through penalty area
      Vector middle_of_inner_defense;
      middle_of_inner_defense.setXY( -ServerOptions::pitch_length/2. + inner_defense_x/2., 0. );
      if(Tools::intersection(middle_of_inner_defense, inner_defense_x,
			     inner_defense_y,my_pos,
			     state.my_team[recv_idx].pos) == true){
      	DBLOG_POL(2,<<"Pass to receiver "<<state.my_team[recv_idx].number
		<<" intersects penalty area -> DO NOT PASS");
	continue;
      }
    }

    if ((state.ball.pos.getX() >5.) && (state.my_team[recv_idx].pos.getX() < 0.)){
      DBLOG_POL(2,<<"Pass to receiver "<<state.my_team[recv_idx].number
		<<" intersects midfield -> DO NOT PASS");
      continue;  // do not allow passes across the middelfieldline
    }

/*TG_OSAKA: just a question: should i not prefer to hand over 'receiver->vel'
 * in the next call instead of Vector(0.0) - this parameter is expected to be a 'relative target'*/

    if (Planning::check_action_pass(state,candidate_action,recv_idx, Vector(0,0))){
      DBLOG_POL(0,<<"Checking Pass receiver "<<state.my_team[recv_idx].number<<" SUCCESS");
      actions[num_actions++] = candidate_action; // insert action
    } else{
      DBLOG_POL(0,<<"Checking Pass receiver "<<state.my_team[recv_idx].number<<" FAILURE");
    }

    
    
  } // for all players
}


void NeuroWball::generate_risky_passes(const AState &state, AAction *actions, 
				       int & num_actions){
  AAction candidate_action;
  const double scoring_area_width = 18.;
  const double scoring_area_depth = 11.;
  const XYRectangle2d scoring_area( Vector(FIELD_BORDER_X - scoring_area_depth, -scoring_area_width*0.5),
				    Vector(FIELD_BORDER_X, scoring_area_width * 0.5)); 

  DBLOG_DRAW(0,scoring_area);

  for(int recv_idx=0;recv_idx<11;recv_idx++){
    if(state.my_team[recv_idx].valid == false)
      continue;
    if(state.my_idx == recv_idx) // passes to myself are handled below
      continue;
    if(state.my_team[recv_idx].number == WSinfo::ws->my_goalie_number)
      //never pass to own goalie
      continue;
    #if 0
    if(scoring_area.inside(state.my_team[recv_idx].pos) == false)
      continue;
#endif

    if(Tools::is_a_scoring_position(state.my_team[recv_idx].pos) == false)
       continue;

    Vector receiverpos = state.my_team[recv_idx].pos;
    Angle dir = ( receiverpos-state.ball.pos).arg();
    double arrival_speed = 1.5; // play sharp
    float speed = Planning::compute_pass_speed_with_arrival_vel(state.ball.pos,receiverpos,receiverpos, 
								2.5, arrival_speed);
    //speed = adjust_speed(state.ball.pos,dir,speed);

    if (Planning::check_action_laufpass(state,candidate_action,speed,dir,true)){
      DBLOG_POL(0,<<"Checking RISKY PASS Pass receiver "<<state.my_team[recv_idx].number<<" : SUCCESS");
      actions[num_actions++] = candidate_action; // insert action
      //DBLOG_POL(0,<<"FOR CONTROL PURPOSE. print action data");
      //print_action_data(state,candidate_action,0,0.0,0);
    } 
    else{
      DBLOG_POL(0,<<"Checking RISKY PASS Pass receiver "<<state.my_team[recv_idx].number<<" : FAILURE");
    }
  } // for all players
}

void NeuroWball::generate_passes(const AState &state, AAction *actions, 
					   int & num_actions,const int save_time){
  AAction candidate_action;
  Vector test_pass[30];
  double inner_defense_x = 16.;
  double inner_defense_y = 32.;

  for(int recv_idx=0;recv_idx<11;recv_idx++){
    if(state.my_team[recv_idx].valid == false)
      continue;
    if(state.my_idx == recv_idx) // passes to myself are handled below
      continue;
    if(state.my_team[recv_idx].number == WSinfo::ws->my_goalie_number)
      //never pass to own goalie
      continue;

    if((WSinfo::me->pos.getX() < - ServerOptions::pitch_length/2. + inner_defense_x)
       && (fabs(WSinfo::me->pos.getY()) <inner_defense_y/2.0)){
      // I am in my penalty area
      if( (state.my_team[recv_idx].pos.getX() < - ServerOptions::pitch_length/2.+inner_defense_x) &&
	  (fabs(state.my_team[recv_idx].pos.getY()) < inner_defense_y/2.0)){
	// receiver is within penalty area
	DBLOG_POL(BASELEVEL,<<"Pass receiver "<<state.my_team[recv_idx].number
		<<" also within penalty area -> DO NOT PASS");
	continue;
      }
      if((WSinfo::me->pos.getY() <0)
	 && (WSinfo::me->pos.getY() < state.my_team[recv_idx].pos.getY())) {
      	DBLOG_POL(BASELEVEL,<<"Pass receiver "<<state.my_team[recv_idx].number
		<<" too far left -> DO NOT PASS");
	continue;
      }
      if((WSinfo::me->pos.getY() >0)
	 && (WSinfo::me->pos.getY() > state.my_team[recv_idx].pos.getY())) {
      	DBLOG_POL(BASELEVEL,<<"Pass receiver "<<state.my_team[recv_idx].number
		<<" too far right -> DO NOT PASS");
	continue;
      }
    }
    else{ // I am not in my penalty area, but I do want to  pass through penalty area
      Vector middle_of_inner_defense;
      middle_of_inner_defense.setXY( -ServerOptions::pitch_length/2. + inner_defense_x/2., 0. );
      if(Tools::intersection(middle_of_inner_defense, inner_defense_x,
			     inner_defense_y,WSinfo::me->pos,
			     state.my_team[recv_idx].pos) == true){
      	DBLOG_POL(BASELEVEL,<<"Pass to receiver "<<state.my_team[recv_idx].number
		<<" intersects penalty area -> DO NOT PASS");
	continue;
      }
    }

    if((WSinfo::ball->pos.getX() >5.) && (state.my_team[recv_idx].pos.getX() < 0.))
      continue;  // do not allow passes across the middelfieldline

    // passes to other players 
    // the order determines the priority
    bool forward_pass = (state.my_team[recv_idx].pos.getX() >
			 state.my_team[state.my_idx].pos.getX());
    int i = 0;
    test_pass[i++] = Pass_Direct;

    if(save_time == 0){
      if(state.my_team[recv_idx].pos.getX() <0){ // receiver is in own half - be cautious
	test_pass[i++] = Pass_Forward;
	test_pass[i++] = Pass_LeftForward;
	test_pass[i++] = Pass_RightForward;
	if (forward_pass) test_pass[i++] = Pass_Left;
	if (forward_pass) test_pass[i++] = Pass_Right;
	test_pass[i++] = 1.5 * Pass_Forward;
	if (forward_pass) test_pass[i++] = 1.5 * Pass_LeftForward;
	if (forward_pass) test_pass[i++] = 1.5 * Pass_RightForward;
	if (forward_pass) test_pass[i++] = 1.5 * Pass_Left;
	if (forward_pass) test_pass[i++] = 1.5 * Pass_Right;
      }
      else{
	if(state.my_team[recv_idx].pos.getY() <0){ // receiver is right from goal
	  test_pass[i++] = Pass_Forward;
	  test_pass[i++] = Pass_LeftForward;
	  test_pass[i++] = Pass_RightForward;
	  if (forward_pass) test_pass[i++] = Pass_Left;
	  if (forward_pass) test_pass[i++] = Pass_Right;
	  test_pass[i++] = 1.5 * Pass_Forward;
	  if (forward_pass) test_pass[i++] = 1.5 * Pass_LeftForward;
	  if (forward_pass) test_pass[i++] = 1.5 * Pass_RightForward;
	  if (forward_pass) test_pass[i++] = 1.5 * Pass_Left;
	  if (forward_pass) test_pass[i++] = 1.5 * Pass_Right;
	}
	else{
	  test_pass[i++] = Pass_Forward;
	  test_pass[i++] = Pass_RightForward;
	  test_pass[i++] = Pass_LeftForward;
	  if (forward_pass) test_pass[i++] = Pass_Right;
	  if (forward_pass) test_pass[i++] = Pass_Left;
	  test_pass[i++] = 1.5 * Pass_Forward;
	  if (forward_pass) test_pass[i++] = 1.5 * Pass_RightForward;
	  if (forward_pass) test_pass[i++] = 1.5 * Pass_LeftForward;
	  if (forward_pass) test_pass[i++] = 1.5 * Pass_Right;
	  if (forward_pass) test_pass[i++] = 1.5 * Pass_Left;
	}
      }
    }
    // test_pass[i++] = Pass_Backward;
    int number_passpos = i;

    for(int i=0;i<number_passpos;i++){
      // DBLOG_POL(0,<<"in generate pass ALT ");
      if(Planning::check_action_pass(state,candidate_action,recv_idx,
				     test_pass[i])){
	actions[num_actions++] = candidate_action; // insert action
	break;  // if one found, stop testing  
      }
    }

    // check for one forward pass. In contrast to a deadly pass, from there planning continues
    int forward_passes = 0;
    test_pass[forward_passes++] = Pass_Forward;
    test_pass[forward_passes++] = Pass_LeftForward;
    test_pass[forward_passes++] = Pass_RightForward;

    for(int i=0;i<forward_passes;i++){
      //      DBLOG_POL(0,<<"in generate pass ALT  Forward");

      if(Planning::check_action_pass(state,candidate_action,recv_idx,
				     test_pass[i])){
	actions[num_actions++] = candidate_action; // insert action
	break;  // if one found, stop testing  
      }
    }
  }
}


double NeuroWball::adjust_speed(const Vector ballpos, const double dir, const double speed){
  Vector ip;
  double suggested_speed;
#define DIST_FROM_BORDER 5.

#if 0
  if(RIGHT_PENALTY_AREA.inside(WSinfo::me->pos) == false){
    ip = Tools::ip_with_right_penaltyarea(ballpos, dir);
    if(ip.x != 0 || ip.y !=0){ // ip is 0 if no interception w. penalty area is found
      DBLOG_DRAW(0,C2D(ip.x,ip.y,0.3, "red"));
      suggested_speed= Tools::ballspeed_for_dist((ballpos - ip).norm() - DIST_FROM_BORDER);
      if(suggested_speed < speed){
	//    DBLOG_POL(1,"adapted speed "<< suggested_speed<<" tested speed "<<speed);
	return suggested_speed;
      }
    }
  }  
#endif

  ip = Tools::ip_with_fieldborder(ballpos, dir);
  //DBLOG_DRAW(0,C2D(ip.x,ip.y,0.3, "red"));
  suggested_speed= Tools::ballspeed_for_dist((ballpos - ip).norm() - DIST_FROM_BORDER);
  if(suggested_speed < speed){
    //    DBLOG_POL(1,"adapted speed "<< suggested_speed<<" tested speed "<<speed);
    return suggested_speed;
  }
  return speed;
}


void NeuroWball::generate_laufpasses(const AState &state, AAction *actions, 
				  int &num_actions, const int save_time){

  long ms_time= MYGETTIME;

  AObject me = state.my_team[state.my_idx];

  AAction candidate_action;
  float dir;
  float min_angle = -90;
  float max_angle = 90;
  int min_advantage_level = 2;  // avoid unsafe laufpasses: ridi: confirmed; otherwise there are too many losses



  Vector mypos = me.pos;
  if(mypos.getX() > 40){
    if(mypos.getY() < -3){
      max_angle = 138;
    }
    if(mypos.getY() > 3){
      min_angle = -135;
    }
  }


  if (DeltaPositioning::get_role(me.number) == 0) {// I'm a defender: laufpasses go forward and nowhere else
    if(mypos.getY() < -20){
      min_angle = -30;
      max_angle = 30;
    }
    else  if(mypos.getY() < 0){
      min_angle = 0;
      max_angle = 30;
    }
    else  if(mypos.getY() < 20){
      min_angle = -30;
      max_angle = 0;
    }
    else{
      min_angle = -30;
      max_angle = 30;
    }
  }
  
  DBLOG_POL(0,<<"Checking LAUFPASSES, save_time:  "<<save_time);

  // normal mode
  for(float speed=2.0;speed<2.6;speed+=.5){
    for(float angle=min_angle;angle<max_angle;angle+=5){
      if(save_time == 1){
	if(angle == min_angle && speed == 2.0)
	  angle += 5; // have a different starting point for speed = 2.0 and speed = 2.5
	else if(angle > min_angle) // 
	  angle += 5;

      }
      dir = angle/180. *PI;
      // test for regular laufpass
      if(Planning::check_action_laufpass(state,candidate_action,
	   adjust_speed(state.ball.pos,dir,speed),dir)){
	if(candidate_action.advantage >= min_advantage_level){
	  DBLOG_POL(0,<<"Checking LAUFPASS, dir  "<<RAD2DEG(dir)
		    <<" speed "<<speed<<" : SUCCESS");
	  actions[num_actions++] = candidate_action; // insert action
	  continue;  // Laufpass found, continue with next dir and speed
	}
      }
      // regular laufpass not possible, check for risky laufpass
      if(me.pos.getX() > 36. && fabs(me.pos.getY()) <20 ){
	bool risky_pass = true;
	if(Planning::check_action_laufpass(state,candidate_action,
					   adjust_speed(state.ball.pos,dir,speed),dir,
					   risky_pass)){
	  if(candidate_action.actual_resulting_position.getX() > 39.32 &&
	     fabs(candidate_action.actual_resulting_position.getY())<10){
	    DBLOG_POL(0,<<"Checking RISKY LAUFPASS, dir  "<<RAD2DEG(dir)
		      <<" speed "<<speed<<" : SUCCESS");
	    actions[num_actions++] = candidate_action; // insert action
	  }
	}
      }

      // in this direction, no laufpass is possible
      DBLOG_POL(2,<<"Checking LAUFPASS, dir  "<<RAD2DEG(dir)
		<<" speed "<<speed<<" : FAILURE");
    }
  }

  ms_time = MYGETTIME - ms_time;
  LOGNEW_POL(0, << "checking laufpasses needed " << ms_time << "millis");

}


bool NeuroWball::is_dribblestraight_possible( const AState & state ) {
  AObject me = state.my_team[state.my_idx];
  ANGLE my_body_angle = ANGLE(me.body_angle);

  ANGLE opgoaldir = (Vector(52,0) - me.pos).ARG(); //go towards goal

  int my_role = DeltaPositioning::get_role(me.number);

  if (my_role == 0){// I'm a defender: 
    if (me.pos.getX() > -5)
      return false;
    int stamina = Stamina::get_state();
    //DBLOG_POL(0,<<"Check selfpass: check stamina state "<<staminaa);
    if((stamina == STAMINA_STATE_ECONOMY || stamina == STAMINA_STATE_RESERVE )
       && me.pos.getX() >-35){// stamina low
      DBLOG_POL(0,<<"check selfpass: I'm a defender and stamina level not so good -> do not advance");
      return false;
    }
  }

  if ((my_body_angle.diff(ANGLE(0))>20/180.*PI || me.pos.getX() > 45) &&
      (my_body_angle.diff(opgoaldir)>20/180.*PI)){
    return false;
  }

  DBLOG_POL(1,<<"me->ang "<<RAD2DEG(me.body_angle)<<" OK for Dribble");


  if (dribblestraight->is_dribble_safe(state, 0)) {
    /*
    if (my_blackboard.pass_or_dribble_intention.valid_since() < WSinfo::ws->time) {
      // check area before me
      WSpset pset = WSinfo::valid_opponents;
      Vector endofregion;
      endofregion.init_polar(4, my_body_angle);
      endofregion += WSinfo::me->pos;
      Quadrangle2d check_area = Quadrangle2d(me.pos, endofregion, 2);
      DBLOG_DRAW(0, check_area );
      pset.keep_players_in(check_area);
      if (pset.num > 0) {
	DBLOG_POL(0,<<"Player directly before me and pass intention is set -> do not dribble");
	return false; // go on in any case
      }
    }
    */
    return true;
  }
  return false;
}


void NeuroWball::generate_selfpasses( const AState & state, AAction *actions, int &num_actions ) {
  AObject me = state.my_team[state.my_idx];
  ANGLE my_body_angle = ANGLE(me.body_angle);

  /*
  AAction candidate_action;
  AbstractMDP::set_action(candidate_action, AACTION_TYPE_WAITANDSEE, 
			  me.number, me.number, me.pos, 0.0, 0.0);    	
  actions[num_actions++] = candidate_action;
  */

  bool is_dribble_ok = is_dribblestraight_possible(state);

  if (is_dribble_ok && me.pos.getX() + 3 < WSinfo::his_team_pos_of_offside_line()) {
    PlayerSet pset = WSinfo::valid_opponents;
    Vector endofregion;
    const double scanrange = 10;
    endofregion.init_polar(scanrange, my_body_angle);
    endofregion += me.pos;
    Quadrangle2d check_area = Quadrangle2d(me.pos, endofregion,scanrange/2., scanrange);
    //    DBLOG_DRAW(0, check_area );
    pset.keep_players_in(check_area);
    if(pset.num > 0){
      if(pset.num > 1 || pset[0] != WSinfo::his_goalie){
	DBLOG_POL(0,<<"Test advance: Area before me is crowded -> rather dribble ");
	Vector target;
	target.init_polar(2.0, my_body_angle); 

	AAction candidate_action;
	AbstractMDP::set_action(candidate_action, AACTION_TYPE_DRIBBLE, 
				me.number, me.number, target, 0.0, 0.0);    	
	actions[num_actions++] = candidate_action;
	return;
      }
    }
  }


#define NUM_DIRS 20

  const ANGLE opgoaldir = (Vector(47.,0) - me.pos).ARG(); //go towards goal
  ANGLE targetdir;
  ANGLE testdir[NUM_DIRS];
  int num_dirs = 0;
  double speed;
  int steps, op_number;
  Vector ipos, op_pos;

  testdir[num_dirs ++] = ANGLE(0); // go straight
  testdir[num_dirs ++] = ANGLE(45/180.*PI); 
  testdir[num_dirs ++] = ANGLE(-45/180.*PI);
  testdir[num_dirs ++] = ANGLE(30/180.*PI); 
  testdir[num_dirs ++] = ANGLE(-30/180.*PI);
  testdir[num_dirs ++] = my_body_angle;
  testdir[num_dirs ++] = opgoaldir; // close to goal: default: go towards goal


  int mode = 0;  //default: 0: evaluation by position; 1: 1vs1-Situation

  // chck to switch to 1vs1 mode:

  bool goalie_close = false;
  if (WSinfo::his_goalie){
    if (me.pos.distance(WSinfo::his_goalie->pos) < 6.0)
      goalie_close = true;
  }

  if ((me.pos.getX()  > FIELD_BORDER_X - PENALTY_AREA_LENGTH -5) &&
      (goalie_close == true)){ 
    Quadrangle2d check_area = Quadrangle2d(me.pos, Vector(52.0,0), 5., 14);
    //DBLOG_DRAW(0, check_area );
    PlayerSet pset = WSinfo::valid_opponents;
    pset.keep_players_in(check_area);
    if(pset.num == 0){
      mode = 1;
      DBLOG_POL(0,"Get Best Selfpass:Nobody before me, switch to 1vs1 mode");
    }
    else if(pset.num == 1 && pset[0] == WSinfo::his_goalie){
      mode = 1;
      DBLOG_POL(0,"Get Best Selfpass:Only goalie before me, switch to 1vs1 mode");
    }
  }


  if(mode ==1){ 
    //DBLOG_DRAW(0,C2D(50,30,.4,"red") );
  }

  if(mode ==1){
    testdir[num_dirs ++] = ANGLE(90/180.*PI); 
    testdir[num_dirs ++] = ANGLE(-90/180.*PI);
    testdir[num_dirs ++] = ANGLE(120/180.*PI); 
    testdir[num_dirs ++] = ANGLE(-120/180.*PI);
    testdir[num_dirs ++] = ANGLE(179/180.*PI);
    testdir[num_dirs ++] = ANGLE(150/180.*PI); 
    testdir[num_dirs ++] = ANGLE(-150/180.*PI);
  }


  for(int i=0; i<num_dirs; i++){
    
    targetdir = testdir[i];
    if (mode ==0 && selfpass_dir_ok(state, targetdir) == false) // in mode 0, check selfpass directions
      continue;
    //    DBLOG_POL(0,<<"selfpass DIRECTION ok : "<<RAD2DEG(targetdir.get_value()));

    if (check_selfpass(state, targetdir, speed, ipos, steps, op_pos, op_number) == false)
      continue;
    
    AAction candidate_action;
    AbstractMDP::set_action(candidate_action, AACTION_TYPE_SELFPASS, 
			    me.number, me.number, ipos, speed, targetdir.get_value(), op_pos, Vector(0,0), steps);
    actions[num_actions++] = candidate_action;
  }

  if (is_dribble_ok && num_dirs == 0) {
    DBLOG_POL(0,<<"Test advance: No chance for selfpasses, but dribble is possible ");
    Vector target;
    target.init_polar(2.0,my_body_angle); 
    AAction candidate_action;
    AbstractMDP::set_action(candidate_action, AACTION_TYPE_DRIBBLE, 
			    me.number, me.number, target, 0.0, 0.0);    	
    actions[num_actions++] = candidate_action;
  }

  /*
  if ( num_actions == 0 ) {
    LOG_DEB(0, << "Emergency selfpass activated");
    WSpset pset = WSinfo::valid_opponents;
    PPlayer p_tmp = pset.closest_player_to_point(me.pos);
    targetdir = (me.pos - p_tmp->pos).ARG();
    LOG_DEB(0, << "op_pos is " << op_pos);
    if ( (p_tmp->pos - me.pos).norm() < 0.8 ) {
      LOG_DEB(0, << "Emergency selfpass away from goal allowed!");
      Vector tmp;
      AAction candidate_action;

      tmp.init_polar(4.0, targetdir);
      if (FIELD_AREA.inside(me.pos + tmp)) {
	tmp.init_polar(20.0, targetdir);
	AbstractMDP::set_action(candidate_action, AACTION_TYPE_SELFPASS, 
				me.number, me.number, me.pos + tmp, 0.5, targetdir.get_value(), 
				me.pos + tmp, Vector(0,0), 2);
	actions[num_actions++] = candidate_action; 
      }

      tmp.init_polar(4.0, targetdir.get_value()+M_PI/2.0);
      if (FIELD_AREA.inside(me.pos + tmp)) {
	tmp.init_polar(20.0, targetdir.get_value()+M_PI/2.0);
	AbstractMDP::set_action(candidate_action, AACTION_TYPE_SELFPASS, 
				me.number, me.number, me.pos + tmp, 0.5, targetdir.get_value()+PI/2.0, 
				me.pos + tmp, Vector(0,0), 2);
	actions[num_actions++] = candidate_action; 
      }

      tmp.init_polar(4.0, targetdir.get_value()-M_PI/2.0);
      if (FIELD_AREA.inside(me.pos + tmp)) {
	tmp.init_polar(20.0, targetdir.get_value()-M_PI/2.0);
	AbstractMDP::set_action(candidate_action, AACTION_TYPE_SELFPASS, 
				me.number, me.number, me.pos + tmp, 0.5, targetdir.get_value()-PI/2.0, 
				me.pos + tmp, Vector(0,0), 2);
	actions[num_actions++] = candidate_action; 
      }

    }
  }
  */

}


bool NeuroWball::selfpass_dir_ok( const AState & state, const ANGLE dir ) {
  AObject me = state.my_team[state.my_idx];

  const ANGLE opgoaldir = (Vector(47.,0) - me.pos).ARG(); //go towards goal

#if 0
  if (me.pos.y < -(FIELD_BORDER_Y - 10.) 
      && dir.get_value_mPI_pPI()<0.)
    return false;

  if (me.pos.y > (FIELD_BORDER_Y - 10.) 
      && dir.get_value_mPI_pPI()>0.)
    return false;

#endif

  if(me.pos.getX() < 45 && dir.diff(ANGLE(0))<25/180.*PI )
    return true;
  if (dir.diff(opgoaldir)<20/180.*PI)
    return true;
  return false;
}

bool NeuroWball::check_selfpass( const AState & state, const ANGLE targetdir, double &ballspeed, Vector &target, int &steps,
				 Vector &op_pos, int &op_num ) {

  AObject me = state.my_team[state.my_idx];
  ANGLE my_body_angle = ANGLE(me.body_angle);

  //  double op_time2react = 1.0;
  double op_time2react = 0.0;
  /* op_time2react is the time that is assumed the opponents need to react. 0 is worst case, that they
     are maximally quick. This means that the game is less aggressive and much less effective
     1 assumes
     that ops. need 1 cycle to react. This is already pretty aggressive and (nearly) safe.
     Maybe this could be improved by selecting op_time2react depending on the position on the field
     or by using dribbling in hard cases.
  */

  int my_role = DeltaPositioning::get_role(me.number);

  if (my_role == 0) {// I'm a defender: 
    if(me.pos.getX() > -5)
      return false;

    int stamina = Stamina::get_state();
    //DBLOG_POL(0,<<"Check selfpass: check stamina state "<<staminaa);
    if((stamina == STAMINA_STATE_ECONOMY || stamina == STAMINA_STATE_RESERVE )
       && me.pos.getX() >-35){// stamina low
      DBLOG_POL(0,<<"check selfpass: I'm a defender and stamina level not so good -> do not advance");
      return false;
    }

  }

  int max_dashes = 7; // be a little cautious anytime

  PlayerSet pset;


  if (me.pos.getX() + 3. < WSinfo::his_team_pos_of_offside_line()) {
    max_dashes = 4;
    pset= WSinfo::valid_opponents;
    Vector endofregion;
    double scanrange = 20;
    for(int i=0; i<2; i++){
      if(i==0)
	scanrange = 20;
      else	
	scanrange = 10;
      endofregion.init_polar(scanrange, targetdir);
      endofregion += me.pos;
      Quadrangle2d check_area = Quadrangle2d(me.pos, endofregion, scanrange/2., scanrange);
      //      DBLOG_DRAW(1, check_area );
      pset.keep_players_in(check_area);
      if(pset.num >0){
	if(pset.num == 1 && pset[0] == WSinfo::his_goalie){
	  DBLOG_POL(3,<<"Only goalie before me -> do not reduce steps");
	  max_dashes = 100;
	}
	else{
	  if(i==0)
	    max_dashes = 3;
	  else
	    max_dashes = 2;
	}
      }
    }
    DBLOG_POL(3,"Selfpass targetdir "<<RAD2DEG(targetdir.get_value())<<" I'm behind offside line;reducing max_dashes to "<<max_dashes);
  }

  // check if I should risk something
  if (me.pos.getX()  > 0 && targetdir.diff(ANGLE(0)) <90./180.*PI  ){
    double scanrange = 80;
    Vector endofregion;
    endofregion.init_polar(scanrange, targetdir);
    endofregion += me.pos;
    Quadrangle2d check_area = Quadrangle2d(me.pos, endofregion, 5., scanrange);
    //DBLOG_DRAW(0, check_area );
    pset.keep_players_in(check_area);
    if(pset.num == 1){
      if(pset[0] == WSinfo::his_goalie){
	op_time2react = 1.0;
	//op_time2react = 2.0;
	DBLOG_POL(3,"Selfpass targetdir "<<RAD2DEG(targetdir.get_value())
		  <<" Only goalie is before me -> risk something ");
      }
    }
    else if(pset.num == 0){
      if(WSinfo::his_goalie){// if goalie pointer is defined
	if(WSinfo::his_goalie->age > 2){
	  DBLOG_POL(3,"Selfpass targetdir "<<RAD2DEG(targetdir.get_value())
		    <<" goalie age > 2 -> reduce max_dashes ");
	  max_dashes = 4;
	}
      }
      else{
	DBLOG_POL(3,"Selfpass targetdir "<<RAD2DEG(targetdir.get_value())
		  <<" goalie pointer not defined/ not known) -> reduce max_dashes ");
	max_dashes = 4;
      }
      op_time2react = 1.0;
      DBLOG_POL(3,"Selfpass targetdir "<<RAD2DEG(targetdir.get_value())
		<<" Nobody is before me -> risk something ");
    }
  }

  if (me.pos.getX() + 3. > WSinfo::his_team_pos_of_offside_line()) {
    DBLOG_POL(3,"Selfpass targetdir "<<RAD2DEG(targetdir.get_value())
	      <<" I'm at offside line -> risk something ");
    op_time2react = 1.0;
  }

  if (my_body_angle.diff(targetdir) >90./180.*PI ||
     WSmemory::last_seen_in_dir(targetdir) >1){
    max_dashes = 1;
    DBLOG_POL(3,"Selfpass targetdir "<< RAD2DEG(targetdir.get_value())
	      << " Bad body dir or see too old "<<WSmemory::last_seen_in_dir(targetdir)
	      <<" reduce number of dashes to "<<max_dashes);
  }


  op_time2react = 0.0; // ridi: reset, because of self determination in selfpass
  bool result = selfpass->is_selfpass_safe(state, targetdir, ballspeed, target, steps, op_pos, op_num,
					   max_dashes,op_time2react);
  pset= WSinfo::valid_teammates_without_me;
  pset.keep_players_in_circle(target,2);
  if (pset.num >0){
    if (result == true) {
      DBLOG_POL(3,"Selfpass ok, but target too close to teammate");
    }
    return false;
  }

  if(my_role == 0){// I'm a defender: 
    if(target.getX() >-10)
      return false;
  }

  if (result == false)
    return false;
  return true;
}


int NeuroWball::generate_action_set(const AState &state, AAction *actions){
  int num_actions = 0;

  Planning::unmark_all_pass_receiver(); // reset pass receiver array
  Planning::mark_pass_receiver(state.my_idx); // avoid to get ball back!
  
  generate_safe_passes(state, actions, num_actions);
  generate_risky_passes(state, actions, num_actions);
  /*
  int save_time = 0;
  long ms_time= Tools::get_current_ms_time();
#define CRITICAL_TIME 45
#define CRITICAL_TIME2 60
  if(  ms_time - WSinfo::ws->ms_time_of_sb
     > (CRITICAL_TIME2 * ServerOptions::slow_down_factor)){// do not generate laufpasses then!
    LOGNEW_POL(0,"Beforee CHECK Laufpasses, critical limit reached. Do not check "
	       <<ms_time - WSinfo::ws->ms_time_of_sb);
    return num_actions;
  } 

  if(  ms_time - WSinfo::ws->ms_time_of_sb
     > (CRITICAL_TIME * ServerOptions::slow_down_factor )){
    save_time = 1;
  }
  */

  //  generate_laufpasses(state, actions, num_actions, save_time);
  //  generate_laufpasses2(state,actions, num_actions,0);
  generate_laufpasses2(state,actions, num_actions,0);
  generate_penaltyarea_passes(actions, num_actions);

//  generate_laufpasses(state, actions, num_actions, 0);
  return num_actions;
}


/***********************************************+******/

/* Osaka 05 */

/***********************************************+******/

bool NeuroWball::evaluate_passes(AAction & best_aaction_P, AState &current_state){
  AAction action_set[MAX_AACTIONS];
  AAction best_aaction;
  int action_set_size;

  LOG_POL(0, << "Neuro Wball: Evaluate Passes DEACTIVATED. Use PassSelection instead ");
  return false;

  action_set_size = generate_action_set(current_state, action_set);

  LOGNEW_POL(0, << "Action set generation needed " << MYGETTIME- ms_time << " millis");

  // 3. select best action

  if(select_best_aaction2(current_state,action_set,action_set_size,best_aaction) == false)
    return false;

  LOGNEW_POL(0, << "select best needed " <<MYGETTIME- ms_time << " millis");
  best_aaction_P = best_aaction;
  return true;
}

Vector NeuroWball::compute_potential_pos(const AState current_state,const AAction action){
  AState successor_state;

  if(action.action_type == AACTION_TYPE_LAUFPASS){
    // for laufpasses, potential position is resulting position
    return action.actual_resulting_position;
  }
  AbstractMDP::model(current_state, action,successor_state); // compute state after action is executed
  return Planning2::compute_potential_pos(successor_state);   // compute potential of that state
}


bool NeuroWball::is_better2(const AState &state, const AAction *actions, const int idx, const int best_idx){
  if(best_idx <0)
    return true;

  double evaluation_delta;

  if(Tools::compare_positions(actions[idx].potential_position,actions[best_idx].potential_position, evaluation_delta) == FIRST)
    return true;
  if(Tools::compare_positions(actions[idx].potential_position,actions[best_idx].potential_position, evaluation_delta) == SECOND)
    return false;
  // both positions are in the same 'equivalence class', now check the fine difference for decision.
  if (evaluation_delta >0)
    return true;

  return false;


  /*
  if(max_evaluation_of_candidate_action > max_evaluation_of_best_action)
    return true;
  */


  return false;
}  


bool NeuroWball::is_safe(const AAction action){
  if (action.action_type == AACTION_TYPE_PASS) {
    if(action.risky_pass == true)
      return false;
    else
      return true;
  }
  else if (action.action_type == AACTION_TYPE_LAUFPASS) {
    if(action.advantage >2)
      return true;
  }
  return false;
}

void NeuroWball::check_for_best(const AAction *actions, const int idx, int & best_safe, int & best_risky ){
  double evaluation_delta;

  if (is_safe(actions[idx])){  // check, if candidate idx improve best safe action
     if (best_safe <0){  // no best safe found yet
       best_safe = idx;
       return;
     }
     if(Tools::compare_positions(actions[idx].actual_resulting_position,
				 actions[best_safe].actual_resulting_position, evaluation_delta) == FIRST){
       best_safe = idx;
       return;
     }
     else if(Tools::compare_positions(actions[idx].actual_resulting_position,
				 actions[best_safe].actual_resulting_position, evaluation_delta) == SECOND){
       // best_safe remains best
       return;
     }
     else{ // both resulting positions are equal
       if(actions[idx].actual_resulting_position.getX() >= WSinfo::ball->pos.getX()){ // the pass goes forward
	 if (Tools::is_pos1_better(actions[idx].potential_position,actions[best_safe].potential_position) == true){
	   // compare potential positions
	   best_safe = idx;
	   return;
	 }
       }
       else{ // the pass does not go forward
	 if(evaluation_delta >0){
	   best_safe = idx;
	   return;
	 }
       }
     }  // both resulting positions are equal
     return;
   }  // is a safe pass. 

   /*********************************************************************/
   /* compare risky passes */
   /*********************************************************************/

   else{ // it's a risky pass
     if (best_risky <0){  // no risky pass found yet
       best_risky = idx;
       return;
     }
     if(Tools::compare_positions(actions[idx].actual_resulting_position,
				 actions[best_risky].actual_resulting_position, evaluation_delta) == FIRST){
       best_risky = idx;
       return;
     }
     else if(Tools::compare_positions(actions[idx].actual_resulting_position,
				 actions[best_risky].actual_resulting_position, evaluation_delta) == SECOND){
       // best_risky remains 
       return;
     }
     else{  // EQUAL risky pass -> now improve advantage
       if(actions[idx].advantage >actions[best_risky].advantage){
	 best_risky = idx;
	 return;
       }
     } // end EQUAL
   }// end risky pass
}  






void NeuroWball::check_to_improve_action(AAction &action){
  // example; might be refined by more angles, more speeds...., could compare evaluations...

  if(action.action_type == AACTION_TYPE_LAUFPASS){
    int tmp_advantage = action.advantage;
    double tmp_kickdir = action.kick_dir;
    AAction candidate_action;

    double test_speed;
    double test_angle = tmp_kickdir;

    test_speed = 2.0;
    if(Planning::check_action_laufpass2(candidate_action,test_speed,test_angle)){
      LOGNEW_POL(0,"improve action? speed "<<test_speed<<" possible "<<" eval. resultingpos "
		<<Tools::evaluate_pos(candidate_action.actual_resulting_position)
		<<"advantage: "<<candidate_action.advantage);
      if(candidate_action.advantage >= tmp_advantage || candidate_action.advantage >=2){
	LOGNEW_POL(0,"candidate action: possible with speed "<<test_speed<<" CHANGE");
	candidate_action.potential_position = candidate_action.actual_resulting_position;
	action=candidate_action;
      }
    }
  }
}

bool NeuroWball::select_best_aaction2(const AState current_state,
				      AAction *action_set,
				      int action_set_size,
				      AAction &best_action){
  long ms_time= MYGETTIME;

  /* search for Abstract action with best evaluation */
  int best_safe=-1;
  int best_risky=-1;
  int action_idx_best = -1;
  
  //1. check all actions in set:
  for(int a=0;a<action_set_size;a++){
    Vector potential_pos =compute_potential_pos(current_state, action_set[a]);

    if(Tools::is_pos1_better(potential_pos, action_set[a].actual_resulting_position)){
      action_set[a].potential_position = potential_pos;
    }
    else{
      action_set[a].potential_position = action_set[a].actual_resulting_position;
    }


    //    LOGNEW_POL(0,"potential pos: "<<action_set[a].potential_position<<" res. pos "<<action_set[a].actual_resulting_position);
#if 1
    print_action_data(current_state,action_set[a],a,0.0,0,1);
#endif
    //4. check if evaluation is actually better than current best
    check_for_best(action_set, a, best_safe, best_risky);
  } // for all actions

  LOGNEW_POL(0,"Select Best. Best safe: "<<best_safe<<" best risky "<<best_risky);

  double evaluation_delta;

  if(best_safe <0){
    action_idx_best = best_risky;
  }
  else if (best_risky <0){
    action_idx_best = best_safe;
  }
  else if (Tools::compare_positions(action_set[best_safe].actual_resulting_position, 
				    action_set[best_risky].actual_resulting_position, evaluation_delta) == FIRST){
    LOGNEW_POL(0,"Select Best. Best safe: "<<best_safe<<" is best");
    action_idx_best = best_safe;
  }
  else if (Tools::compare_positions(action_set[best_safe].actual_resulting_position, 
				    action_set[best_risky].actual_resulting_position, evaluation_delta)==EQUAL){
    LOGNEW_POL(0,"Select Best. Best safe: "<<best_safe<<" are EQUAL. Choose safe");
    action_idx_best = best_safe;
  }
  else{  // risky pass is definitely better, so choose risky pass!
    LOGNEW_POL(0,"Select Best. Best Risky: "<<best_risky<<" is best");
    action_idx_best = best_risky;
  }





  if(action_idx_best <0){ // no action found!
    LOGNEW_POL(NEWBASELEVEL+1,<<"No PASS found");
    return false;
  }

  best_action = action_set[action_idx_best];
  check_to_improve_action(best_action);

  double evaluation= Tools::evaluate_pos(action_set[action_idx_best].potential_position);

  LOGNEW_POL(NEWBASELEVEL+0, << "Select PASS NO: " <<action_idx_best<<" evaluation: "
	     <<evaluation<< "as follows: ");
  print_action_data(current_state,best_action,action_idx_best,evaluation,0,1);
  //  print_action_data(current_state,action_set[action_idx_best],action_idx_best,Vbest,0);

  ms_time= MYGETTIME - ms_time;
  DBLOG_POL(1,<< "WBALL03: Action selection  needed " << ms_time  << " ms");
  if (ms_time > (40*ServerOptions::slow_down_factor)) {
    LOGNEW_ERR(0,<< "WBALL03: Action selection needed " << ms_time  << " ms");
  }
  return true;
}


void NeuroWball::generate_laufpasses2(const AState &state, AAction *actions, 
				  int &num_actions, const int save_time){

  //  LOGNEW_POL(0, << "Entered laufpass2");
  long ms_time= MYGETTIME;
  AAction candidate_action;
  double min_angle = -90/180.*PI;
  double max_angle = 90/180.*PI;

  Vector mypos = WSinfo::me->pos;
  if(mypos.getX() > 40){
    if(mypos.getY() < -3){
      min_angle = 0/180.*PI;
    }
    if(mypos.getY() > 3){
      max_angle = 0/180.*PI;
    }
  }

  if (DeltaPositioning::get_role(WSinfo::me->number) == 0) {// I'm a defender: laufpasses go forward and nowhere else
    if(mypos.getY() < -20){
      min_angle = -30/180.*PI;
      max_angle = 30/180.*PI;
    }
    else  if(mypos.getY() < 0){
      min_angle = -30/180.*PI;
      max_angle = 0/180.*PI;
    }
    else  if(mypos.getY() < 20){
      min_angle = 0/180.*PI;
      max_angle = 30/180.*PI;
    }
    else{
      min_angle = -30/180.*PI;
      max_angle = 30/180.*PI;
    }
  }
  
  for(double angle=min_angle;angle<max_angle;angle+=10/180.*PI){
  //  for(double angle=min_angle;angle<max_angle;angle+=20/180.*PI){
    //    LOGNEW_POL(0, << "check angle "<<RAD2DEG(angle));
    double speed = ServerOptions::ball_speed_max; // maximum
    speed = adjust_speed2(WSinfo::ball->pos,angle,speed);
    if(speed < ServerOptions::ball_speed_max){
      LOGNEW_POL(0, << "Adjusted speed in dir "<<RAD2DEG(angle)<<" to "<< speed);
    }

    if(Planning::check_action_laufpass2(candidate_action,speed,angle)){
      actions[num_actions++] = candidate_action; // insert action
      angle += 30/180.*PI;
    } // succesful laufpass found
  }  // for all angles
  ms_time = MYGETTIME - ms_time;
  LOGNEW_POL(0, << "checking laufpasses needed " << ms_time << "millis");
}

void NeuroWball::generate_penaltyarea_passes(AAction *actions, int &num_actions){

  //  LOGNEW_POL(0, << "Entered penaltyarea passes");
  long ms_time= MYGETTIME;

  AAction candidate_action;
  double min_angle = -90/180.*PI;
  double max_angle = 90/180.*PI;

  Vector mypos = WSinfo::me->pos;
  if(mypos.getX() < 35 || fabs(mypos.getY()) >20)
    return;

  if(mypos.getY() < -10){
    min_angle = 0/180.*PI;
    if(mypos.getX()>45)
      max_angle=120/180.*PI;
    else
      max_angle = 100/180.*PI;
  }
  else  if(mypos.getY() < 0){
    if(mypos.getX()>45)
      min_angle=-120/180.*PI;
    else
      min_angle = -100/180.*PI;
    max_angle = 0/180.*PI;
  }
  
  for(double angle=min_angle;angle<max_angle;angle+=5/180.*PI){
    //    LOGNEW_POL(0, << "check angle "<<RAD2DEG(angle));
    double speed = ServerOptions::ball_speed_max; // maximum
    if(Planning::check_action_penaltyareapass(candidate_action,speed,angle)){
      //LOGNEW_POL(0,"Checking Pen are, dir  "<<RAD2DEG(angle)<<" speed "<<speed<<" : SUCCESS");
      actions[num_actions++] = candidate_action; // insert action
      angle += 20/180.*PI;
    } // succesful laufpass found
  }  // for all angles
  ms_time = MYGETTIME - ms_time;
  LOGNEW_POL(0, << "checking penalty area passes needed " << ms_time << "millis");
}

double NeuroWball::adjust_speed2(const Vector ballpos, const double dir, const double speed){
  Vector ip;

  ip = Tools::ip_with_fieldborder(ballpos, dir);
  //  LOGNEW_DRAW(0,C2D(ip.x,ip.y,0.3, "red"));
  if(ballpos.distance(ip) <5.)
    return 0.5;
  else if(ballpos.distance(ip) <10.)
    return 1.0;
  else if(ballpos.distance(ip) <15.)
    return 1.5;
  else if(ballpos.distance(ip) <20.)
    return 2.0;
  else if(speed > 3.0)//TG08: NeuroKick cannot kick as hard, yet !!!
    return 3.0;
  else
    return speed;
}



/*********************************************************************************************/

bool NeuroWball::select_best_aaction06(const AState current_state,
				      AAction *action_set,
				      int action_set_size,
				      AAction &best_action){
  long ms_time= MYGETTIME;

  /* search for Abstract action with best evaluation */
  int best_safe=-1;
  int best_risky=-1;
  int action_idx_best = -1;
  
  //1. check all actions in set:
  for(int a=0;a<action_set_size;a++){
    Vector potential_pos =compute_potential_pos(current_state, action_set[a]);

    if(Tools::is_pos1_better(potential_pos, action_set[a].actual_resulting_position)){
      action_set[a].potential_position = potential_pos;
    }
    else{
      action_set[a].potential_position = action_set[a].actual_resulting_position;
    }


    //    LOGNEW_POL(0,"potential pos: "<<action_set[a].potential_position<<" res. pos "<<action_set[a].actual_resulting_position);
#if 1
    print_action_data(current_state,action_set[a],a,0.0,0,1);
#endif
    //4. check if evaluation is actually better than current best
    check_for_best(action_set, a, best_safe, best_risky);
  } // for all actions

  DBLOG_POL(0,"Select Best. Best safe: "<<best_safe<<" best risky "<<best_risky);

  double evaluation_delta;

  if(best_safe <0){
    action_idx_best = best_risky;
  }
  else if (best_risky <0){
    action_idx_best = best_safe;
  }
  else if (Tools::compare_positions(action_set[best_safe].actual_resulting_position, 
				    action_set[best_risky].actual_resulting_position, evaluation_delta) == FIRST){
    DBLOG_POL(0,"Select Best. Best safe: "<<best_safe<<" is best");
    action_idx_best = best_safe;
  }
  else if (Tools::compare_positions(action_set[best_safe].actual_resulting_position, 
				    action_set[best_risky].actual_resulting_position, evaluation_delta)==EQUAL){
    DBLOG_POL(0,"Select Best. Best safe: "<<best_safe<<" are EQUAL. Choose safe");
    action_idx_best = best_safe;
  }
  else{  // risky pass is definitely better, so choose risky pass!
    DBLOG_POL(0,"Select Best. Best Risky: "<<best_risky<<" is best");
    action_idx_best = best_risky;
  }

  if(action_idx_best <0){ // no action found!
    LOGNEW_POL(NEWBASELEVEL+1,<<"No PASS found");
    return false;
  }

  best_action = action_set[action_idx_best];
  check_to_improve_action(best_action);

  double evaluation= Tools::evaluate_pos(action_set[action_idx_best].potential_position);

  DBLOG_POL(NEWBASELEVEL+0, << "Select PASS NO: " <<action_idx_best<<" evaluation: "
	     <<evaluation<< "as follows: ");
  print_action_data(current_state,best_action,action_idx_best,evaluation,0,1);
  //  print_action_data(current_state,action_set[action_idx_best],action_idx_best,Vbest,0);

  ms_time= MYGETTIME - ms_time;
  DBLOG_POL(1,<< "WBALL03: Action selection  needed " << ms_time  << " ms");
  if (ms_time > (40*ServerOptions::slow_down_factor)) {
    LOGNEW_ERR(0,<< "WBALL03: Action selection needed " << ms_time  << " ms");
  }
  return true;
}


bool NeuroWball::evaluate_passes06(AAction & best_aaction_P, AState &current_state){
  AAction action_set[MAX_AACTIONS];
  AAction best_aaction;
  int action_set_size;

  action_set_size = generate_action_set06(current_state, action_set);
  DBLOG_POL(0, << "Action set generation needed " << MYGETTIME- ms_time << " millis");

  // 3. select best action

  if(select_best_aaction06(current_state,action_set,action_set_size,best_aaction) == false)
    return false;

  LOGNEW_POL(0, << "select best needed " <<MYGETTIME- ms_time << " millis");
  best_aaction_P = best_aaction;
  return true;
}



int NeuroWball::generate_action_set06(const AState &state, AAction *actions){
  int num_actions = 0;

  Planning::unmark_all_pass_receiver(); // reset pass receiver array
  Planning::mark_pass_receiver(state.my_idx); // avoid to get ball back!
  
  generate_safe_passes06(state, actions, num_actions);
  generate_laufpasses06(state, actions, num_actions,0);
  generate_penaltyarea_passes06(actions, num_actions);
  return num_actions;
}


void NeuroWball::generate_safe_passes06(const AState &state, AAction *actions, 
				   int & num_actions){
  AAction candidate_action;
  double inner_defense_x = 16.;
  double inner_defense_y = 32.;

  Vector my_pos = state.my_team[state.my_idx].pos;



  for(int recv_idx=0;recv_idx<11;recv_idx++){
    // dont play backpasses, if I'm not advanced enough
    if((WSinfo::me->pos.getX() < -20)
       && (state.my_team[recv_idx].pos.getX() < my_pos.getX()))
      continue;

    if(state.my_team[recv_idx].valid == false)
      continue;
    if(state.my_idx == recv_idx) // passes to myself are handled below
      continue;
    if(state.my_team[recv_idx].number == WSinfo::ws->my_goalie_number)
      //never pass to own goalie
      continue;

    //    DBLOG_POL(0,<<"generate safe passes. check receiver "<<state.my_team[recv_idx].number);


    if ((my_pos.getX() < - ServerOptions::pitch_length/2. + inner_defense_x)
       && (fabs(my_pos.getY()) <inner_defense_y/2.0)){
      // I am in my penalty area
      if( (state.my_team[recv_idx].pos.getX() < - ServerOptions::pitch_length/2.+inner_defense_x) &&
	  (fabs(state.my_team[recv_idx].pos.getY()) < inner_defense_y/2.0)){
	// receiver is within penalty area
	DBLOG_POL(2,<<"Pass receiver "<<state.my_team[recv_idx].number
		<<" also within penalty area -> DO NOT PASS");
	continue;
      }
      if ((my_pos.getY() <0)
	 && (my_pos.getY() < state.my_team[recv_idx].pos.getY())) {
      	DBLOG_POL(2,<<"Pass receiver "<<state.my_team[recv_idx].number
		<<" too far left -> DO NOT PASS");
	continue;
      }
      if ((my_pos.getY() >0)
	 && (my_pos.getY() > state.my_team[recv_idx].pos.getY())) {
      	DBLOG_POL(2,<<"Pass receiver "<<state.my_team[recv_idx].number
		<<" too far right -> DO NOT PASS");
	continue;
      }
    }
    else{ // I am not in my penalty area, but I do want to  pass through penalty area
      Vector middle_of_inner_defense;
      middle_of_inner_defense.setXY( -ServerOptions::pitch_length/2. + inner_defense_x/2., 0. );
      if(Tools::intersection(middle_of_inner_defense, inner_defense_x,
			     inner_defense_y,my_pos,
			     state.my_team[recv_idx].pos) == true){
      	DBLOG_POL(2,<<"Pass to receiver "<<state.my_team[recv_idx].number
		<<" intersects penalty area -> DO NOT PASS");
	continue;
      }
    }

    if ((state.ball.pos.getX() >5.) && (state.my_team[recv_idx].pos.getX() < 0.)){
      DBLOG_POL(2,<<"Pass to receiver "<<state.my_team[recv_idx].number
		<<" intersects midfield -> DO NOT PASS");
      continue;  // do not allow passes across the middelfieldline
    }

/*TG_OSAKA: just a question: should i not prefer to hand over 'receiver->vel'
 * in the next call instead of Vector(0.0) - this parameter is expected to be a 'relative target'*/

    double pass_speed = 2.0;  // todo: refine this ;-)
    if(state.my_team[recv_idx].pos.distance(state.ball.pos) >10.)
      pass_speed = 2.5;

    double pass_dir = (state.my_team[recv_idx].pos + state.my_team[recv_idx].vel  - state.ball.pos).arg();
    
    if (Planning::check_candidate_pass(candidate_action,state.ball.pos, pass_speed, pass_dir) == true){
      DBLOG_POL(0,<<"Checking Pass receiver "<<state.my_team[recv_idx].number<<" SUCCESS");
      DBLOG_POL(0,<<"Resulting pos: "<<candidate_action.actual_resulting_position
		<<" Receiver number "<<candidate_action.targetplayer_number
		<<" Receiver idx "<<candidate_action.target_player);

      actions[num_actions++] = candidate_action; // insert action
    } else{
      DBLOG_POL(0,<<"Checking Pass receiver "<<state.my_team[recv_idx].number<<" FAILURE");
    }

    
    
  } // for all players
}

void NeuroWball::generate_laufpasses06(const AState &state, AAction *actions, 
				  int &num_actions, const int save_time){

  //  LOGNEW_POL(0, << "Entered laufpass2");
  long ms_time= MYGETTIME;
  AAction candidate_action;
  double min_angle = -90/180.*PI;
  double max_angle = 90/180.*PI;

  Vector mypos = WSinfo::me->pos;
  if(mypos.getX() > 40){
    if(mypos.getY() < -3){
      min_angle = 0/180.*PI;
    }
    if(mypos.getY() > 3){
      max_angle = 0/180.*PI;
    }
  }

  if (DeltaPositioning::get_role(WSinfo::me->number) == 0) {// I'm a defender: laufpasses go forward and nowhere else
    if(mypos.getY() < -20){
      min_angle = -30/180.*PI;
      max_angle = 30/180.*PI;
    }
    else  if(mypos.getY() < 0){
      min_angle = -30/180.*PI;
      max_angle = 0/180.*PI;
    }
    else  if(mypos.getY() < 20){
      min_angle = 0/180.*PI;
      max_angle = 30/180.*PI;
    }
    else{
      min_angle = -30/180.*PI;
      max_angle = 30/180.*PI;
    }
  }
  
  for(double angle=min_angle;angle<max_angle;angle+=10/180.*PI){
  //  for(double angle=min_angle;angle<max_angle;angle+=20/180.*PI){
    //    LOGNEW_POL(0, << "check angle "<<RAD2DEG(angle));
    double speed =ServerOptions::ball_speed_max; // maximum
    speed = adjust_speed2(WSinfo::ball->pos,angle,speed);
    if(speed < ServerOptions::ball_speed_max){
      LOGNEW_POL(0, << "Adjusted speed in dir "<<RAD2DEG(angle)<<" to "<< speed);
    }

    if (Planning::check_candidate_pass(candidate_action,state.ball.pos, speed, angle) == true){
      DBLOG_POL(0,<<"Checking LAUFPASS: SUCCESS");
      DBLOG_POL(0,<<"Resulting pos: "<<candidate_action.actual_resulting_position
		<<" Receiver number "<<candidate_action.targetplayer_number
		<<" Receiver idx "<<candidate_action.target_player);
      candidate_action.subtype = SUBTYPE_LAUFPASS;  // just to note
      actions[num_actions++] = candidate_action; // insert action
      angle += 30/180.*PI;
    } // succesful laufpass found
  }  // for all angles
  ms_time = MYGETTIME - ms_time;
  LOGNEW_POL(0, << "checking laufpasses needed " << ms_time << "millis");
}

void NeuroWball::generate_penaltyarea_passes06(AAction *actions, int &num_actions){

  //  LOGNEW_POL(0, << "Entered penaltyarea passes");
  long ms_time= MYGETTIME;

  AAction candidate_action;
  double min_angle = -90/180.*PI;
  double max_angle = 90/180.*PI;

  Vector mypos = WSinfo::me->pos;
  if(mypos.getX() < 35 || fabs(mypos.getY()) >20)
    return;

  if(mypos.getY() < -10){
    min_angle = 0/180.*PI;
    if(mypos.getX()>45)
      max_angle=120/180.*PI;
    else
      max_angle = 100/180.*PI;
  }
  else  if(mypos.getY() < 0){
    if(mypos.getX()>45)
      min_angle=-120/180.*PI;
    else
      min_angle = -100/180.*PI;
    max_angle = 0/180.*PI;
  }
  
  for(double angle=min_angle;angle<max_angle;angle+=5/180.*PI){
    //    LOGNEW_POL(0, << "check angle "<<RAD2DEG(angle));
    double speed = 2.7;//ServerOptions::ball_speed_max; // maximum //TG08: NeuroKick cannot kick as hard, yet !!!
    if(Planning::check_action_penaltyareapass06(WSinfo::ball->pos,candidate_action,speed,angle)){
      //LOGNEW_POL(0,"Checking Pen are, dir  "<<RAD2DEG(angle)<<" speed "<<speed<<" : SUCCESS");
      actions[num_actions++] = candidate_action; // insert action
      angle += 20/180.*PI;
    } // succesful laufpass found
  }  // for all angles
  ms_time = MYGETTIME - ms_time;
  LOGNEW_POL(0, << "checking penalty area passes needed " << ms_time << "millis");
}



