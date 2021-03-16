#include "pass_selection.h"

#define MAX(X,Y) ((X>Y)?X:Y)

#if 1
#define LOGNEW_POL(LLL,XXX) LOG_POL(LLL,<<"PassSelection: " XXX)
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

PassSelection::PassSelection() {
  //  ValueParser vp(CommandLineOptions::policy_conf,"PassSelection");
  ValueParser vp(CommandLineOptions::policy_conf,"PassSelection");

  selfpass = new Selfpass;
  dribblestraight = new DribbleStraight;
}


PassSelection::~PassSelection() {
  delete selfpass;
  delete dribblestraight;
}

double PassSelection::adjust_speed2(const Vector ballpos, const double dir, const double speed){
  Vector ip;

  ip = Tools::ip_with_fieldborder(ballpos, dir);
  //  LOGNEW_DRAW(0,C2D(ip.getX(),ip.getY(),0.3, "red"));
  if(ballpos.distance(ip) <5.)
    return 0.5;
  else if(ballpos.distance(ip) <10.)
    return 1.0;
  else if(ballpos.distance(ip) <15.)
    return 1.5;
  else if(ballpos.distance(ip) <20.)
    return 2.0;
  else
    return speed;
}


Vector PassSelection::compute_potential_pos(const AState current_state,const AAction action){
  AState successor_state;

  AbstractMDP::model(current_state, action,successor_state); // compute state after action is executed
  return Planning2::compute_potential_pos(successor_state);   // compute potential of that state
}


void PassSelection::print_action_data(const Vector ballpos, const AAction action, const int color)
{
//  const char* COLORS[]={"000000","707070","FFFFFF","000060","000080","0000A0","0000C0","0000F0"};
#if LOGGING && BASIC_LOGGING
  const char* COLORS[]={"black","grey","white","cyan","orange","blue","red","green"};
#endif

  if(action.action_type == AACTION_TYPE_PASS){
    LOGNEW_DRAW(0, L2D(ballpos.getX(), ballpos.getY(),
					    action.target_position.getX(),
					    action.target_position.getY(),
					    COLORS[color]));
    if(action.subtype == SUBTYPE_PASS){
      LOGNEW_DRAW(0, C2D(action.actual_resulting_position.getX(),
			 action.actual_resulting_position.getY(), 0.2 * color,COLORS[color]));
      //      LOGNEW_DRAW(0, C2D(action.potential_position.getX(), action.potential_position.getY(), 0.5,"blue"));
    }
    else{ // Laufpass
      LOGNEW_DRAW(0, C2D(action.actual_resulting_position.getX(),
			 action.actual_resulting_position.getY(), 0.2 * color,COLORS[color]));
      //      LOGNEW_DRAW(0, C2D(action.potential_position.getX(),	 action.potential_position.getY(), 0.5,COLORS[color]));
    }
    
    LOGNEW_POL(0, << "Action:  PASS "<<" subtype: "<<action.subtype
	       <<" Receiver: "
	       <<action.targetplayer_number
	       << " Target: "<<action.target_position
	       << " speed: "<<action.kick_velocity
	       << " kickdir: "<<RAD2DEG(action.kick_dir)
	       << " Resulting Position: "<<action.actual_resulting_position<<" advantage: "<<action.advantage);
  }
  else{
    LOGNEW_ERR(0, <<"Strange Action type "<<action.action_type);
    LOGNEW_POL(0, << "Strange ActionType "<<action.action_type);
  }
}



bool PassSelection::is_safe(const AAction action){
  if (action.action_type != AACTION_TYPE_PASS) 
    return false;
  if(action.risky_pass == true)  // this is the case for penalty area passes only
    return false;
  if(action.advantage >=2)
    return true;
  return false;
}


/*********************************************************************************************/

bool PassSelection::select_best_aaction06(const AState current_state,
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
    print_action_data(current_state.ball.pos, action_set[a],5);
#endif

    //4. check if evaluation is actually better than current best
    if (is_safe(action_set[a])){  // candiate pass is risky:
      if(best_safe <0)
	best_safe = a;
      else if(is_pass1_better(current_state, action_set[a],action_set[best_safe]) == true){
	best_safe= a;
      }
    }
    else{ // candiate pass is risky:
      if(best_risky < 0)
	best_risky = a;
      else if(is_pass1_better(current_state, action_set[a],action_set[best_risky]) == true){
	best_risky = a;
      }
    }
  } // for all actions

  LOGNEW_POL(0,"Select Best. Best safe: "<<best_safe<<" best risky "<<best_risky);

  double evaluation_delta;

  if(best_safe <0){
    action_idx_best = best_risky;
  }
  else if (best_risky <0){
    action_idx_best = best_safe;
  }  
  else{  // both risky and safe are defined
    int result = Tools::compare_positions(action_set[best_safe].actual_resulting_position, 
					  action_set[best_risky].actual_resulting_position, evaluation_delta) ;
    if (result == FIRST){
      LOGNEW_POL(0,"Select Best. Best safe: "<<best_safe<<" is best");
      action_idx_best = best_safe;
    }
    else if (result ==EQUAL && action_set[best_safe].actual_resulting_position.getX() > action_set[best_risky].actual_resulting_position.getX() -2.){
      LOGNEW_POL(0,"Select Best. Best safe: "<<best_safe<<" are EQUAL. Choose safe");
      action_idx_best = best_safe;
    }
    else{  // risky pass is definitely better, so choose risky pass!
      LOGNEW_POL(0,"Select Best. Best Risky: "<<best_risky<<" is best");
      action_idx_best = best_risky;
    }
  }

  if(action_idx_best <0){ // no action found!
    LOGNEW_POL(NEWBASELEVEL+1,<<"No PASS found");
    return false;
  }

  best_action = action_set[action_idx_best];
  LOGNEW_POL(NEWBASELEVEL+0, << "Select INTERMEDIATE PASS NO: " <<action_idx_best<< " as follows: ");
  print_action_data(current_state.ball.pos,best_action,3);

#if 1
  if (WSinfo::ws->play_mode != PM_my_GoalKick) //TG17: No improvements unde goal kicks; too risky.
    try_to_improve_pass(current_state, best_action);
#endif

  if(best_action.subtype == SUBTYPE_PASS)
    improve_direct_pass(current_state, best_action);
  //  double evaluation= Tools::evaluate_pos(action_set[action_idx_best].potential_position); // time consuming

  LOGNEW_POL(NEWBASELEVEL+0, << "Select PASS NO: " <<action_idx_best<< " as follows: ");
  print_action_data(current_state.ball.pos,best_action,0);
  //  print_action_data(current_state,action_set[action_idx_best],action_idx_best,Vbest,0);

  ms_time= MYGETTIME - ms_time;
  LOGNEW_POL(1,<< "PASS_SELECTION: Action selection  needed " << ms_time  << " ms");
  if (ms_time > (40*ServerOptions::slow_down_factor)) {
    LOGNEW_ERR(0,<< "PASS_SE�ECTION: Action selection needed " << ms_time  << " ms");
  }
  return true;
}


bool PassSelection::evaluate_passes06(AAction & best_aaction_P, AState &current_state){
  AAction action_set[MAX_AACTIONS];
  AAction best_aaction;
  int action_set_size;

#if LOGGING && BASIC_LOGGING
  long ms_time = MYGETTIME;
#endif

  action_set_size = generate_action_set06(current_state, action_set);
  LOGNEW_POL(0, << "Action set generation needed " << MYGETTIME- ms_time << " millis (#passes="<<action_set_size<<")");

  // 3. select best action

  if(select_best_aaction06(current_state,action_set,action_set_size,best_aaction) == false)
    return false;

  LOGNEW_POL(0, << "select best needed " <<MYGETTIME- ms_time << " millis");
  best_aaction_P = best_aaction;
  return true;
}



int PassSelection::generate_action_set06(const AState &state, AAction *actions){
  int num_actions = 0;

  Planning::unmark_all_pass_receiver(); // reset pass receiver array
  Planning::mark_pass_receiver(state.my_idx); // avoid to get ball back!
  
  generate_safe_passes06(state, actions, num_actions);
  generate_dream_passes06(state, actions, num_actions);
  generate_laufpasses06(state, actions, num_actions,0);
#if 1
  generate_penaltyarea_passes06(state,actions, num_actions);
#endif
  return num_actions;
}


void PassSelection::generate_safe_passes06(const AState &state, AAction *actions, 
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

    //    LOGNEW_POL(0,<<"generate safe passes. check receiver "<<state.my_team[recv_idx].number);


    if ((my_pos.getX() < - ServerOptions::pitch_length/2. + inner_defense_x)
       && (fabs(my_pos.getY()) <inner_defense_y/2.0)){
      // I am in my penalty area
      if( (state.my_team[recv_idx].pos.getX() < - ServerOptions::pitch_length/2.+inner_defense_x) &&
	  (fabs(state.my_team[recv_idx].pos.getY()) < inner_defense_y/2.0)){
	// receiver is within penalty area
	LOGNEW_POL(2,<<"Pass receiver "<<state.my_team[recv_idx].number
		<<" also within penalty area -> DO NOT PASS");
	continue;
      }
      /*TG17 if ((my_pos.getY() <0)
	 && (my_pos.getY() < state.my_team[recv_idx].pos.getY())) {
      	LOGNEW_POL(2,<<"Pass receiver "<<state.my_team[recv_idx].number
		<<" too far left -> DO NOT PASS");
	continue;
      }*/
      /*TG17if ((my_pos.getY() >0)
	 && (my_pos.getY() > state.my_team[recv_idx].pos.getY())) {
      	LOGNEW_POL(2,<<"Pass receiver "<<state.my_team[recv_idx].number
		<<" too far right -> DO NOT PASS");
	continue;
      }*/
    }
    else{ // I am not in my penalty area, but I do want to  pass through penalty area
      Vector middle_of_inner_defense = Vector( -ServerOptions::pitch_length/2. + inner_defense_x/2., 0. );
      if(Tools::intersection(middle_of_inner_defense, inner_defense_x,
			     inner_defense_y,my_pos,
			     state.my_team[recv_idx].pos) == true){
      	LOGNEW_POL(2,<<"Pass to receiver "<<state.my_team[recv_idx].number
		<<" intersects penalty area -> DO NOT PASS");
	continue;
      }
    }

    if ((state.ball.pos.getX() >5.) && (state.my_team[recv_idx].pos.getX() < 0.)){
      LOGNEW_POL(2,<<"Pass to receiver "<<state.my_team[recv_idx].number
		<<" intersects midfield -> DO NOT PASS");
      continue;  // do not allow passes across the middelfieldline
    }

/*TG_OSAKA: just a question: should i not prefer to hand over 'receiver->vel'
 * in the next call instead of Vector(0.0) - this parameter is expected to be a 'relative target'*/

    double pass_speed = compute_direct_pass_speed(state.ball.pos,
						 state.my_team[recv_idx].pos);


    double pass_dir = (state.my_team[recv_idx].pos + state.my_team[recv_idx].vel  - state.ball.pos).arg();
    
    if (Planning::check_candidate_pass(candidate_action,state.ball.pos, 
				       pass_speed, pass_dir, state.my_team[recv_idx].number) == true){
#if 1
      LOGNEW_POL(0,<<"Checking Direct Pass receiver "<<state.my_team[recv_idx].number<<" SUCCESS");
      //      print_action_data(state.ball.pos,candidate_action,4);

#endif

      actions[num_actions++] = candidate_action; // insert action
    } else{
      LOGNEW_POL(0,<<"Checking Pass receiver "<<state.my_team[recv_idx].number<<" FAILURE");
      ;
    }

    
    
  } // for all players
}


void PassSelection::generate_dream_passes06(const AState &state, AAction *actions, 
					    int & num_actions){
  AAction candidate_action;

  Vector mypos = state.my_team[state.my_idx].pos;

  PlayerSet pset = WSinfo::valid_teammates_without_me;
  pset.keep_and_sort_players_by_x_from_right(4);

  double min_x = 0;
  if(pset.num>0){
    min_x = pset[pset.num-1]->pos.getX();
  }

  LOGNEW_DRAW(0, C2D(min_x,0,1.0, "magenta"));
  for(int i=0;i<11;i++){
    Vector tmpos = state.my_team[i].pos;
    if(i==state.my_idx)
      continue;

    if(tmpos.getX() < min_x){  // only consider the 4 most advanced players
      continue;
    }
    Vector targetpos = Vector(48.0, tmpos.getY());
    if( RIGHT_PENALTY_AREA.inside(targetpos) == true)
    {
      targetpos.setX( FIELD_BORDER_X- 19.0 );
      if (WSinfo::his_goalie) //TG17
        targetpos.setX( (tmpos.getX() + WSinfo::his_goalie->pos.getX()) / 2.0 );
    }
    /*TG17 if(tmpos.getX() > targetpos.getX()-3.){ // forget it, already too close
      continue;
    }*/
    
    if (   WSinfo::ws->play_mode != PM_PlayOn //TG17: no dreampasses in standard sits
        &&   state.my_team[state.my_idx].pos.distance( targetpos ) // with me nearer
           < state.my_team[i].pos.distance(targetpos)) // to tgt pos than receiver
      continue;

    LOGNEW_DRAW(0, C2D(targetpos.getX(),targetpos.getY(),0.3, "magenta"));

    PPlayer teammate= WSinfo::get_teammate_by_number(state.my_team[i].number);
    if(teammate== NULL)
      continue;

    Vector tmp_ipos; // not used;
    //    int mytime2react = 1; // be a bit pessimistic
    int mytime2react = 1; // be a bit pessimistic
    int steps2pos = Policy_Tools::get_time2intercept_hetero(tmp_ipos, targetpos,0.0, 0.0,teammate,mytime2react); 
    // trick: ballpos is target; ballvel is 0

    double pass_speed =Tools::get_ballspeed_for_dist_and_steps(mypos.distance(targetpos), steps2pos);

    double pass_dir = (targetpos - mypos).arg();

    LOGNEW_POL(0,"Dreampass: player "<<teammate->number<<" needs "<<steps2pos
	       <<" steps to go 2 target. targetdir: "<<RAD2DEG(pass_dir)<<" speed "<<pass_speed);

    if(pass_speed > 3.0)//ServerOptions::ball_speed_max)/TG08: NeuroKick cannot kick as hard, yet !!!
      pass_speed = 3.0;//ServerOptions::ball_speed_max;/TG08: NeuroKick cannot kick as hard, yet !!!

    
    if (Planning::check_candidate_pass(candidate_action,state.ball.pos, 
				       pass_speed, pass_dir, state.my_team[i].number) == true){
#if 1
      LOGNEW_POL(0,<<"Checking DREAM Pass receiver "<<state.my_team[i].number<<" SUCCESS");
      print_action_data(state.ball.pos,candidate_action,4);
#endif
      actions[num_actions++] = candidate_action; // insert action
    }     
  } // for all players
}

void PassSelection::generate_laufpasses06(const AState &state, AAction *actions, 
				  int &num_actions, const int save_time){

  //  LOGNEW_POL(0, << "Entered laufpass2");
  long ms_time= MYGETTIME;
  AAction candidate_action;
  double min_angle = -120/180.*PI;//TG17: -90->120
  double max_angle = 120/180.*PI;//TG17: 90->120

  Vector mypos = WSinfo::me->pos;
  if(mypos.getX() > 40){
    if(mypos.getY() < -3){
      min_angle = -30./180.*PI; //TG17: 0->30
    }
    if(mypos.getY() > 3){
      max_angle = 30./180.*PI; //TG17: 0->30
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
  
  for(double angle=min_angle;angle<max_angle;angle+=20/180.*PI){
  //  for(double angle=min_angle;angle<max_angle;angle+=20/180.*PI){
    //    LOGNEW_POL(0, << "check angle "<<RAD2DEG(angle));
    double speed = ServerOptions::ball_speed_max; // maximum
    speed = adjust_speed2(WSinfo::ball->pos,angle,speed);
      
    //    Tools::display_direction(state.ball.pos,ANGLE(angle), 2*speed, 3);

    if (Planning::check_candidate_pass(candidate_action,state.ball.pos, speed, angle) == true){
#if 1
      LOGNEW_POL(0,<<"Checking LAUFPASS: SUCCESS");
      LOGNEW_POL(0,<<"Resulting pos: "<<candidate_action.actual_resulting_position
		<<" Receiver number "<<candidate_action.targetplayer_number
		<<" Receiver idx "<<candidate_action.target_player);
#endif
      candidate_action.subtype = SUBTYPE_LAUFPASS;  // just to note
      actions[num_actions++] = candidate_action; // insert action
      angle += 30/180.*PI;
    } // succesful laufpass found
  }  // for all angles
  ms_time = MYGETTIME - ms_time;
  LOGNEW_POL(0, << "checking laufpasses needed " << ms_time << "millis");
}


#if 0 // old version
void PassSelection::generate_penaltyarea_passes06(const AState & state, AAction *actions, int &num_actions){

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
    Tools::display_direction(state.ball.pos,ANGLE(angle), 4*speed, 3);

    if(Planning::check_action_penaltyareapass06(state.ball.pos, candidate_action,speed,angle)){
      //LOGNEW_POL(0,"Checking Pen are, dir  "<<RAD2DEG(angle)<<" speed "<<speed<<" : SUCCESS");
      actions[num_actions++] = candidate_action; // insert action
      angle += 20/180.*PI;
    } // succesful laufpass found
  }  // for all angles
  ms_time = MYGETTIME - ms_time;
  LOGNEW_POL(0, << "checking penalty area passes needed " << ms_time << "millis");
}

#endif



void PassSelection::generate_penaltyarea_passes06(const AState & state, AAction *actions, int &num_actions){

  //  LOGNEW_POL(0, << "Entered penaltyarea passes");
  long ms_time= MYGETTIME;
  AAction candidate_action;

  // 1. determine all players in the hot scoring zone and check whether they can immediately score.
  

  for(int recv_idx=0;recv_idx<11;recv_idx++){
    if(state.my_idx == recv_idx) // passes to myself are handled below
      continue;
    if(state.my_team[recv_idx].valid == false)
      continue;
    if(state.my_team[recv_idx].number == WSinfo::ws->my_goalie_number)      //never pass to own goalie
      continue;
    Vector receiverpos = state.my_team[recv_idx].pos;
    Vector mypos = state.my_team[state.my_idx].pos;
    if(receiverpos.getX() < FIELD_BORDER_X - 13 || fabs(receiverpos.getY())> 12.0) // not in hot scoring zone
      continue;
    if(Tools::can_score(receiverpos) == false) // not worth playing a risky pass
      continue;
    // found an interesting receiver!
    
    LOGNEW_DRAW(0,C2D(receiverpos.getX(),receiverpos.getY(),1.3, "magenta"));


    double min_angle = (receiverpos - mypos).arg() - 10./180.*PI;
    double max_angle = min_angle + 20./180. *PI;
  
    for(double angle=min_angle;angle<max_angle;angle+=5/180.*PI){
      //    LOGNEW_POL(0, << "check angle "<<RAD2DEG(angle));
      //double speed = ServerOptions::ball_speed_max; // maximum
      double speed = compute_direct_pass_speed(state.ball.pos, receiverpos);


      if(fabs(Tools::get_angle_between_mPI_pPI(angle)) > 150./180. *PI)  // too far back, too dangerous
	continue;
      Tools::display_direction(state.ball.pos,ANGLE(angle), 4*speed, 3);
      if(Planning::check_action_penaltyareapass06(state.ball.pos, candidate_action,speed,angle) == true){
	LOGNEW_POL(0,"Checking Pen are, dir  "<<RAD2DEG(angle)<<" speed "<<speed<<" : SUCCESS");
	actions[num_actions++] = candidate_action; // insert action
      } // succesful pass found
    }  // for all angles
  } // for all teammates
  ms_time = MYGETTIME - ms_time;
  LOGNEW_POL(0, << "checking NEW penalty area passes needed " << ms_time << "millis");
}












/*********************************************************************************************/

/* check to improve */

/*********************************************************************************************/



bool PassSelection::is_pass1_better(const AState &state,  AAction &candidate_pass, AAction &best_pass){
  // do not consider safety/ risk here; should be made elsewhere
  double evaluation_delta;

  if(state.ball.pos.getX() < FIELD_BORDER_X - 15.){ // allow passes in the back only opponents penalty area
    if(candidate_pass.subtype == SUBTYPE_PASS_IN_BACK){
      LOGNEW_POL(0,<<"Found a pass in the back (orange): DO NOT CONSIDER");
      print_action_data(state.ball.pos,candidate_pass,4);
      return false;
    }
    if(best_pass.subtype == SUBTYPE_PASS_IN_BACK){
      LOGNEW_POL(0,<<"Best pass so far was a pass in the back: replace");
      print_action_data(state.ball.pos,best_pass,4);
      return true;
    }
  }


  //  print_action_data(state.ball.pos,candidate_pass,4);
  int result =Tools::compare_positions(candidate_pass.actual_resulting_position,
				       best_pass.actual_resulting_position, evaluation_delta) ;

  if(result == SECOND){  // 
    LOGNEW_POL(0,<<"is pass1 better?: Current best pass wins EINDEUTIG!: NOT BETTER");
    return false;
  }

  if(result == FIRST){ // ipos is untouched; but we know now, that pass is safe
    LOGNEW_POL(0,<<"is pass1 better?: Candidate pass wins EINDEUTIG!:  BETTER");
    //    print_action_data(state.ball.pos,candidate_pass,5);
    return true;
  }

  // both passes are more or less equal, decide by evaluation
  // it is important to distinguish also the evaluation; otherwise all passes (even backpasses, are in the same class!)
  if(candidate_pass.subtype == SUBTYPE_PASS && best_pass.subtype != SUBTYPE_PASS){
    if(candidate_pass.actual_resulting_position.distance( best_pass.actual_resulting_position) <5.){
      LOGNEW_POL(0,<<"is pass 1 better: passes go to similar positions, but candidate is a DIRECT pass: BETTER");
      return true;
    }
  }
  if(candidate_pass.subtype != SUBTYPE_PASS && best_pass.subtype == SUBTYPE_PASS){
    if(candidate_pass.actual_resulting_position.distance( best_pass.actual_resulting_position) <5.){
      LOGNEW_POL(0,<<"is pass 1 better: passes go to similar positions, but current best is a DIRECT pass: BETTER");
      return false;
    }
  }

  //TG17
  if (WSinfo::ws->play_mode == PM_my_KickIn)
  {
    if (! (   best_pass.actual_resulting_position.getX() >= WSinfo::his_team_pos_of_offside_line()
           && candidate_pass.actual_resulting_position.getX() >= WSinfo::his_team_pos_of_offside_line() ))
    {
      if (best_pass.actual_resulting_position.getX() >= WSinfo::his_team_pos_of_offside_line())
      {
        LOGNEW_POL(0,<<"is pass 1 better: best pass so far is already a pass at kick-in behind his offside line: NOT BETTER");
        return false;
      }
      if (candidate_pass.actual_resulting_position.getX() >= WSinfo::his_team_pos_of_offside_line())
      {
        LOGNEW_POL(0,<<"is pass 1 better: pass at kick-in behind his offside line: BETTER");
        return true;
      }
    }
  }

  if(evaluation_delta >0){
    LOGNEW_POL(0,<<"is pass 1 better: Candidate pass does not improve significantly, but is evaluated better: BETTER");
    //    print_action_data(state.ball.pos,candidate_pass,5);
    return true;
  }
  return false;
}  

bool PassSelection::is_pass_an_improvement(const AState &state, AAction &candidate_action, AAction &current_pass){
  if(candidate_action.subtype == SUBTYPE_PASS && current_pass.subtype == SUBTYPE_PASS){
    return false;
  }
  if(is_pass1_better(state,  candidate_action, current_pass) == false)
    return false;
  if(Planning::check_pass_with_known_ipos(candidate_action, state.ball.pos) == false){
    // print_action_data(state.ball.pos,candidate_action,6);
    return false;
  }
  if(candidate_action.advantage < 2 && candidate_action.advantage < current_pass.advantage)
    return false;
  return true;
}

void PassSelection::try_to_improve_pass(const AState &state, AAction &current_pass){
  if(current_pass.action_type != AACTION_TYPE_PASS){
    LOGNEW_POL(0,"check2improve: action not a pass. Return");
    return;
  }

#if 0 // if activated, then direct passes are always preferred. That's probably too cautious
  if(current_pass.subtype == SUBTYPE_PASS){
    LOGNEW_POL(0,"check2improve: PASS is a direct pass. Do not refine!");
    return;
  }
#endif

  AAction candidate_action;
  double delta_angle = 0./180.*PI;//TG17: orig_20
  double current_dir = Tools::get_angle_between_mPI_pPI(current_pass.kick_dir);
  int teststeps = 0;//TG17: orig_1

  if (WSinfo::ws->play_mode == PM_my_KickOff || WSinfo::ws->play_mode == PM_my_BeforeKickOff)
  {
    delta_angle = 5.; teststeps = 0;
  }

  double min_angle =  current_dir -teststeps*delta_angle;
  double max_angle = current_dir + teststeps*delta_angle +.2 * delta_angle;

  if(current_dir >20./180. *PI){
    min_angle = current_dir - (teststeps+1)*delta_angle - .2 * delta_angle;
    max_angle =  current_dir;
  }
  else if(current_dir <- 20./180. *PI){
    min_angle =  current_dir;
    max_angle = current_dir + (teststeps+1)*delta_angle +.2 * delta_angle;
  }

  if(max_angle<min_angle){
    double tmp = min_angle;
    min_angle = max_angle;
    max_angle = tmp;
  }
  double speed = 3.0;//ServerOptions::ball_speed_max; // maximum/TG08: NeuroKick cannot kick as hard, yet !!!

  LOGNEW_POL(0,"START LOOP IN try_to_improve_pass");
  for(int speedcounter = 0; speedcounter <2; speedcounter ++){  //save time
    for(double angle=min_angle;angle<max_angle;angle+= delta_angle){
      LOGNEW_POL(0,"looping over pass modifications: speed_"<<speedcounter<<", "
        <<RAD2DEG(current_pass.kick_dir)<<"to  ang_"<<RAD2DEG(angle));
      if(speedcounter == 1)
	speed = 2.0;  // slow pass
      if(fabs(angle - current_dir)>2.0/180. *PI || fabs(speed - current_pass.kick_velocity) > .2){
	speed = adjust_speed2(WSinfo::ball->pos,angle,speed);
	// Tools::display_direction(state.ball.pos,ANGLE(angle), 2*speed, 3);
	if(Planning::compute_ipos_of_candidate_pass(candidate_action, state.ball.pos, speed, angle) == true){
	  // our player gets the ball, so check
	  // print_action_data(state.ball.pos,candidate_action,1);
	  if(is_pass_an_improvement(state, candidate_action, current_pass) == true){
	    LOGNEW_POL(0,"candidate pass improves over current pass. REPLACE");
	    print_action_data(state.ball.pos,candidate_action,7);
	    current_pass = candidate_action;
	  }// pass improves
	  else
	  {
  	    LOGNEW_POL(0,"candidate pass does not improve over current pass :-/");
	  }
	}// our team intercepts
      }// angle or kick ar different
    }  // for all angles
    min_angle += delta_angle/2.; // slight Verschiebung
  }  // for all speeds
}

double PassSelection::compute_direct_pass_speed(const Vector ballpos,
					       const Vector teammatepos){
  double pass_speed = 3.0;//ServerOptions::ball_speed_max;  // todo: refine this ;-)//TG08: NeuroKick cannot kick as hard, yet !!!
#if 0 // previous computation
  if(teammate->pos.distance(state.ball.pos) <2.)
    pass_speed = 1.5;
  else if(teammate->pos.distance(state.ball.pos) <4.)
    pass_speed = 1.8;
  else if(teammate->pos.distance(state.ball.pos) <6.)
    pass_speed = 2.2;

#endif 

  double balldist2tm = ballpos.distance(teammatepos);
  int steps4distance = Tools::get_steps_for_ball_distance(balldist2tm, ServerOptions::ball_speed_max);
  if(steps4distance<4){ // for short passes, try to kick exact
    balldist2tm -= 0.3; // roughly the half kickrange -> play in closer half of player.
  }
  if(steps4distance > 10)  //if I need more than 5 steps, try to make it 5
    steps4distance = 10;

  // probably assume a minimum number of steps here (to give tm time to react)
  pass_speed = Tools::get_ballspeed_for_dist_and_steps( balldist2tm, steps4distance);

#if 0  
  int check_only = Tools::get_steps_for_ball_distance(balldist2tm, pass_speed);

  LOGNEW_POL(0,"direct pass distance: "<<balldist2tm<<" cyles2go: "
	    <<steps4distance<<" pass_speed: "<<pass_speed<<" check only: "<<check_only);
#endif 

  if(pass_speed > 3.0)//ServerOptions::ball_speed_max)//TG08: NeuroKick cannot kick as hard, yet !!!
    pass_speed = 3.0;//ServerOptions::ball_speed_max;//TG08: NeuroKick cannot kick as hard, yet !!!

  return pass_speed;
}

void PassSelection::improve_direct_pass(const AState &state, AAction &direct_pass){
  // might be refined...
  PPlayer teammate = WSinfo::valid_teammates.get_player_by_number(direct_pass.targetplayer_number);
  if(teammate == NULL){ // not found
    LOGNEW_POL(0,"Improve direct pass: Teammate not found. no improvement");
    return;
  }
  LOGNEW_POL(0,"START IMPROVEMENT OF DIRECT PASSES via improve_direct_pass");
  double pass_speed = compute_direct_pass_speed(state.ball.pos, teammate->pos);
  
  Vector target = teammate->pos;
  if(teammate->age_vel <=1)
    target += teammate->vel;
  double pass_dir = (target  - state.ball.pos).arg();


  AAction candidate_action;
  if (Planning::check_candidate_pass(candidate_action,state.ball.pos, pass_speed, pass_dir, direct_pass.targetplayer_number) == true){
    LOGNEW_POL(0,"END OF improve direct pass. Successfully found direct pass");
    direct_pass = candidate_action; // take this candidate
    return;
  } 
  LOGNEW_POL(0,"END OF improve direct pass. Improved direct pass not successful. Taking Default");
}

