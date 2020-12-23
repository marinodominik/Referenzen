#include "../policy/planning.h"
#include "ws_info.h"
#include "neuro_positioning.h"
#include "../policy/positioning.h"
#include "sort.h"
#include "ws_memory.h"


#define BASELEVEL 0 // Baselevel for Logging : Should be 3 for quasi non logging
#if 1
#define DBLOG_POL(LLL,XXX) LOG_POL(LLL,XXX)
#define DBLOG_ERR(LLL,XXX) LOG_ERR(LLL,XXX)
#define DBLOG_DRAW(LLL,XXX) LOG_POL(LLL,<<_2D <<XXX)
#else
#define DBLOG_POL(LLL,XXX)
#define DBLOG_ERR(LLL,XXX) 
#define DBLOG_DRAW(LLL,XXX)
#endif

#define MIN_POSITION 0

/***********************************************************************************/

/*   value based positioning */

/***********************************************************************************/


NeuroPositioning::NeuroPositioning() {
  num_pj_attackers = 7;
  min_dist2teammate = 4.0;
  ValueParser vp(CommandLineOptions::policy_conf,"NeuroPositioning");
  vp.get("num_pj_attackers",num_pj_attackers);
  vp.get("min_dist2teammate",min_dist2teammate);

}


/* returns an array of indices corresponding to permutation number m */
void  NeuroPositioning::get_permutation(const int m, const int no_sets, const int n[], int idx[]){
  if(n[0] == 0){
    cout<<"Permutation: Error: n[0] = 0!"<<endl;
    return;
  }
  idx[0] = m % n[0];
  int tmp = 1;
  for(int k=1;k<no_sets;k++){
    if(n[k] == 0){
      cout << "Nr " << WSinfo::me->number << ": Permutation: Error: n[" << k << "] = 0!" << endl;
      DBLOG_ERR(0, << " Permutation: Error: n["<< k << "] = 0!" << endl);
      DBLOG_POL(0, << " Permutation: Error: n["<< k << "] = 0!"
	      << " m " << m << " no_sets " << no_sets << endl);
      return;
    }
    tmp *= n[k-1];
    if(tmp == 0) {
      DBLOG_POL(0,<<" tmp "<<tmp);
    }
    idx[k] = (int)(m / tmp)%n[k];
  }
}



float NeuroPositioning::evaluate(const AState& state,const AAction jointaction[]){
  
  float J;
  AState newstate;


  AbstractMDP::model(state,jointaction,newstate);

#if 0 // be careful in activating - takes a lot of time ?!
  char buffer[200];
  char buffer_2d[2048];
  strstream stream_2d(buffer_2d,2048);
  sprintf(buffer," ");
  sprintf(buffer_2d," ");
  for(int k=0;k<PLAYERS;k++){
    stream_2d <<P2D(newstate.my_team[k].pos.x, newstate.my_team[k].pos.y,"yellow"); 
    sprintf(buffer,"%s %.1f %.1f ",buffer,newstate.my_team[k].pos.x, 
	    newstate.my_team[k].pos.y);
  }
  stream_2d << '\0';
  DBLOG_POL(1,<< _2D << buffer_2d);
  DBLOG_POL(1,"PJ: checking joint action "<<buffer);
#endif

  J= Planning::Jnn(newstate,true,jas1.my_idx); // outsourcing

  return J;
}

void NeuroPositioning::get_jointaction(const int idx[],AAction jointaction[]){
  for(int k=0;k<PLAYERS;k++){
    jointaction[k] = jas1.a[k][idx[k]];
  }
}

bool NeuroPositioning::is_relevant(const AState& state,int player){
  if (!state.my_team[player].valid)
    return false;
  if (state.my_team[player].number == WSinfo::me->number)
    return true;

  if (state.my_team[player].number == WSinfo::ws->my_goalie_number)
    return false;
  return false;
}

Vector rel_target_arr[] = {Forward, Left, Right, Backward, LeftForward, RightForward, RightBackward, LeftBackward};

int NeuroPositioning::determine_all_jointactions(const AState& state, const XYRectangle2d *constraints_P, const Vector *home_positions_P){
  int no_jas1 = 1;
  AAction candidate_action;


  for(int k=0;k<PLAYERS;k++){
    int actionsk=0;
    jas1.a[k][actionsk].action_type = AACTION_TYPE_STAY;
    jas1.a[k][actionsk].target_position = state.my_team[k].pos;
    actionsk++;

    if (is_relevant(state,k)) {
      jas1.a[k][actionsk].action_type = AACTION_TYPE_GOTO;
      jas1.a[k][actionsk].target_position = home_positions_P[k];
      actionsk++;
      
      DBLOG_POL(BASELEVEL+0,"Player "<<k<<" considered relevant");

      Vector player_pos = state.my_team[k].pos;

      for (int i = 0; i < 8; i++) {
	if (constraints_P[k].inside(player_pos + rel_target_arr[i])) {
	  jas1.a[k][actionsk].action_type = AACTION_TYPE_GOTO;
	  jas1.a[k][actionsk].target_position = player_pos + rel_target_arr[i];
	  actionsk++;
	}
      }
      for (int i = 0; i < 8; i++)
	if (constraints_P[k].inside(player_pos + 0.5 * rel_target_arr[i])) {
	  jas1.a[k][actionsk].action_type = AACTION_TYPE_GOTO;
	  jas1.a[k][actionsk].target_position = player_pos + 0.5 * rel_target_arr[i];
	  actionsk++;
	}
    }
    jas1.n[k]=actionsk;
    if(actionsk >=MAXACTIONS){
      DBLOG_ERR(0,"ERROR PJ: Max Number of Actions exceeded!");
    }
    no_jas1 *= actionsk;
  }
  return no_jas1;
}


bool NeuroPositioning::shall_player_do_neuropositioning(int number_P, Vector homepos_P) {

  int role = DeltaPositioning::get_role(number_P);

  if (role == 2) { // I am an attacker
#ifdef TRAINING
    //Daniel
    return true;
#endif
    if (homepos_P.getX() > FIELD_BORDER_X - 24.0 &&  // if homepos is very close2 goal
       fabs(homepos_P.getY() ) < 20.0)
      return true;
    return false;
  }

#ifdef TRAINING
  if (role == 1) {
    return true;
  }
#endif

  // I am a midfielder
  return false; // currently, neuro positioning makes no sense!

  if (number_P == 6) { // I am the midfielder on the left wing
    if (WSinfo::ball->pos.getY() > 0)  // Ball is in left half
      return true;
    return false;
  }
  else if(number_P == 7) { // I am the midfielder in the middle position
    if(WSinfo::ball->pos.getY() > -10 && WSinfo::ball->pos.getY() < 10  )  // Ball is in middle of field
      return true;
    return false;
  }
  else if(WSinfo::me->number == 8){ // I am the midfiedler on the right half
    if(WSinfo::ball->pos.getY() < 0)  // Ball is in left half
      return true;
    return false;
  }

  return false;
}


bool NeuroPositioning::am_I_neuroattacker() {
  AState state;
  AbstractMDP::copy_mdp2astate(state);
  int teammates = 0;
  Sort teamsort = Sort(PLAYERS);
  Vector sort_reference;

  //  mdpInfo::memory->update();

  int my_role = DeltaPositioning::get_role(WSinfo::me->number);

  if (my_role == 0) { // I'm a defender
    DBLOG_POL(0, << "PJ Check I'm a defender "
	      <<" my role: "<< my_role);
    return false;
  }

  if ((WSinfo::me->pos.getX() < MIN_POSITION) || // I'm not in on the opponent's half
      (WSinfo::ball->pos.getX() < MIN_POSITION)){
    DBLOG_POL(0, << "PJ Check for Neuro failed: "
	      << " team last at ball " << WSmemory::team_last_at_ball()
	      << " my role: " << my_role);
    
    return false;
  }

  if(WSmemory::team_last_at_ball() != 0) {// &&  // other team might be in attack
     //mdpInfo::ball_pos_abs().x < 25. &&          // the ball position is less than 25 m
     //mdpInfo::my_pos_abs().x > mdpInfo::ball_pos_abs().x){ // I am before ball position
      DBLOG_POL(0, << "PJ Check for Neuro failed: "
		<< " team last at ball " << WSmemory::team_last_at_ball()
		<< " my role: "<< my_role);
    return false;
  }


  sort_reference = HIS_GOAL_CENTER;

  for(int i=0;i<PLAYERS;i++){
    if(!state.my_team[i].valid)
      continue;
    teammates ++;
    Vector ppos = state.my_team[i].pos;
    teamsort.add(i,ppos.sqr_distance(sort_reference));
  }
  teamsort.do_sort();

  for(int i=0;i<num_pj_attackers;i++){
    if(i>=teammates)
      continue;
    if(state.my_team[teamsort.get_key(i)].number == WSinfo::me->number){
      DBLOG_POL(0, << "PJ_noBall: CHECK NEUROATTACKER = TRUE! I`m among the "<<num_pj_attackers
	      <<" closest players to reference point "<<sort_reference
	      <<" my role: "<< my_role);
      return true;
    }
  }
  DBLOG_POL(0, << "PJ_noBall: I`m NOT NOT NOT among the "<<num_pj_attackers
	    <<" closest players to reference point "<<sort_reference
	    <<" my role: "<< my_role);
  return false;
}



bool NeuroPositioning::get_neuro_position(Vector &target_P, XYRectangle2d *constraints_P, Vector *home_positions_P){

  for (int i = 1; i < 7; i++) {
    LOG_DEB(0, << _2D << constraints_P[i]);
    LOG_DEB(0, << _2D << VC2D(home_positions_P[i], 1.0, "blue"));
  }

  Vector my_homepos = home_positions_P[WSinfo::me->number-1];//DeltaPositioning::attack433.get_grid_pos(WSinfo::me->number);  

  DBLOG_POL(0,<<"Do neuropositionig ");

  //Daniel: auskommentiert, damits erstmal laeuft
#if 0 // ridi: test only 
  //  if(1){ // always go home -> test only
  if(position_check(WSinfo::me->pos) == false){
    DBLOG_POL(0,"Ouch - my current position violates constraints -> go home !");
    if(Stamina::get_state() == STAMINA_STATE_RESERVE)
      return do_waitandsee(cmd);
    return go2pos_withcare(cmd, my_homepos);
    //    return go2pos_economical(cmd,my_homepos);
  }

#endif 

  AState state;
  AAction jointaction[PLAYERS];
  int idx[PLAYERS];
  float J;
  float Jmin = 1000;
  AAction winner;


  AbstractMDP::copy_mdp2astate(state);
  for(int k=0;k<PLAYERS;k++){ // search for my index
    if(state.my_team[k].valid && state.my_team[k].number == WSinfo::me->number)
      jas1.my_idx = k;
  }

  int no_joint_actions = determine_all_jointactions(state, constraints_P, home_positions_P);
  DBLOG_POL(BASELEVEL+1, << "Number of jointactions " << no_joint_actions);


  for ( int m = 0; m < no_joint_actions; m++ ){ // for all combinations of actions
    get_permutation(m, PLAYERS, jas1.n, idx);
    get_jointaction(idx, jointaction);

    J = evaluate(state, jointaction);
    DBLOG_POL(BASELEVEL+0, << "Action "<< jointaction[jas1.my_idx].target_position - 
	    state.my_team[jas1.my_idx].pos
	    << " J: " << J<<" action type "<< jointaction[jas1.my_idx].action_type);
    //DBLOG_POL(1, << "J "<<J);
    if(J < Jmin){
      winner = jointaction[jas1.my_idx];
      Jmin = J;
    }
  }

  winner.dash_power= Stamina::dash_power();

  DBLOG_POL(BASELEVEL+0, << "PJ: Positioning Target: "
	  << winner.target_position-state.my_team[jas1.my_idx].pos
	  << " Abs: "<< winner.target_position<<" action type "<<winner.action_type);
  DBLOG_POL(BASELEVEL+0,<< _2D 
	  << VC2D(winner.target_position ,1,"red"));


  target_P = winner.target_position;

  if (!constraints_P[WSinfo::me->number-1].inside(target_P))
    target_P = my_homepos;

  return true;

}


/*
bool NeuroPositioning::position_check(const Vector targetpos, Set2d &allowed_area_P){


  Vector mypos = WSinfo::me->pos;

  Vector checkpos;
  Vector deltapos = targetpos - mypos;
  deltapos.normalize(1.1); // Max. distance I could actually go in one step
  checkpos = mypos + deltapos;

  if (allowed_area_P.inside(checkpos)) return false;

  if(!(FIELD_AREA.inside(checkpos))){
    DBLOG_POL(BASELEVEL+0,"Pos Check: REMOVE rel Target "<<deltapos<<" pos "<<checkpos<<" not in pitch "
	      <<" mypos: "<<mypos);
    return false;
  }

  // consider offside
  if(checkpos.x > critical_offside_line){
    DBLOG_POL(BASELEVEL+0,"Pos Check: REMOVE rel Target "
	      <<deltapos<<" Targetpos "<<checkpos<<" behind offside line "
	      <<critical_offside_line <<" mypos: "<<mypos);
    return false;
  }


  // check teammates;
  WSpset pset = WSinfo::valid_teammates_without_me;
  
  for (int idx = 0; idx < pset.num; idx++) {
    if(pset[idx]->pos.distance(checkpos) < min_dist2teammate){
      DBLOG_POL(BASELEVEL+0,"Pos Check: REMOVE rel Target "<<deltapos<<" Targetpos "<<checkpos<<" too close to teammate "
		<<pset[idx]->pos<<" mypos: "<<mypos);
      return false;
    }
  }


  return true; // got everything right!
}
*/
