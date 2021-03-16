#include "noball03_attack_bmp.h"
#include "ws_info.h"
#include "ws_memory.h"
#include "log_macros.h"
#include "mdp_info.h"
#include <stdlib.h>
#include <stdio.h>
#include "tools.h"
#include "valueparser.h"
#include "options.h"
#include "log_macros.h"
#include "../policy/planning.h"
#include "intention.h"
#include "geometry2d.h"
#include "ws_memory.h"

bool Noball03_Attack::initialized=false;
const float Noball03_Attack::GO2BALL_TOLERANCE = 0.5;

//#define DBLOG_INFO2 DBLOG_INFO  // log information
#define DBLOG_INFO2(X) // do not log information
//#define DBLOG_INFO3 DBLOG_INFO  // log information
#define DBLOG_INFO3(X) // do not log information
//#define DBLOG_INFO4 DBLOG_INFO  // log information
#define DBLOG_INFO4(X) // do not log information
#define BASELEVEL 0 // Baselevel for Logging : Should be 3 for quasi non logging
//#define BASELEVEL 3 // Baselevel for Logging : Should be 3 for quasi non logging
#if 0
#define DBLOG_POL(LLL,XXX) LOG_POL(LLL,XXX)
#define DBLOG_ERR(LLL,XXX) LOG_ERR(LLL,XXX)
#define DBLOG_DRAW(LLL,XXX) LOG_POL(LLL,<<_2D <<XXX)
#else
#define DBLOG_POL(LLL,XXX)
#define DBLOG_ERR(LLL,XXX) 
#define DBLOG_DRAW(LLL,XXX)
#endif

#define MAX_STEPS2GO 2 // defines the max. number of steps until the move is terminated and redecided

#define OPPONENTS 8

#define PLAYERS 11

#define POSITION_TOLERANCE 1.

int Noball03_Attack::num_pj_attackers;


bool Noball03_Attack::aaction2cmd(AAction action, Cmd &cmd){

  switch(action.action_type){
  case AACTION_TYPE_STAY:
    mdpInfo::set_my_intention(DECISION_TYPE_WAITANDSEE);
    face_ball->turn_to_ball();
    face_ball->get_cmd(cmd);
    //    LOG_ERR(0,<<"aaction2cmd: requires face-ball, but not yet implemented");
    return true;
    break;
  case AACTION_TYPE_GOTO: 
  case AACTION_TYPE_GOHOME:
    //LOG_DEB(0, << "Noball03_attack aaction2cmd: go2pos, target (" << action.target_position.x << "," << action.target_position.y << ")");
    mdpInfo::set_my_intention(DECISION_TYPE_RUNTO);
    return go2pos_economical(cmd, action.target_position);
    /*      
    go2pos->set_target( action.target_position );
    go2pos->get_cmd(cmd);
    return true;
    */
    break;
  case AACTION_TYPE_GO2BALL:
    //LOG_DEB(0, << "Noball03_attack aaction2cmd: go2ball");
    mdpInfo::set_my_intention(DECISION_TYPE_INTERCEPTBALL);
    //intercept->get_cmd(cmd);
    neurointercept->get_cmd(cmd);
    return true;
    break;
  default:
    LOG_ERR(0,<<"aaction2cmd: AActionType not known");
    return false;
  }
}


bool Noball03_Attack::get_cmd(Cmd & cmd) {
  LOG_POL(0, << "In NOBALL03_ATTACK_BMC : ");

  if(WSinfo::is_ball_kickable())
    return false;
  //my_role = DeltaPositioning::attack433.get_role(WSinfo::me->number);
  my_role = DeltaPositioning::get_role(WSinfo::me->number);

  if(my_role ==0 ) { // I'm a defender
    return false;
  }
  switch(WSinfo::ws->play_mode) {
  case PM_PlayOn:
    return playon(cmd);
    break;
  default:
    return false;  // behaviour is currently not responsible for that case
  }
  return false;  // behaviour is currently not responsible for that case
}


bool Noball03_Attack::am_I_attacker(){ 

  //return am_I_neuroattacker();  // this was used before 28.05.03

  //  my_role = DeltaPositioning::attack433.get_role(WSinfo::me->number);
  int role = DeltaPositioning::get_role(WSinfo::me->number);


  if(role == 0) { // I'm a defender
    DBLOG_POL(0, << "PJ Check I'm a defender "<<" my role: "<< role);
    return false;
  }

  if(role == 1 &&  // I'm a midfielder
     (WSinfo::ball->pos.getX() <-40 || WSmemory::team_last_at_ball() != 0)){
    DBLOG_POL(0, << "PJ Check I'm a midfielder and ballpos < 0 or other team has ball : I'm not an attacker ")
    return false;
  }
  DBLOG_POL(0, << "Positioning: I am an attacker ");
  return true;
}



#define MAX_STR_LEN 200
//Noball03_Attack::Noball03_Attack(): Main_Policy() {
Noball03_Attack::Noball03_Attack(){
  /* set default defend params */
  go2ball_tolerance_teammate = 0.0; // ridi: if 1.0, then often conflicts occur
  go2ball_tolerance_opponent = -1.0;

  do_receive_pass=false;
  time_of_last_receive_pass = -1;

  target = Vector(0,0);
  last_neuro_positioning = -10;

  /* read defend params from config file*/
  ValueParser vp(CommandLineOptions::policy_conf,"noball03_attack_bmp");
  vp.get("do_receive_pass",do_receive_pass);

//  cout<<"\nPJ_nb status Date: 31.05.02"
//      <<"do_receive_pass= " << do_receive_pass << " "<<endl;

  go2pos = new NeuroGo2Pos;
  intercept = new InterceptBall;
  neurointercept = new NeuroIntercept;
  face_ball = new FaceBall;
  basiccmd = new BasicCmd;

  neuro_wball = new NeuroWball;

  neuro_positioning = new NeuroPositioning;
  attack_move1_nb = new Attack_Move1_Nb;

}


Noball03_Attack::~Noball03_Attack() {
  delete go2pos;
  delete intercept;
  delete neurointercept;
  delete face_ball;
  delete neuro_positioning;

  delete attack_move1_nb;
  delete neuro_wball;
}


void Noball03_Attack::check_correct_homepos(Vector &homepos){
  if(homepos.getX() > critical_offside_line){
    homepos.setX( critical_offside_line );
  }
  if(WSinfo::me->number == 6){ // I am the midfielder on the left wing
    if(WSinfo::ball->pos.getY() > 0 && // Ball is in left half
       WSinfo::ball->pos.getX() > WSinfo::me->pos.getX()){ // Ball is before me
      if(homepos.getX() < critical_offside_line - 8.)
	homepos.setX( critical_offside_line - 8. );
      const double y_dist2ball =7.0;
      if(WSinfo::ball->pos.getY() > 10 && // Ball is left
	 WSinfo::ball->pos.getY() - homepos.getY() < y_dist2ball){
	homepos.subFromY( y_dist2ball ); // go 2 middle
	if(homepos.getY() < 15.)
	  homepos.setY( 15 );
      }
    }
  }
  else   if(WSinfo::me->number == 7){ // I am the midfielder in the middle position
    if(WSinfo::ball->pos.getY() > -20 && WSinfo::ball->pos.getY() < 20  &&  // Ball is in middle of field
       WSinfo::ball->pos.getX() > WSinfo::me->pos.getX()){ // Ball is before me
      if(homepos.getX() < critical_offside_line - 10.)
	homepos.setX( critical_offside_line - 10. );
    }
  }
  else   if(WSinfo::me->number == 8){ // I am the midfiedler on the right half
    if(WSinfo::ball->pos.getY() < 0 && // Ball is in right half
       WSinfo::ball->pos.getX() > WSinfo::me->pos.getX()){ // Ball is before me
      if(homepos.getX() < critical_offside_line - 8.)
	homepos.setX( critical_offside_line - 8. );
    }
  }
}

#ifdef TRAINING
#define DIST2OFFSIDELINE 0.0
#else
#define DIST2OFFSIDELINE 1.0
#endif

void Noball03_Attack::reset_intention() {
  critical_offside_line = WSinfo::his_team_pos_of_offside_line() - DIST2OFFSIDELINE;
  last_neuro_positioning = -10;
  target = Vector (0,0);
  LOG_POL(0, << "Noball03_Attack reset intention");
}


bool Noball03_Attack::playon(Cmd &cmd){
  long ms_time= Tools::get_current_ms_time();

#ifndef TRAINING
  if(WSmemory::get_his_offsideline_movement() <0){
    DBLOG_POL(0,"offside line moves towards us, increase critical distance!");
    critical_offside_line = WSinfo::his_team_pos_of_offside_line() - 2* DIST2OFFSIDELINE;
  }
  else
    critical_offside_line = WSinfo::his_team_pos_of_offside_line() - DIST2OFFSIDELINE;
  if(critical_offside_line < 0. - DIST2OFFSIDELINE){
    critical_offside_line = -DIST2OFFSIDELINE;
  }
#else
  critical_offside_line = WSinfo::his_team_pos_of_offside_line() - DIST2OFFSIDELINE;
#endif

  if (critical_offside_line > FIELD_BORDER_X - 5.5)
    critical_offside_line = FIELD_BORDER_X - 5.5;

  DBLOG_POL(BASELEVEL+0,<< _2D << L2D( WSinfo::his_team_pos_of_offside_line(),-30,
				       WSinfo::his_team_pos_of_offside_line(),30,"red"));
  DBLOG_POL(BASELEVEL+0,<< _2D << L2D( critical_offside_line,-30,
				       critical_offside_line,30,"green")) 


  /* analyse current game state. who will get the ball fastest? */
  compute_steps2go();
  
  // probably sets turn-neck intention:
  Policy_Tools::check_if_turnneck2ball((int)steps2go.teammate,(int) steps2go.opponent);

  go_home_is_possible = true; 
  should_care4clearance = false;

  //Iam_a_neuroattacker = am_I_neuroattacker();

  //Daniel: changed for learning!!!

#ifndef TRAINING
  my_homepos = DeltaPositioning::attack433.get_grid_pos(WSinfo::me->number);

#else

  if (WSinfo::me->number >= 2 && WSinfo::me->number <= 4)
    my_homepos = DeltaPositioning::attack433.get_grid_pos(WSinfo::me->number + 7);
  else if (WSinfo::me->number >= 5 && WSinfo::me->number <= 7)
    my_homepos = DeltaPositioning::attack433.get_grid_pos(WSinfo::me->number + 1);
  else
    my_homepos = DeltaPositioning::attack433.get_grid_pos(WSinfo::me->number);

  //Daniel end
#endif

  check_correct_homepos(my_homepos);

  DBLOG_POL(BASELEVEL+0,<< _2D << C2D(my_homepos.x, my_homepos.y ,1.2,"orange")); 
  DBLOG_POL(BASELEVEL+0,<<"My (corrected) homepos :"<<my_homepos<<" current pos "
	    <<WSinfo::me->pos<<" my role "<<my_role);


  /* updateing game_state */
  //int myteam_steps = (int)Tools::min(steps2go.me, steps2go.teammate);
  //mdpInfo::update_game_state((int) myteam_steps,(int) steps2go.opponent);

  /*
  if(myteam_steps == 0){
    ourteam_holdsball_for_cycles++;
  }
  else{
    ourteam_holdsball_for_cycles=0;
    }*/

  DBLOG_POL(BASELEVEL+0,"PJNB: stamina state: "<<Stamina::get_state());
  
  if (  1 && (test_ballpos_valid(cmd) == true) ){
    DBLOG_POL(0,"Ball pos invalid");
    return true;
  }
#if 1 // test formation
  else if (  1 && (test_go2ball_value(cmd) == true) ){
    DBLOG_POL(0,"Go to ball");
    return true;
  }
#endif
  else if (  1 && (test_go4pass(cmd) == true) ){
    DBLOG_POL(0,"Go for pass");
    return true;
  }
  else if(attack_move1_nb->get_cmd(cmd) == true){
    DBLOG_POL(0,"attack positioning: Called attack move 1 position" );
    return true;
  }
  else if (  1 && (test_offside(cmd) == true) ){
    DBLOG_POL(0,"disable offside");
    return true;
  }
  else if (  1 && (test_wait4clearance(cmd) == true) ){
    return true;
  }
#ifndef TRAINING
  else if (  1 && (test_support_attack(cmd)== true) ){
    return true;
  }
#endif
#if 0
  else if (  1 && (test_scoringarea_positioning(cmd)== true) ){
    return true;
  }
#endif
  else if (  1 && (test_neuro_positioning(cmd)== true) ){
    DBLOG_INFO3("neuro positioning");
    return true;
  }
  else if (  1 && (test_analytical_positioning(cmd)== true) ){
    DBLOG_INFO3("analytical positioning");
    return true;
  }
  // if I decided to stay, than I test if I should go home instead
  else if (  1 && (test_gohome(cmd) == true) ){ 
    DBLOG_INFO3("test going home");
    return true;
  }
  else if ( (test_default(cmd) == true) )
    return true;
  //  else lookaround->get_cmd(cmd);

  ms_time = Tools::get_current_ms_time() - ms_time;
  DBLOG_POL(BASELEVEL+1, << "PJ policy needed " << ms_time << "millis to decide");
  return false;
}


bool Noball03_Attack::test_tackle(Cmd &cmd) {
  // ridi 04: be more agressive
  double suc = Tools::get_tackle_success_probability(WSinfo::me->pos, WSinfo::ball->pos, WSinfo::me->ang.get_value());
  //  LOG_RIDI(0, << "Test Tackle: probability: " << suc);

  if (WSinfo::ws->play_mode != PM_PlayOn) {
    return 0;
  }
  if (WSinfo::is_ball_kickable()) {
    return 0;
  }
  if ((steps2go.me <= steps2go.opponent) && 
      (steps2go.me <= 2) &&  // I get the ball within the next two cycles
      (!steps2go.ball_kickable_for_opponent)) {
    return 0;
  }

  // compute if opponent has the ball
  bool opponent_has_ball = false;
  PlayerSet pset= WSinfo::valid_opponents;
  pset.keep_players_in_circle(WSinfo::ball->pos, ServerOptions::kickable_area); 
  // considr only close ops. for correct
  pset.keep_and_sort_closest_players_to_point(1,WSinfo::ball->pos);
  if ( pset.num > 0 )
    opponent_has_ball = true;

  float tackle_thresh = .75;

  if(opponent_has_ball == true && suc>tackle_thresh){
    basiccmd->set_tackle(100);
    basiccmd->get_cmd(cmd);
    DBLOG_POL(0, << "Test Tackle: Opponent has ball and suc > threshold -> tackle ");
    return true;
  }

  DBLOG_POL(0, << "Test Tackle: NO tackle situation ");

  return false;
}


bool Noball03_Attack::test_ballpos_valid(Cmd &cmd){
  if(WSinfo::is_ball_pos_valid())
    return false;
  mdpInfo::set_my_intention(DECISION_TYPE_SEARCH_BALL);
  face_ball->turn_to_ball();
  face_ball->get_cmd(cmd);
  LOG_POL(BASELEVEL,<<"Noball03_attack: Don't knwo ballpos, face ball!");
  return true;
}

   

void Noball03_Attack::compute_steps2go(){
  /* calculate number of steps to intercept ball, for every player on the pitch, 
     return sorted list */
  Policy_Tools::go2ball_steps_update();
  go2ball_list = Policy_Tools::go2ball_steps_list();

  steps2go.me = 1000;
  steps2go.opponent = 1000;
  steps2go.opponent_number = 0;
  steps2go.teammate = 1000;
  steps2go.teammate_number = 0;
  steps2go.my_goalie = 1000;

  
  /* get predicted steps to intercept ball for me, my goalie , fastest 
     teammate and fastest opponent */
  for(int i=0;i<22;i++){
    if(go2ball_list[i].side == Go2Ball_Steps::MY_TEAM){
      if(WSinfo::me->number == go2ball_list[i].number){
	steps2go.me = go2ball_list[i].steps;
      }
      else if(WSinfo::ws->my_goalie_number == go2ball_list[i].number){
	steps2go.my_goalie = go2ball_list[i].steps;
	
      }
      else if(steps2go.teammate == 1000){
	steps2go.teammate = go2ball_list[i].steps;
	steps2go.teammate_number = go2ball_list[i].number;
      }
    }
    if(go2ball_list[i].side == Go2Ball_Steps::THEIR_TEAM){
      if(steps2go.opponent == 1000){
	steps2go.opponent = go2ball_list[i].steps;
	steps2go.opponent_number = go2ball_list[i].number;
      }
    } 
  }
  int mysteps2intercept =  neurointercept->get_steps2intercept();
  if(mysteps2intercept >0 && mysteps2intercept !=steps2go.me ){
    DBLOG_POL(0,"WARNING! Old intercept estimation "<<steps2go.me  
	      <<" differs from exact estimation "<<mysteps2intercept<<" DO CORRECTION");
    steps2go.me = mysteps2intercept;
  }

  char buffer[200];
  sprintf(buffer,"Go2Ball Analysis: Steps to intercept (Me %.2d) (Teammate %d %.2d) (Opponent %d %.2d)", 
	  steps2go.me, steps2go.teammate_number, steps2go.teammate,  steps2go.opponent_number,  
	  steps2go.opponent);  
  DBLOG_POL(BASELEVEL+0, << buffer);

  PPlayer p_tmp = WSinfo::get_opponent_by_number(steps2go.opponent_number);
  if (p_tmp != NULL) {
    steps2go.opponent_pos = p_tmp->pos;
    steps2go.ball_kickable_for_opponent = WSinfo::is_ball_kickable_for(p_tmp);
  }
  else {
    steps2go.opponent_pos = Vector(0.0, 0.0);
    steps2go.ball_kickable_for_opponent = false;
  }

  p_tmp = WSinfo::get_teammate_by_number(steps2go.teammate_number);
  if (p_tmp != NULL) {
    steps2go.teammate_pos = p_tmp->pos;
    steps2go.ball_kickable_for_teammate = WSinfo::is_ball_kickable_for(p_tmp);
  }
  else {
    steps2go.teammate_pos = Vector(0.0, 0.0);
    steps2go.ball_kickable_for_teammate = false;
  }


}

bool Noball03_Attack::i_am_fastest_teammate(){
  if(steps2go.me < steps2go.teammate+go2ball_tolerance_teammate) // I am the fastest
    return true;
  if(steps2go.me == steps2go.teammate+go2ball_tolerance_teammate) // Me and teammate have same distance
    //if(steps2go.teammate_number>WSinfo::me->number)
    return true;
  return false;
}

bool Noball03_Attack::my_free_kick_situation(){
  if(WSinfo::ws->play_mode == PM_my_KickIn
     || WSinfo::ws->play_mode == PM_my_FreeKick
     || WSinfo::ws->play_mode == PM_my_CornerKick
     ){
    return true;
  }
  
  return false;
}

bool Noball03_Attack::opening_seq(const Vector ballpos, const Vector ballvel, const Vector ipos, Cmd &cmd){

  Vector mynewpos, dummy1;
  Angle dummy2;

  DBLOG_POL(BASELEVEL+0,"PJ: Can get expected pass: Ipos behind offside line ");
  mdpInfo::set_my_intention(DECISION_TYPE_INTERCEPTBALL); // tell neck to turn2 ball
  neurointercept->set_virtual_state(ballpos,ballvel );
  neurointercept->get_cmd(cmd);
  Tools::model_player_movement(WSinfo::me->pos,WSinfo::me->vel, WSinfo::me->ang.get_value(), 
			       cmd.cmd_body, mynewpos, dummy1, dummy2);

  if(mynewpos.getX() > critical_offside_line){
    LOG_POL(BASELEVEL+0,"PJ: Opening Sequence: My next pos is behind critical offside line -> wait! ");
    cmd.cmd_body.unset_lock();
    cmd.cmd_body.unset_cmd();
    basiccmd->set_turn(0);
    basiccmd->get_cmd(cmd);
    return true;
  }

  LOG_POL(BASELEVEL+0,"PJ: Opening Sequence: Goint towards intercept position ");
  return true;
}

bool Noball03_Attack::test_go4pass(Cmd &cmd){
  Vector ipos,ballpos,ballvel;

  if(Policy_Tools::check_go4pass(ipos,ballpos,ballvel) == false)
    return false;
  if(WSinfo::me->pos.getX() > critical_offside_line){
    DBLOG_POL(0,"WAITING FOR PASS. , But have to disable offside first!");
    Vector targetpos;
    targetpos.setXY( WSinfo::me->pos.getX() -10., WSinfo::me->pos.getY() );
    go2pos->set_target( targetpos );
    go2pos->get_cmd(cmd);
    return true;
  }

  if(ipos.getX() >critical_offside_line){
    return opening_seq(ballpos,ballvel,ipos,cmd);
  }

  int cycles2wait;

  if(WSinfo::me->vel.norm() > .1){
    DBLOG_POL(0,"GO4PASS: Winkel zur Ipos: "
	      <<RAD2DEG((ipos - WSinfo::me->pos).arg())
	      <<" Speedrichtung:  "<<RAD2DEG(WSinfo::me->vel.arg())
	      <<" absdiff: "<<RAD2DEG(Tools::get_abs_angle((ipos - WSinfo::me->pos).arg() - WSinfo::me->vel.arg())));
    
    if((WSinfo::me->pos.distance(ipos) <= WSinfo::me->kick_radius) 
       ||(Tools::get_abs_angle((ipos - WSinfo::me->pos).arg() - WSinfo::me->vel.arg()) > 45./180. *PI)){
      DBLOG_POL(0,"Expected Icpt-pos is behind my current speed, so stop first.");
      double stop_dash=Tools::get_dash2stop();
      basiccmd->set_dash(stop_dash);
      basiccmd->get_cmd(cmd);
      return true;
    }
  }

  if ((WSinfo::ball->pos - ipos).norm() < 10. && (WSinfo::me->pos - ipos).norm()<5. &&
      (WSinfo::ball->pos - WSinfo::me->pos).norm() < 5.){
    LOG_POL(0, << "I am passreceiver, but I get too close to ball -> Turn2ball " );
    face_ball->turn_to_ball();
    face_ball->get_cmd(cmd);
  }



#ifndef TRAINING
  if(Tools::shall_I_wait_for_ball(ballpos,ballvel,cycles2wait) == true){
      DBLOG_POL(0,"Expected pass is coming right to me, and no op. close to me -> just face ball");
      if(WSinfo::me->pos.distance(WSinfo::ball->pos) < ServerOptions::visible_distance){
	DBLOG_POL(0,"Ball is coming towards me and I can feel it now, let's turn forward!");
	if(WSinfo::me->pos.getX() < 35)
	  face_ball->turn_in_dir(ANGLE(0));
	else
	  face_ball->turn_in_dir((Vector(52,0)-WSinfo::me->pos).ARG());
	face_ball->get_cmd(cmd);
	return true;
      }
      face_ball->turn_to_ball();
      face_ball->get_cmd(cmd);
      return true;
  }
#endif

  LOG_POL(BASELEVEL+0,"PJ: Can get expected pass: Intercepting virtual ball ");
  mdpInfo::set_my_intention(DECISION_TYPE_INTERCEPTBALL); // tell neck to turn2 ball
  neurointercept->set_virtual_state(ballpos,ballvel );
  neurointercept->get_cmd(cmd);
  return true;
}

bool Noball03_Attack::test_go2ball_value(Cmd &cmd){

  if(!i_am_fastest_teammate()){
    /* dont go to ball because a teammate is faster!*/
    return false;
  }
  // ridi: save energy, if no chance to get ball
  // if you change this, also change code in pvq_no_ball
  if((Stamina::get_state()==STAMINA_STATE_RESERVE) &&
     (WSinfo::ball->pos.getX() >20) &&
     ((steps2go.me > steps2go.opponent + 2) || steps2go.opponent == 0)){
    DBLOG_POL(BASELEVEL+0,"PJ go2ball: Stamina low, my steps to ball "<<steps2go.me
	    <<" op is faster "<<steps2go.opponent<<" ->FACE_BALL");
#if 0
    DBLOG_ERR(0,"PJ go2ball: Stamina low, my steps to ball "<<steps2go.me
	    <<" op is faster "<<steps2go.opponent<<" ->FACE_BALL");
#endif
    face_ball->turn_to_ball();
    face_ball->get_cmd(cmd);
    return true;
  }

  int cycles2wait;
  if(Tools::shall_I_wait_for_ball(WSinfo::ball->pos, WSinfo::ball->vel,cycles2wait) == true){
    //    if(steps2go.me +2 <= steps2go.opponent &&
       //       steps2go.me >1){ // if Ball is close, start intercept
    // steps2go.me >-1){ // if Ball is close, start intercept
    if(cycles2wait +2 <= steps2go.opponent){
      if(WSinfo::me->pos.distance(WSinfo::ball->pos) < ServerOptions::visible_distance ||
	 WSinfo::me->pos.distance((WSinfo::ball->pos+WSinfo::ball->vel)) < ServerOptions::visible_distance-.2){
	DBLOG_POL(0,"Ball is coming towards me and I can feel it now or next time step, let's turn forward!");
	if(WSinfo::me->pos.getX() < 35)
	  face_ball->turn_in_dir(ANGLE(0));
	else
	  face_ball->turn_in_dir((Vector(52,0)-WSinfo::me->pos).ARG());
	face_ball->get_cmd(cmd);
	return true;
      }

      DBLOG_POL(0,"I am the fastest, but ball comes directly to me and no op. close to me -> just face ball");
      face_ball->turn_to_ball();
      face_ball->get_cmd(cmd);
      return true;
    }
    DBLOG_POL(0,"I am the fastest, ball comes directly, but opponent needs only 1 step more than me -> intercept");
  }
  return intercept_ball(cmd);
}


bool Noball03_Attack::intercept_ball(Cmd &cmd){
  mdpInfo::set_my_intention(DECISION_TYPE_INTERCEPTBALL);
  neurointercept->get_cmd(cmd);
  return true;
}



/** test_offside()
    if player is in offside this function will initiate a Neuro_Go2Pos 
    move to disable offside position
*/
bool Noball03_Attack::test_offside(Cmd & cmd){

  if(WSinfo::me->pos.getX() > critical_offside_line){
    Vector targetpos;
    targetpos.setXY( WSinfo::me->pos.getX() -10., WSinfo::me->pos.getY() );
    PlayerSet pset= WSinfo::valid_teammates_without_me;
    Quadrangle2d check_area = Quadrangle2d(WSinfo::me->pos, targetpos, 10.);
    //  DBLOG_DRAW(0,  C2D(endofregion.x, endofregion.y, 1., "red")); 
    DBLOG_DRAW(0, check_area );
    pset.keep_players_in(check_area);
    pset.keep_players_in_circle(WSinfo::ball->pos, 2.0);
    if(pset.num >0){
      DBLOG_POL(0,"Disable offside: Ballholder within my target region -> modify target");
      targetpos.setX( WSinfo::me->pos.getX() );
      if(pset[0]->pos.getY() > targetpos.getY())
	targetpos.setY( WSinfo::me->pos.getY() - 10 );
      else
	targetpos.setY( WSinfo::me->pos.getY() + 10 );
    }

    DBLOG_POL(BASELEVEL+0, << "noballattack: Disable Offside. My Pos"<<WSinfo::me->pos
	      <<" offence line "<< critical_offside_line<<" mytarget: "<<targetpos);
    mdpInfo::set_my_intention(DECISION_TYPE_LEAVE_OFFSIDE,
			      (critical_offside_line));
    // go one cycle to a position behind my current position
    go2pos->set_target( targetpos );
    go2pos->get_cmd(cmd);
    return true;
  }
  return false;
}


bool Noball03_Attack::shall_I_go(const Vector target){
  int stamina = Stamina::get_state();

  if(WSmemory::team_last_at_ball() != 0 && // other team attacks
     WSinfo::ball->pos.getX() < WSinfo::me->pos.getX() && // ball is already behind me
     target.getX() > WSinfo::me->pos.getX()){ // target lies before me
    return false;
  }

  if(stamina == STAMINA_STATE_FULL) // I can go wherever I want
    return true;

  if(stamina == STAMINA_STATE_OK){// in any other stamina state, think before acting
    if(my_role == 2 && // I'm an attacker
       WSinfo::me->pos.getX() < target.getX() - 2. &&
       WSinfo::me->pos.distance(WSinfo::ball->pos) < 30. &&
       WSinfo::ball->pos.getX() > WSinfo::me->pos.getX() - 20.){ // ball is before me or not far behind me
      return true;
    }
    else if(WSinfo::me->pos.distance(target) > 6. &&  // I'm not close 2 target
	    WSinfo::ball->pos.getX() > WSinfo::me->pos.getX() - 5.){ // ball is before me or not far behind me
      return true;
    }
  }

  if(stamina == STAMINA_STATE_ECONOMY || stamina == STAMINA_STATE_RESERVE ){// stamina low
    if(WSinfo::me->pos.distance(target) > 10. &&  // I'm not close 2 target
       WSinfo::ball->pos.getX() > WSinfo::me->pos.getX() - 5.){ // ball is before me or not far behind me
      return true;
    }
  }
  
  return false;
}

bool Noball03_Attack::go2pos_withcare(Cmd &cmd, const Vector target){

  LOG_DEB(0, << "target in go2pos_withcare " << target);


  if((WSinfo::me->pos -target).norm()<10){
    // first check if there's someone between me and my target
    PlayerSet pset= WSinfo::valid_teammates_without_me;
    Vector endofregion = target - WSinfo::me->pos;
    if(endofregion.norm() > 3 * ServerOptions::kickable_area)
      endofregion.normalize(3 * ServerOptions::kickable_area);
    endofregion += WSinfo::me->pos;
    Quadrangle2d check_area = Quadrangle2d(WSinfo::me->pos, endofregion, 4 * ServerOptions::kickable_area);
    DBLOG_DRAW(1, check_area );
    pset.keep_players_in(check_area);
    if(pset.num >0){
      DBLOG_POL(1,"There's a teammate on my way to my target, Wait and See ");
      return do_waitandsee(cmd);
    }
  }
  go2pos->set_target(target);
  if(go2pos->get_cmd(cmd) == false){ // no cmd was set; probably close enough
    return do_waitandsee(cmd);
  }
  return true;
}

bool Noball03_Attack::go2pos_economical(Cmd &cmd, const Vector target){

  if(should_care4clearance && // I should probably help a teammate in trouble
     target.distance(WSinfo::ball->pos)>  // and new target is farther away from ballpos
     WSinfo::me->pos.distance(WSinfo::ball->pos)){
    DBLOG_POL(0,"Go2Pos economical -> care 4 Clearance");
    mdpInfo::set_my_intention( DECISION_TYPE_EXPECT_PASS);
    face_ball->turn_to_ball();
    face_ball->get_cmd(cmd);
    return true;
  }

  if(shall_I_go(target) == true){
    DBLOG_POL(0,"No need to save stamina or positioning urgent. state : "<<Stamina::get_state());
    go2pos_withcare(cmd,target);
    return true;
  }

  DBLOG_POL(0,"saving stamina or currently go2pos is not possible state : "<<Stamina::get_state());
  do_waitandsee(cmd);
  return true;
}

bool Noball03_Attack::do_waitandsee(Cmd &cmd){
  mdpInfo::set_my_intention(DECISION_TYPE_WAITANDSEE);
  if(WSinfo::me->pos.getX() + 3. > critical_offside_line){
    DBLOG_POL(0, << "noballattack: DO WAIT AND SEE towards 180degree");
    double desired_turn = PI - WSinfo::me->ang.get_value();
    basiccmd->set_turn_inertia(desired_turn);
    basiccmd->get_cmd(cmd);
    return true;
  }
  else{
    DBLOG_POL(0, << "noballattack - DO WAIT AND SEE Face Ball");
    face_ball->turn_to_ball();
    face_ball->get_cmd(cmd);
    Tools::set_neck_request(NECK_REQ_NONE);  // delete neck request; turn neck should be free!
  }
  return true;
}



/** test_default()
    initiated default move if all test_... functions return null
*/
bool Noball03_Attack::test_default(Cmd &cmd){
  // this is the default move 
  //mdpInfo::set_my_intention(DECISION_TYPE_FACE_BALL);
  return do_waitandsee(cmd);
}


bool Noball03_Attack::is_position_covered(const Vector pos){
  PlayerSet pset= WSinfo::valid_opponents;
  pset.keep_players_in_circle(pos, 1.8 * WSinfo::me->kick_radius);
  if(pset.num == 0) // teammate is not attacked
    return false;
  return true;
}


void Noball03_Attack::update_pinfo(XYRectangle2d rect, const Vector basepos, Vector *pos_arr, int size){
  //pinfo.basepos = basepos; // update
  if(pinfo.valid_at == WSinfo::ws->time-1){ // target was valid at last time step, just take
    pinfo.valid_at = WSinfo::ws->time;
    return;
  }
  select_new_pinfo_pos(rect, basepos, pos_arr, size);
  //pinfo.relpos = Vector(0,0); // default
  pinfo.valid_at = WSinfo::ws->time;
}

Vector Noball03_Attack::select_good_position(XYRectangle2d rect, const Vector basepos, Vector *pos_arr, int size){

  //first is base position, second and third are relativ  positions
  update_pinfo(rect, basepos, pos_arr, size);
  Vector current_targetpos = pinfo.basepos + pinfo.relpos;
  DBLOG_POL(BASELEVEL+0,<< _2D << C2D(current_targetpos.x, current_targetpos.y ,.3,"red")); 
  DBLOG_POL(0,<<"Select good position, rel target: "<<pinfo.relpos);
  if(WSinfo::me->pos.distance(current_targetpos) > WSinfo::me->kick_radius) // I am still not at target
    return current_targetpos; // old target is new target
  // I am close 2 my target
  if(!is_position_free(current_targetpos)){
    DBLOG_POL(0,<<"current targetpos is covered, select a new one");
    select_new_pinfo_pos(rect, basepos, pos_arr, size);
  } else
    select_new_pinfo_pos(rect, basepos, pos_arr, size);
    
  DBLOG_POL(0,<<"Select good position,(MODIFIED) rel target: "<<pinfo.relpos);
  current_targetpos = pinfo.basepos + pinfo.relpos;
  DBLOG_POL(BASELEVEL+0,<< _2D << C2D(current_targetpos.x, current_targetpos.y ,.6,"red")); 
  return current_targetpos; // old target is new target
}


bool Noball03_Attack::choose_free_position(XYRectangle2d rect, Vector &res, Vector *pos_arr, int size) {
  for (int i = 0; i < size; i++) {
    if (rect.inside(WSinfo::me->pos + pos_arr[i]) && is_position_free(WSinfo::me->pos + pos_arr[i])) {
      res = pos_arr[i];
      return true;
    }
  }
  return false;
}

bool Noball03_Attack::is_position_free(Vector position) {
  Vector next_i_pos = Policy_Tools::next_intercept_pos();
  Vector ball_to_pos = position - next_i_pos;
  
  ball_to_pos.normalize();

  LOG_DEB(0, << "next_i_pos ist " << next_i_pos);

  LOG_DEB(0, << "ball_to_pos ist " << ball_to_pos);

  LOG_DEB(0, << "position ist " << position);

  Quadrangle2d quad(next_i_pos + 3.0 * ball_to_pos, position, 2.4);

  LOG_DEB(0, << _2D << quad);
  LOG_DEB(0, << " rechteck ang");

  
  PlayerSet pset_tmp = WSinfo::valid_opponents;

  if ((next_i_pos - position).sqr_norm() < 16.0) {
    //quad = (next_i_pos + ball_to_pos, position, 1.2);
    Circle2d circle(position, 1.8);
    pset_tmp.keep_players_in(circle);
  } else {
    pset_tmp.keep_players_in(quad);
  }

  if (pset_tmp.num > 0)
    return false;
  else {
    //LOG_DEB(0, << _2D << quad);
    return true;
  }
}

void Noball03_Attack::select_new_pinfo_pos(XYRectangle2d rect, const Vector basepos, Vector *pos_arr, int size){
  Vector result;
  result = Vector(0,0); // default position

  LOG_DEB(0, << _2D << VC2D(DeltaPositioning::get_position(WSinfo::me->number), 3.0, "#ffffff"));

  pinfo.basepos = WSinfo::me->pos;

  if (!choose_free_position(rect, result, pos_arr, size))
    pinfo.basepos = basepos; // no free position found
    

  /*
  if(rect.inside(my_pos + p1) && is_position_covered(my_pos + p1) == false)
    result = p1;
  else
    pinfo.basepos = basepos; // update
  */

  pinfo.relpos = result; // default
  pinfo.valid_at = WSinfo::ws->time;
}

bool Noball03_Attack::scoringarea_positioning_for_lwa(Cmd &cmd){
  Vector targetpos;
  
  // default: go to left corner of goal:
  targetpos.setXY( critical_offside_line, 8.0 );

  if (WSinfo::ball->pos.getY() > 10.0)
    targetpos.addToY( 4.0 );
  else if (WSinfo::ball->pos.getY() < -10.0)
    targetpos.subFromY( 4.0 );

  Vector saved_targetpos = targetpos;

  XYRectangle2d rect;
  
  if (critical_offside_line > FIELD_BORDER_X - PENALTY_AREA_LENGTH + 5.0)
    rect = XYRectangle2d(Vector(targetpos.getX(), targetpos.getY() + 4.0),
			 Vector(FIELD_BORDER_X - PENALTY_AREA_LENGTH + 2.0, targetpos.getY() - 4.0));
  else
    rect = XYRectangle2d(Vector(targetpos.getX(), targetpos.getY() + 4.0),
			 Vector(critical_offside_line - 3.0, targetpos.getY() - 4.0));

  LOG_DEB(0, _2D << rect);
		     
  // if Ball is between me and targetpos, go behind ball
  if(WSinfo::ball->pos.getY() >7 && WSinfo::ball->pos.getY() < WSinfo::me->pos.getY()
     && WSinfo::ball->pos.getX() > critical_offside_line - 6.)
    targetpos = Vector(WSinfo::ball->pos.getX() - 10, WSinfo::ball->pos.getY());

  Vector candidate_positions[] = {Vector(5,0), Vector(0,- 5), Vector(-4, -4), Vector(-5, 0),  Vector(4, -4)};


  targetpos = select_good_position(rect, targetpos, candidate_positions, 5);


  if (Stamina::get_state() == STAMINA_STATE_ECONOMY ||
      Stamina::get_state() == STAMINA_STATE_RESERVE) {
    if (((saved_targetpos - targetpos).norm() < 3.0 &&
	(WSinfo::me->pos - saved_targetpos).norm() > 3.0) ||
	!rect.inside(WSinfo::me->pos)) {
      go2pos_withcare(cmd, Vector(saved_targetpos.getX()-2.0, saved_targetpos.getY()));
      return true;
    } 
    return do_waitandsee(cmd);
  }
    
  go2pos_withcare(cmd,targetpos);
  return true;
}
  
bool Noball03_Attack::scoringarea_positioning_for_rwa(Cmd &cmd){
  Vector targetpos;
  
  // default: go to right corner of goal:
  targetpos.setXY( critical_offside_line, -8 );

  if (WSinfo::ball->pos.getY() > 10.0)
    targetpos.addToY( 4.0 );
  else if (WSinfo::ball->pos.getY() < -10.0)
    targetpos.subFromY( 4.0 );

  Vector saved_targetpos = targetpos;

  XYRectangle2d rect;

  if (critical_offside_line > FIELD_BORDER_X - PENALTY_AREA_LENGTH + 5.0)
    rect = XYRectangle2d(Vector(targetpos.getX(), targetpos.getY() + 4.0),
			 Vector(FIELD_BORDER_X - PENALTY_AREA_LENGTH + 2.0, targetpos.getY() - 4.0));
  else
    rect = XYRectangle2d(Vector(targetpos.getX(), targetpos.getY() + 4.0),
			 Vector(critical_offside_line - 3.0, targetpos.getY() - 4.0));

  LOG_DEB(0, _2D << rect);

  // if Ball is between me and targetpos, go behind ball
  if(WSinfo::ball->pos.getY() <-7 && WSinfo::ball->pos.getY() > WSinfo::me->pos.getY()
     && WSinfo::ball->pos.getX() > critical_offside_line - 6.)
    targetpos = Vector(WSinfo::ball->pos.getX() - 10, WSinfo::ball->pos.getY());

  Vector candidate_positions[] = {Vector(5, 0), Vector(0,+ 5), Vector(-4, 4), Vector(4, 4), Vector(-5, 0)};

  targetpos = select_good_position(rect, targetpos, candidate_positions, 5);
  
  if (Stamina::get_state() == STAMINA_STATE_ECONOMY ||
      Stamina::get_state() == STAMINA_STATE_RESERVE) {
    if (((saved_targetpos - targetpos).norm() < 3.0 && (WSinfo::me->pos - saved_targetpos).norm() > 3.0) ||
	!rect.inside(WSinfo::me->pos)) {
      go2pos_withcare(cmd, Vector(saved_targetpos.getX()-2.0, saved_targetpos.getY()));
      return true;
    } 
    return do_waitandsee(cmd);
  }
  
  go2pos_withcare(cmd,targetpos);
  return true;
}


bool Noball03_Attack::scoringarea_positioning_for_ma(Cmd &cmd){
  Vector targetpos;

  // default: go to left corner of goal:
  targetpos.setXY( critical_offside_line, 0 );

  Vector p1, p2, p3, p4, p5; // reserve position

  XYRectangle2d rect;

  if (WSinfo::ball->pos.getY() > 10.0)
    targetpos.addToY( 4.0 );
  else if (WSinfo::ball->pos.getY() < -10.0)
    targetpos.subFromY( 4.0 );

  if (critical_offside_line > FIELD_BORDER_X - PENALTY_AREA_LENGTH + 5.0)
    rect = XYRectangle2d(Vector(targetpos.getX(), targetpos.getY() + 4.0),
			 Vector(FIELD_BORDER_X - PENALTY_AREA_LENGTH + 2.0, targetpos.getY() - 4.0));
  else
    rect = XYRectangle2d(Vector(targetpos.getX(), targetpos.getY() + 4.0),
			 Vector(critical_offside_line - 3.0, targetpos.getY() - 4.0));

  Vector saved_targetpos = targetpos;

  // if Ball is between me and targetpos, go behind ball
  if(WSinfo::ball->pos.getY() >0) {
    p1 = Vector(0, 5);
    p3 = Vector(-4, 4);
    p4 = Vector(4, 4);
  } else {
    p1 = Vector(0, -5);
    p3 = Vector(-4, -4);
    p4 = Vector(4, -4);
  }

  LOG_DEB(0, _2D << rect);

  p2 = Vector(-5, 0);
  p5 = Vector(5, 0);

  Vector candidate_positions[] = {p5, p1, p2, p3, p4};

  targetpos = select_good_position(rect, targetpos, candidate_positions, 5);
  
  if (Stamina::get_state() == STAMINA_STATE_ECONOMY ||
      Stamina::get_state() == STAMINA_STATE_RESERVE) {  
    if (((saved_targetpos - targetpos).norm() < 3.0 && (WSinfo::me->pos - saved_targetpos).norm() > 3.0) ||
	!rect.inside(WSinfo::me->pos)) {
      go2pos_withcare(cmd, Vector(saved_targetpos.getX()-2.0, saved_targetpos.getY()));
      return true;
    }  
    return do_waitandsee(cmd);
  }
  
  
  go2pos_withcare(cmd,targetpos);
  return true;
}



bool Noball03_Attack::test_scoringarea_positioning(Cmd &cmd){
  if (WSinfo::ball->pos.getX() < FIELD_BORDER_X - PENALTY_AREA_LENGTH - 8.0)
    return false;
  if (critical_offside_line < FIELD_BORDER_X - PENALTY_AREA_LENGTH)
    return false;

  if (fabs(WSinfo::ball->pos.getY()) > PENALTY_AREA_WIDTH/2.0 + 3.0)
    return false;

  if(WSinfo::me->number==9)// I'm left wing attacker
    return scoringarea_positioning_for_lwa(cmd);
  else if(WSinfo::me->number==10)// I'm left wing attacker
    return scoringarea_positioning_for_ma(cmd);
  else if(WSinfo::me->number==11)// I'm left wing attacker
    return scoringarea_positioning_for_rwa(cmd);
  else 
    return false; // nothing defined for that player
}



bool Noball03_Attack::test_support_attack_for_midfielders(Cmd &cmd){
  DBLOG_POL(0, << "Test SUPPORT ATTACK");

  if(my_role != 1)  // this applies only for midfielders
    return false;
  if(WSinfo::me->pos.getX() >FIELD_BORDER_X - 10.){
    DBLOG_POL(0, << "Support attack: I'm too advanced -> no support");
    return false;
  }
  if(WSinfo::me->pos.getX() > critical_offside_line){
    DBLOG_POL(0, << "Support attack: Before offsideline ! -> no support");
    return false;
  }
  if(WSmemory::get_his_offsideline_movement() <=0){
    DBLOG_POL(0, << "Support attack: offside line doesnt move! -> no support");
    return false;
  }


#if 0
  PlayerSet pset= WSinfo::valid_teammates_without_me;
  pset.keep_players_in_circle(WSinfo::ball->pos,5.0);
  if (pset.num ==0){ // no teammate controls the ball
    DBLOG_POL(0, << "Support attack: No teammate controls the ball ! -> no support");
    return false;
  }
#endif

  Vector suggested_pos;
  const double y_dist2ball =7.0;
  double x_dist2ball =1.0;

  if(WSinfo::me->number==6){// I'm left midfieler
    if(WSinfo::ball->pos.getY() <10. ||
       WSinfo::ball->pos.getX() < WSinfo::me->pos.getX())
      return false;
    
    suggested_pos.setX( FIELD_BORDER_X - 15. );
    suggested_pos.setY( my_homepos.getY() ); // default
    if(WSinfo::ball->pos.getY() - suggested_pos.getY() < 0.75 * y_dist2ball){
      suggested_pos.subFromY( y_dist2ball );
      if(suggested_pos.getY() < 12.){
	suggested_pos.setY( 12 );
	x_dist2ball =7.0;
      }
    }
    if(WSinfo::ball->pos.getX() - suggested_pos.getX() < 0.75 * x_dist2ball)
      suggested_pos.subFromX( x_dist2ball ); // default
  }
  else if(WSinfo::me->number==8){// I'm right midfieler
    if(WSinfo::ball->pos.getY() >-10. ||
       WSinfo::ball->pos.getX() < WSinfo::me->pos.getX())
      return false;
    
    suggested_pos.setX( FIELD_BORDER_X - 15. );
    suggested_pos.setY( my_homepos.getY() ); // default
    if(fabs( suggested_pos.getY() - WSinfo::ball->pos.getY()) < 0.75 * y_dist2ball){
      suggested_pos.addToY( y_dist2ball );
      if(suggested_pos.getY() > - 12.){
	suggested_pos.setY( - 12 );
	x_dist2ball =7.0;
      }
    }
    if(WSinfo::ball->pos.getX() - suggested_pos.getX() < 0.75 * x_dist2ball)
      suggested_pos.subFromX( x_dist2ball ); // default
  }
  else
    return false; // currently, nothing is defined for other players


  DBLOG_POL(BASELEVEL+0, << "PJ: Support Attack");
  DBLOG_POL(BASELEVEL+0,<< _2D << C2D(suggested_pos.x,suggested_pos.y ,1,"orange")); 

  go2pos_withcare(cmd,suggested_pos);
  return true;
}


bool Noball03_Attack::test_support_attack(Cmd &cmd){
  if (WSmemory::team_last_at_ball() != 0) // other team attacks
    return false;
  if(my_role == 1)  // special movement for midfielders
    return test_support_attack_for_midfielders(cmd);
  if(my_role != 2)  // this applies only for attackers!
    return false;
  if(WSinfo::me->pos.getX() >40.)
    return false;
  if(WSmemory::get_his_offsideline_movement() <=0){
    DBLOG_POL(0, << "Support attack: offside line doesnt move! -> no support");
    return false;
  }
  if(WSinfo::me->pos.getX() >= critical_offside_line){
    DBLOG_POL(0, << "Support attack: Before offsideline ! -> no support");
    return false;
  }
  if(WSinfo::me->pos.getX() <15.){
    DBLOG_POL(0, << "Support attack: Too far behind ! -> no support");
    return false;
  }

  PlayerSet pset= WSinfo::valid_teammates_without_me;
  pset.keep_players_in_circle(WSinfo::ball->pos,5.0);
  if (pset.num ==0){ // no teammate controls the ball
    DBLOG_POL(0, << "Support attack: No teammate controls the ball ! -> no support");
    return false;
  }

  Vector suggested_pos;
  if(my_homepos.getY() >10){ // I'm on the left wing
    if(WSinfo::ball->pos.getY() >0.){ // ball is on 'my' side, not responsible
      DBLOG_POL(0, << "Support attack: Ballpos too close to my side  -> no support");
      return false;
    }
    suggested_pos = Vector(42,9.);
  }
  else if(my_homepos.getY() <-10){
    if(WSinfo::ball->pos.getY() <0.){ // ball is on 'my' side, not responsible
      DBLOG_POL(0, << "Support attack: Ballpos too close to my side  -> no support");      
      return false;
    }
    suggested_pos = Vector(42,-9.);
  }
  else{
    DBLOG_POL(0, << "Support attack: Homepos already close to middle  -> no extra support");      
    DBLOG_POL(BASELEVEL+0,<< _2D 
	      << C2D(suggested_pos.x,suggested_pos.y ,1,"orange")); 
    return false; // homepos already ok
  }
  DBLOG_POL(BASELEVEL+0, << "PJ: Support Attack");
  DBLOG_POL(BASELEVEL+0,<< _2D 
	  << C2D(suggested_pos.x,suggested_pos.y ,1,"orange")); 

  go2pos_withcare(cmd,suggested_pos);
  return true;
}
  

bool Noball03_Attack::is_mypos_ok(const Vector & targetpos) {
  // determine whether to move or to stay
  Vector mypos = WSinfo::me->pos;   // targetposition; default: my current position
  float max_y_tolerance = 2; // could be made player dependend
  
  if(mypos.getX() <= targetpos.getX())  // too far behind
    return false;
  if(mypos.getY() > targetpos.getY() + max_y_tolerance) // too far left
    return false;
  if(mypos.getY() < targetpos.getY() - max_y_tolerance) // too far right
    return false;
  if(mypos.getX() >critical_offside_line)
    return false;
  return true;
}




DashPosition Noball03_Attack::attack_positioning_for_middlefield_player( const DashPosition & pos ) {
  // ridi 04: attack positioning for midfield player
  // here we enter, if our team was last at ball -> we attack
  Vector tp;
  PPlayer teammate=NULL;
    
  tp.clone( pos ); // homeposition

  // First, do a correction of the homeposition for attack

  if(WSinfo::me->number == 8){ // right side
    WSinfo::get_teammate(7,teammate); // default teammate
    tp.setY( -20.0 );
    if(WSinfo::ball->pos.getY() > 0){
      tp.setY( WSinfo::ball->pos.getY() -25. ); // default, if teammate not known
      if(teammate){
	tp.setY( teammate->pos.getY() - 15. );
      }
      tp.setY( Tools::min(tp.getY(),0.0) ); // do not go further than that
    }
  } // number 8
  if(WSinfo::me->number == 6){ // left side
    WSinfo::get_teammate(7,teammate); // default teammate
    tp.setY( 20.0 );
    if(WSinfo::ball->pos.getY() < 0){
      tp.setY( WSinfo::ball->pos.getY() + 25. ); // default, if teammate not known
      if(teammate){
	tp.setY( teammate->pos.getY() + 15. );
      }
      tp.setY( Tools::max(tp.getY(),0.0) ); // do not go further than that
    }
  } // number 6
  if(WSinfo::me->number == 7){ // centre midfielder
    if(WSinfo::ball->pos.getY() <0){ // Ball is on the right side
      tp.setY( WSinfo::ball->pos.getY() + 15.0 ); // stay 15 meters left of Ball
      tp.setY( Tools::max(tp.getY(), -12.0) ); // do not go further than 15m left
    }
    else{ // Ball is on left side
      tp.setY( WSinfo::ball->pos.getY() - 15.0 ); // stay 15 meters left of Ball
      tp.setY( Tools::min(tp.getY(), 12.0) ); // do not go further than 15m left
    }
  }

  // now check, if I should find a better position within my region if I could probably get a pass
  if(WSinfo::me->pos.distance(WSinfo::ball->pos) < 35.){
#define MAXPOSITIONS 20
    Vector testpos[MAXPOSITIONS];
    int i=0;
    float y_variation = 5.; // could be made player dependend
    float x_variation = 4.; // could be made player dependend

    if(is_mypos_ok(tp)) // if it's possible, stay where you are.
      testpos[i++] = WSinfo::me->pos;  
    testpos[i++] = Vector(tp.getX() + x_variation, tp.getY());
    testpos[i++] = Vector(tp.getX(),               tp.getY());
    testpos[i++] = Vector(tp.getX() - x_variation, tp.getY());
    testpos[i++] = Vector(tp.getX() + x_variation, tp.getY() + y_variation);
    testpos[i++] = Vector(tp.getX(),               tp.getY() + y_variation);
    testpos[i++] = Vector(tp.getX() - x_variation, tp.getY() + y_variation);
    testpos[i++] = Vector(tp.getX() + x_variation, tp.getY() - y_variation);
    testpos[i++] = Vector(tp.getX(),               tp.getY() - y_variation);
    testpos[i++] = Vector(tp.getX() - x_variation, tp.getY() - y_variation);
    testpos[i++] = Vector(tp.getX() - x_variation, tp.getY() + 1.5 * y_variation);
    testpos[i++] = Vector(tp.getX() - x_variation, tp.getY() - 1.5 * y_variation);

    Tools::get_optimal_position(tp,testpos, i, NULL);

  }  // end I am a potential candidate for a pass

  LOG_POL(0,<< _2D << VC2D(tp ,1.5,"red"));
  LOG_POL(0,<< _2D << VC2D(tp ,1.4,"blue"));
  LOG_POL(0,"Attack positioning for Midfielder. Best position  "<< tp);

  DashPosition new_pos;
  new_pos.clone( tp );


  if(WSinfo::me->pos.distance(new_pos) < 1.0){
    new_pos.dash_power = 0.0;
  }  
  if(WSinfo::ws->play_mode != PM_PlayOn && WSinfo::me->pos.distance(new_pos) < 4.0){
    new_pos.dash_power = 0.0;
  }
  return new_pos;
}


DashPosition Noball03_Attack::positioning_for_middlefield_player( const DashPosition & pos ) {

  if(WSmemory::team_last_at_ball() == 0) // we attack
    return attack_positioning_for_middlefield_player(pos ) ;



  DashPosition new_pos = pos;
  
  PlayerSet pset_tmp;
  PPlayer p_tmp;

  // Annahme: wenn wir hier reinkommen, sind wir immer in der Verteidigung

  pset_tmp = WSinfo::valid_opponents;
  p_tmp = pset_tmp.closest_player_to_point(pos);
  
  int opp = -1;
  if (p_tmp != NULL)
    opp = p_tmp->number;

  if(opp != -1){
    Vector lfp = Tools::get_Lotfuss(WSinfo::ball->pos, p_tmp->pos, WSinfo::me->pos);
    if(lfp.distance(pos)<pos.radius && 
       lfp.distance(WSinfo::ball->pos) < p_tmp->pos.distance(WSinfo::ball->pos) &&
       lfp.distance(p_tmp->pos) < p_tmp->pos.distance(WSinfo::ball->pos)) {
      //Zielposition liegt innerhalb des Radiuses
      //        cout << "\nPlayer: " << WSinfo::me->number << " verhindert Pass.";
      new_pos.clone( lfp );
    } else{
      //ich bin hinter dem Spieler, also gehe auf herkoemmliche Weise zwischen ihn und den Ball
      Vector a = WSinfo::ball->pos - p_tmp->pos;
      Vector b = p_tmp->pos + 0.1 * a;
      if(b.distance(pos) < pos.radius){
	//          cout << "\nPlayer: " << WSinfo::me->number << " verhindert Pass.";
	new_pos.clone( b );
      }
    }
  }

  if(WSinfo::me->pos.distance(new_pos) < 1.0){
    new_pos.dash_power = 0.0;
  }  
  return new_pos;
}


DashPosition Noball03_Attack::positioning_for_offence_player( const DashPosition & pos ) {  
#if 0 // ridi: code before Lisbon 04
  DashPosition ret_pos = pos; // default
  if ( ret_pos.x > critical_offside_line ) //but avoid offside positions
    ret_pos.x= critical_offside_line;
  return ret_pos;
#endif
  DashPosition ret_pos = pos; // default
  if(WSinfo::me->pos.distance(WSinfo::ball->pos) > 35.)
    return ret_pos; // don't worry about clever positioning

  Vector tp(pos);

  Vector testpos[20];
  int i=0;
  float y_var = 5.; // could be made player dependend
  float x_var = 2.; // could be made player dependend

  if(WSinfo::me->number == 9){ // iam left wing attacker
       testpos[i++] = Vector(tp.getX(),tp.getY());
       testpos[i++] = Vector(tp.getX(),tp.getY() + y_var);
       testpos[i++] = Vector(tp.getX(),tp.getY() - y_var);
       testpos[i++] = Vector(tp.getX()-x_var,tp.getY());
  }
  else   if(WSinfo::me->number == 11){ // iam right wing attacker
       testpos[i++] = Vector(tp.getX(),tp.getY());
       testpos[i++] = Vector(tp.getX(),tp.getY() - y_var);
       testpos[i++] = Vector(tp.getX(),tp.getY() + y_var);
       testpos[i++] = Vector(tp.getX()-x_var,tp.getY());
  }
  else{  // default: middle
       testpos[i++] = Vector(tp.getX(),tp.getY());
       testpos[i++] = Vector(tp.getX(),tp.getY() - y_var);
       testpos[i++] = Vector(tp.getX(),tp.getY() + y_var);
       testpos[i++] = Vector(tp.getX()-x_var,tp.getY());
  }

  Tools::get_optimal_position(tp,testpos, i, NULL);

  ret_pos.clone( tp );
  return ret_pos;
}




bool Noball03_Attack::test_analytical_positioning(Cmd &cmd){

  if(Iam_a_neuroattacker == true){ // this should be done for attackers only; all others go home!
    DBLOG_POL(0,"I'm a neuro attacker, no analytical positiong");
    return false;    
  }
  
  DashPosition my_form;
  DashPosition my_fine;
  my_form.clone( my_homepos );
  my_form.dash_power = 100;
  
  if (my_role == 1)
    my_fine = positioning_for_middlefield_player( my_form );
  if (my_role == 2)
    my_fine = positioning_for_offence_player( my_form );


  if((my_fine.dash_power == 0.0) ||
     (WSinfo::me->pos.distance(Vector(my_fine)) < 2.0)){ // close enough
    DBLOG_POL(0,<<"Analytical positioning: close enough to target, face ball ");

    DBLOG_POL(BASELEVEL+0,<< _2D << C2D(my_fine.x, my_fine.y ,1.3,"aaaaaa")); 
    return do_waitandsee(cmd);
  }
  
  DBLOG_POL(BASELEVEL+0,<< _2D << C2D(my_fine.x, my_fine.y ,1.3,"red")); 

  return go2pos_economical(cmd,my_fine);
}


void Noball03_Attack::determine_positioning_constraints(XYRectangle2d *constraints_P, Vector *home_positions_P) {
#ifdef TRAINING
  for (int i = 0; i < 11; i++) {
      home_positions_P[i] = Vector(0,0);
      constraints_P[i] = XYRectangle2d( Vector(0, 0),
					Vector(5, 5) );
  }

#else
  for (int i = 0; i < 5; i++) {
    home_positions_P[i] = DeltaPositioning::attack433.get_grid_pos(i+1);
    constraints_P[i] = XYRectangle2d( Vector( home_positions_P[i].getX() - 5.0, home_positions_P[i].getY() - 5.0 ),
				      Vector( home_positions_P[i].getX() + 5.0, home_positions_P[i].getY() + 5.0 ) );
  }

  // ++++++ midfielders ++++++
  for (int i = 5; i < 8; i++) {
    home_positions_P[i] = DeltaPositioning::attack433.get_grid_pos(i+1);
    constraints_P[i] = XYRectangle2d( Vector( home_positions_P[i].getX() - 1.0, home_positions_P[i].getY() - 15.0),
				      Vector( home_positions_P[i].getX() + 30.0, home_positions_P[i].getY() + 15.0) );
  }
#endif

#ifdef TRAINING
  // ++++++ attackers ++++++
  if (critical_offside_line < FIELD_BORDER_X - PENALTY_AREA_LENGTH + 5.0) {

    for (int i = 1; i < 4; i++) { // 9..11
      home_positions_P[i] = DeltaPositioning::attack433.get_grid_pos(i+1 + 7);
      constraints_P[i] = XYRectangle2d( Vector( home_positions_P[i].x - 6.0, home_positions_P[i].y - 6.0),
					Vector( home_positions_P[i].x, home_positions_P[i].y + 6.0) );
    }
    for (int i = 4; i < 7; i++) { // 6..8
      home_positions_P[i] = DeltaPositioning::attack433.get_grid_pos(i + 2);
      constraints_P[i] = XYRectangle2d( Vector( home_positions_P[i].x + 8.0, home_positions_P[i].y - 6.0),
					Vector( home_positions_P[i].x + 14.0, home_positions_P[i].y + 6.0) );
      home_positions_P[i].x = home_positions_P[i].x + 11.0;
    }
  } else {

    for (int i = 1; i < 4; i++) { // 9..11
      home_positions_P[i] = DeltaPositioning::attack433.get_grid_pos(i+1 + 7);
      constraints_P[i] = XYRectangle2d( Vector( FIELD_BORDER_X - PENALTY_AREA_LENGTH, home_positions_P[i].y - 7.0),
					Vector( home_positions_P[i].x, home_positions_P[i].y + 7.0) );
    }
    for (int i = 4; i < 7; i++) { // 6..8
      home_positions_P[i] = DeltaPositioning::attack433.get_grid_pos(i + 2);
      constraints_P[i] = XYRectangle2d( Vector( home_positions_P[i].x + 8.0, home_positions_P[i].y - 7.0),
					Vector( home_positions_P[i].x + 14.0, home_positions_P[i].y + 7.0) );
      home_positions_P[i].x = home_positions_P[i].x + 11.0;
    }
  }
#else
  // ++++++ attackers ++++++
  if (critical_offside_line < FIELD_BORDER_X - PENALTY_AREA_LENGTH + 5.0)
    for (int i = 8; i < 11; i++) {
      home_positions_P[i] = DeltaPositioning::attack433.get_grid_pos(i+1);
      constraints_P[i] = XYRectangle2d( Vector( home_positions_P[i].getX() - 6.0, home_positions_P[i].getY() - 5.0),
					Vector( home_positions_P[i].getX(), home_positions_P[i].getY() + 5.0) );
    }
  else
    for (int i = 8; i < 11; i++) {
      home_positions_P[i] = DeltaPositioning::attack433.get_grid_pos(i+1);
      constraints_P[i] = XYRectangle2d( Vector( FIELD_BORDER_X - PENALTY_AREA_LENGTH + 2.0, home_positions_P[i].getY() - 5.0),
					Vector( home_positions_P[i].getX(), home_positions_P[i].getY() + 5.0) );
    }  
#endif

}


bool Noball03_Attack::test_neuro_positioning(Cmd &cmd) {

  if(neuro_positioning->am_I_neuroattacker() == false){ // this should be done for attackers only; all others go home!
    DBLOG_POL(0,"I'm not a neuro attacker, return");
    return false;    
  }


  if ( neuro_positioning->shall_player_do_neuropositioning(WSinfo::me->number, my_homepos) == false ) {
      DBLOG_POL(0,<<"Shall I do neuropositioning is FALSE ");
      return false;
  }


#if 0 // test only ridi04: cosidered harmful
  if ( neuro_positioning->shall_player_do_neuropositioning(WSinfo::me->number, my_homepos) == false ) {
    if ( WSinfo::me->pos.distance(Vector(my_homepos.x, my_homepos.y)) < 2.0 ) { // close enough
      DBLOG_POL(0,<<"Do neuropositionig is false: close to target, face ball ");
      DBLOG_POL(BASELEVEL+0,<< _2D << C2D(my_homepos.x, my_homepos.y ,1.3,"aaaaaa")); 
      return do_waitandsee(cmd);
    }
  
    DBLOG_POL(0,<<"Do neuropositionig is false: go home econmical ");
    DBLOG_POL(BASELEVEL+0,<< _2D << C2D(my_homepos.x, my_homepos.y ,1.3,"red")); 

    return go2pos_economical(cmd,my_homepos);
  }
#endif

  XYRectangle2d constraints[11];
  Vector home_positions[11];

  DBLOG_POL(0, << " NEURO_POSITIONING is active!!!");

  determine_positioning_constraints(constraints, home_positions);

  if ( (WSinfo::me->pos - target).norm() <= 1.0 || !constraints[WSinfo::me->number-1].inside(target) || 
       last_neuro_positioning < WSinfo::ws->time - 1) {

    /* *********************** call neuro positioning ************************** */
    
    if (!neuro_positioning->get_neuro_position(target, constraints, home_positions)) {
      DBLOG_POL(0, << " neuro_positioning returned false");
      return false;
    }
    
    /* ************************************************************************* */
    
  }

  DBLOG_POL(0, << " target in NEURO_POS: " << target);
  last_neuro_positioning = WSinfo::ws->time;

  if(Stamina::get_state() == STAMINA_STATE_RESERVE){
    DBLOG_POL(BASELEVEL+0,"PJ: in stamina RESERVE state -> Recover and Face Ball");
    return do_waitandsee(cmd);
  }

  /*
  if(winner.action_type == AACTION_TYPE_STAY){
    DBLOG_POL(BASELEVEL+0,"Decided to stay; check if I should go home instead");
    return false;
    }*/


  go2pos_withcare(cmd, target);
  return true;

  //  return aaction2cmd(winner,cmd);
  //  return AbstractMDP::aaction2move(winner);



}


//#define GOHOMEDASHPOWER 100

bool Noball03_Attack::test_gohome(Cmd &cmd){
  bool gohome = false;
  if(my_role == 1) { // I'm a midfielder
    if(WSinfo::me->pos.getX() < my_homepos.getX() -  midfielder_x_backward_tolerance)
      gohome = true;
    if(WSinfo::me->pos.getX() > my_homepos.getX() + 2.0)
      gohome = true;
    if(fabs(WSinfo::me->pos.getY() - my_homepos.getY()) > y_homepos_tolerance){
      gohome = true;
    }
  }
  if(my_role == 2) { // I'm an attacker
    if(WSinfo::me->pos.getX() < my_homepos.getX() -  attacker_x_backward_tolerance)
      gohome = true;
    if(fabs(WSinfo::me->pos.getY() - my_homepos.getY()) > y_homepos_tolerance){
      gohome = true;
    }
  }

  if(gohome == true){
    DBLOG_POL(BASELEVEL+0,<< _2D << C2D(my_homepos.x, my_homepos.y ,1.3,"red")); 
    return go2pos_economical(cmd,my_homepos);
  }
  DBLOG_POL(0,<<"GoHome: Not needed; already close enough");
  return false;
 
}

bool Noball03_Attack::test_wait4clearance(Cmd &cmd){

  should_care4clearance = false;
  PlayerSet pset= WSinfo::valid_teammates_without_me;
  pset.keep_players_in_circle(WSinfo::ball->pos, 3.0);
  pset.keep_and_sort_closest_players_to_point(1,WSinfo::ball->pos);
  if(pset.num == 0) // no teammate has the ball
    return false;

  Vector ballholder_pos = pset[0]->pos;
  pset= WSinfo::valid_opponents;
  pset.keep_players_in_circle(ballholder_pos, 1.8 * ServerOptions::kickable_area);
  if(pset.num == 0) // teammate is not attacked
    return false;

  if(Planning::is_player_a_passcandidate(WSinfo::me->pos)>0.0){
    DBLOG_POL(BASELEVEL+0,<<"I am a pass candidate and my teammate is attacked -> set variable ");
#if 0
    mdpInfo::set_my_intention( DECISION_TYPE_EXPECT_PASS);
    face_ball->turn_to_ball();
    face_ball->get_cmd(cmd);
    return true;
#endif
    should_care4clearance = true;
    return false;
  }
  return false;
}


