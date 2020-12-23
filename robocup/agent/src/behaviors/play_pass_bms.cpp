#include "play_pass_bms.h"
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
#include "geometry2d.h"

#include "../policy/planning.h"
#include "../policy/policy_tools.h"
#include "../policy/positioning.h"  // get_role()
#include "globaldef.h"
#include "ws_info.h"
//#include "pos_tools.h"
#include "blackboard.h"

#define LOG_INFO2(X)
#define LOG_INFO3(X)

#if 0
//#define DBLOG_POL(LLL,XXX) LOG_POL(LLL,XXX)
#define DBLOG_POL(LLL,XXX) LOG_POL(LLL,<<"PlayPass: " XXX)
#define DBLOG_DRAW(LLL,XXX) LOG_POL(LLL,<<_2D <<XXX)
#define DBLOG_ERR(LLL,XXX) LOG_ERR(LLL,XXX)
#define MYGETTIME (Tools::get_current_ms_time())
#define BASELEVEL 0 // level for logging; should be 3 for quasi non-logging
//#define LOG_DAN(YYY,XXX) LOG_DEB(YYY,XXX)
#define LOG_DAN(YYY,XXX)
#else
#define DBLOG_POL(LLL,XXX)
#define DBLOG_DRAW(LLL,XXX)
#define DBLOG_ERR(LLL,XXX) 
#define MYGETTIME (0)
#define BASELEVEL 3 // level for logging; should be 3 for quasi non-logging
#define LOG_DAN(YYY,XXX)
#endif


/* constructor method */
PlayPass::PlayPass() {
  /* read with ball params from config file*/
  cyclesI_looked2goal = 0;
  use_handicap_goalshot_test = false;
  handicap = 3.6;
  safety_handicap_sub = 0.7;
  riskyDisToPole5 = 0.3;
  riskyDisToPole30 = 1.0;
  lastTimeLookedForGoalie = -1;
  lastTimeLookedForGoal = -1;
  use_handicap_selfpasses= false;
  selfpasses_handicap = 2.0;
  selfpasses_safety_level = 3.0;
  selfpasses_SHORT_safety_level = 0.5;
  selfpasses_SHORT_max_speed = 0.71;
  reconsider_goalshot = false;
  wait_and_see_patience = 1000; // default: be very patient
  wait_and_see_clearanceline = 25; // default: if position is less than that -> clear
  flank_param = 0.1;
  success_threshold=0.8;
  last_at_ball = -100;
  at_ball_for_cycles = 0;
  at_ball_patience = 1000;
  cycles_in_waitandsee = 0;
  last_waitandsee_at = -1;
  do_refine_selfpasses = true;

  ValueParser vp(CommandLineOptions::policy_conf,"wball03_bmp");
  vp.get("use_handicap_goalshot_test", use_handicap_goalshot_test);
  vp.get("handicap", handicap);
  vp.get("safety_handicap_sub", safety_handicap_sub);
  vp.get("risky_dis_to_pole_5", riskyDisToPole5);
  vp.get("risky_dis_to_pole_30", riskyDisToPole30);
  vp.get("reconsider_goalshot",reconsider_goalshot);
  vp.get("wait_and_see_patience",wait_and_see_patience);
  vp.get("wait_and_see_clearanceline",wait_and_see_clearanceline);
  vp.get("at_ball_patience",at_ball_patience);
  vp.get("flank_param", flank_param);
  vp.get("use_handicap_selfpasses", use_handicap_selfpasses);
  vp.get("selfpasses_handicap", selfpasses_handicap);
  vp.get("selfpasses_safety_level", selfpasses_safety_level);
  vp.get("selfpasses_SHORT_safety_level",selfpasses_SHORT_safety_level);
  vp.get("selfpasses_SHORT_max_speed", selfpasses_SHORT_max_speed);
  vp.get("do_refine_selfpasses", do_refine_selfpasses);
  vp.get("success_threshold", success_threshold);
  dribble_success_threshold=success_threshold;
  vp.get("dribble_success_threshold", dribble_success_threshold);

  decisionHandicap = handicap;

  neurokick = new NeuroKick05;
  basiccmd = new BasicCmd;
  onestepkick = new OneStepKick;
  oneortwo = new OneOrTwoStepKick;
  neuro_wball = new NeuroWball;
  onetwoholdturn = new OneTwoHoldTurn;

  my_blackboard.pass_or_dribble_intention.reset();
  my_blackboard.intention.reset();
}

PlayPass::~PlayPass() {
  delete neurokick;
  delete basiccmd;
  delete onestepkick;
  delete oneortwo;
  delete onetwoholdturn;
  delete neuro_wball;
}


bool PlayPass::get_cmd(Cmd & cmd) {

  Intention intention;

  LOG_POL(BASELEVEL, << "In PLAYPASS_BMS : ");
  if(!WSinfo::is_ball_kickable())
    return false;

  if(get_intention(intention)){
    intention2cmd(intention,cmd);
    DBLOG_POL(BASELEVEL, << "PLAYPASS: intention was set! ");
    return true;
  }
  else{
    DBLOG_POL(BASELEVEL, << "PLAYPASS: no cmd was set ");
    return false;
  }

  return false;  // behaviour is currently not responsible for that case

}


bool PlayPass::get_intention(Intention &intention){

  long ms_time= MYGETTIME;
  DBLOG_POL(BASELEVEL+2, << "Entering PlayPass");

  if(last_at_ball == WSinfo::ws->time -1)
    at_ball_for_cycles ++;
  else
    at_ball_for_cycles =1;
  last_at_ball = WSinfo::ws->time;

  if(last_waitandsee_at == WSinfo::ws->time -1) // waitandsee selected again
    cycles_in_waitandsee ++;
  else
    cycles_in_waitandsee = 0; // reset

  if(cycles_in_waitandsee >0){
    DBLOG_POL(BASELEVEL,<<"cylces in wait and see: "<<cycles_in_waitandsee);
  }


  my_role= DeltaPositioning::get_role(WSinfo::me->number);
  //  my_role= 2; // test only !!!!
  // determine closest opponent (frequently needed)
  PlayerSet pset = WSinfo::valid_opponents;
  pset.keep_and_sort_closest_players_to_point(1, WSinfo::me->pos);
  if(pset.num >0)
    closest_opponent = pset[0];
  else
    closest_opponent = NULL;

  if(onestepkick->can_keep_ball_in_kickrange() == false) {
    DBLOG_POL(0,<<"WBALL03: CANNOT keep ball in kickrange");
  }


  Intention tmp, tmp_previous;

  tmp_previous = my_blackboard.pass_or_dribble_intention; // copy old intention

  if(my_blackboard.intention.get_type() == OPENING_SEQ){
    my_blackboard.pass_or_dribble_intention.reset(); // 
    DBLOG_POL(0,<<"Current intention is opening_seq: invalidated original pass intention");
  }
  else if(check_previous_intention(my_blackboard.pass_or_dribble_intention,
				   my_blackboard.pass_or_dribble_intention) == true){
    DBLOG_POL(0,<<"Pass or Dribble Intention was already set, still ok, just take it!");
  }
  
  if(my_blackboard.pass_or_dribble_intention.valid_at() != WSinfo::ws->time){ // not set yet
    if(get_pass_or_dribble_intention(tmp)){ // check for a new intention.
      if(tmp_previous.valid_at() == WSinfo::ws->time -1){// if previous intention was set in last cycle
	DBLOG_POL(0,"previous pass intention was set, dir "<<RAD2DEG(tmp_previous.kick_target.ARG().get_value()));
	if(tmp_previous.kick_target.ARG().diff(tmp.kick_target.ARG()) < 20./180. *PI){
	  tmp.set_valid_since(WSinfo::ws->time-1);  // indicate
	  DBLOG_POL(0,"current and previous pass direction have not changed too much, current: "
		    <<RAD2DEG(tmp.kick_target.ARG().get_value()));
	}
      } // previous intention was valid last cycle
      my_blackboard.pass_or_dribble_intention = tmp;
    }
    else
      my_blackboard.pass_or_dribble_intention.reset();      
  }
  DBLOG_POL(0,<<"Pass or Dribble Intention valid since "<<my_blackboard.pass_or_dribble_intention.valid_since());

  //PASS!: get_best_selfpass();
  //is_planned_pass_a_killer = is_pass_a_killer(); // attention: depends on best selfpass!

  bool result = false;
  if ( (result = check_previous_intention(my_blackboard.intention, intention)) );
  else if ( (result = test_two_teammates_control_ball(intention)) );
  else if ( (result = test_in_trouble(intention)) );
  else if ( (result = test_priority_pass(intention)) );
  //D else if ( (result = test_advance(intention)));
  //D else if ( (result = test_holdturn(intention)));
  else result = test_pass_or_dribble(intention);

  my_blackboard.intention = intention;
  check_write2blackboard();

  ms_time = MYGETTIME - ms_time;
  DBLOG_POL(BASELEVEL+1, << "WBALL03 policy needed " << ms_time << "millis to decide");
  return result;
}


bool PlayPass::check_previous_intention(Intention prev_intention, Intention  &new_intention){
  if(prev_intention.valid_at() < WSinfo::ws->time -1)
    return false;
  if(is_planned_pass_a_killer){
    if(prev_intention.get_type() != PASS &&
       prev_intention.get_type() != LAUFPASS){
      DBLOG_POL(0,<<"Check previous intention: planned pass is a killer -> RESET intention");
      return false;
    }
  }

  int targetplayer_number = prev_intention.target_player;
  if (targetplayer_number >11) targetplayer_number = 0; // invalid then
  float speed = prev_intention.kick_speed;
  Vector target = prev_intention.kick_target;
  Angle dir = (target - WSinfo::me->pos).arg(); // direction of the target
  ANGLE targetdir = prev_intention.target_body_dir; // for selfpasses
  Vector ballpos = WSinfo::ball->pos;
  //  Vector mypos = WSinfo::me->pos;
  Vector interceptpos, playerpos;  // return parameters; not considered here
  int advantage, number;
  //int mytime2intercept;  // return parameters; not considered here
  bool myteam; // return pararamter, not considered here
  Vector receiverpos; // used 
  Vector op_pos; // return parameter for selfpass, not used
  bool stillok = false;
  int risky_pass = prev_intention.risky_pass;

  switch(prev_intention.get_type()){
  case PASS:
    receiverpos = target; // default setting
    if(targetplayer_number >0){
      DBLOG_POL(BASELEVEL,
		<<"WBall03: Check previous intention: PASS: correct target, old: "<<target);
      PPlayer p= WSinfo::get_teammate_by_number(targetplayer_number);
      if(p){
	receiverpos = WSinfo::get_teammate_by_number(targetplayer_number)->pos;
	if(receiverpos.getX() > WSinfo::his_team_pos_of_offside_line()){
	  DBLOG_POL(BASELEVEL,
		    <<"WBall03: Check Pass considered harmful, offside of receiver -> RESET");
	  stillok = false;
	  break;
	}
	dir = (receiverpos - WSinfo::me->pos).arg(); // direction of the target
      }
      stillok = Planning::is_pass_successful_hetero(ballpos, speed, dir, p, interceptpos, advantage); // use player info
    }
    else{ // player number not known
      stillok = Planning::is_pass_successful(ballpos, speed, dir, receiverpos, interceptpos, advantage); // 
    }
    if(stillok == false){
      DBLOG_POL(BASELEVEL,
		<<"WBall03: Check previous intention: Pass considered harmful -> RESET");
    }
    else{
      DBLOG_POL(BASELEVEL,
		<<"WBall03: Pass considered ok, speed "<<speed<<" dir "<<RAD2DEG(dir));
    }
    target = receiverpos;
    break;
  case LAUFPASS:
    stillok = Planning::is_laufpass_successful(ballpos,speed, 
					       dir,interceptpos,advantage,
					       number,playerpos,risky_pass);
    if(playerpos.getX() > WSinfo::his_team_pos_of_offside_line()){
      DBLOG_POL(BASELEVEL,
		<<"WBall03: Check Laufpass considered harmful, offside of receiver -> RESET");
      stillok = false;
      break;
    }

    if (advantage < 2)
      stillok = false;
    if(stillok == false) {
      DBLOG_POL(BASELEVEL,
		<<"WBall03: Check previous intention: Pass considered harmful -> RESET");
    }
    break;
  case KICKNRUSH:
    stillok = true; // kicknrush remains ok, since this is default action!
    break; 
  case OPENING_SEQ:
    stillok = true; // kicknrush remains ok, since this is default action!
    if(onetwoholdturn->is_holdturn_safe() == false){
      DBLOG_POL(BASELEVEL, <<"WBALL 03: opening is interrupt, since holdturn not safe");
      stillok = false; 
    }  
    Policy_Tools::earliest_intercept_pos(WSinfo::me->pos,speed,dir,
					 interceptpos, myteam, number, advantage);
    if(interceptpos.getX() < WSinfo::his_team_pos_of_offside_line()){
      DBLOG_POL(BASELEVEL, <<"WBALL 03: opening is interrupt, since ipos of pass < offsideline");
      stillok = false; 
    }
    break; 
  case HOLDTURN:
  case TURN_AND_DASH:
    stillok = false;
    break;
  default:
    stillok = false;
    break;
  }
  if(stillok){
    DBLOG_POL(0,<<"Previous Intention still ok, just take it!");
    new_intention.confirm_intention(prev_intention,WSinfo::ws->time );
    new_intention.correct_target(target);
    new_intention.correct_speed(speed);
    DBLOG_DRAW(BASELEVEL, VL2D(WSinfo::ball->pos,target,"lightblue"));
    return true;
  }
  return false;
}


bool PlayPass::get_pass_or_dribble_intention(Intention &intention){
  AState current_state;
  AbstractMDP::copy_mdp2astate(current_state);

  return get_pass_or_dribble_intention(intention, current_state);
}

bool PlayPass::get_pass_or_dribble_intention(Intention &intention, AState &state){
  AAction best_aaction;

  intention.reset();

  if (neuro_wball->evaluate_passes_and_dribblings(best_aaction, state) == false) // nothing found
    return false;
	// CIJAT OSAKA
	current_advantage = best_aaction.advantage;

  if(best_aaction.action_type == AACTION_TYPE_WAITANDSEE)
    last_waitandsee_at = WSinfo::ws->time;
  
  aaction2intention(best_aaction, intention);

  // found a pass or dribbling
  return true;
}


bool PlayPass::test_opening_seq(Intention &intention)
{
  if(onetwoholdturn->is_holdturn_safe() == false){
    DBLOG_POL(BASELEVEL, <<"WBALL 03: Opening not possible, since holdturn not safe");
    return false;
  }

  if (my_blackboard.pass_or_dribble_intention.valid_at() == WSinfo::ws->time){ // pass or dribble int. is set
    if(my_blackboard.pass_or_dribble_intention.kick_target.getX()>WSinfo::me->pos.getX()){
      DBLOG_POL(BASELEVEL+0,<<"WBALL03 No Kickrush, Pass is planned, and kick target before me!"
		<<my_blackboard.pass_or_dribble_intention.kick_target);
      return false;
    }
  }

#if 0
  if(WSinfo::me->pos.getX() <-10){
    DBLOG_POL(BASELEVEL+0,<<"WBALL03 No Opening, too far back");
    return false;
  }
#endif
  if(WSinfo::me->pos.getX() >ServerOptions::pitch_length - 20){
    DBLOG_POL(BASELEVEL+0,<<"WBALL03 No Opening, too far advanced");
    return false;
  }

  //  DBLOG_POL(BASELEVEL, <<"WBALL 03: Start Opening Seq");

  Vector target;
  double speed, dir;
  Vector ipos; // intercept position
  int advantage, closest_teammate;
  
  get_opening_pass_params(speed,dir,ipos,advantage,closest_teammate);
  if(speed <0)
    return false; // no kickrush params found

  if(ipos.getX() < WSinfo::his_team_pos_of_offside_line()){
    DBLOG_POL(BASELEVEL+0,<<"WBALL03 Opening sequence NOT possible, best ipos "<<ipos
	      <<" behind offsideline "<<WSinfo::his_team_pos_of_offside_line());
    return false;
  }
    
  target.init_polar(speed/(1-ServerOptions::ball_decay),dir);
  target += WSinfo::me->pos;
  
  //  speed = 3.0; // artifical speed to communicate opening seq.
  intention.set_opening_seq(target,speed, WSinfo::ws->time);
  DBLOG_POL(BASELEVEL+0,<<"WBALL03  Start Opening sequence to "<<target<<" dir "
	    <<RAD2DEG(dir)<<" w.speed "<<speed
	    <<" best ipos "<<ipos
	    <<" before offsideline "<<WSinfo::his_team_pos_of_offside_line());
  //DBLOG_POL(0, << _2D << C2D(target.x, target.y, 1, "#00ffff"));
  return true;
}



/** selects one player who is responsible for the ball */
bool PlayPass::test_two_teammates_control_ball(Intention &intention){
  // test if Ball is in kickrange of me and a teammate!

  Vector ballpos = WSinfo::ball->pos;
  PPlayer p_tmp = WSinfo::valid_teammates_without_me.closest_player_to_point(ballpos);
  if (p_tmp == NULL) return false;
  int teammate = p_tmp->number;
  float dist = (p_tmp->pos - ballpos).norm();

  if((dist < .95*ServerOptions::catchable_area_l) && (teammate == WSinfo::ws->my_goalie_number)){
    LOG_ERR(0,<<"WBALL03: Our goalie has the ball, I could kick - Maybe I should back up?!");
    DBLOG_POL(BASELEVEL,<<"WBALL03: Our goalie has the ball, I could kick - Maybe I should back up?! (not yet implemented)");
  }

  if((dist < ServerOptions::kickable_area) && (WSinfo::me->number > teammate)){
    LOG_ERR(0,<<"WBALL03: Another teammate with lower number  has the ball, I could kick - Maybe I should back up?!");
    DBLOG_POL(BASELEVEL,<<"WBALL03: I could kick Another teammate with lower number  has the ball - Maybe I should back up?! (not yet implemented)");
  }

  intention.set_backup(WSinfo::ws->time);
  return false;
}



bool PlayPass::test_priority_pass(Intention &intention){

  // first, check for killer passes
  if(is_planned_pass_a_killer){ // planned pass is a killer
    Vector respos = my_blackboard.pass_or_dribble_intention.resultingpos;
    if(fabs(respos.getY())<8 &&
       respos.getX()>42){ // in hot scoring area
      DBLOG_POL(BASELEVEL, <<"Teammate in hot scoring area -> pass IMMEDIATELY");
      Vector target;
      double speed;
      my_blackboard.pass_or_dribble_intention.get_kick_info(speed, target);
      ANGLE targetdir = (target-WSinfo::me->pos).ARG();
      my_blackboard.neckreq.set_request(NECK_REQ_LOOKINDIRECTION,targetdir.get_value());
      intention = my_blackboard.pass_or_dribble_intention;
      return true;
    } 

    if(my_blackboard.pass_or_dribble_intention.valid_since() < WSinfo::ws->time){ // it has been already communicated
      // this might be replaced by == -> play killer passes immediately, if I have looked in the direction
      Vector target;
      double speed;
      my_blackboard.pass_or_dribble_intention.get_kick_info(speed, target);
      ANGLE targetdir = (target-WSinfo::me->pos).ARG();
      if(WSmemory::last_seen_in_dir(targetdir) >1){ 
	my_blackboard.neckreq.set_request(NECK_REQ_LOOKINDIRECTION,targetdir.get_value());
	DBLOG_POL(0,<<"Planned pass is a killer, but I haven't seen in targetdir "<<RAD2DEG(targetdir.get_value())<<" since "
		  <<WSmemory::last_seen_in_dir(targetdir)<<" cycles -> wait and look ");
	if(onetwoholdturn->is_holdturn_safe() == false){
	  DBLOG_POL(BASELEVEL, <<"Priority Pass: HoldTurn NOT possible. Not Safe");
	  return false; // alternatively: immediately play killer pass then
	}
	intention.set_holdturn(targetdir, WSinfo::ws->time);
	return true;
      }
      DBLOG_POL(BASELEVEL, <<"pass intention is a KILLER and I looked into direction -> Priority pass");
      intention = my_blackboard.pass_or_dribble_intention;
      return true;
    } // end pass intention was valid for at least 1 cycle
    else{ // pass intention was just set -> wait
      DBLOG_POL(0,<<"Planned pass is a killer, wait one cycle");
      if(onetwoholdturn->is_holdturn_safe() == false){
	DBLOG_POL(BASELEVEL, <<"Priority Pass: HoldTurn NOT possible. Not Safe");
	return false; // alternatively: immediately play killer pass then
      }
      intention.set_holdturn(ANGLE(0.), WSinfo::ws->time);
      return true;
    }
  }


  // now check passing due to other situations
  const double scoring_area_width = 20.;
  const double scoring_area_depth = 15.;


  const XYRectangle2d scoring_area( Vector(FIELD_BORDER_X - scoring_area_depth, -scoring_area_width*0.5),
				    Vector(FIELD_BORDER_X, scoring_area_width * 0.5)); 

  DBLOG_DRAW(0,scoring_area);
  if(scoring_area.inside(WSinfo::me->pos)){
    DBLOG_POL(0,"I am within scoring area, do not do prioriy pass");
    return false;
  }

  // I am not in scoring area

  PlayerSet pset= WSinfo::valid_opponents;
  pset.keep_players_in_circle(WSinfo::me->pos, 2 * ServerOptions::kickable_area + 2 * ServerOptions::player_speed_max); 
  bool I_am_attacked = (pset.num >0);
  
  pset= WSinfo::valid_opponents;
  Vector endofregion;
  endofregion.init_polar(5, 0);
  endofregion += WSinfo::me->pos;
  Quadrangle2d check_area = Quadrangle2d(WSinfo::me->pos, endofregion, 5);
  //  DBLOG_DRAW(1, check_area );
  pset.keep_players_in(check_area);
  bool area_infront_is_free = (pset.num == 0);
  
  if((I_am_attacked) && // opponent is close and pass intention communicated
     (my_blackboard.pass_or_dribble_intention.valid_since() < WSinfo::ws->time)){ 

    if(my_role == 0 ||
       my_blackboard.pass_or_dribble_intention.kick_target.getX() > WSinfo::me->pos.getX() ||
       area_infront_is_free == false){
      DBLOG_POL(BASELEVEL, <<"I am attacked, pass intention is set -> Priority pass");
      intention = my_blackboard.pass_or_dribble_intention;
      return true;
    }
  }

  return false;

}

/** What to do if two opponents attack me */
bool PlayPass::test_in_trouble(Intention &intention){
  // test for situation 'a': Ball is in kickrange of me and opponent!

  Vector target;


  PlayerSet pset= WSinfo::valid_opponents;
  pset.keep_players_in_circle(WSinfo::ball->pos, ServerOptions::kickable_area); 
  // considr only close ops. for correct
  pset.keep_and_sort_closest_players_to_point(1,WSinfo::ball->pos);
  if ( pset.num == 0 )
    return false; // no op. has ball in kickrange



  DBLOG_POL(BASELEVEL, << "WBALL03: Ball is in kickrange of me and opponent "<<pset[0]->number);
  if(WSinfo::me->pos.getX() >-10){
    DBLOG_POL(BASELEVEL, << "WBALL03: I dont care for trouble. I'm too far advanced, just cont.");
    return false;
  }

  // Danger: Opponent can also kick -> kick ball straight away
  if(WSinfo::me->pos.getX() <10){ // I am far from opponents goal
    target = Vector(52.5,0);
  }
  else if(WSinfo::me->pos.getX() <43.5){ // I have more than 7 meters to goal
    if(WSinfo::me->pos.getY() >0)
      target = Vector(52.5,ServerOptions::pitch_width/2.-5.);
    else
      target = Vector(52.5,-ServerOptions::pitch_width/2.+5.);
  }
  else{// I have less than 17 meters to goal)
    //target = Vector(35.,0);
    if(WSinfo::me->pos.getY() >0)
      target = Vector(52.5,ServerOptions::goal_width/2.-.5);
    else
      target = Vector(52.5,-ServerOptions::goal_width/2.+.5);
  }
  DBLOG_POL(BASELEVEL, << "WBALL03: panic: Kick Ball with maxpowert to target "<<target);
  intention.set_panic_kick(target, WSinfo::ws->time);
  return true;
}


void PlayPass::get_clearance_params(double &speed, double &dir){
  int advantage, closest_teammate;
  Vector ipos;

  return get_opening_pass_params(speed,dir, ipos, advantage, closest_teammate);
}


void PlayPass::get_opening_pass_params(double &speed, double &dir, Vector &ipos, int &advantage,
				      int & closest_teammate){
  float min_angle = -50;
  float max_angle = 50;

  const int max_targets = 200;
  double testdir[max_targets];
  double testspeed[max_targets];

  int i= 0;
  float tmp_speed;
  Angle tmp_dir;
  for(float angle=min_angle;angle<max_angle;angle+=5){
    for(float speed=2.5;speed<2.6;speed+=.5){
      if (i>=max_targets)
	break;
      tmp_speed = speed;
      tmp_dir = angle/180. *PI;
      if(Planning::is_kick_possible(tmp_speed,tmp_dir) == false){
	continue;
      }
      testdir[i]= tmp_dir;
      testspeed[i]= tmp_speed;
      i++;
    }
  }

  Policy_Tools::get_best_kicknrush(WSinfo::me->pos,i,testdir,
				   testspeed,speed,dir,ipos,advantage, closest_teammate);
  if(speed >0){
    DBLOG_POL(0,<<"found kicknrush with advantage "<<advantage<<" resulting pos : "
	      <<ipos<<" closest teammate "<<closest_teammate);
    DBLOG_DRAW(0,VC2D(ipos,1.0,"00FFFF"));
    DBLOG_DRAW(0,VL2D(WSinfo::ball->pos,
		     ipos,"00FFFF"));
  }
  else{
    DBLOG_POL(0,<<"PlayPass: NO kicknrush found ");
  }


}

bool PlayPass::test_pass_or_dribble(Intention &intention){
  if(my_blackboard.pass_or_dribble_intention.valid_at() == WSinfo::ws->time){
    intention = my_blackboard.pass_or_dribble_intention;
    return true;
  }
  return false;
}


void PlayPass::check_write2blackboard(){
  int main_type = my_blackboard.intention.get_type();
  bool main_type_is_pass = false;

  if (main_type == PASS || main_type == LAUFPASS || 
      main_type == KICKNRUSH) 
    main_type_is_pass = true;
  
  // check for turn neck requests first
  if (my_blackboard.neckreq.is_set()){
    DBLOG_POL(0,<<"Write 2 Blackboard: Neck Request is SET to dir :"<<RAD2DEG(my_blackboard.neckreq.get_param()));
    Blackboard::set_neck_request(my_blackboard.neckreq.get_type(),my_blackboard.neckreq.get_param());
  }
  
  // check communication request for main type
  if(main_type == OPENING_SEQ || main_type == KICKNRUSH){
    Vector target;
    double speed;
    my_blackboard.intention.get_kick_info(speed, target);
    Blackboard::pass_intention = my_blackboard.intention;
    DBLOG_POL(1,<<"PlayPass: Check write2blackboard: intention opening seq / kicknrush Set blackboard intention: valid at "
	      << Blackboard::pass_intention.valid_at());
    return;
  }

  if (my_blackboard.pass_or_dribble_intention.valid_at() != WSinfo::ws->time){
    DBLOG_POL(1,<<"PlayPass: Check write2blackboard: no pass or dribble intention is set");
    return;
  }

  // now, check for turn neck request
  int pass_type = my_blackboard.pass_or_dribble_intention.get_type();
  if(pass_type == PASS || pass_type == LAUFPASS || pass_type == KICKNRUSH){
    // turn neck to pass receiver
    double dummy;
    if(Blackboard::get_neck_request(dummy) == NECK_REQ_NONE){
      Vector target;
      double speed;
      my_blackboard.pass_or_dribble_intention.get_kick_info(speed, target);
      ANGLE ball2targetdir = (target - WSinfo::ball->pos).ARG(); // direction of the target
      if(WSmemory::last_seen_in_dir(ball2targetdir) >0 || main_type_is_pass == true){
	DBLOG_POL(1,<<"Neck intn. not set.  and haven't looked in pass dir. look in dir of pass "<<RAD2DEG(ball2targetdir.get_value()));
	Blackboard::set_neck_request(NECK_REQ_LOOKINDIRECTION, ball2targetdir.get_value());
      }
    }
    //    mdpInfo::set_my_neck_intention(NECK_INTENTION_LOOKINDIRECTION, ball2targetdir);
  }


  PlayerSet pset= WSinfo::valid_opponents;
  pset.keep_players_in_circle(WSinfo::me->pos, 2* ServerOptions::kickable_area + 2* ServerOptions::player_speed_max); 
  bool opps_around_me = (pset.num >0);

  pset = WSinfo::valid_opponents;
  Vector endofregion;
  const double scanrange = 7;
  endofregion.init_polar(scanrange,(my_blackboard.intention.kick_target -  WSinfo::me->pos).ARG());
  endofregion += WSinfo::me->pos;
  Quadrangle2d check_area = Quadrangle2d(WSinfo::me->pos, endofregion, 5., scanrange);
  DBLOG_DRAW(0, check_area );
  pset.keep_players_in(check_area);

  bool opps_in_targetdir = (pset.num >0);

  // if opponent could get in my kickrange, then keep pass or dribble intention  
  if((main_type == SELFPASS || main_type == DRIBBLE) && // I'm dribbling
     is_planned_pass_a_killer == false && // do not reset killer passes 
     //(opps_around_me == true || opps_in_targetdir == true)){ // I have enough space around me
     (opps_around_me == false && opps_in_targetdir == false)){ // I have enough space around me
    DBLOG_POL(0,<<"I do selfpass or dribble and I'm not attacked -> do not communicate AND reset pass intent.");
    my_blackboard.pass_or_dribble_intention.reset();
    return;
  }


  // now, check for communication request

  if(pass_type == PASS || pass_type == LAUFPASS || pass_type == KICKNRUSH){
    DBLOG_POL(1,<<"PlayPass: Check write2blackboard: pass intention is set");
    // for now, communication is done indirectly; should be improved
    Blackboard::pass_intention = my_blackboard.pass_or_dribble_intention;
  }
}




void PlayPass::aaction2intention(const AAction &aaction, Intention &intention){

  double speed = aaction.kick_velocity;
  Vector target = aaction.target_position;
  double kickdir = aaction.kick_dir;

  switch(aaction.action_type){
  case  AACTION_TYPE_PASS:
    intention.set_pass(target,speed, WSinfo::ws->time, aaction.targetplayer_number, 0, 
		       aaction.actual_resulting_position);
    break;
  case  AACTION_TYPE_LAUFPASS:
    intention.set_laufpass(target,speed, WSinfo::ws->time, aaction.targetplayer_number, 0, 
			   aaction.actual_resulting_position, aaction.risky_pass);
    break;
  case AACTION_TYPE_SELFPASS:
    target.init_polar(5.,kickdir);
    target += WSinfo::ball->pos;
    intention.set_selfpass(ANGLE(kickdir), target,speed, WSinfo::ws->time);
    break;
  case AACTION_TYPE_IMMEDIATE_SELFPASS:
    target.init_polar(5.,kickdir);
    target += WSinfo::ball->pos;
    intention.set_immediateselfpass(target,speed, WSinfo::ws->time);
    break;
  case  AACTION_TYPE_WAITANDSEE:
    intention.set_waitandsee(WSinfo::ws->time);
    break;
  case AACTION_TYPE_TURN_AND_DASH:
    intention.set_turnanddash(WSinfo::ws->time);
    break;
  case  AACTION_TYPE_DRIBBLE:
    intention.set_dribble(target, WSinfo::ws->time);
    break;
  case  AACTION_TYPE_DRIBBLE_QUICKLY:
    intention.set_dribblequickly(target, WSinfo::ws->time);
    break;
  default:
    DBLOG_POL(0,<<"WBALL03 aaction2intention: AActionType not known");
    LOG_ERR(0,<<"WBALL03 aaction2intention: AActionType not known");
  }
}

bool PlayPass::get_turn_and_dash(Cmd &cmd){
  // used for turn_and_dash
  Vector dummy1, dummy3;
  Vector mynewpos,ballnewpos;
  double dummy2;
  int dash = 0;  
  Cmd_Body testcmd;
  double required_turn;
  bool dash_found;

  required_turn = 
    Tools::get_angle_between_null_2PI(WSinfo::ball->vel.arg() - 
				      WSinfo::me->ang.get_value());
  // If I can get ball by dashing, dash
  dash_found=false;
  if(Tools::get_abs_angle(required_turn)<40./180. *PI){
    for(dash=100;dash>=30;dash-=10){
      testcmd.unset_lock();
      testcmd.unset_cmd();
      testcmd.set_dash(dash);
      Tools::model_cmd_main(WSinfo::me->pos, WSinfo::me->vel, WSinfo::me->ang.get_value(), 
			    WSinfo::ball->pos,
			    WSinfo::ball->vel,
			    testcmd, mynewpos, dummy1, dummy2, ballnewpos, dummy3);
      if((mynewpos-ballnewpos).norm()<0.8*ServerOptions::kickable_area){
	dash_found=true;
	break;
      }
      }
  }
  if(dash_found ==true){
    LOG_POL(BASELEVEL,<<"TURN_AND_DASH: I found a dash "<<dash);
    basiccmd->set_dash(dash);
    basiccmd->get_cmd(cmd);
    return true;
    }
  
  if(Tools::get_abs_angle(required_turn)>10./180. *PI){
    LOG_POL(BASELEVEL,<<"Intention2cmd: Turn and Dash: turn 2 balldir "
	    <<RAD2DEG(WSinfo::ball->vel.arg())<<" Have to turn by "
	    <<RAD2DEG(required_turn));
    basiccmd->set_turn_inertia(required_turn);
    basiccmd->get_cmd(cmd);
    return true;
  }
  // dash forward
  basiccmd->set_dash(100);
  basiccmd->get_cmd(cmd);
  return true;
}

bool PlayPass::get_opening_seq_cmd( const float  speed, const Vector target,Cmd &cmd){

  Vector ipos;
  bool myteam;
  int number;
  int advantage;
  Angle dir = (target-WSinfo::ball->pos).arg();
  DBLOG_POL(BASELEVEL+0,<<"Get Command for Opening sequence to "<<target<<" dir "
	    <<RAD2DEG(dir)<<" w.speed "<<speed);
  DBLOG_DRAW(0, VC2D(target, 1, "#00ffff"));

  Policy_Tools::earliest_intercept_pos(WSinfo::me->pos,speed,dir,
				       ipos, myteam, number, advantage);


  if((advantage >0) && (my_blackboard.intention.valid_since() < WSinfo::ws->time)){
    // pass possible and I decided it at least one cycle ago -> already communicated 
    my_blackboard.intention.set_kicknrush(target,speed, WSinfo::ws->time); // directly set new intent.
    DBLOG_POL(BASELEVEL+0,<<"WBALL03 Opening sequence DO LAUFPASS, advantage ok "<<advantage
	      <<target<<" dir "
	      <<RAD2DEG(dir)<<" w.speed "<<speed<<" receiver "<<number);
    DBLOG_DRAW(0, VC2D(target, 1, "#00ffff"));
    neurokick->kick_to_pos_with_initial_vel(speed,target);
    neurokick->get_cmd(cmd);
    return true;
  }

  DBLOG_POL(0,<<"wball03: get opening seq cmd: do holdturn");
  onetwoholdturn->get_cmd(cmd);
  return true;  
}

bool PlayPass::intention2cmd(Intention &intention, Cmd &cmd){
  double speed;
  Vector target;
  speed = intention.kick_speed;
  target = intention.kick_target;
  ANGLE targetdir = intention.target_body_dir; // for selfpasses
#if LOGGING && BASIC_LOGGING
  int targetplayer_number = intention.target_player;
#endif
  //  double ball2targetdir = (target - WSinfo::ball->pos).arg(); // direction of the target
  bool cankeepball = onestepkick->can_keep_ball_in_kickrange();
  //double speed1step = Move_1Step_Kick::get_vel_in_dir(speed, ball2targetdir);
  //bool need_only1kick = (fabs(speed -speed1step) < .2);
  double opposite_balldir;
  double kick_dir;

  switch(intention.get_type()){
  case  PASS:
  case  LAUFPASS:
    speed = intention.kick_speed;
    target = intention.kick_target;
    DBLOG_POL(BASELEVEL,<<"WBALL03: AAction2Cmd: passing to teammate "<<targetplayer_number
	      //<<" onestepkick is possible (0=false) "<<need_only1kick
	      <<" speed "<<speed<<" to target "<<target.getX()<<" "<<target.getY());
    if(intention.risky_pass == true){
      DBLOG_POL(BASELEVEL,<<"WBALL03: RISKY PASS!!!");
    }
    neurokick->kick_to_pos_with_initial_vel(speed,target);
    neurokick->get_cmd(cmd);
    return true;
    break;
  case  OPENING_SEQ:
    speed = intention.kick_speed;
    target = intention.kick_target;
    DBLOG_POL(BASELEVEL,<<"WBALL03: AAction2Cmd: Opening Seq for teammate "<<targetplayer_number
	      //<<" onestepkick is possible (0=false) "<<need_only1kick
	      <<" speed "<<speed<<" to target "<<target.getX()<<" "<<target.getY());
    return get_opening_seq_cmd(speed,target,cmd);
    break;
    break;
  case  WAITANDSEE:
    if(cankeepball){
      mdpInfo::set_my_intention(DECISION_TYPE_WAITANDSEE);
      onetwoholdturn->get_cmd(cmd);
      last_waitandsee_at = WSinfo::ws->time;
      return true;
    }
    else{
      opposite_balldir = WSinfo::ball->vel.arg() + PI;
      opposite_balldir = opposite_balldir - WSinfo::me->ang.get_value();
      LOG_ERR(0,<<"Can't keep the ball in my kickrange, and I do not plan to pass. Kick to "
	      <<RAD2DEG(opposite_balldir));
      basiccmd->set_kick(100,ANGLE(opposite_balldir));
      basiccmd->get_cmd(cmd);
      return true;
    }

    break;
  case TURN_AND_DASH:
    return get_turn_and_dash(cmd);
    break;
  case KICKNRUSH:
    speed = intention.kick_speed;
    target = intention.kick_target;
    mdpInfo::set_my_intention(DECISION_TYPE_KICKNRUSH, speed,0,0,0, 0);
    LOG_POL(BASELEVEL,<<"WBALL03 aaction2cmd: kicknrush w speed "<<speed<<" to target "<<target);
    neurokick->kick_to_pos_with_initial_vel(speed,target);
    neurokick->get_cmd(cmd);
    return true;
    break;
  case PANIC_KICK:
    target = intention.kick_target;
    kick_dir = (target - WSinfo::me->pos).arg() - WSinfo::me->ang.get_value();
    basiccmd->set_kick(100,ANGLE(kick_dir));
    basiccmd->get_cmd(cmd);
    return true;
    break;
  case BACKUP:
    LOG_POL(BASELEVEL,<<"WBALL03 aaction2cmd: back up (two teammates at ball)  not yet implemented");
    LOG_ERR(BASELEVEL,<<"WBALL03 aaction2cmd: back up (two teammates at ball)  not yet implemented");
    //ridi03: todo
    return false;
    break;
  case HOLDTURN:
    DBLOG_POL(BASELEVEL,<<"WBALL03 Intention: holdturn in dir "<<RAD2DEG(intention.target_body_dir.get_value()));
    if(cankeepball){
      if(onetwoholdturn->is_holdturn_safe() == false){
	DBLOG_POL(BASELEVEL,<<"WBALL03 Intention: holdturn NOT safe, relaxed trial (should only occur in troubled sits)");
	last_waitandsee_at = WSinfo::ws->time;
	onetwoholdturn->get_cmd_relaxed(cmd);
	return true;
      }
      last_waitandsee_at = WSinfo::ws->time;
      onetwoholdturn->get_cmd(cmd,intention.target_body_dir);
      return true;
    }
    else{
      opposite_balldir = WSinfo::ball->vel.arg() + PI;
      opposite_balldir = opposite_balldir - WSinfo::me->ang.get_value();
      LOG_ERR(0,<<"Can't keep the ball in my kickrange, and I do not plan to pass. Kick to "
	      <<RAD2DEG(opposite_balldir));
      basiccmd->set_kick(100,ANGLE(opposite_balldir));
      basiccmd->get_cmd(cmd);
      return true;
    }


    return true;
  default:
    DBLOG_POL(0,<<"WBALL03 aaction2cmd: AActionType not known");
    LOG_ERR(0,<<"WBALL03 aaction2cmd: AActionType not known");
    return false;
  }
  return false;
}
