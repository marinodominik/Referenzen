#include "wball06_bmp.h"
#include "ws_info.h"
#include "ws_memory.h"
#include "log_macros.h"
#include <stdlib.h>
#include <stdio.h>

#include "../basics/wmoptions.h"
#include "tools.h"
#include "valueparser.h"
#include "options.h"
#include "geometry2d.h"
#include "mdp_info.h"

#include "../policy/planning.h"
#include "../policy/policy_tools.h"
#include "../policy/positioning.h"  // get_role()
#include "globaldef.h"
#include "ws_info.h"
#include "blackboard.h"

#define MIN_SELFPASS_MOVEMENT 2.0
#define MIN_SELFPASS_MOVEMENT_GOALAREA 3.0 //otherwise advance by tingletangle



#define WBALL06LOGGING 1

#if WBALL06LOGGING
#define MYLOG_POL(LLL,XXX) LOG_POL(LLL,<<"WBALL06: " XXX)
#define MYLOG_DRAW(LLL,XXX) LOG_POL(LLL,<<_2D <<XXX)
#define MYLOG_ERR(LLL,XXX) LOG_ERR(LLL,XXX)
#define MYGETTIME (Tools::get_current_ms_time())
#else
#define MYLOG_POL(LLL,XXX)
#define MYLOG_DRAW(LLL,XXX)
#define MYLOG_ERR(LLL,XXX) 
#define MYGETTIME (0)
#endif

#if WBALL06LOGGING  // reduced logging: testintrouble and 2tmmctrlball only
#define   TGLOGPOL(YYY,XXX)        LOG_POL(YYY,XXX)
#else
#define   TGLOGPOL(YYY,XXX)
#endif

#define MIN(X,Y) ((X<Y)?X:Y)


/* constructor method */
Wball06::Wball06() {

  neurokick = new NeuroKick05;
  neurokickwrapper = new NeuroKickWrapper;
  //dribble_between = DribbleBetween06::getInstance();
  dribble_between = DribbleBetween::getInstance();
  tingletangle = TingleTangle::getInstance();
  selfpass2 = new Selfpass2;
  basiccmd = new BasicCmd;
  onestepkick = new OneStepKick;
  oneortwo = new OneOrTwoStepKick;
  onetwoholdturn = new OneTwoHoldTurn;
  ballzauber = new Ballzauber;
  score = new Score;
  score05_sequence = new Score05_Sequence;
  onestepscore = new OneStepScore;
  neurodribble2017 = new NeuroDribble2017;
  //  neuro_wball = new NeuroWball;
  pass_selection = new PassSelection;
  tt_last_active_at = 0;
  memory.intention.reset();
}

Wball06::~Wball06() {
	// do not delete dribble_between: singleton!
  delete neurokick;
  delete neurokickwrapper;
  delete selfpass2;
  delete basiccmd;
  delete onestepkick;
  delete oneortwo;
  delete onetwoholdturn;
  delete ballzauber;
  delete score;
  delete score05_sequence;
  delete onestepscore;
  //  delete neuro_wball;
  delete pass_selection;
  delete ballzauber;
  delete neurodribble2017;
}

void Wball06::reset_intention() {
  memory.intention.reset();
}


void Wball06::foresee_modify_cmd(Cmd &cmd, const double targetdir){
  Vector newmypos, newmyvel, newballpos, newballvel;
  ANGLE newmyang;

  if (cmd.cmd_body.get_type() == Cmd_Body::TYPE_TURN){
    LOG_POL(0,"Foresee: Next command is a turn command; modify if possible!");
    Cmd tmp_cmd;
    Angle turn2dir_angle = targetdir  -WSinfo::me->ang.get_value();
    turn2dir_angle = Tools::get_angle_between_mPI_pPI(turn2dir_angle);
    turn2dir_angle=turn2dir_angle*(1.0+(WSinfo::me->inertia_moment*
					(WSinfo::me->vel.norm())));
    if (turn2dir_angle > 3.14) turn2dir_angle = 3.14;
    if (turn2dir_angle < -3.14) turn2dir_angle = -3.14;
    turn2dir_angle = Tools::get_angle_between_null_2PI(turn2dir_angle);
    tmp_cmd.cmd_body.unset_lock();
    tmp_cmd.cmd_body.set_turn(turn2dir_angle);
    if (Tools::is_ball_kickable_next_cycle(tmp_cmd,newmypos,newmyvel,newmyang,newballpos,newballvel)){
      LOG_POL(0,"Foresee: WOW, I can turn 2 MODIFIED target direction and still get the ball!");
      cmd.cmd_body = tmp_cmd.cmd_body;
    }
  } // cmd was a turn (intercept can spend its last cycle on turning!
}

#if 1 // new version

void Wball06::foresee(Cmd &cmd){
  Intention tmp_intention;
  Intention pass_option, selfpass_option;

  Vector newmypos, newmyvel, newballpos, newballvel;
  ANGLE newmyang;

  if(WSinfo::is_ball_pos_valid() == false || WSinfo::is_ball_kickable() == true)
    return;  // not a foresee situation
  if (Tools::is_ball_kickable_next_cycle(cmd,newmypos,newmyvel,newmyang,newballpos,newballvel) == false)
    return; // ball is not kickable next cycle.

  MYLOG_POL(0, << "foresee: Ball is KICKABLE next time, check new intention");
  

  /* todo: virtuelles dribblen, um M?glichkeiten zu checken. Aktuell: Annahme dribbeln ok -> priority pass nur, wenn Verbesserung, nicht aus Not */
  status.is_dribble_ok = true;
  status.is_dribble_safe = true;
  status.is_holdturn_safe = true;
  status.is_tt_safe = true;
  status.tt_dir = newmyang.get_value();
  bool set = false;

#if 1  // test: might take too much time...
  if(score05_sequence->test_shoot2goal(tmp_intention, &cmd) == true){ // false: no recursiveness...
   MYLOG_POL(0, << "foresee: WOW, I will SCORE next time!!!");
    set = true;
  }
#endif

  determine_best_selfpass(selfpass_option,newmypos,newmyvel, newmyang, WSinfo::me->stamina, newballpos,newballvel );
  determine_pass_option(pass_option,newmypos,newmyvel, newmyang, newballpos,newballvel);

  if(!set && test_priority_pass(tmp_intention, pass_option, selfpass_option) == true){
    // passing is useful! Schedule it as the intention for the next step, announce and turn neck
    set_neck_pass(pass_option);
    set_communication(pass_option, pass_option, newmypos, cmd);
    display_intention(pass_option);
    //    memory.intention = pass_option;
    MYLOG_POL(0,<<"foresee: test_priority_pass successful");
    set=true;
  }
  else{ // forget about this pass, not good enough
    set_communication(tmp_intention, pass_option, newmypos, cmd);
    pass_option.reset();
  }

  if(!set && selfpass_option.valid_at() == WSinfo::ws->time){
    // found a selfpass, plan to advance. Todo: Turn into the desired direction
    set_neck_selfpass(selfpass_option);
    foresee_modify_cmd(cmd, selfpass_option.target_body_dir.get_value());
    MYLOG_POL(0,<<"foresee: selfpass successful");
    set = true;
  }

  double targetdir = 0.;
  if(newmypos.getX() >45.)
    targetdir = (Vector(52.0) - newmypos).arg();

  if(!set){
    MYLOG_POL(0,<<"foresee: try to modify_cmd");
    foresee_modify_cmd(cmd,targetdir); // puts turns to target dir in cmd if necessary
  }

  if (   WMoptions::use_server_based_collision_detection == false //TG+JTS08   
      &&   newmypos.distance(newballpos)
         < WSinfo::me->radius + ServerOptions::ball_size+.1 )
  {
    MYLOG_POL(0, << "foresee: Ball might collide with me next cycle!");
    Angle toball = (newballpos-newmypos).arg();

    if(Tools::could_see_in_direction(toball)){
      MYLOG_POL(0, << "foresee: Ball might collide with me next cycle, I can "
        <<"look towards it w/ angle="<<toball);
      Tools::set_neck_request(NECK_REQ_LOOKINDIRECTION, toball, true); // forced
      set = true;
    }
  }
  if(set) return;

  //check for looking to the goalie (without forcing)
  int goalieAgeThreshold = 1;
  if (    WSinfo::his_goalie 
       && WSinfo::his_goalie->pos.distance(WSinfo::me->pos) < 8.0 )//TG08:6->8 
    goalieAgeThreshold = 0;
  if (   WSinfo::me->pos.distance(HIS_GOAL_CENTER) < 22.0
      && WSinfo::his_goalie
      && WSinfo::his_goalie->pos.distance(WSinfo::me->pos) < 12.0
      && WSinfo::his_goalie->age > goalieAgeThreshold)
  {
    ANGLE toGoalie = (WSinfo::his_goalie->pos - WSinfo::me->pos).ARG();
    if (Tools::could_see_in_direction( toGoalie ) )
    {
      MYLOG_POL(0,<<"TG@WBALL06: Set neck to goalie: look to dir "
        <<RAD2DEG(toGoalie.get_value_mPI_pPI()));
      Tools::set_neck_request(NECK_REQ_LOOKINDIRECTION, toGoalie);
      return;
    }
  }

  MYLOG_POL(0,<<"Foresee: turn and look into DEFAULT targetdirection: "<<RAD2DEG(targetdir));
  Tools::set_neck_request(NECK_REQ_LOOKINDIRECTION, targetdir, true); // forced
}


#endif 

bool Wball06::get_cmd(Cmd & cmd) {
  Intention intention;
  Intention pass_option, selfpass_option;
  intention.reset(); //to make it blank!


  MYLOG_POL(0, << "Entered");

  if(!WSinfo::is_ball_kickable()){
    MYLOG_ERR(0,<<"Ball not kickable");
    return false;
  }
  if(WSinfo::ws->play_mode != PM_PlayOn && WSinfo::ws->play_mode != PM_my_GoalKick){
    MYLOG_ERR(0,<<"Unknown Playmode");
    return false;
  }

  /* get straightforward decision */ //TG09
  if(get_simple_intention(intention, pass_option, selfpass_option) == true)
  {
    MYLOG_ERR(0,<<"I found a simple intention!");
    memory.intention = intention;
  }
  else
  {
    MYLOG_ERR(0,<<"Could not find simple intention");
    
    //continue the default way

    /* compute some predicates */
    if(last_waitandsee_at == WSinfo::ws->time -1) // waitandsee selected again
      cycles_in_waitandsee ++;
    else
      cycles_in_waitandsee = 0; // reset
    I_am_heavily_attacked_since();  //* call at least once to update

    // first, determine all options
    int idbp = is_dribblebetween_possible();
    switch(idbp){
    case 0: status.is_dribble_ok = false;
      status.is_dribble_safe = false;
      break;
    case 1: status.is_dribble_ok = true;
      status.is_dribble_safe = false;
      break;
    case 2: status.is_dribble_ok = true;
      status.is_dribble_safe = true;
      break;
    }
    // status.is_dribble_ok = WSinfo::me->stamina > MIN_STAMINA_DRIBBLE && status.is_dribble_ok; // war fr dribble_between06 auskommentiert. Why?
    status.is_dribble_safe = WSinfo::me->stamina > MIN_STAMINA_DRIBBLE && status.is_dribble_safe;
    status.is_holdturn_safe = onetwoholdturn->is_holdturn_safe();

    status.tt_dir = get_prefered_tingletangle_dir();
    status.is_tt_safe = can_tingletangle2dir(status.tt_dir);

    determine_best_selfpass(selfpass_option,WSinfo::me->pos, WSinfo::me->vel, WSinfo::me->ang, WSinfo::me->stamina, 
	  		  WSinfo::ball->pos, WSinfo::ball->vel);
    determine_pass_option(pass_option,WSinfo::me->pos, WSinfo::me->vel, WSinfo::me->ang, WSinfo::ball->pos, WSinfo::ball->vel);
    check_adapt_pass_speed(pass_option);

    /* get decision */
    if(get_intention(intention, pass_option, selfpass_option) == false)
    {
      MYLOG_ERR(0,<<"Can not find (complex) intention");
      return false;
    }
  
  } //end of get simple intention

  Cmd_Say savedCmdSay = cmd.cmd_say;
  intention2cmd(intention,cmd);
  cmd.cmd_say = savedCmdSay;
  set_neck(intention, pass_option, selfpass_option);
  set_communication(intention, pass_option, WSinfo::me->pos, cmd);

  if(cmd.cmd_body.get_type() == Cmd_Body::TYPE_KICK ||
     cmd.cmd_body.get_type() == Cmd_Body::TYPE_TURN)
    last_waitandsee_at = WSinfo::ws->time;
  
  Blackboard::main_intention = intention; 
  display_intention(intention);
  return true;
}


bool Wball06::get_simple_intention(Intention &intention, Intention &pass_option, Intention &selfpass_option )
{

  bool result = false;
  
  // priority queue
   MYLOG_POL(0,"******** Entering [SIMPLE] daisy chain. [intention.wait_then_pass="<<intention.wait_then_pass
                                                    <<",pass_option.wait_then_pass="<<pass_option.wait_then_pass
                                                    <<",selfpass_option.wait_then_pass="<<selfpass_option.wait_then_pass<<"]");
  if ( 0 )
  {}
  else if ( (result = score05_sequence->test_shoot2goal( intention )) ){
    MYLOG_POL(0,"Daisy chain: test SCORE05_SEQUENCE successful");
    TGLOGPOL(0,"Daisy chain: test SCORE05_SEQUENCE successful");
  }
  else if ( (result = onestepscore->test_score_now( intention )) ){
    MYLOG_POL(0,"Daisy chain: test ONE_STEP_SCORE successful");
    TGLOGPOL(0,"Daisy chain: test ONE_STEP_SCORE successful");
  }
#if 1 // old score
  else if ( (result = score->test_shoot2goal(intention)) ){
    MYLOG_POL(0,"Daisy chain: test SCORE successful");
    TGLOGPOL(0,"Daisy chain: test SCORE successful");
  }
#endif 
  else if ( (result = check_previous_intention(memory.intention, intention, pass_option, selfpass_option)) ){
    MYLOG_POL(0,"Daisy chain: test CHECK PREVIOUS INTENTION successful");
    TGLOGPOL(0,"Daisy chain: test CHECK PREVIOUS INTENTION successful");
  }
  else if ( (result = test_two_teammates_control_ball(intention))){
    MYLOG_POL(0,"Daisy chain: test TWO TEAMMATES CONTROL BALL successful");
    TGLOGPOL(0,"Daisy chain: test TWO TEAMMATES CONTROL BALL successful");
  }
  else if ( (result = test_in_trouble(intention))){
    MYLOG_POL(0,"Daisy chain: test IN TROUBLE successful");
    TGLOGPOL(0,"Daisy chain: test IN TROUBLE successful");
  }
  else result = false; // modified version

  return result;
}

bool Wball06::get_intention(Intention &intention, Intention &pass_option, Intention &selfpass_option )
{

  bool result = false;
  
  // priority queue
  MYLOG_POL(0,"******** Entering [COMPLEX] daisy chain. [intention.wait_then_pass="<<intention.wait_then_pass
                                                    <<",pass_option.wait_then_pass="<<pass_option.wait_then_pass
                                                    <<",selfpass_option.wait_then_pass="<<selfpass_option.wait_then_pass<<"]");
  
  if ( 0 )
  {}
  else if ( (result = test_1vs1(intention, pass_option, selfpass_option))){
    MYLOG_POL(0,"Daisy chain: test 1vs1 successful");
    TGLOGPOL(0,"Daisy chain: test 1vs1 successful");
  }
  else if ((result = test_priority_pass(intention, pass_option, selfpass_option))){
    MYLOG_POL(0,"Daisy chain: test PRIORITY PASS successful");
    TGLOGPOL(0,"Daisy chain: test PRIORITY PASS successful");
  }
  else if ( (result = test_dream_selfpass(intention, WSinfo::me->pos, WSinfo::ball->pos, WSinfo::ball->vel))){
    MYLOG_POL(0,"Daisy chain: test DREAM SELFPASSES successful");
    TGLOGPOL(0,"Daisy chain: test DREAM SELFPASSES successful");
  }
  else if ( (result = test_save_turn_in_opp_pen_area(intention, selfpass_option))){
    MYLOG_POL(0,"Daisy chain: test SAVE TURN IN OPP PEN AREA successful");
    TGLOGPOL(0,"Daisy chain: test SAVE TURN IN OPP PEN AREA successful");
  }
  else if ( (result = test_pass_when_playing_against_aggressive_opponents_08
                     (intention, pass_option, selfpass_option))){
    MYLOG_POL(0,"Daisy chain: test PASS (AGGR_OPP_08) successful");
    TGLOGPOL(0,"Daisy chain: test PASS (AGGR_OPP_08) successful");
  }
  else if ( (result = test_advance_selfpasses(intention, selfpass_option))){
    MYLOG_POL(0,"Daisy chain: test SELFPASSES successful "
      <<"[intention.wait_then_pass="<<intention.wait_then_pass
      <<"selfpass_option.wait_then_pass="<<selfpass_option.wait_then_pass<<"]");
    TGLOGPOL(0,"Daisy chain: test SELFPASSES successful "
      <<"[intention.wait_then_pass="<<intention.wait_then_pass
      <<"selfpass_option.wait_then_pass="<<selfpass_option.wait_then_pass<<"]");
  }
  else if ( (result = test_tingletangle(intention)) ){
    MYLOG_POL(0,"Daisy chain: test TINGLETANGLE successful");
    TGLOGPOL(0,"Daisy chain: test TINGLETANGLE successful");
    return true;
  }
  else if ((result = test_pass_under_attack(intention, pass_option, selfpass_option))){
    MYLOG_POL(0,"Daisy chain: test PASS under ATTACK successful");
    TGLOGPOL(0,"Daisy chain: test PASS under ATTACK successful");
  }
#if 0 // trust or not trust dribbling in goal area
  else if ( (result = test_prefer_holdturn_over_dribbling(intention))){
    MYLOG_POL(0,"Daisy chain: test PREFER HOLDTURN successful");
    TGLOGPOL(0,"Daisy chain: test PREFER HOLDTURN successful");
  }
#endif
  else if ( (result = test_holdturn(intention))){
    MYLOG_POL(0,"Daisy chain: test HOLDTURN successful");
    TGLOGPOL(0,"Daisy chain: test HOLDTURN successful");
  }
  else if ((result = test_pass(intention, pass_option, selfpass_option))){
    MYLOG_POL(0,"Daisy chain: test PASS successful");
    TGLOGPOL(0,"Daisy chain: test PASS successful");
  }
  // ridi 06: no dribbling is only done, if no other option. advancing done by tingletangle
  else if ( (result = test_dribbling(intention)) ){
    MYLOG_POL(0,"Daisy chain: test DRIBBLING successful");
    TGLOGPOL(0,"Daisy chain: test DRIBBLING successful");
  }
  else result = test_default(intention); // modified version
  memory.intention = intention;

  /* TODO: setze Neck und communication. Fr?her bekannt als
     check_write2blackboard();
  */
  return result;
}


bool Wball06::check_previous_intention(Intention prev_intention, Intention  &new_intention, Intention &pass_option, 
				       Intention &selfpass_option )
{
  MYLOG_POL(0,<<"Entering Check previous intention; valid at: "<<prev_intention.valid_at());
  if(prev_intention.valid_at() < WSinfo::ws->time -1){
    MYLOG_POL(0,<<"Check previous intention: Previous intention not valid!");
    return false;
  }

  if (test_two_teammates_control_ball(prev_intention) == true){
    MYLOG_POL(0,<<"Check previous intention: Two teammates control ball, previous intention is invalidated!");
    return false;
  }
  
  // toga05: check if 'i am in trouble' (meaning: ball is in kick area of me _and_ opponent player)
  PlayerSet ballControllingOpponents = WSinfo::valid_opponents;
  ballControllingOpponents.keep_players_in_circle(WSinfo::ball->pos, 2.0*ServerOptions::kickable_area);
  ballControllingOpponents.keep_and_sort_closest_players_to_point(1, WSinfo::ball->pos);
  if (    ballControllingOpponents.num > 0
       &&   WSinfo::ball->pos.distance(ballControllingOpponents[0]->pos)
          < 1.1 * ballControllingOpponents[0]->kick_radius ){ //10% additional safety margin
    MYLOG_POL(0,<<"Check previous intention: I am in trouble since an opponent has ball control, also!");
    return false;
  }


  int targetplayer_number = prev_intention.target_player;
  if (targetplayer_number >11) targetplayer_number = 0; // invalid then
  //  float speed = prev_intention.kick_speed;
  float speed = prev_intention.kick_speed;
  Vector target = prev_intention.kick_target;
  Angle dir = (target - WSinfo::me->pos).arg(); // direction of the target
  double passdir = (target - WSinfo::me->pos).arg(); // direction of the target
  ANGLE targetdir = prev_intention.target_body_dir; // for selfpasses
  Vector ballpos = WSinfo::ball->pos;
  //  Vector mypos = WSinfo::me->pos;
  Vector interceptpos, playerpos;  // return parameters; not considered here
  int advantage, number;
  //int mytime2intercept;  // return parameters; not considered here
  bool myteam; // return pararamter, not considered here
  Vector receiverpos; // used 
  //  int steps; // return parameter for selfpass, not used
  Vector op_pos; // return parameter for selfpass, not used
  bool stillok = false;
  // double ballspeed;
  //int op_number;
  //  int risky_pass = prev_intention.risky_pass;
  double new_kickspeed;
  int attacker_num = prev_intention.attacker_num;
  double passspeed =prev_intention.kick_speed;
  Vector dummy_playerpos;

  int tmp_advantage;

  if(prev_intention.wait_then_pass == true){
    MYLOG_POL(0,<<"Check previous intention. Was WAIT THEN PASS, now check pass without waiting ");
    prev_intention.wait_then_pass = false;  // Waited one cylcle, now do pass!
    MYLOG_POL(0,<<"Check previous intention. Was WAIT THEN PASS, return false. DECIDE FROM NEW ");
    return false;
  }

  switch(prev_intention.get_type())
  {
    case PASS:
      receiverpos = target; // default setting
      MYLOG_POL(0,<<"Check previous intention: Check PASS: target player: "<<targetplayer_number);
      if(prev_intention.subtype == SUBTYPE_PASS && targetplayer_number >0)
      { // do target correction
        MYLOG_POL(0,<<"Check previous intention: PASS: correct target, old: "<<target);
        PPlayer p= WSinfo::get_teammate_by_number(targetplayer_number);
        if(p) 
        {
          receiverpos = p->pos;
          if(p->age_vel <=1)
          receiverpos += p->vel;	  
          passdir = (receiverpos - WSinfo::me->pos).arg(); // direction of the target
          passspeed = pass_selection->compute_direct_pass_speed(WSinfo::ball->pos, receiverpos);
        }
      }  // end pass dir correction

      if (prev_intention.subtype == SUBTYPE_PENALTYAREAPASS)
        stillok = Planning::is_penaltyareapass_safe(ballpos,  passspeed,passdir, advantage, targetplayer_number, dummy_playerpos);
      else 
      if (prev_intention.subtype == SUBTYPE_SELFPASS)
      {
          MYLOG_POL(0,"Previous Intention ws: Dream Selfpass. Still OK: "<<stillok
          <<"   [prev.WTB="<<prev_intention.wait_then_pass<<",new.WTB="<<new_intention.wait_then_pass<<"]");
        stillok = is_dream_selfpass_to_target_possible(new_intention, 
                                                       prev_intention.kick_target, 
                                                       WSinfo::me->pos, 
                                                       WSinfo::ball->pos, 
                                                       WSinfo::ball->vel,
                                                       true, //prev. intention mode
                                                       prev_intention.kick_speed
                                                      );
        MYLOG_POL(0,"Previous Intention ws: Dream Selfpass. Still OK: "<<stillok
          <<"   [prev.WTB="<<prev_intention.wait_then_pass<<",new.WTB="<<new_intention.wait_then_pass<<"]");
        if ( new_intention.get_type() == SELFPASS )
        {
          prev_intention.confirm_intention(new_intention,WSinfo::ws->time);
          receiverpos = new_intention.kick_target;
          passspeed = new_intention.kick_speed;
        }
        MYLOG_POL(0,"Previous Intention ws: Dream Selfpass. Still OK: "<<stillok
          <<"   [prev.WTB="<<prev_intention.wait_then_pass<<",new.WTB="<<new_intention.wait_then_pass<<"]");
      }
      else
        stillok = Planning::is_pass_safe(ballpos,  passspeed,passdir, interceptpos, advantage, targetplayer_number);

      target = receiverpos;
      speed = passspeed;
      break;

    case LAUFPASS:
      MYLOG_POL(0,<<"Check previous intention: Check LAUFPASS: target player: "<<targetplayer_number);
      stillok = Planning::is_laufpass_successful2(ballpos,speed, dir);
      if (stillok == false && WSinfo::me->pos.getX() >35. && fabs(WSinfo::me->pos.getY()) <20.)
      {
        // second chance for penaltyareapasses
        stillok = Planning::is_penaltyareapass_successful(ballpos,speed,dir);
      }

      if (stillok == false) {
        MYLOG_POL(0, <<"Check previous intention: LAUFPASS considered harmful -> RESET");
      }
      break;
    case KICKNRUSH:
      /*TG08*/ stillok = false; // kicknrush remains ok, since this is default action! //TG08 ZUI
      break; 
    case OPENING_SEQ:
      stillok = true; // kicknrush remains ok, since this is default action!
      if (onetwoholdturn->is_holdturn_safe() == false)
      {
        MYLOG_POL(0, <<" opening is interrupt, since holdturn not safe");
        stillok = false; 
      }  
      Policy_Tools::earliest_intercept_pos(WSinfo::me->pos,speed,dir,
                                           interceptpos, myteam, number, advantage);
      if (interceptpos.getX() < WSinfo::his_team_pos_of_offside_line())
      {
        MYLOG_POL(0, <<"opening is interrupt, since ipos of pass < offsideline");
        stillok = false; 
      }
      break; 
    case SELFPASS:
      // ridi 06
      stillok = false;  // always interrupt selfpasses
      break; /*!!! => selfpass_option is not used, actually!*/
    
      if(selfpass_option.valid_at() == WSinfo::ws->time)
      {  // found a new selfpass that is possible
        if(Tools::evaluate_pos_analytically(selfpass_option.resultingpos) > 
           Tools::evaluate_pos_analytically(prev_intention.kick_target) + 3.)
        {
          // only stop a successful selfpass if a new selfpass improves considerably
          stillok = false;
          MYLOG_POL(0,<<"Check previous intention: New Selfpass improves evaluation. Abort");
          break;
        }
      }
      if(selfpass2->is_selfpass_still_safe(tmp_advantage, prev_intention.target_body_dir, new_kickspeed, attacker_num)==true)
      {
        speed = new_kickspeed;       // speed now has new kickspeed ; it is set below,  (see new_intention.correct_speed(speed))
        stillok = true;
        MYLOG_POL(0,<<"Check previous intention: SELFPASS to dir "
		   <<RAD2DEG(prev_intention.target_body_dir.get_value())<<" is still ok"<<" speed "<<speed);
      }
      else
      {
        stillok = false;
        MYLOG_POL(0,<<"Check previous intention: Selfpass not safe. Aborting");
      }
      break;
  case IMMEDIATE_SELFPASS:
    stillok = false;
    MYLOG_POL(0,<<"Check previous intention: Immediate Selfpass ALWAYS reconsidered harmful -> RESET");
    break;
  case SCORE:
    /* ridi 2006: old version.
    stillok = true;
    if (prev_intention.immediatePass==true)
    {
      stillok = false;
      MYLOG_POL(0,<<"Check previous intention: Score05_Sequence ALWAYS reconsidered harmful -> RESET");
    }
    */
    // ridi 2006: new version:
    stillok = false;
    MYLOG_POL(0,<<"Check previous intention: Score05_Sequence ALWAYS reconsidered harmful -> RESET");
    break;    
  case TINGLETANGLE:
  case DRIBBLE:
  case DRIBBLE_QUICKLY:
    stillok = false;
    break;
  case HOLDTURN:
  case TURN_AND_DASH:
    stillok = false;
    break;
  default:
    MYLOG_POL(0,<<"Previous Intention SPECIAL TYPE "<<prev_intention.type<<" NOT CHECKED. RETURN FALSE");
    stillok = false;
    break;
  }
  if(stillok)
  {
    MYLOG_POL(0,<<"Previous Intention still ok, just take it!");
    new_intention.confirm_intention(prev_intention,WSinfo::ws->time );
    new_intention.correct_target(target);
    new_intention.correct_speed(speed);
    new_intention.attacker_num = attacker_num;
    MYLOG_DRAW(0, VL2D(WSinfo::ball->pos,target,"lightblue"));
    return true;
  }
  return false;
}




bool Wball06::holdturn_intention2cmd(Intention &intention, Cmd &cmd){

  
  MYLOG_POL(0,<<"HOLDTURN_intention2cmd: holdturn in dir "<<
	    RAD2DEG(intention.target_body_dir.get_value()));


  if(am_I_attacked(3.0) == false){
    if(ballzauber->isBallzauberToTargetDirectionPossible(intention.target_body_dir) == true){
      MYLOG_POL(0,<<" intention2cmd: BALLZAUBEr possible. Do turn to dir: "<<RAD2DEG(intention.target_body_dir.get_value()));
      ballzauber->get_cmd(cmd,intention.target_body_dir);
      return true;
    }
  }

  if(onestepkick->can_keep_ball_in_kickrange()){ // can keep ball in kickrange
    if(status.is_holdturn_safe == false){
      MYLOG_POL(0,<<" intention2cmd: holdturn NOT safe, relaxed trial (should only occur in troubled sits)");
      onetwoholdturn->get_cmd_relaxed(cmd);
      return true;
    }
    onetwoholdturn->get_cmd(cmd,intention.target_body_dir);
    return true;
  }
  else{
    double opposite_balldir = WSinfo::ball->vel.arg() + PI;
    opposite_balldir = opposite_balldir - WSinfo::me->ang.get_value();
    MYLOG_POL(0,<<" intention2cmd: holdturn. Can't keep the ball in my kickrange, and I do not plan to pass. Kick to "<<RAD2DEG(opposite_balldir));
    LOG_ERR(0,<<"Can't keep the ball in my kickrange, and I do not plan to pass. Kick to "
	    <<RAD2DEG(opposite_balldir));
    basiccmd->set_kick(100,ANGLE(opposite_balldir));
    basiccmd->get_cmd(cmd);
    return true;
  }
  return true;
}


bool Wball06::intention2cmd(Intention &intention, Cmd &cmd){
  double speed;
  Vector target;
  speed = intention.kick_speed;
  target = intention.kick_target;
  ANGLE targetdir = intention.target_body_dir; // for selfpasses
  double kick_dir;

  /*
  double toRightCornerFlag = abs((Vector(FIELD_BORDER_X,-FIELD_BORDER_Y) - WSinfo::me->pos).ARG().get_value_mPI_pPI());
  double toLeftCornerFlag = abs((Vector(FIELD_BORDER_X,FIELD_BORDER_Y) - WSinfo::me->pos).ARG().get_value_mPI_pPI());
  double toGoal = abs(WSinfo::me->ang.get_value_mPI_pPI());
  */

  //tell the blackboard's pass intention whether this pass is really
  //intended to be actively played (i.e. a kick is already initiated
  //within the current cycle
  if (intention.is_pass2teammate() == true ||
      intention.get_type() == PANIC_KICK)
    Blackboard::pass_intention.immediatePass = true; // communicate!
  else
    Blackboard::pass_intention.immediatePass = false;
  
  Vector tttarget; // tingletangle target
  ANGLE ttdir;

  switch(intention.get_type()){
  case  PASS:
  case  LAUFPASS:
    if(intention.wait_then_pass == true){
      return holdturn_intention2cmd(intention,cmd);
    }
    speed = intention.kick_speed;
    target = intention.kick_target;
    /*
    MYLOG_POL(0,<<"WBALL03: Intention2cmd: passing to teammate "<<targetplayer_number
	      //<<" onestepkick is possible (0=false) "<<need_only1kick
	      <<" speed "<<speed<<" to target "<<target.x<<" "<<target.y);
    */
    if(intention.risky_pass == true){
      MYLOG_POL(0,<<"WBALL03: RISKY PASS!!!");
    }
    neurokickwrapper->kick_to_pos_with_initial_vel(speed,target);
    neurokickwrapper->get_cmd(cmd);
    return true;
    break;
  case SELFPASS:
    MYLOG_POL(0,<<"Intention type SELFPASS2 to target "<<target
	       <<" w. speed "<<speed<<" targetdir "<<RAD2DEG(intention.target_body_dir.get_value()));
    if(intention.wait_then_pass == true){
      MYLOG_POL(0,<<"Wait then selfpass. ");
      // ridi06: 13.6.:
      // first, test to tingletangle to go on (less stops)
      ttdir = WSinfo::me->ang;
      if(ttdir.diff(intention.target_body_dir)> 100/180.*PI)
	ttdir = intention.target_body_dir;
      tttarget.init_polar(30,ttdir);
      tttarget += WSinfo::me->pos;
      MYLOG_POL(0,"tingle tangle WAIT N SEE targe dir: "<<RAD2DEG(ttdir.get_value()));
      MYLOG_DRAW(0,VC2D(target,1.3,"magenta"));
      
      tingletangle->setTarget(tttarget);
      if(tingletangle->isSafe()== true){
	tingletangle->get_cmd(cmd);
	return true;
      }
      
      // second chance: only Ballzauber
      if(ballzauber->isBallzauberToTargetDirectionPossible(intention.target_body_dir) == true){
	MYLOG_POL(0,<<" intention2cmd: BALLZAUBEr possible. Do turn to dir: "<<RAD2DEG(intention.target_body_dir.get_value()));
	ballzauber->get_cmd(cmd,intention.target_body_dir);
	return true;
      }
      else{
	MYLOG_POL(0,<<" intention2cmd: BALLZAUBEr NOT possible. Holdturn to dir: "
		  <<RAD2DEG(intention.target_body_dir.get_value()));
	return holdturn_intention2cmd(intention,cmd);
      }
    }  // wait then self pass
    selfpass2->get_cmd(cmd,intention.target_body_dir, intention.kick_target, intention.kick_speed);
    return true;
    break;
  case  DRIBBLE:  
    MYLOG_POL(0,"DRIBBLING BETWEEN ");
    dribble_between->set_target(intention.player_target);
    dribble_between->get_cmd(cmd);
    return true;
    break;
  case TINGLETANGLE:
    if (neurodribble2017->is_safe()) //TG17
    {
      MYLOG_POL(0,"NEURO DRIBBLING (2017) to target: "<<intention.player_target<<", ang: "
                   << (intention.player_target - WSinfo::me->pos).ARG() <<", age_vel="<<WSinfo::ball->age_vel );
      neurodribble2017->set_target( (intention.player_target - WSinfo::me->pos).ARG() );
      //neurodribble2017->set_target( ANGLE( -PI/2.0 ) );
      neurodribble2017->get_cmd( cmd );
      //safety check: sim command
      ANGLE dummyANG;
      Vector myNewPos, myNewVel, ballNewPos, ballNewVel;
      Tools::model_cmd_main( WSinfo::me->pos,
                             WSinfo::me->vel,
                             WSinfo::me->ang,
                             WSinfo::ball->pos,
                             WSinfo::ball->vel,
                             cmd.cmd_body,
                             myNewPos,
                             myNewVel,
                             dummyANG,
                             ballNewPos,
                             ballNewVel);
      double dist2BallNextStep = myNewPos.distance(ballNewPos);
      //ND17 has been trained with a safety margin of 0.1,
      //cf. CommandAnalysis::NEURODRIBBLE2017_BALL_SAFETY_MARGIN
      double allowedDist = WSinfo::me->kick_radius - 0.10;
      if (WSinfo::ball->age_vel > 0) allowedDist -= 0.02;
      if ( dist2BallNextStep >= allowedDist )
      {
        MYLOG_POL(0,"NeuroDribble is likely to lose the ball (dist="
          <<dist2BallNextStep<<", r="<<WSinfo::me->radius<<", kr="
          <<WSinfo::me->kick_radius<<"): Fall back to TT!");
      }
      else
      if (   fabs( myNewPos.getX() ) > FIELD_BORDER_X - 1.5
          || fabs( myNewPos.getY() ) > FIELD_BORDER_Y - 1.5 )
      {
        MYLOG_POL(0,"NeuroDribble is likely to get too near to field border (x="
          <<myNewPos.getX()<<", y="<<myNewPos.getY()<<"): Fall back to TT!");
      }
      else
      {
        MYLOG_POL(0,"NeuroDribble is doing fine (dist="
          <<dist2BallNextStep<<", r="<<WSinfo::me->radius<<", kr="
          <<WSinfo::me->kick_radius<<"): Use ND17+++++!");
        return true;
      }
    }
    MYLOG_POL(0,"TINGLE TANGLing to target: "<<intention.player_target<<", ang: "
                << (intention.player_target - WSinfo::me->pos).ARG() );
    tingletangle->setTarget(intention.player_target);
    tingletangle->get_cmd(cmd);
    return true;
    break;
  case SCORE:
    speed = intention.kick_speed;
    target = intention.kick_target;
    if (intention.immediatePass == false){
      mdpInfo::set_my_intention(DECISION_TYPE_SCORE, speed, 0, target.getX(), target.getY(),0);
      MYLOG_POL(0,<<"intention2cmd: try to score w speed "<<speed<<" to target "<<target);
      neurokickwrapper->kick_to_pos_with_initial_vel(speed,target);
      neurokickwrapper->get_cmd(cmd);
    }
    else{
      MYLOG_POL(0,<<"intention2cmd: try to score by sequence (or onestepscore), using onetstepkick with speed "<<speed<<" to target "<<target);
      onestepkick->kick_to_pos_with_initial_vel(speed,target);      
      onestepkick->get_cmd(cmd);
    }
    return true;
    break;
  case KICKNRUSH:
    speed = intention.kick_speed;
    target = intention.kick_target;
    mdpInfo::set_my_intention(DECISION_TYPE_KICKNRUSH, speed,0,0,0, 0);
    MYLOG_POL(0,<<"intention2cmd: kicknrush w speed "<<speed<<" to target "<<target);
    neurokickwrapper->kick_to_pos_with_initial_vel(speed,target);
    neurokickwrapper->get_cmd(cmd);
    return true;
    break;
  case PANIC_KICK:
    MYLOG_POL(0,<< "intention2cmd: PANIC KICK.");
    target = intention.kick_target;
    kick_dir = (target - WSinfo::me->pos).arg() - WSinfo::me->ang.get_value();
    basiccmd->set_kick(100,ANGLE(kick_dir));
    basiccmd->get_cmd(cmd);
    return true;
    break;
  case TACKLING:
    speed = intention.kick_speed;
    MYLOG_POL(0,<< "intention2cmd: Make an angular Tackling (TG).");
    basiccmd->set_tackle(speed);
    basiccmd->get_cmd(cmd);
    return true;
    break;
  case BACKUP:
    MYLOG_POL(0,<<"intention2cmd: back up (two teammates at ball)  not yet implemented");
    LOG_ERR(0,<<"WBALL03 intention2cmd: back up (two teammates at ball)  not yet implemented");
    //ridi03: todo
    return false;
    break;
  case HOLDTURN:
    return holdturn_intention2cmd(intention,cmd);
    break;
  case TURN_AND_DASH:
  {
    ANGLE myAngleToHisGoal = ( HIS_GOAL_CENTER - WSinfo::me->pos ).ARG();
    ANGLE turnAngle = myAngleToHisGoal - WSinfo::me->ang;
    basiccmd->set_turn_inertia( turnAngle.get_value_mPI_pPI() );
    MYLOG_POL(0,<<"intention2cmd: turn to his goal");
    return basiccmd->get_cmd( cmd );
    break;
  }
  default:
    MYLOG_POL(0,<<"intention2cmd: Intention not known");
    LOG_ERR(0,<<" intention2cmd: Intention not known");
    return false;
  }
}


/************************************************************************************************************/

/*  Determine and evaluate Options  */

/************************************************************************************************************/

void Wball06::determine_pass_option(Intention &pass_option,const Vector mypos, const Vector myvel,const ANGLE myang, 
				    const Vector ballpos,const Vector ballvel){
  AState current_state;
  AbstractMDP::copy_mdp2astate(current_state);

  current_state.ball.pos = ballpos;
  current_state.ball.vel = ballvel;
  current_state.my_team[current_state.my_idx].pos = mypos;
  current_state.my_team[current_state.my_idx].vel = myvel;
  determine_best_pass(pass_option, current_state);
}


void Wball06::determine_best_pass(Intention &intention, AState &state){
  AAction best_aaction;

  long ms_time= MYGETTIME;

  intention.reset();

  if (pass_selection->evaluate_passes06(best_aaction, state) == false){ // nothing found
    MYLOG_POL(0,<<"Determine Pass failed: No pass found");
    ms_time = MYGETTIME - ms_time;
    //MYLOG_POL(0, << "Determine Pass intention needed " << ms_time << " millis to decide****************");
    return;
  }

  aaction2intention(best_aaction, intention);

  MYLOG_POL(0,<<"Determine Pass successful: Found a pass (black). Taget "<<intention.kick_target
	     <<" speed "<<intention.kick_speed<<" Value: "<<best_aaction.V);

  MYLOG_DRAW(0, VL2D(WSinfo::ball->pos,
				  intention.kick_target,
				  "black"));

  ms_time = MYGETTIME - ms_time;
  //   MYLOG_POL(0, << "Get Pass intention needed " << ms_time << " millis to decide****************");

  // found a pass or dribbling
  return;
}



/************************************************************************************************************/

/*  Selfpass - stuff  */

/************************************************************************************************************/

bool Wball06::advance_selfpass_area_is_free(const ANGLE dir, const Vector ballpos){

  PlayerSet pset= WSinfo::valid_opponents;
  Vector endofregion;
  double length = 15.; //TG09: ZUI geaendert von 35.0 auf 15.0
  //  double startwidth = 1.;
  double startwidth = 5.0; //TG09: alt 0.2;
  //  double endwidth = 1.5*length;
  double endwidth = 1*length + startwidth;

  if(ballpos.getX() + 5.0 > WSinfo::his_team_pos_of_offside_line()){ // test only
    MYLOG_POL(0,<<"I am close to offsideline, so check more relaxed");
    length = 5.; // check only, if nobody's directly in front of me
    endwidth = 3.;
  }
  endofregion.init_polar(length, dir);
  endofregion += WSinfo::me->pos;
  //  Quadrangle2d check_area = Quadrangle2d(WSinfo::me->pos, endofregion, startwidth, 1.25*length);
  Quadrangle2d check_area = Quadrangle2d(WSinfo::me->pos, endofregion, startwidth, endwidth);
    MYLOG_DRAW(0, check_area );
  pset.keep_players_in(check_area);
  if(pset.num >0){
    if(pset.num <= 1 && pset[0] == WSinfo::his_goalie){
      MYLOG_POL(0,<<"Advance Selfpass area in dir "<<RAD2DEG(dir.get_value())<<" is free: Only goalie before me: free ");
    }
    else{
      MYLOG_POL(0,<<"Advance Selfpass area in dir  "<<RAD2DEG(dir.get_value())<<" is NOT free");
      return false;
    }
  }
  return true;
}


bool Wball06::prefered_direction_is_free(const ANGLE dir, const double length, const double width_factor, const bool consider_goalie){

  PlayerSet pset= WSinfo::valid_opponents;
  Vector endofregion;
  double endwidth = width_factor *length;
  double startwidth = .2;

  endofregion.init_polar(length, dir);
  endofregion += WSinfo::me->pos;
  Quadrangle2d check_area = Quadrangle2d(WSinfo::me->pos, endofregion, startwidth, endwidth);
  MYLOG_DRAW(0, check_area );
  pset.keep_players_in(check_area);
  if(pset.num >0){
    if(pset.num <= 1 && pset[0] == WSinfo::his_goalie && consider_goalie == false){
      MYLOG_POL(0,<<"prefered dir "<<RAD2DEG(dir.get_value())<<" is free: Only goalie before me: free ");
      return true;
    }
    else{
      MYLOG_POL(0,<<"prefered  dir  "<<RAD2DEG(dir.get_value())<<" is NOT free");
      return false;
    }
  }
  MYLOG_POL(0,<<"prefered dir "<<RAD2DEG(dir.get_value())<<" is free: No one before me ");
  return true;
}



bool Wball06::is_goalarea_free(){

  PlayerSet pset= WSinfo::valid_opponents;
  Vector endofregion;
  double length = 30.0;
  double endwidth = 1.5*length;
  double startwidth = 2.0;

  ANGLE dir =(Vector(52.5,0)- WSinfo::me->pos).ARG();

  endofregion.init_polar(length, dir);
  endofregion += WSinfo::me->pos;
  Quadrangle2d check_area = Quadrangle2d(WSinfo::me->pos, endofregion, startwidth, endwidth);
  //  MYLOG_DRAW(0, check_area );
  pset.keep_players_in(check_area);
  if(pset.num >0){
    if(pset.num <= 1 && pset[0] == WSinfo::his_goalie){
      MYLOG_POL(0,<<"prefered dir "<<RAD2DEG(dir.get_value())<<" is free: Only goalie before me: free ");
    }
    else{
      MYLOG_POL(0,<<"prefered  dir  "<<RAD2DEG(dir.get_value())<<" is NOT free");
      return false;
    }
  }
  MYLOG_POL(0,<<"prefered dir "<<RAD2DEG(dir.get_value())<<" is free: No one before me ");
  return true;
}





bool Wball06::selfpass_close2goalline_is_free(const ANGLE dir, const Vector ballpos){

  PlayerSet pset= WSinfo::valid_opponents;
  Vector endofregion;
  //  double length = 35.;
  double length = 10.; // check more relaxed
  //  double startwidth = 1.;
  double startwidth = .2;
  //  double endwidth = 1.5*length;
  double endwidth = 1*length;

  if(ballpos.getX() + 5.0 > WSinfo::his_team_pos_of_offside_line()){ // test only
    MYLOG_POL(0,<<"I am close to offsideline, so check more relaxed");
    length = 5.; // check only, if nobody's directly in front of me
    endwidth = 3.;
  }
  endofregion.init_polar(length, dir);
  endofregion += WSinfo::me->pos;
  //  Quadrangle2d check_area = Quadrangle2d(WSinfo::me->pos, endofregion, startwidth, 1.25*length);
  Quadrangle2d check_area = Quadrangle2d(WSinfo::me->pos, endofregion, startwidth, endwidth);
  //  MYLOG_DRAW(0, check_area );
  pset.keep_players_in(check_area);
  if(pset.num >0){
    if(pset.num <= 1 && pset[0] == WSinfo::his_goalie){
      MYLOG_POL(0,<<" Selfpass close2goalline in dir "<<RAD2DEG(dir.get_value())<<" is free: Only goalie before me: free ");
    }
    else{
      MYLOG_POL(0,<<"Selfpass close2goalline in dir  "<<RAD2DEG(dir.get_value())<<" is NOT free");
      return false;
    }
  }
  return true;
}

bool Wball06::selfpass_goalarea_is_free(const ANGLE dir, const Vector ballpos){

  PlayerSet pset= WSinfo::valid_opponents;
  Vector endofregion;
  double length = 8.; // check more relaxed
  //  double startwidth = 1.;
  double startwidth = .2;
  //  double endwidth = 1*length;
  double endwidth = 1.5*length; //ridi: really free: now have tingletangel

  endofregion.init_polar(length, dir);
  endofregion += WSinfo::me->pos;
  //  Quadrangle2d check_area = Quadrangle2d(WSinfo::me->pos, endofregion, startwidth, 1.25*length);
  Quadrangle2d check_area = Quadrangle2d(WSinfo::me->pos, endofregion, startwidth, endwidth);
  //MYLOG_DRAW(0, check_area );
  pset.keep_players_in(check_area);
  if(pset.num >0){
    if(pset.num <= 1 && pset[0] == WSinfo::his_goalie){
      MYLOG_POL(0,<<" Selfpass GOAL area in dir "<<RAD2DEG(dir.get_value())<<" is free: Only goalie before me: free ");
    }
    else{
      MYLOG_POL(0,<<"Selfpass GOAL area in dir  "<<RAD2DEG(dir.get_value())<<" is NOT free");
      return false;
    }
  }
  return true;
}


bool Wball06::selfpass_area_is_free(const ANGLE dir){

  PlayerSet pset= WSinfo::valid_opponents;
  Vector endofregion;
  double length = 35.;
  double startwidth = 1.;
  endofregion.init_polar(length, dir);
  endofregion += WSinfo::me->pos;
  //  Quadrangle2d check_area = Quadrangle2d(WSinfo::me->pos, endofregion, startwidth, 1.25*length);
  Quadrangle2d check_area = Quadrangle2d(WSinfo::me->pos, endofregion, startwidth, 1.5*length);
  //LOGNEW_DRAW(NEWBASELEVEL, check_area );
  pset.keep_players_in(check_area);
  if(pset.num >0){
    if(pset.num <= 1 && pset[0] == WSinfo::his_goalie){
      MYLOG_POL(5,<<"Selfpass area in dir "<<RAD2DEG(dir.get_value())<<" is free: Only goalie before me: free ");
    }
    else{
      MYLOG_POL(5,<<"Selfpass area in dir  "<<RAD2DEG(dir.get_value())<<" is NOT free");
      return false;
    }
  }
  return true;
}

bool Wball06::selfpass_dir_ok(const ANGLE dir){

  const ANGLE opgoaldir = (Vector(47.,0) - WSinfo::me->pos).ARG(); //go towards goal

  if(dir.diff(ANGLE(0))<50/180.*PI )
    return true;
  if(WSinfo::me->pos.getX()>30 && dir.diff(opgoaldir)<20/180.*PI )
    return true;

  return false;
}


bool Wball06::I_am_close2goalline(){

  if(WSinfo::me->pos.getX() >FIELD_BORDER_X -20.)
    return true;
  return false;
}


bool Wball06::I_am_in_goalarea(){

  if(WSinfo::me->pos.getX() > FIELD_BORDER_X - 18 && fabs(WSinfo::me->pos.getY()) <10)
    return true;
  //BEGIN TG-HANNOVER07
  PlayerSet oppsNearMe = WSinfo::valid_opponents;
  if (WSinfo::his_goalie) oppsNearMe.remove(WSinfo::his_goalie);
  oppsNearMe.keep_and_sort_closest_players_to_point( 1, WSinfo::me->pos );
  if (    WSinfo::me->pos.distance( HIS_GOAL_CENTER ) < 25.0 
       && oppsNearMe.num > 0
       && oppsNearMe[0]->pos.distance(WSinfo::me->pos) > oppsNearMe[0]->age
       && oppsNearMe[0]->pos.distance(WSinfo::me->pos) > 3.0
       && WSinfo::his_team_pos_of_offside_line() - WSinfo::me->pos.getX() < 2.5
     )
  {
    MYLOG_POL(2,<<"TG@wball06: I AM IN GOALAREA."); 
    return true;
  }
  //END TG-HANNOVER07
  return false;
}




void Wball06::determine_best_selfpass(Intention &selfpass_option,const Vector mypos, const Vector myvel, const ANGLE myang,
				       const double mystamina,
					   const Vector ballpos, const Vector ballvel){  
#define NUM_DIRS 20

  if(I_am_in_goalarea()){
    return determine_best_selfpass_in_goalarea(selfpass_option,mypos,myvel,myang, mystamina, ballpos,ballvel);
  }


  if(I_am_close2goalline()){
    return determine_best_selfpass_close2goalline(selfpass_option,mypos,myvel,myang, mystamina, ballpos,ballvel);
  }

  if(mypos.getX() <-30.){// do this only in an advance position
    MYLOG_POL(0," deteremine selfpass: pos < -30, false: "<<mypos.getX());

    selfpass_option.reset();
    return;
  }

  if(DeltaPositioning::get_role(WSinfo::me->number) == 0){ // defender
    MYLOG_POL(0," determine selfpass: Im a defender, false");
    selfpass_option.reset();
    return;
  }

#if 0  
  determine_advance_selfpass(selfpass_option,mypos,myvel,myang, mystamina, ballpos,ballvel);
  return;
#endif

  ANGLE targetdir;
  ANGLE testdir[NUM_DIRS];
  int num_dirs = 0;

  // order the directions to to priority. reason: make selection as stable as possible

  testdir[num_dirs ++] = ANGLE(0); // go straight
#if 1
  if(mypos.getY() >0){  // I am on the left side
    if(myang.get_value_mPI_pPI() >10./ 180. *PI && myang.get_value_mPI_pPI() <  40./ 180. *PI ){
      testdir[num_dirs ++] = myang;
    }
    else
    {
      testdir[num_dirs ++] = ANGLE(20./180.*PI);
    }
    testdir[num_dirs ++] = ANGLE(45./180.*PI);
    testdir[num_dirs ++] = ANGLE(-20./180.*PI);
    // tend to go to left (via wings)
  }
  //  if(mypos.y <+10||  mypos.x >FIELD_BORDER_X - 10  ){ // I am more on the right side
  if(mypos.getY() <0){ // I am more on the right side
    if(myang.get_value_mPI_pPI() <-10./ 180. *PI && myang.get_value_mPI_pPI() >  -40./ 180. *PI ){
      testdir[num_dirs ++] = myang;
    }
    else
    {
      testdir[num_dirs ++] = ANGLE(-20./180.*PI);
    }
    testdir[num_dirs ++] = ANGLE(-45./180.*PI);
    testdir[num_dirs ++] = ANGLE(20./180.*PI);
    
    // tend to go to right or straight (via wings)
  }
#endif


  ANGLE bestdir;
  double bestspeed = 0;
  Vector besttargetpos, bestattackerpos;
  double max_evaluation;
  int bestattacker_num = 0;

  max_evaluation=-1000;
  int reduce_dashes = 0;

  int tmp_advantage;

  for(int i=0; i<num_dirs; i++)
  {
    
    targetdir = testdir[i];
    MYLOG_POL(0,"Check selfpass in dir"<<RAD2DEG(targetdir.get_value()));
    // if the following is activated, selfpasses are rather conservative ...
    //    if(selfpass_area_is_free(targetdir)== true){
    if(advance_selfpass_area_is_free(targetdir,ballpos)== false){
      MYLOG_POL(0,"SELFPASSES area is occupied. not advanced. Do NOT consider");
      continue;
    }
   

    Vector targetpos;
    double kickspeed;
    int steps2go;
    Vector oppos;
    int opnum;
    
    if ( selfpass2->is_selfpass_safe(tmp_advantage,targetdir, kickspeed,
                                     targetpos,steps2go, oppos, opnum,
                                     mypos,myvel, myang,mystamina,ballpos, 
                                     ballvel, false, reduce_dashes))
    {
      double evaluation = Tools::evaluate_pos_analytically(targetpos);

      if(WSinfo::me->pos.distance(targetpos) <MIN_SELFPASS_MOVEMENT)
      {  
        MYLOG_POL(0,"SELFPASS in dir "<<RAD2DEG(targetdir.get_value())
          <<"is safe. BUT MOVEMENT TOO SMALL");
        continue;
      }
      
      //TG09: begin
      PlayerSet tmms = WSinfo::valid_teammates_without_me;
      tmms.keep_and_sort_closest_players_to_point( 1, targetpos );
      if (    tmms.num > 0 
           &&   tmms[0]->pos.distance(targetpos) + (double)tmms[0]->age + 1.0
              < WSinfo::me->pos.distance(targetpos) )
      {  
        MYLOG_POL(0,"SELFPASS in dir "<<RAD2DEG(targetdir.get_value())
          <<"is safe. BUT A TEAMMATE IS NEARER TO TARGET POSITION.");
        continue;
      }
      //TG09: end
      
      double advantageConsideringEvaluation = evaluation + tmp_advantage;

      MYLOG_POL(0,"SELFPASS in dir "<<RAD2DEG(targetdir.get_value())
               <<"is safe.kickspeed"<<kickspeed
             <<" Evaluation "<<evaluation<<" advantage "<<tmp_advantage
             <<" => "<<advantageConsideringEvaluation);

      if (advantageConsideringEvaluation > max_evaluation)
      {
        bestdir = targetdir;
        bestspeed = kickspeed;
        besttargetpos = targetpos;
        max_evaluation = advantageConsideringEvaluation;
        bestattackerpos = oppos;
        bestattacker_num = opnum;
//TG09 ZUI        break;  // ridi 06: take the first available selfpass
      }
    }// selfpass is safe
    else{
      MYLOG_POL(0,"SELFPASS in dir "<<RAD2DEG(targetdir.get_value())
		<<"NOT SAFE");
      ;
    }
  } // for all dirs
  

  if(max_evaluation > -1000) {
    MYLOG_POL(0,"Determine best selfpass finished succesfully "
	       <<" targetdir "<<RAD2DEG(bestdir.get_value()));
    /* directly set intention parameters. can be replaced by set_selfpass(...) */
    selfpass_option.valid_at_cycle = WSinfo::ws->time;
    selfpass_option.valid_since_cycle = WSinfo::ws->time;
    selfpass_option.resultingpos = besttargetpos;
    selfpass_option.kick_speed = bestspeed;
    selfpass_option.target_body_dir = bestdir;
    selfpass_option.V = max_evaluation;
    selfpass_option.attacker_num = bestattacker_num;
    selfpass_option.advantage = tmp_advantage;    
    
  }
  else{
    selfpass_option.reset();
    MYLOG_POL(0,"Determine best selfpass failed: no direction found! ");
  }

  return;
}



void Wball06::determine_best_selfpass_close2goalline(Intention &selfpass_option,const Vector mypos, const Vector myvel, const ANGLE myang,
						   const double mystamina,
						   const Vector ballpos, const Vector ballvel){  
  // side effect: fill out scheduled selfpass formular
  // can be also called with vitual positions.

#define NUM_DIRS 20


  ANGLE targetdir;
  ANGLE testdir[NUM_DIRS];
  int num_dirs = 0;




  // idea: reduce the number of directions as much as possible
  if(mypos.getY() >0){  // I am on the left side -> go right (towards goal)
    testdir[num_dirs ++] = ANGLE(-90/180.*PI);
    if(myang.get_value_mPI_pPI() <-45./180.*PI && myang.get_value_mPI_pPI() <-90./180.*PI)
      testdir[num_dirs ++] = myang;
    else
      testdir[num_dirs ++] = ANGLE(-45/180.*PI);
  }
  
  if(mypos.getY() <0){ // I am more on the right side
    testdir[num_dirs ++] = ANGLE(90/180.*PI);
    if(myang.get_value_mPI_pPI() >45./180.*PI && myang.get_value_mPI_pPI() <90./180.*PI)
      testdir[num_dirs ++] = myang;
    else
      testdir[num_dirs ++] = ANGLE(45/180.*PI);
  }
  
  if(mypos.getX() < FIELD_BORDER_X - 7.0)
  {
    if ( fabs( WSinfo::me->ang.get_value_mPI_pPI() ) < 20./180.*PI ) //TG09
      testdir[num_dirs ++] = WSinfo::me->ang; //TG09
    else
      testdir[num_dirs ++] = ANGLE(0); // go straight
  }



  ANGLE bestdir;
  double bestspeed = 0;
  Vector besttargetpos, bestattackerpos;
  double max_evaluation;
  int bestattacker_num= 0;

  max_evaluation=-1000;
  int reduce_dashes = 0;

  for(int i=0; i<num_dirs; i++){
    targetdir = testdir[i];
    MYLOG_POL(0,"Check  CLOSE2GOALLINE selfpass in dir"<<RAD2DEG(targetdir.get_value()));
    // if the following is activated, selfpasses are rather conservative ...
    if(selfpass_close2goalline_is_free(targetdir, ballpos)== false){
      MYLOG_POL(0,"SELFPASSES  CLOSE2GOALLINE area is occupied. Do NOT consider");
      continue;
    }

    Vector targetpos;
    double kickspeed;
    int steps2go;
    Vector oppos;
    int opnum;
    int tmp_advantage;
    if(selfpass2->is_selfpass_safe(tmp_advantage,targetdir, kickspeed,targetpos,steps2go, oppos, opnum,
				   mypos,myvel, myang,mystamina,ballpos, ballvel, false, reduce_dashes)){

      if(WSinfo::me->pos.distance(targetpos) <MIN_SELFPASS_MOVEMENT){ // 
	MYLOG_POL(0,"SELFPASS in dir "<<RAD2DEG(targetdir.get_value())
		  <<"is safe. BUT MOVEMENT TOO SMALL");
	continue;
      }

      double evaluation= Tools::evaluate_pos_analytically(targetpos);


      MYLOG_POL(0,"SELFPASS CLOSE2GOALLINE in dir "<<RAD2DEG(targetdir.get_value())
		 <<"is safe.  Evaluation "<<evaluation<<" advantage "<<tmp_advantage);
      if(evaluation > max_evaluation){
	bestdir = targetdir;
	bestspeed = kickspeed;
	besttargetpos = targetpos;
	max_evaluation=evaluation;
	bestattackerpos = oppos;
	bestattacker_num = opnum;
      }
    }// selfpass is safe
    else{
      MYLOG_POL(0,"SELFPASS CLOSE2GOALLINE in dir "<<RAD2DEG(targetdir.get_value())
		 <<"NOT SAFE. ");
      ;

    }
  } // for all dirs
  

  if(max_evaluation > -1000) {
    MYLOG_POL(0,"Determine best selfpass "
	       <<" targetdir "<<RAD2DEG(bestdir.get_value()));
    selfpass_option.valid_at_cycle = WSinfo::ws->time;
    selfpass_option.valid_since_cycle = WSinfo::ws->time;
    selfpass_option.resultingpos = besttargetpos;
    selfpass_option.kick_speed = bestspeed;
    selfpass_option.target_body_dir = bestdir;
    selfpass_option.V = max_evaluation;
    selfpass_option.attacker_num = bestattacker_num;
  }
  else{
    selfpass_option.reset();
    MYLOG_POL(0,"TEST SELFPASSES: no direction found! ");
  }

  return;
}



void Wball06::determine_best_selfpass_in_goalarea(Intention &selfpass_option,const Vector mypos, const Vector myvel, const ANGLE myang,
						   const double mystamina,
						   const Vector ballpos, const Vector ballvel){  
  // side effect: fill out scheduled selfpass formular
  // can be also called with vitual positions.

#define NUM_DIRS 20


  ANGLE targetdir;
  ANGLE testdir[NUM_DIRS];
  int num_dirs = 0;

  int direction = 1;

  if(fabs(mypos.getY()) <5.){ // in the middle: let current angle decide about direction
    if(myang.get_value_mPI_pPI() > 0) // I am looking up, so go up
      direction = 1;
    else
      direction = -1;
  }
  else if(mypos.getY() >0) // I am up -> go down
    direction = -1;
  else
    direction = 1;


  bool myANGAdded = false;
  if(mypos.getX() < FIELD_BORDER_X - 7.0)
  {
    if ( fabs( WSinfo::me->ang.diff(ANGLE(0)) ) < 30./180.*PI ) //TG09
    {
      testdir[num_dirs ++] = WSinfo::me->ang;
      myANGAdded = true;
    }
    else
    {
      testdir[num_dirs ++] = ANGLE(0); // go straight
    }
  }
    
  if (    myANGAdded == false
       && fabs( WSinfo::me->ang.diff(ANGLE(direction*45./180.*PI)) ) < 15./180.*PI )
  {
    testdir[num_dirs ++] = WSinfo::me->ang;
    myANGAdded = true;
  }
  else         
    testdir[num_dirs ++] = ANGLE(direction * 45./180.*PI);
    
  //TG09: begin
  if (    myANGAdded == false
       && fabs( WSinfo::me->ang.diff(ANGLE(-direction*45./180.*PI)) ) < 15./180.*PI )
  {
    testdir[num_dirs ++] = WSinfo::me->ang;
    myANGAdded = true;
  }
  else           
    testdir[num_dirs ++] = ANGLE(-direction * 45./180.*PI); //TG09
  //TG09: end

  if(mypos.getX() > FIELD_BORDER_X - 8.0)
  {
    if (    myANGAdded == false
         && fabs( WSinfo::me->ang.diff(ANGLE(direction*90./180.*PI)) ) < 15./180.*PI )
    {
      testdir[num_dirs ++] = WSinfo::me->ang;
      myANGAdded = true;
    }
    else           
      testdir[num_dirs ++] = ANGLE(direction * 90./180.*PI);
  }

  ANGLE bestdir;
  double bestspeed = 0;
  Vector besttargetpos, bestattackerpos;
  double max_evaluation;
  int bestattacker_num= 0;

  max_evaluation=-1000;
  int reduce_dashes = 0;

  for(int i=0; i<num_dirs; i++){
    targetdir = testdir[i];
    MYLOG_POL(0,"Check  GOALAREA selfpass in dir"<<RAD2DEG(targetdir.get_value()));
    // if the following is activated, selfpasses are rather conservative ...
    if(selfpass_goalarea_is_free(targetdir, ballpos)== false){
      MYLOG_POL(0,"SELFPASSES  Goalarea area is occupied. Do NOT consider");
      continue;
    }

    Vector targetpos;
    double kickspeed;
    int steps2go;
    Vector oppos;
    int opnum;
    int tmp_advantage;
    //    double max_dist = 12.0;  // do not move further than that
    double max_dist = 100.0;  // do not restrict, or do it more carefully

    if(selfpass2->is_selfpass_safe_max_advance(tmp_advantage, max_dist, targetdir, kickspeed,targetpos,steps2go, oppos, opnum,
				   mypos,myvel, myang,mystamina,ballpos, ballvel, false, reduce_dashes)){


      //      double evaluation= Tools::evaluate_pos_analytically(targetpos);

      if(mypos.distance(targetpos) <MIN_SELFPASS_MOVEMENT_GOALAREA){ // 
	MYLOG_POL(0,"SELFPASS in dir "<<RAD2DEG(targetdir.get_value())
		  <<"is safe. Improvement NOT ok. Continue");
	continue;
      }

      //      double evaluation= Tools::evaluate_pos_selfpass_neuro06(targetpos);
      double evaluation= Tools::evaluate_pos_analytically(targetpos); //check, probably also analytically

      MYLOG_POL(0,"SELFPASS Goalarea in dir "<<RAD2DEG(targetdir.get_value())
		 <<"is safe.  Evaluation "<<evaluation<<" advantage "<<tmp_advantage
		<<" Improvement: "<<mypos.distance(targetpos));
      if(evaluation > max_evaluation){
	bestdir = targetdir;
	bestspeed = kickspeed;
	besttargetpos = targetpos;
	max_evaluation=evaluation;
	bestattackerpos = oppos;
	bestattacker_num = opnum;
#if 0
	if(WSinfo::me->pos.distance(targetpos) >MIN_SELFPASS_MOVEMENT){ // 
	  MYLOG_POL(0,"SELFPASS in dir "<<RAD2DEG(targetdir.get_value())
		    <<"is safe. Improvement ok. So take this one (priority)");
	  break;
	}
#endif
      }
    }// selfpass is safe
    else{
      MYLOG_POL(0,"SELFPASS goalarea area in dir "<<RAD2DEG(targetdir.get_value())
		 <<"NOT SAFE. ");
      ;

    }
  } // for all dirs
  

  if(max_evaluation > -1000) {
    MYLOG_POL(0,"Determine best selfpass goalarea area "
	       <<" targetdir "<<RAD2DEG(bestdir.get_value()));
    selfpass_option.valid_at_cycle = WSinfo::ws->time;
    selfpass_option.valid_since_cycle = WSinfo::ws->time;
    selfpass_option.resultingpos = besttargetpos;
    selfpass_option.kick_speed = bestspeed;
    selfpass_option.target_body_dir = bestdir;
    selfpass_option.V = max_evaluation;
    selfpass_option.attacker_num = bestattacker_num;
  }
  else{
    selfpass_option.reset();
    MYLOG_POL(0,"TEST SELFPASSES goalarea area: no direction found! ");
  }

  return;
}




/************************************************************************************************************/

/*  Test for a concrete intention                                                                                                                                         */

/************************************************************************************************************/

bool Wball06::is_selfpass_lookfirst_possible(Intention &intention, Intention & selfpass_option){ 

  //  if(status.is_holdturn_safe == true){
  if(ballzauber->isBallzauberToTargetDirectionPossible(selfpass_option.target_body_dir) == true){
    MYLOG_POL(0,"Selfpass scheduled, but information might be old. Ballzauber is possible. So do it."); 
    intention.set_selfpass(selfpass_option.target_body_dir, selfpass_option.resultingpos,selfpass_option.kick_speed, WSinfo::ws->time,selfpass_option.attacker_num);
    intention.wait_then_pass= true;
    return true;
  }
  MYLOG_POL(0,"Selfpass scheduled, but information is old. Ballzauber NOT safe. Do not play.");      	
  return false;
}



bool Wball06::is_selfpass_lookfirst_needed(Intention & selfpass_option){ 
  int targetdir_age = WSmemory::last_seen_in_dir(selfpass_option.target_body_dir);

#if 0  // ridi06: 12.6: make this a little bit more restrictive: only play so risky in advanced situations, otherwise probably look first!
  //  if(WSinfo::me->pos.x + 5.0 > WSinfo::his_team_pos_of_offside_line()){
  if(WSinfo::me->pos.x + 2. > WSinfo::his_team_pos_of_offside_line()){
    MYLOG_POL(0,"Selfpass lookfirst needed: close to offside line -> play immediately");
    return false;
  }
#endif

  if (   (    WSinfo::me->pos.getX() + 5.0 > WSinfo::his_team_pos_of_offside_line()
           && WSinfo::me->pos.getX() >(FIELD_BORDER_X - 45) //TG09: 25->45
           && fabs(selfpass_option.target_body_dir.get_value_mPI_pPI()) < 65.0*PI/180.0//TG09: neu
         )
      || (    WSinfo::me->pos.getX() + 10.0 > WSinfo::his_team_pos_of_offside_line() //TG09: neu
           && WSinfo::me->pos.getY() > 0.5*FIELD_BORDER_Y
           && selfpass_option.target_body_dir.get_value_0_p2PI() < 90.0*PI/180.0
           && selfpass_option.target_body_dir.get_value_0_p2PI() > 10.0*PI/180.0
         )
      || (    WSinfo::me->pos.getX() + 10.0 > WSinfo::his_team_pos_of_offside_line() //TG09: neu
           && WSinfo::me->pos.getY() < -0.5*FIELD_BORDER_Y
           && selfpass_option.target_body_dir.get_value_0_p2PI() > 270.0*PI/180.0  
           && selfpass_option.target_body_dir.get_value_0_p2PI() < 350.0*PI/180.0  
         )
     )
  {
    MYLOG_POL(0,"Selfpass lookfirst needed: close to offside line -> play immediately");
    return false;
  }


  PlayerSet opset= WSinfo::valid_opponents;
  PPlayer attacker = opset.get_player_by_number(selfpass_option.attacker_num);
  if(attacker != NULL){
    int maxopage = 2;
    ANGLE opdir = (attacker->pos  - WSinfo::me->pos).ARG();
    int opage = attacker->age;
    if (   opdir.diff(selfpass_option.target_body_dir) < 100./180.*PI  
        && opage > 1 + selfpass_option.advantage /*TG09: bisher: maxopage*/  
        && WSmemory::last_seen_in_dir(opdir) > maxopage
       )
    {
      MYLOG_POL(0,"Selfpass lookfirst needed: opponent ("
        <<selfpass_option.attacker_num<<") too old (he is "<<opage<<", advantage="
        <<selfpass_option.advantage<<")");
      return true; // should look to opponent first
    }
  }
  if(targetdir_age > 2){
    MYLOG_POL(0,"Selfpass lookfirst needed: targetdir too old");
    return true;
  }
  return false;
}


bool Wball06::test_1vs1(Intention &intention, Intention & pass_option, Intention & selfpass_option){ 
  // currently: try to score by selfpasses; might be extended by dribbling
  
  if(I_am_in_goalarea()== false){
    MYLOG_POL(0,"goalarea: not in area ");
    return false;
  }

  if(pass_option.valid_at() == WSinfo::ws->time){
    if(Tools::can_score(pass_option.resultingpos)){
      MYLOG_POL(0,"goalarea: passreceiver might score -> do not play selfish ");
      return false;
    }
  }

  if(is_goalarea_free() == false){
    MYLOG_POL(0,"goalarea: Area is not free ");
    return false;
  }


  if(selfpass_option.valid_at() != WSinfo::ws->time)
  {
    if ( test_tingletangle_in_scoring_area( intention ) == true )
    {
      MYLOG_POL(0,"goalarea: NO selfpass option, BUT tingletangle IS possible!");
      return true;
    }
    else
    {
      MYLOG_POL(0,"goalarea: NO selfpass option, NO tingletangle possible, do something else ");
      return false;
    }
  }

  // selfpass is possible, so do it
  selfpass_option.wait_then_pass = false; //TG09
  if(is_selfpass_lookfirst_needed(selfpass_option)){ // I should look before I selfpass
    return is_selfpass_lookfirst_possible(intention, selfpass_option);
  }
  intention.set_selfpass(selfpass_option.target_body_dir, selfpass_option.resultingpos ,
			 selfpass_option.kick_speed, WSinfo::ws->time,selfpass_option.attacker_num);

  MYLOG_POL(0,"goalarea: scheduled selfpass successful.");
  return true;


}

bool 
Wball06::test_save_turn_in_opp_pen_area(Intention &intention, Intention & selfpass_option)
{
  Vector nextBallPos = WSinfo::ball->pos + WSinfo::ball->vel,
         nextMyPos   = WSinfo::me->pos + WSinfo::me->vel;
  PlayerSet oppsInFeelRange = WSinfo::valid_opponents;
  oppsInFeelRange.keep_players_in_circle( WSinfo::me->pos, 3.0 );
  ANGLE     hisGoalLeftPostAngle
              = (HIS_GOAL_LEFT_CORNER - nextMyPos).ARG(),
            hisGoalRightPostAngle
              = (HIS_GOAL_RIGHT_CORNER - nextMyPos).ARG();
   
  if (   1
      && WSinfo::ball->pos.distance( HIS_GOAL_CENTER ) < 16.0 
      && WSinfo::ball->age == 0 
      && WSinfo::ball->age_vel == 0
      && fabs(WSinfo::me->ang.get_value_mPI_pPI()) > PI*0.5 
      && oppsInFeelRange.num == 0
      && nextMyPos.distance( nextBallPos ) < WSinfo::me->kick_radius * 0.9
      && (    Tools::could_see_in_direction(hisGoalLeftPostAngle) == false
           || Tools::could_see_in_direction(hisGoalRightPostAngle) == false )
     )
  {
    MYLOG_POL(0,"TEST SAVE TURN IN OPP PEN AREA: YES, use turnanddash intention");
    intention.set_turnanddash( WSinfo::ws->time );
    return true;
  }
  /*MYLOG_POL(0,"TEST SAVE TURN IN OPP PEN AREA: NOPE "
  <<" 1 "<< (WSinfo::ball->age == 0)
  <<" 2 "<< (WSinfo::ball->age_vel == 0)
  <<" 3 "<< fabs(WSinfo::me->ang.get_value_mPI_pPI())
  <<" 4 "<< oppsInFeelRange.num  
  <<" 5 "<< (nextMyPos.distance( nextBallPos ) < WSinfo::me->kick_radius * 0.9)
  <<" 6 "<< (    Tools::could_see_in_direction(hisGoalLeftPostAngle) == false
           || Tools::could_see_in_direction(hisGoalRightPostAngle) == false )
  );*/
  return false;
}

bool Wball06::test_advance_selfpasses(Intention &intention, Intention & selfpass_option){ 

  if(selfpass_option.valid_at() != WSinfo::ws->time){
    MYLOG_POL(0,"TEST ADVANCE SELFPASSES: NO selfpass option ");
    return false;
  }

  if(WSinfo::me->pos.distance(selfpass_option.resultingpos) <MIN_SELFPASS_MOVEMENT){ // 

#if 0 // ridi06 (12.6.): no second chance: that's probably too slow!
    if(WSinfo::me->ang.diff(selfpass_option.target_body_dir) > 10./180 *PI){ // second chance turn to that dir
      Vector targetpos;  // return values of is_turn2dir_safe
      double kickspeed; // return values of is_turn2dir_safe
      int steps2go; // return values of is_turn2dir_safe
      Vector oppos; // return values of is_turn2dir_safe
      int opnum; // return values of is_turn2dir_safe
      int tmp_advantage;
      if(selfpass2->is_turn2dir_safe(tmp_advantage,selfpass_option.target_body_dir, kickspeed,targetpos,steps2go, oppos, opnum)){
	MYLOG_POL(0,"Cannot really advance in scheduled direction, but might turn"<<tmp_advantage);      
	intention.set_selfpass(selfpass_option.target_body_dir, targetpos,kickspeed, WSinfo::ws->time,opnum);
	return true;
      }
    }
#endif // second chance
    MYLOG_POL(0,"TEST SELFPASSES: scheduled; but NO real movement possible. ");
    return false;
  }

  
  selfpass_option.wait_then_pass = false; //TG09
  if(is_selfpass_lookfirst_needed(selfpass_option)){ // I should look before I selfpass
    return is_selfpass_lookfirst_possible(intention, selfpass_option);
  }



  intention.set_selfpass(selfpass_option.target_body_dir, selfpass_option.resultingpos ,
			 selfpass_option.kick_speed, WSinfo::ws->time,selfpass_option.attacker_num);
  intention.wait_then_pass = selfpass_option.wait_then_pass; //TG09

  MYLOG_POL(0,"TEST ADVANCE SELFPASSES: scheduled and Improvement. DO it.");
  return true;
}





bool Wball06::test_tingletangle(Intention& intention){

/*  if(DeltaPositioning::get_role(WSinfo::me->number)  == 0 &&
     (WSinfo::me->pos.x > 15.)){
    MYLOG_POL(0,"test tingle: Im a defender and too advanced. NO TINGLETANGLE");
    return false;
  }*/


#if 0
  double dir = get_prefered_tingletangle_dir();
  Vector target;
  target.init_polar(30,dir);
  target += WSinfo::me->pos;
  MYLOG_POL(0,"tingle tangle targe dir: "<<RAD2DEG(dir));
  MYLOG_DRAW(0,C2D(target.x,target.y,1.3,"magenta"));

  tingletangle->setTarget(target);
  if(tingletangle->isSafe()== false)
    return false;
#endif

  if(status.is_tt_safe == false){
    MYLOG_POL(0,"TINGLETANGLE NOT SAFE");
    return false;
  }

  double dir = status.tt_dir;
  Vector target;
  target.init_polar(30,dir);
  target += WSinfo::me->pos;
  MYLOG_POL(0,"tingle tangle targe dir: "<<RAD2DEG(dir));
  MYLOG_DRAW(0,VC2D(target,1.3,"magenta"));
  tingletangle->setTarget(target);
  intention.set_tingletangle(target, WSinfo::ws->time);
  tt_last_active_at = WSinfo::ws->time;
  return true;
}


bool Wball06::test_dribbling(Intention& intention){
	// I just assume that the target has been set by is_dribblebetween_possible().

  if(!status.is_dribble_ok)
    return false;
  intention.set_dribble(dribble_between->get_target(), WSinfo::ws->time);
  /* Todo: do set neck at a central place :*/
  if(dribble_between->is_neck_req_set())
    //    Tools::set_neck_request(NECK_REQ_LOOKINDIRECTION, dribble_between->get_neck_req());
    status.dribble_neckreq.set_request(NECK_REQ_LOOKINDIRECTION, dribble_between->get_neck_req().get_value());
  return true;
}


bool Wball06::test_priority_pass(Intention &intention, Intention &pass_option, Intention &selfpass_option){

  if(pass_option.valid_at() != WSinfo::ws->time) // no pass option available
    return false;

  Vector pass_target;
  double pass_speed;
  pass_option.get_kick_info(pass_speed, pass_target);
  Vector pass_resulting_pos = pass_option.resultingpos;
  Vector pass_potential_pos = pass_option.resultingpos;
  ANGLE pass_dir = (pass_target-WSinfo::me->pos).ARG();

  MYLOG_POL(0,<<"Testing the following priority pass. Taget "<<pass_option.kick_target
	     <<" speed "<<pass_option.kick_speed
	     <<" Respos: "<<pass_resulting_pos<<" potentialpos: "<<pass_potential_pos);

  /*
  MYLOG_DRAW(0, L2D(WSinfo::ball->pos.x, WSinfo::ball->pos.y, 
				  pass_option.kick_target.x,pass_option.kick_target.y,
				  "red"));
  */

  // compute all the predicates first
  bool overcome_offside_tm = Tools::can_advance_behind_offsideline(pass_potential_pos);
  bool attacked = am_I_attacked(4.0, 2.0);//TG17
  bool selfpass_possible = (selfpass_option.valid_at()==WSinfo::ws->time);
  // ranges of pass quality:  ok, cool, supercool, killer
  bool pass_quality_cool =false;


  //  Vector my_potential_pos = Tools::check_potential_pos(WSinfo::me->pos, 5.0); // check potential advance with max. 5 m
  Vector my_potential_pos;
  if(selfpass_option.valid_at()==WSinfo::ws->time){ // selfpass is an option
    /* Todo: check this replacement:
    my_potential_pos = scheduled_selfpass.targetpos; 
    */ ;
    my_potential_pos = selfpass_option.resultingpos;    
  }
  else{
    my_potential_pos = WSinfo::me->pos;  // my pot. pos might be harmful
  }

  // determine principle pass quality:
  double evaluation_delta;

  
  ANGLE myprefered_dir = ANGLE(0);
  ANGLE myprefered_ttdir = ANGLE(status.tt_dir);

  if(WSinfo::me->pos.getX()>35.)
    myprefered_dir =(Vector(52.5,0)- WSinfo::me->pos).ARG();

  bool potential2advance = status.is_dribble_ok || selfpass_possible;


  if(Tools::compare_positions(pass_potential_pos, my_potential_pos,evaluation_delta) == EQUAL){
    // positions are in the same equivalence class
    if(pass_resulting_pos.getX()>WSinfo::me->pos.getX()){
      // I don't have to play a back pass to get this potential position.
      if(evaluation_delta >=0.){
	MYLOG_POL(0, <<"I can at least improve a bit by passing.  QUALITY ok.");
      }
      if(evaluation_delta >10. && 
	 ((DeltaPositioning::get_role(WSinfo::me->number) == 0) 
	  || potential2advance == false)){
	MYLOG_POL(0, <<" I cannot advance or I am a defender and pass is ok. Cool pass");
	pass_quality_cool= true;
      }
    } // resulting pos is ok.
    // enables many passes:, avoids selfish play
    // ridi06: 15.6. change this at offsideline and check resulting position
    // be more selfish, if tingletangle is true!
    // also, dream selfpasses might be possible
    // if activated, it should be at least restriced(e.g. pass goes
    // forward, not breakthrough, not a scoring sit,...)
#if 1
  PlayerSet pset= WSinfo::valid_opponents;
  double radius_of_attacked_circle =  3.5 * ServerOptions::kickable_area;
  pset.keep_players_in_circle(WSinfo::me->pos,radius_of_attacked_circle);
    
  if(      ( attacked == true  )
        && selfpass_possible == false 
	&& evaluation_delta > 0
//        && WSinfo::me->pos.x > FIELD_BORDER_X - 12.0 
//        && WSinfo::me->pos.x < WSinfo::his_team_pos_of_offside_line() - 2.0 
    )  
  {
      MYLOG_POL(0, <<" I am attacked and evaluation >0. No selfpass possible: Cool pass");
      pass_quality_cool = true;
  }
    
#endif
  }  // pass position and my position are EQUAL


  MYLOG_POL(0, <<"Compare  potential passpos:"<<pass_potential_pos
	    <<" to my potential pos "<<my_potential_pos<<" wins: "
	    <<Tools::compare_positions(pass_potential_pos, my_potential_pos,evaluation_delta)
	    <<" eval. delta: "<<evaluation_delta);


  // avoid to get stuck in the corner before the goal:
  if(WSinfo::me->pos.getX() > FIELD_BORDER_X - 5.  &&   // I am very far advanced
     prefered_direction_is_free(myprefered_dir, 7.0, 1.0, true) == false && // direction to the goal is blocked by opponent or goalie
     pass_resulting_pos.getX() > FIELD_BORDER_X - 20  && // pass goal is not too far back
     (fabs(pass_resulting_pos.getY()) < fabs(WSinfo::me->pos.getY()) +6. && // pass has to go to the middle and a little back
      pass_resulting_pos.getX() < WSinfo::me->pos.getX() -3)
     ){
    MYLOG_POL(0, <<" I am far advanced, goalie is before me pass found -> cool");
    pass_quality_cool = true; // spiele, egal welche y-Koordinate mein Mitspieler hat
  }


  if(fabs(WSinfo::me->pos.getY()) > 16.){ // play passes quickly from outside
    // * consider situations, when I am very far advanced I cannot move to the middle and a pass can be played
    // Situation 'vorne, aussen: Spiele nach hinten um aufzul?sen.
    if(WSinfo::me->pos.getX() > FIELD_BORDER_X - 10.  &&
       prefered_direction_is_free(myprefered_dir, 4.0) == false && // direction to the goal is blocked by opponent other than goalie
       pass_resulting_pos.getX() > FIELD_BORDER_X - 20  && // pass goal is not too far back
       (fabs(pass_resulting_pos.getY()) < fabs(WSinfo::me->pos.getY()) || // either pass goes to the middle
	pass_resulting_pos.getX() < WSinfo::me->pos.getX() -5)  // or pass is not a parallel pass to the goalline (that brings nothing)
       ){
      MYLOG_POL(0, <<" I am far advanced, cannot go towards goal, and pass found -> cool");
      pass_quality_cool = true; // spiele, egal welche y-Koordinate mein Mitspieler hat
    }
    

    // * consider situations, when I am very far advanced I cannot move to the middle and a pass can be played
    // Situation 'vorne, aussen: Spiele nach hinten um aufzul?sen.
    // this rule makes the game faster, if a teammate is not too bad positioned an I cannot go further
    if(WSinfo::me->pos.getX() > FIELD_BORDER_X - 10.  &&
       prefered_direction_is_free(myprefered_dir, 8.0) == false && // direction to the goal is blocked by opponent other than goalie
       pass_resulting_pos.getX() > FIELD_BORDER_X - 20  && // pass goal is not too far back
       (fabs(pass_resulting_pos.getY()) < fabs(WSinfo::me->pos.getY()) +6. && // pass has to go to the middle and a little back
	pass_resulting_pos.getX() < WSinfo::me->pos.getX() -5)
       ){
      MYLOG_POL(0, <<" I am far advanced, cannot go towards goal, and pass found -> cool");
      pass_quality_cool = true; // spiele pss, da Mitspieler gut postiert
    }
  }
  else{ // vor dem Tor spiele ein bisschen egoistischer!
    if(WSinfo::me->pos.getX() > FIELD_BORDER_X - 10.  &&
       prefered_direction_is_free(myprefered_dir, 2.0, 1.0, true) == false && // direction to the goal is blocked by opponent
       prefered_direction_is_free(myprefered_ttdir, 2.0) == false && // tingle tangle direction is blocked by opponent other than goalie
       pass_resulting_pos.getX() > FIELD_BORDER_X - 20  && // pass goal is not too far back
       (fabs(pass_resulting_pos.getY()) < fabs(WSinfo::me->pos.getY()) +3. && // pass has to go to the middle and a little back
	pass_resulting_pos.getX() < WSinfo::me->pos.getX() -3)
       ){
      MYLOG_POL(0, <<" I am far advanced, cannot go towards goal, and pass found -> cool");
      pass_quality_cool = true; // spiele pss, da Mitspieler gut postiert
    }

  }

#if LOGGING && BASIC_LOGGING && WBALL06LOGGING
  bool iCanScore         = Tools::can_score(WSinfo::me->pos);
#endif
  bool iCanActuallyScore = Tools::can_actually_score(WSinfo::me->pos);
#if LOGGING && BASIC_LOGGING && WBALL06LOGGING
  bool teammateCanScore  = Tools::can_score(pass_potential_pos);
#endif
  bool teammateCanActuallyScore = Tools::can_actually_score(pass_potential_pos);

  MYLOG_POL(0, " I can score "<< iCanScore
	    << " I can acutally score "<< iCanActuallyScore
	     <<" Teammate CAN SCORE: "<< teammateCanScore
	    <<" Teammate can ACTUALLY score: "<<teammateCanActuallyScore);


  //  if(Tools::can_score(pass_potential_pos) == true && potential2advance == false){ 
#if 0 
  if(Tools::can_actually_score(pass_potential_pos) == true){  // ridi06: was can_score() ; relaxed checking without goalie
    // if I could score, than I would have done goalarea (previous in priority queue). So pass
#endif
    // ridi06. 16.6: play more selfish
  if(teammateCanActuallyScore == true &&
     iCanActuallyScore == false && status.is_tt_safe){  
    MYLOG_POL(0, <<"Teammate can score, I not (otherwise would have done). pass quality cool!");
    pass_quality_cool = true;
  }

  if(Tools::compare_positions(pass_potential_pos, my_potential_pos,evaluation_delta) == FIRST){
    MYLOG_POL(0, <<"Teammate has greater potential than I: pass quality cool");
    pass_quality_cool = true;
  }

  if(attacked == true && potential2advance == false && 
     overcome_offside_tm == true &&  WSinfo::his_team_pos_of_offside_line() < FIELD_BORDER_X - 12.){  // ridi06: 9.6. be more selfish
    MYLOG_POL(0, <<"I am attacked and teammate can overcome offside: pass quality cool");
    pass_quality_cool = true;
  }

#if 1
  //TG+MR, 26.06.2007: Unterstuetzung des Noball07AttackScoringArea-seitigen
  //"help 1vs1"-Submodus
  PlayerSet oppsNearTeammate = WSinfo::valid_opponents;
  oppsNearTeammate.keep_players_in_circle( pass_potential_pos, 3.0 );
  Vector passTarget = pass_potential_pos - WSinfo::me->pos;
  passTarget.normalize( passTarget.norm() + 1.0 );
  passTarget += WSinfo::me->pos;
  PlayerSet oppsInPassWay = WSinfo::valid_opponents;
  Quadrangle2d passWay( WSinfo::me->pos, passTarget, 2.0, 4.0 );
  oppsInPassWay.keep_players_in( passWay );
  if (    iCanActuallyScore 
       && teammateCanActuallyScore
       && WSinfo::me->pos.distance( pass_potential_pos ) < 5.0
       && oppsNearTeammate.num == 0 
       && oppsInPassWay.num == 0 )
  { 
    MYLOG_POL(0, <<"me and my teammate can score, the teammate tries to help me -> pass");
    TGLOGPOL(0, <<"me and my teammate can score, the teammate tries to help me -> pass");
    pass_quality_cool = true;    
  }
#endif

#if 0 // ridi06 12.6.: too restrictive: can still try to tingletangle. do this only if I cant move at all or pass is really good
  if(Tools::close2_goalline(WSinfo::me->pos) && (attacked == true ||potential2advance == false) 
     && pass_resulting_pos.x > 25.){ 
    // I am already close to goalline, now play a pass if reasonable
    MYLOG_POL(0, <<"I am close2 goalline and am attacked -> pass");
    pass_quality_cool = true;    
  }
#endif


  bool do_priority_pass = false;

  if(pass_quality_cool == true){
    MYLOG_POL(0, <<"PRIORITY PASS: Pass quality_cool");
    do_priority_pass = true;
  }

  if(do_priority_pass == false){
    MYLOG_POL(0, <<"PRIORITY PASS: Scheduled pass not considered cool! Do NOT play PRIORITY PASS");
    return false;
  }

  // * Play this pass. Just check if wait or play immediately.
  intention = pass_option;
  MYLOG_POL(0, <<"PRIORITY PASS:  Do play PRIORITY PASS");
  check_if_wait_then_pass(intention);
  return true;
}

bool
Wball06::test_pass_when_playing_against_aggressive_opponents_08
   (Intention &intention, Intention &pass_option, Intention &selfpass_option)
{
  //exclude the case where i have no pass option
  if(pass_option.valid_at() != WSinfo::ws->time)
  {
    MYLOG_POL(0, <<"TG@Wball06-passAggrOpp: NOPE, I have no pass option.");
    return false;
  }
  //exclude the case where i have no opponent in my feel range
  PlayerSet oppsInFeelRange = WSinfo::valid_opponents;
  PlayerSet oppsAlmostInFeelRange = WSinfo::valid_opponents; 
  oppsInFeelRange.keep_players_in_circle
                  ( WSinfo::me->pos, 
                    ServerOptions::visible_distance );
  oppsAlmostInFeelRange.keep_players_in_circle
                  ( WSinfo::me->pos, 
                    ServerOptions::visible_distance * 1.5 );
  if ( oppsAlmostInFeelRange.num == 0 )
  {
    MYLOG_POL(0, <<"TG@Wball06-passAggrOpp: NOPE, I have no opps around me.");
    return false;
  }
  int numberOfHighDangerOpponents = 0;
  for (int i=0; i<oppsAlmostInFeelRange.num; i++)
  {
    Vector nearOppsVec2Me = WSinfo::me->pos - oppsAlmostInFeelRange[i]->pos;
    if (   oppsAlmostInFeelRange[i]->age == 0
        && (  (   oppsAlmostInFeelRange[i]->age_ang == 0
               && fabs( (   nearOppsVec2Me.ARG()
                          - oppsAlmostInFeelRange[i]->ang ).get_value_mPI_pPI() ) < PI/6.0
              ) 
            ||
              (   oppsAlmostInFeelRange[i]->age_vel == 0
               && oppsAlmostInFeelRange[i]->vel.norm() > 0.2
               && fabs( (   nearOppsVec2Me.ARG()
                          - oppsAlmostInFeelRange[i]->vel.ARG() ).get_value_mPI_pPI() ) < PI/6.0 
              )      
           )
       )
    {
      numberOfHighDangerOpponents ++ ;
      break;
    }
  }
  if (    numberOfHighDangerOpponents == 0
       && oppsInFeelRange.num == 0 )
  {
    MYLOG_POL(0, <<"TG@Wball06-passAggrOpp: NOPE, there is no high danger opponent.");
    return false;
  }
  //exlcude the case where i am so advanced that i may soon score
  if (    WSinfo::me->pos.getX() > WSinfo::his_team_pos_of_offside_line() - 2.0
       || WSinfo::me->pos.distance(HIS_GOAL_CENTER) < 22.0 ) //TG09: changed from && to ||
  {
    MYLOG_POL(0, <<"TG@Wball06-passAggrOpp: NOPE, I want to score soon.");
    return false;
  }
  //exclude the case where an up-to-date opponent is not heading towards me
  Vector oppVectorTowardsMe
    = WSinfo::me->pos - oppsInFeelRange[0]->pos; 
  if (    (   oppsInFeelRange[0]->age_vel == 0
           && oppsInFeelRange[0]->pos.distance(WSinfo::me->pos) > 2.5//TG09: 2.0->2.5
           && oppsInFeelRange[0]->vel.norm() < 0.3 )//TG09: 0.6->0.3
       ||
          (   oppsInFeelRange[0]->age_ang == 0
           && fabs( (   oppVectorTowardsMe.ARG()
                      - oppsInFeelRange[0]->ang ).get_value_mPI_pPI() ) > PI/4.0
           && oppsInFeelRange[0]->pos.distance(WSinfo::me->pos) > 2.0 )
     )
  {
    MYLOG_POL(0, <<"TG@Wball06-passAggrOpp: NOPE, opp is no real danger: "
      <<"oppAgeAng="<<oppsInFeelRange[0]->age_ang<<", oppAgeVel="
      <<oppsInFeelRange[0]->age_vel<<", oppVel="
      <<oppsInFeelRange[0]->vel<<".");
    return false;
  }
  double  passVelocity;
  Vector passTarget;
  int    passTargetPlayerNumber;
  pass_option.get_kick_info( passVelocity, passTarget, passTargetPlayerNumber );
  //exclude too outdated target players
  PPlayer passTargetPlayer = WSinfo::valid_teammates.get_player_by_number
                                                     (passTargetPlayerNumber);
  if (    passTargetPlayer == NULL
       || passTargetPlayer->age > 2 )
  {
    MYLOG_POL(0, <<"TG@Wball06-passAggrOpp: NOPE, target player ("
      <<passTargetPlayerNumber<<") too old.");
    return false;
  }
  //exclude highly dangerous passes along my offside line / sechzehner
  XYRectangle2d dangerZone
     ( Vector( -FIELD_BORDER_X, 20.0),
       Vector( WSinfo::my_team_pos_of_offside_line() + 15.0, -20.0 ) );
  if (    dangerZone.inside( passTargetPlayer->pos )
       &&   passTargetPlayer->pos.distance(MY_GOAL_CENTER)
          < WSinfo::ball->pos.distance(MY_GOAL_CENTER) + 5.0 )
  {
    MYLOG_POL(0, <<"TG@Wball06-passAggrOpp: NOPE, I should not pass in the"
      <<" danger zone.");
    return false;
  }
  
  //no exclusion was effective -> play the pass and be happy    
  intention = pass_option;
  MYLOG_POL(0, <<"TG@Wball06-passAggrOpp: YEP, play my pass option (danOppAgeAng="
    <<oppsInFeelRange[0]->age_ang<<" danOppAgeVel="<<oppsInFeelRange[0]->age_ang
    <<" danOppVel="<<oppsInFeelRange[0]->vel<<").");
  return true;
}

bool Wball06::test_pass_under_attack(Intention &intention, Intention &pass_option, Intention &selfpass_option){

  if(pass_option.valid_at() != WSinfo::ws->time) // no pass option available
    return false;

  if(selfpass_option.valid_at()==WSinfo::ws->time){ 
    MYLOG_POL(0, <<"Selfpass is an option. Do not play emergency pass.");
    return false;
  }

  if(Tools::can_score(WSinfo::me->pos) == true){
    MYLOG_POL(0, <<"I can probably score. Do not play an emergency pass..");
    return false;
  }

  bool pass_quality_ok = false;
  if(pass_option.resultingpos.getX()>WSinfo::me->pos.getX()-5.){  // accept even slight back passes
    pass_quality_ok= true;
  }

  bool do_priority_pass = false;

  if(pass_quality_ok){
    if(status.is_dribble_safe == false && status.is_holdturn_safe == false) 
      do_priority_pass = true;      
    if(status.is_dribble_ok == false && is_my_passway_in_danger(pass_option)){ // I cannot go further and pass is in danger: pass
      MYLOG_POL(0, <<"Emergency PASS: Cannot dribble and passway in danger -> pass");
      do_priority_pass = true;
    }
  }

  if(status.is_holdturn_safe == false && status.is_dribble_ok == false){ // holdturn not safe
    MYLOG_POL(0, <<"Emergency pass:  Holdturn not safe -> pass");
    do_priority_pass = true;
  }


  if(do_priority_pass == false){
    MYLOG_POL(0, <<"Emergency pass: Scheduled pass not considered cool! Do NOT play PRIORITY PASS");
    return false;
  }

  intention = pass_option;
  MYLOG_POL(0, <<"PASS UNDER ATTACK:  Do play PASS UNDER ATTACK");
  check_if_wait_then_pass(intention);  // pass should be played immediately; don't wait
  
  return true;
}

void Wball06::check_if_wait_then_pass(Intention &pass_intention)
{
  
  bool pass_is_announced = is_pass_announced(pass_intention); 
  ANGLE pass_dir = (pass_intention.kick_target-WSinfo::me->pos).ARG();
  int last_looked_in_dir = WSmemory::last_seen_in_dir(pass_dir);
  bool view_information_is_fresh = false;
  int kicks2pass = howmany_kicks2pass(pass_intention);
  bool I_can_look_in_passdir = Tools::could_see_in_direction(pass_dir);

  // default: look in my direction
  ANGLE desired_target_body_dir = WSinfo::me->ang;  

  if (    I_can_look_in_passdir == false 
       || WSinfo::me->ang.get_value_mPI_pPI() > 90./180. *PI 
       || WSinfo::me->ang.get_value_mPI_pPI() < -90./180. *PI)
  {
    // turn, but not more than +- 90 degrees!
    desired_target_body_dir = pass_dir;
    if (desired_target_body_dir.get_value_mPI_pPI() >90./ 180. *PI)
      desired_target_body_dir = ANGLE(90./180. *PI);
    if (desired_target_body_dir.get_value_mPI_pPI() <-90./ 180. *PI)
      desired_target_body_dir = ANGLE(-90./180. *PI);
  }

  int teammate_age = 1000;

  PPlayer p= WSinfo::get_teammate_by_number(pass_intention.target_player);
  if (p)
  {
    teammate_age = p->age;
    if (teammate_age <=1)
    { //maybe I heard of him; then maybe I could play immediately: todo
      MYLOG_DRAW(0, VC2D(p->pos, 1.3, "orange"));
    }
  }

  MYLOG_POL(0, <<"Wait Then pass: passdir: "<<RAD2DEG(pass_dir.get_value())
	<<" desired_target body dir: "<<RAD2DEG(desired_target_body_dir.get_value())
    <<" temmate: age: "<<teammate_age);


  if(last_looked_in_dir <= 1)
  {
    view_information_is_fresh = true;
  }

#if 0 // might be activated
  if(teammate_age <= 1)
  {
    view_information_is_fresh = true;
  }
#endif
 
  if (   (   teammate_age <=1 
          || view_information_is_fresh == true ) 
      && pass_intention.subtype == SUBTYPE_PASS)
  {
    MYLOG_POL(0, <<"Wait Then pass?: I know where my teammate is and DIRECT "
      <<"Pass -> play immediately");
    pass_intention.wait_then_pass = false;
    return;
  }

  // 1. check, if everything's ok
  if (   pass_is_announced == true 
      && view_information_is_fresh == true)
  {
    //everythings'  allright, pass
    MYLOG_POL(0, <<"Wait Then pass?: I looked into direction and communicated -> Play Pass");
    pass_intention.wait_then_pass = false;
    return;
  }
  
  //2. pass not accepted yet, maybe I did not look into pass direction
  if (view_information_is_fresh == false)
  {
    if (kicks2pass >= 2 && I_can_look_in_passdir)
    {  // need at least 2 kicks and can look in pass direction
      MYLOG_POL(0, <<"Wait Then pass?: view information old, but I need 2 kicks anyway, and can look in passdir -> DO PRIRORITY PASS");
      pass_intention.wait_then_pass = false;
      return; // alternatively: immediately play killer pass then
    }
    // want to play pass, but need to look first
    if (status.is_holdturn_safe == true)
    {
      MYLOG_POL(0, <<"Wait Then pass?: Haven't looked in dir,  HoldTurn possible. Hold and look");
      pass_intention.target_body_dir = desired_target_body_dir;
      pass_intention.wait_then_pass = true;
      return;
    }
    else
    { // this is a very risky situation: didn't look, maybe not 
      //communicated, but cannot hold ball!!! So I have to play
      MYLOG_POL(0, <<"Wait Then pass?: Haven't looked in dir, but HoldTurn NOT possible. Play immediately");
      pass_intention.wait_then_pass = false;
      return;     
    }
  }
  
  if (pass_is_announced == false)
  {
    if (kicks2pass >= 2)
    { //TG08
      MYLOG_POL(0, <<"Wait Then pass?: Pass not yet communicated, but I need 2 "
        <<"kicks anyway, and can look in passdir -> DO PRIRORITY PASS");
      pass_intention.wait_then_pass = false;
      return; 
    }
    else
    if (status.is_holdturn_safe == true)
    {
      MYLOG_POL(0, <<"Wait Then pass?: Pass not communicated,  HoldTurn possible. Hold and look");
      pass_intention.target_body_dir = desired_target_body_dir;
      pass_intention.wait_then_pass = true;
      return;
    }
    else
    { // this is a very risky situation: didn't look, maybe not 
      //communicated, but cannot hold ball!!! So I have to play
      MYLOG_POL(0, <<"Wait Then pass?: Pass not communicated, but HoldTurn NOT possible. Play immediately");
      pass_intention.wait_then_pass = false;
      return;     
    }
  }
}





/** What to do if two opponents attack me */
bool Wball06::test_in_trouble(Intention &intention){
  // test for situation 'a': Ball is in kickrange of me and opponent!

  Vector target;

  PlayerSet pset= WSinfo::valid_opponents;
  pset.keep_players_in_circle(WSinfo::ball->pos, 2.0*ServerOptions::kickable_area); 

  bool allOppsHaveTackled = true;
  for ( int i=0; i<pset.num; i++ )
  {
    if (WSinfo::is_ball_kickable_for( pset[i] ) == false)
      continue;
    int minInact = 0, maxInact;
    WSinfo::get_player_inactivity_interval( pset[i], minInact, maxInact );
    if (minInact <= 0) allOppsHaveTackled = false;
  }
  if (allOppsHaveTackled)
  {
    MYLOG_POL(0, << "IN TROUBLE: All opps have tackled. => NOPE ");
    return false;
  }

  // considr only close ops. for correct
  pset.keep_and_sort_closest_players_to_point(1,WSinfo::ball->pos);
  if ( pset.num == 0 )
    return false; // no op. has ball in kickrange
  else
  {
    if ( WSinfo::ball->pos.distance( pset[0]->pos )//TG08: reduced from 10 to 7.5 
         > 1.075 * pset[0]->kick_radius ) //7.5% additional safety margin
    {
      return false;
    }
  }

  MYLOG_POL(0, << "IN TROUBLE: Ball is in kickrange of me and opponent "<<pset[0]->number);
  TGLOGPOL(0, << "WBALL06: Ball is in kickrange of me and opponent "<<pset[0]->number);

  // use this instead:
  if(   WSinfo::me->pos.getX() >-10
     && WSinfo::ball->pos.distance( pset[0]->pos ) > 1.0 * pset[0]->kick_radius ) 
  {//10% additional safety margin
    MYLOG_POL(0, << "I am advanced, and Ball might not be in KR of opponent yet, continue.");
    return false;
  }
  else
  {
    MYLOG_POL(0, << "Wball06: In trouble: opp<->ball="
      <<WSinfo::ball->pos.distance( pset[0]->pos )
      <<", opp.kr="<<pset[0]->kick_radius);
  }


  //////////////////////
  // ADD-ON by TG 04/05
  //////////////////////
  if (   WSinfo::me->pos.sqr_distance( MY_GOAL_CENTER ) < 20*20 
      && WSinfo::ball->age == 0 //TG09
     )
  {
    double tackSuccProb = Tools::get_tackle_success_probability( WSinfo::me->pos, WSinfo::ball->pos, WSinfo::me->ang.get_value_0_p2PI() );
    PlayerSet kickCapableOpponents = WSinfo::valid_opponents;
    kickCapableOpponents.keep_and_sort_closest_players_to_point( 1, WSinfo::ball->pos );
    double oppKickPower = 0.0,
          ownKickPower = 0.0;
    if ( kickCapableOpponents.num > 0 )
    {
      oppKickPower = 100.0;
      Vector oppBallVec = (WSinfo::ball->pos - kickCapableOpponents[0]->pos);
      double  oppBallDistNetto = oppBallVec.norm() - kickCapableOpponents[0]->radius - ServerOptions::ball_size;
      double  oppBallAngle = (oppBallVec.ARG() - kickCapableOpponents[0]->ang).get_value_mPI_pPI();
      oppKickPower *= ServerOptions::kick_power_rate *
        (1 - 0.25*fabs(oppBallAngle)/PI - 0.25*oppBallDistNetto/(kickCapableOpponents[0]->kick_radius - kickCapableOpponents[0]->radius - ServerOptions::ball_size));
    }
    ownKickPower = 100.0;
    Vector ownBallVec = (WSinfo::ball->pos - WSinfo::me->pos);
    double  ownBallDistNetto = ownBallVec.norm() - WSinfo::me->radius - ServerOptions::ball_size;
    double  ownBallAngle = (ownBallVec.ARG() - WSinfo::me->ang).get_value_mPI_pPI();
    ownKickPower *= ServerOptions::kick_power_rate *
        (1 - 0.25*fabs(ownBallAngle)/PI - 0.25*ownBallDistNetto/(WSinfo::me->kick_radius - WSinfo::me->radius - ServerOptions::ball_size));

    //NOTE: A tackling is only advisable if
    //      (a) it has a very high success probability
    //      (b) I am quite weaker in shooting currently
    //      (c) when tackling I am much stronger than the opponent
    double kickPowerDelta = 0.25,
          powerWhenTackling = 100*ServerOptions::kick_power_rate,
          maxPowerWhenTackling = 0.0;
    int goodTackleAngleFound = -1000;
    if ( ClientOptions::server_version >= 12.0 ) //TG09
    {
      for (int checkAngle = 0; checkAngle < 360; checkAngle +=10 )
      {
        //tackle power depends on the tackle angle
        double effectiveTacklePower
          =   ServerOptions::max_back_tackle_power
            +   (ServerOptions::max_tackle_power - ServerOptions::max_back_tackle_power)
              * (fabs(180.0 -  checkAngle )/180.0);
        Vector player_2_ball = WSinfo::ball->pos - WSinfo::me->pos;
        player_2_ball.rotate( - WSinfo::me->ang.get_value_mPI_pPI() );
        ANGLE player_2_ball_ANGLE = player_2_ball.ARG();
        effectiveTacklePower
          *= 1.0 - 0.5 * ( fabs( player_2_ball_ANGLE.get_value_mPI_pPI() ) / PI );
        //double relativeTacklePower = effectiveTacklePower / 100.0;
        powerWhenTackling = effectiveTacklePower * ServerOptions::tackle_power_rate;
        if (powerWhenTackling>maxPowerWhenTackling)
          maxPowerWhenTackling = powerWhenTackling;
        if (    tackSuccProb > 0.9 
             && oppKickPower - ownKickPower > kickPowerDelta
             && powerWhenTackling - oppKickPower > kickPowerDelta
            ) 
        {
          Vector hisKickVector = (MY_GOAL_CENTER - WSinfo::ball->pos);
          hisKickVector.normalize( oppKickPower );
          ANGLE myTackleANGLE = ANGLE(PI*(double)checkAngle/180.0) + WSinfo::me->ang;
          Vector myTackleVector( myTackleANGLE );
          myTackleVector.normalize( powerWhenTackling );
          Vector resultingVector = myTackleVector + hisKickVector + WSinfo::ball->vel;
          Vector ballCrossesTorauslinie
            = Tools::point_on_line( resultingVector, WSinfo::ball->pos, -FIELD_BORDER_X );
          if (    (fabs( ballCrossesTorauslinie.getY() ) > 2.0*7.0 || resultingVector.getX() > 0.0 )
               && resultingVector.norm() > 1.5 )
          {
            goodTackleAngleFound = checkAngle;
            break;
          }
        }
      }
    }

    if (ClientOptions::server_version < 12.0)
    {
      if (    tackSuccProb > 0.9 
           && oppKickPower - ownKickPower > kickPowerDelta
           && powerWhenTackling - oppKickPower > kickPowerDelta
          )  
      {
        double tacklePower;
        if ( fabs( WSinfo::me->ang.get_value_mPI_pPI() ) < PI*0.5 )
          tacklePower = 100;
        else
          tacklePower = -100;
        intention.set_tackling( tacklePower, WSinfo::ws->time );
        MYLOG_POL(0, << "WBALL03: TG: Tackle the ball away (prob="<<tackSuccProb<<").");
        TGLOGPOL(0, << "WBALL03: TG: Tackle the ball away (prob="<<tackSuccProb<<").");
        return true;
      }
    }
    else
    {
      if (goodTackleAngleFound > -1000)
      {
        double tacklePower = -goodTackleAngleFound;
        if (tacklePower <= -180.0) tacklePower += 360.0;
        intention.set_tackling( tacklePower, WSinfo::ws->time );
        MYLOG_POL(0, << "WBALL06: TG: ANGULARLY Tackle the ball away (prob="<<tackSuccProb<<").");
        TGLOGPOL(0, << "WBALL06: TG: ANGULARLY Tackle the ball away (prob="<<tackSuccProb<<").");
        MYLOG_POL(0, << "WBALL03: TG: Tackle the ball away (prob="<<tackSuccProb
          <<",ownKP="<<ownKickPower<<",oppKP="<<oppKickPower<<",ownTP="<<powerWhenTackling
          <<",goodTackleAngleFound="<<tacklePower<<").");
        return true;
      }
    }
    //debug output
    MYLOG_POL(0, << "WBALL03: TG: Tackle the ball away (prob="<<tackSuccProb
      <<",ownKP="<<ownKickPower<<",oppKP="<<oppKickPower<<",maxownTP="<<maxPowerWhenTackling<<").");
    TGLOGPOL(0, << "WBALL03: TG: Tackle the ball away (prob="<<tackSuccProb<<").");
  }
  //////////////////////

  // Danger: Opponent can also kick -> kick ball straight away
  if(WSinfo::me->pos.getX() <10){ // I am far from opponents goal
    //    target = Vector(52.5,0);
    //    target = Vector(52.5,WSinfo::me->pos.y);  // ridi: kick straight
                                                    // TG16: change to kick away from my goal
    target = WSinfo::me->pos + ( WSinfo::me->pos - MY_GOAL_CENTER ).normalize(50.0);
    TGLOGPOL(0, << "WBALL03: TG: I set the initial panic kick target to "<<target);
    TGLOGPOL(0, _2D<<VL2D(WSinfo::me->pos,target,"33fff33"));
      //////////////////////
      // ADD-ON by TG 04/05
      //////////////////////
     if (   WSinfo::me->pos.sqr_distance( MY_GOAL_CENTER ) < 20*20 )
     {
       PlayerSet kickCapableOpponents = WSinfo::valid_opponents;
       kickCapableOpponents.keep_and_sort_closest_players_to_point( 1, WSinfo::ball->pos ); 
       if ( kickCapableOpponents.num > 0 )
       {
         Vector whereTheOpponentWillKick = MY_GOAL_CENTER;
         whereTheOpponentWillKick.setY( kickCapableOpponents[0]->pos.getY() );
         if (whereTheOpponentWillKick.getY() > 6.0) whereTheOpponentWillKick.setY(  6.0 );
         if (whereTheOpponentWillKick.getY() <-6.0) whereTheOpponentWillKick.setY( -6.0 );
         Vector assumedOpponentKickVector 
           = kickCapableOpponents[0]->pos - whereTheOpponentWillKick;
         assumedOpponentKickVector.normalize(50.0);
         target = WSinfo::me->pos + assumedOpponentKickVector;
         //assuming the opponent will also kick, we ought to aim at a position which
         //results in a ball movement away from the opponent (when both ball accelaration vectors are combined)
         if (   kickCapableOpponents[0]->pos.getY() + kickCapableOpponents[0]->vel.getY()
              > WSinfo::me->pos.getY() + WSinfo::me->vel.getY()
            )
           target.subFromY( 30.0 ); //target = Vector(52.5, -40.0);
         else
           target.addToY( 30.0 ); //target = Vector(52.5, 40.0);
         MYLOG_POL(0, << "WBALL03: TG: I modified the target of my panic kick to "<<target<<"."); 
         TGLOGPOL(0, << "WBALL03: TG: I modified the target of my panic kick to "<<target<<"."); 
       }
     }
     //////////////////////
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
  MYLOG_POL(0, << "WBALL03: panic: Kick Ball with maxpowert to target "<<target);
  TGLOGPOL(0, << "WBALL03: panic: Kick Ball with maxpowert to target "<<target);
  intention.set_panic_kick(target, WSinfo::ws->time);
  return true;
}


/** selects one player who is responsible for the ball */
bool Wball06::test_two_teammates_control_ball(Intention &intention){
  // test if Ball is in kickrange of me and a teammate!

  Vector ballpos = WSinfo::ball->pos;
  PPlayer p_tmp = WSinfo::valid_teammates_without_me.closest_player_to_point(ballpos);
  if (p_tmp == NULL) return false;
  int teammate = p_tmp->number;
  float dist = (p_tmp->pos - ballpos).norm();

  if((dist < .95*ServerOptions::catchable_area_l) && (teammate == WSinfo::ws->my_goalie_number)){
    MYLOG_ERR(0,<<"Our goalie has the ball, I could kick - Maybe I should back up?!");
    MYLOG_POL(0,<<"Our goalie has the ball, I could kick - Maybe I should back up?! (not yet implemented)");
    TGLOGPOL(0,<<"Our goalie has the ball, I could kick - Maybe I should back up?! (not yet implemented)");
  }

  if ((dist < ServerOptions::kickable_area) )
  {
    //we should risk nothing and just kick the ball away if we are too near to 
    //our offside line!
    PlayerSet oppsNearby = WSinfo::valid_opponents;
    oppsNearby.keep_players_in_circle(WSinfo::ball->pos, 5.0);
    if (   WSinfo::ball->pos.getX() - WSinfo::my_team_pos_of_offside_line() < 15.0
        && oppsNearby.num > 0 ) //TG17
    {
      intention.set_panic_kick( HIS_GOAL_CENTER, WSinfo::ws->time);
      return true;
    }

    if (fabs(WSinfo::me->pos.getX() - p_tmp->pos.getX()) < 0.05) //TG06
    {
      if ( WSinfo::me->pos.distance(WSinfo::ball->pos) < p_tmp->pos.distance(WSinfo::ball->pos) )
      {
        MYLOG_ERR(0,<<"Ball is kickable for me and teammate who has almost the same x pos. I am nearer to the ball, I can kick!");
        MYLOG_POL(0,<<"Ball is kickable for me and teammate who has almost the same x pos. I am nearer to the ball, I can kick!");
        TGLOGPOL(0,<<"Ball is kickable for me and teammate who has almost the same x pos. I am nearer to the ball, I can kick!");
        //i return false so that i can pass
        return false;
      }
      else
      {
        MYLOG_POL(0,<<"Ball is kickable for me and teammate who has almost the same x pos. I am nearer to the ball, I can kick!");
        TGLOGPOL(0,<<"Ball is kickable for me and teammate who has almost the same x pos. I am nearer to the ball, I can kick!");
        return true;
      }
    }
    else
    {
      if (WSinfo::me->pos.getX() > p_tmp->pos.getX())
      {
        MYLOG_POL(0,<<"Another teammate with smaller x position has the ball, too => my x pos is higher, I can kick!");
        TGLOGPOL(0,<<"Another teammate with smaller x position has the ball, too => my x pos is higher, I can kick!");
        //i return false so that i can pass
        return false;
      }
      else
      {
        //intention.set_backup(WSinfo::ws->time);
        intention.set_waitandsee(WSinfo::ws->time);
        MYLOG_POL(0,<<"I could kick but another teammate with larger x pos has the ball, also => my x pos is smaller, I must retreat! ");
        TGLOGPOL(0,<<"I could kick but another teammate with larger x pos has the ball, also => my x pos is smaller, I must retreat! ");
        return true;
      }
    }
  }
  return false;
}


bool Wball06::test_prefer_holdturn_over_dribbling(Intention &intention){
  if(status.is_holdturn_safe == false){
    MYLOG_POL(0, <<"Prefer HOLDTURN?  NOT Safe");
    return false;
  }
  if(cycles_in_waitandsee >= 50){
    MYLOG_POL(0, <<".Prefer HOLDTURN. wait and  see patience expired");
    return false;
  }
  if(onestepkick->can_keep_ball_in_kickrange() == false){
    MYLOG_POL(0, <<"Prefer HOLDTURN not possible. Can not keep ball in kickrange");
    return false;
  }

  if(am_I_attacked(3.0) == false){ // I am not really attacked, so continue to dribble
    MYLOG_POL(0, <<"Prefer HOLDTURN not needed. Not attacked");
    return false;
  }

  if(WSinfo::me->pos.getX() < FIELD_BORDER_X - 10.0 || fabs(WSinfo::me->pos.getY()) < 10.0){ // I am either not close to the goalline or in hot scorig area
    MYLOG_POL(0, <<"Prefer HOLDTURN not needed. Not far advanced or in hot scoring area");
    return false;
  }

  ANGLE targetdir;

  if(WSinfo::me->pos.getY() > 0){
    targetdir = ANGLE(-90./180. *PI);
  }
  else{
    targetdir = ANGLE(90./180. *PI);
  }
  intention.set_holdturn(targetdir, WSinfo::ws->time);
  MYLOG_POL(0, <<"Prefer HOLDTURN.");
  return true;
}


bool Wball06::is_dream_selfpass_to_target_possible(Intention &intention,  
                                                   Vector targetpos, 
                                                   const Vector mypos, 
                                                   const Vector ballpos, 
                                                   const Vector ballvel,
                                                   bool checkPrevIntentionMode,
                                                   const double selfpassSpeed)
{

  if(mypos.distance(targetpos) < 5.){
    MYLOG_POL(0,"Dream Selfpass: too close to target position; makes no sense");
    return false;
  }

  if(mypos.getX() > targetpos.getX()){
    MYLOG_POL(0,"Dream Selfpass: Do not consider backward passes");
    return false;
  }


  //  MYLOG_DRAW(0, C2D(targetpos.x,targetpos.y,0.3, "green"));

  int mytime2react = 0; 
  int optime2react = 1; // make it a bit risky ;-)

  if(WSinfo::me->pos.getX() > FIELD_BORDER_X -16){
    optime2react = 0; // make it a bit less risky ;-)
    mytime2react = 1; 

    MYLOG_POL(0,"Dream Selfpass: I am in penalty area: Play less risky. Optime2react: "
	      <<optime2react<<" mytime2react: "<<mytime2react);
    int stamina = Stamina::get_state();
    MYLOG_POL(0,<<"Check selfpass: check stamina state "<<stamina);
    if(stamina == STAMINA_STATE_RESERVE ){
      MYLOG_POL(0,"Dream Selfpass not possible: powered out ");
      return false;
    }

  }

  if (checkPrevIntentionMode == true)
  {
    ANGLE desiredDir = (targetpos - ballpos).ARG(),
          currentDir = ballvel.ARG();
    double desiredSpeed = selfpassSpeed,
          currentSpeed = ballvel.norm();
    //the following happens, if, by accident, the ball has not left the player's
    //kick range when having played a dream self pass in the recent cycly
    if (    fabs( (desiredDir-currentDir).get_value_mPI_pPI() ) < PI*15.0/180.0
         && currentSpeed <= 1.05 * desiredSpeed
         && currentSpeed >= 0.8 * desiredSpeed
       )
    {
      int dummyAdvantage, dummyActualSteps, dummyAttackingOpNr;
      double dummyKickSpeed;
      Vector dummyTargetPos, dummyAttackingOp;
      if ( selfpass2->is_selfpass_safe_without_kick(dummyAdvantage, 
                                                    desiredDir, 
                                                    dummyKickSpeed, 
                                                    dummyTargetPos, 
                                                    dummyActualSteps,
                                                    dummyAttackingOp, 
                                                    dummyAttackingOpNr, 
                                                    WSinfo::me->pos, 
                                                    WSinfo::me->vel, 
                                                    WSinfo::me->ang,
                                                    WSinfo::me->stamina,
                                                    ballpos, 
                                                    ballvel) )
      {
        intention.set_selfpass( desiredDir, 
                                dummyTargetPos,
                                0.0, 
                                WSinfo::ws->time,
                                dummyAttackingOpNr);
        MYLOG_POL(0,"Dream Selfpass. Checking previous intention. -> Ball did not leave KR. DOWNGRADE "
          <<"to selfpass ("<<").");
        return true;
      }
    }
  }

  Vector tmp_ipos;
  int steps2pos = Policy_Tools::get_time2intercept_hetero(tmp_ipos, targetpos,0.0, 0.0,WSinfo::me ,mytime2react); 
  // trick: ballpos is target; ballvel is 0

  double pass_speed =Tools::get_ballspeed_for_dist_and_steps(mypos.distance(targetpos), steps2pos);
  
  double pass_dir = (targetpos - mypos).arg();

  MYLOG_POL(0,"Dream Selfpass. I need "<<steps2pos
	     <<" steps to go 2 target. targetdir: "<<RAD2DEG(pass_dir)<<" speed "<<pass_speed);
  
  if(pass_speed > 3.0)//ServerOptions::ball_speed_max)//TG08: NeuroKick cannot kick as hard, yet !!!
    pass_speed = 3.0;//ServerOptions::ball_speed_max;//TG08: NeuroKick cannot kick as hard, yet !!!

  int advantage;
  Vector ipos;
  int receiver;
  Vector playerpos;
  int desired_receiver = WSinfo::me->number;

  if(Policy_Tools::myteam_intercepts_ball_hetero06(ballpos,pass_speed,  pass_dir,ipos,advantage,
						   receiver,playerpos, mytime2react, optime2react, 
						   desired_receiver) == false){
    MYLOG_POL(0,"Dream Selfpass. I do not get this ball first");
    return false;
  }


  if(WSinfo::me->pos.distance(ipos) < 3.0){
    MYLOG_POL(0,"Dream Selfpass. ipos too close to mypos. Probably don't get the ball out of kickrange -> false");
    MYLOG_DRAW(0, VC2D(ipos,0.3, "red"));
    return false;
  }


  // trick: now compute actual pass speed a bit slower than before
  pass_speed =Tools::get_ballspeed_for_dist_and_steps(mypos.distance(targetpos), steps2pos +1);


  MYLOG_POL(0,"Dream Selfpass. Modified pass speed : "<<pass_speed);


  intention.set_pass(targetpos,pass_speed, pass_dir, WSinfo::ws->time, desired_receiver, 0, 
		     ipos, ipos );
  intention.subtype = SUBTYPE_SELFPASS;

  if(WSmemory::last_seen_in_dir(ANGLE(pass_dir)) >1){
    if(status.is_holdturn_safe){
      MYLOG_POL(0,"Dream Selfpass. OK, and status is holdturn safe.but I have to look first!");
      intention.wait_then_pass = true;
      return true;
    }
    // I havent seen to dir too long, and cannot holdturn. 
    MYLOG_POL(0,"Dream Selfpass. OK, but holdturn NOT safe. return false!");
    intention.reset();
    return false;
  }
  else
    intention.wait_then_pass = false;

  MYLOG_POL(0,"Dream Selfpass. I get this ball first");
  return true;

}

bool Wball06::test_dream_selfpass(Intention &intention, const Vector mypos, const Vector ballpos, const Vector ballvel){

  // maybe check targets first, where view information is recent....

  if(DeltaPositioning::get_role(WSinfo::me->number)  == 0){// I'm a defender: 
    return false;
  }
  if(mypos.getX() <0){
    return false;
  }

  struct{
    double evaluation;
    Vector pos;
  } target[10];

  int num = 0;
  double target_x = FIELD_BORDER_X - 8;
  int direction;
  if(mypos.getY() <0)
    direction = -1;
  else
    direction = 1;
  
  if((mypos.getX() > FIELD_BORDER_X -25 && fabs(mypos.getY()) <20) || mypos.getX() > FIELD_BORDER_X -16.){
    // I am either in penalty area or already close to the goal. So 

    if(WSinfo::his_goalie != NULL){
      target_x = mypos.getX() + (WSinfo::his_goalie ->pos.getX() - mypos.getX())/2.0;
      target_x -= 1.5; // stay a bit away from goalie
      if(target_x < mypos.getX())
	target_x = mypos.getX() +3;
    }
    else{
      target_x = mypos.getX() + (FIELD_BORDER_X - mypos.getX())/2.0;
    }

    if(target_x > FIELD_BORDER_X -6)
      target_x = FIELD_BORDER_X -6;

    // find good points in goalarea
    target[num++].pos = Vector(target_x, mypos.getY());  // go straight
    target[num++] .pos= Vector(target_x +1.5, direction * 8. );  // point on the side of the goal
    target[num++].pos = Vector(target_x, 0.0 );  // point in front of the goal
    target[num++].pos = Vector(target_x +1.5, - direction *  8. );

    // sort positions according to neural evaluation of pos
    double tmp_evaluation;
    Vector tmp_position;
    for(int i=0; i<num;i++){ // bubble sort positions
#if 0 // debug only:
      for(int k= 0; k< num; k++){
	MYLOG_POL(0,"target "<<k<<" evaluation "<<target[k].evaluation<<" pos "<<target[k].pos);
    }
#endif
      target[i].evaluation = Tools::evaluate_pos(target[i].pos); // evtl. auch 1vs1 benutzen
      tmp_evaluation = target[i].evaluation;  // new element
      tmp_position = target[i].pos;  // new element
      for(int j=i-1; j>=0;j--){ // bubble sort positions
	if(tmp_evaluation > target[j].evaluation){  //move elements higher
	  target[j+1].evaluation = target[j].evaluation;
	  target[j+1].pos =  target[j].pos;;
	  target[j].evaluation = tmp_evaluation;
	  target[j].pos =  tmp_position;;
	} 
	else{ // I found my place in the list: this element is higher than myself!
	  break;
	}
      } // for all j
    }  // for all  i
#if 1 // debug only:
    for(int k= 0; k< num; k++){
      MYLOG_POL(0,"Sorted target "<<k<<" evaluation "<<target[k].evaluation<<" pos "<<target[k].pos);
    }
#endif

    
  } // end in penalty area
  else{ // in field
    if( target_x > mypos.getX() + 30) // otherwise the goalie gets those always
      target_x = mypos.getX() + 30;
    if(fabs(mypos.getY()) > 20)
      target[num++].pos = Vector(target_x, mypos.getY());
    else
      target[num++].pos = Vector(FIELD_BORDER_X -19.0, mypos.getY());
    // prefer to go via the middle
    target[num++].pos = Vector(FIELD_BORDER_X - 19.0, direction * 10.0 );
    target[num++].pos = Vector(FIELD_BORDER_X - 19.0, direction * 2.0 );
    target[num++].pos = Vector(target_x, direction * (FIELD_BORDER_Y-4.) );
  }


  for(int i= 0; i< num; i++){
    // take the first selfpass that works. 
    if(is_dream_selfpass_to_target_possible(intention, target[i].pos, mypos, ballpos, ballvel)){
      return true;
    }
  }
  return false;
}


bool Wball06::test_holdturn(Intention &intention){
  if(status.is_holdturn_safe == false){
    MYLOG_POL(0, <<"HoldTurn NOT Safe");
    return false;
  }

  if(cycles_in_waitandsee >= 50){
    MYLOG_POL(0, <<"TEST HOLDTURN. wait and  see patience expired");
    return false;
  }
  if(onestepkick->can_keep_ball_in_kickrange() == false){
    MYLOG_POL(0, <<"HoldTurn NOT possible. Can not keep ball in kickrange");
    return false;
  }

  ANGLE targetdir;

  if(WSinfo::me->pos.getX() > 42.0){  // if I'm close to goal, then turn2goal
    targetdir = (Vector(47,0) - WSinfo::me->pos).ARG();
  }
  else{
    targetdir = ANGLE(0.);  // turn straight ahead
  }
  intention.set_holdturn(targetdir, WSinfo::ws->time);
  return true;
}


bool Wball06::test_pass(Intention &intention, Intention & pass_option, Intention & selfpass_option){

  if(pass_option.valid_at() == WSinfo::ws->time)
  {
    //TG17: no extreme back passes
    int recvNr = pass_option.target_player;
    Vector target = pass_option.kick_target;
    PPlayer receiver = WSinfo::get_teammate_by_number(recvNr);
    if (!receiver)
    {
      MYLOG_POL(0, <<"Pass not safe: receiver unknown");
      return false;
    }
    if ( receiver->age > 10 )
    {
      MYLOG_POL(0, <<"Pass not safe: receiver more than 10 steps old");
      return false;
    }
    double backDist = WSinfo::ball->pos.getX() - target.getX();
    if ( backDist * receiver->age > 35.0 )
    {
      MYLOG_POL(0, <<"Pass not safe: backDist*rcvAge="<<backDist<<"*"<<receiver->age
        <<"="<<backDist * receiver->age<<" > 35");
      return false;
    }
    // if nothing applies: play the pass
    intention = pass_option;
    return true;
  }
  return false;
}





/** default move */
bool Wball06::test_default(Intention &intention){

  if(onestepkick->can_keep_ball_in_kickrange() == false){
    PlayerSet pset= WSinfo::valid_opponents;
    Vector endofregion;
    endofregion.init_polar(5, WSinfo::ball->vel.ARG());
    endofregion += WSinfo::me->pos;
    Quadrangle2d check_area = Quadrangle2d(WSinfo::me->pos, endofregion, 3);
    //MYLOG_DRAW(0, check_area );
    pset.keep_players_in(check_area);
    if(pset.num == 0){
      intention.set_holdturn(ANGLE(0.), WSinfo::ws->time); // this will maximally stop the ball
      MYLOG_POL(0,<<"DEFAULT Move - Stop Ball");
      return true;
    }
  }

  Vector opgoalpos = Vector (52.,0); // opponent goalpos
  
  if(cycles_in_waitandsee < 50 && status.is_holdturn_safe == true){
    ANGLE target_dir = (opgoalpos - WSinfo::me->pos).ARG();
    MYLOG_POL(0,<<"DEFAULT Move - Hold Turn is Safe");
    intention.set_holdturn(target_dir, WSinfo::ws->time);
    return true;
  }

  Vector target;
  double speed, dir;
  get_onestepkick_params(speed,dir);
  if (   speed > 0     )
  {
    target.init_polar(speed/(1-ServerOptions::ball_decay),dir);
    Vector targetHalfWay = target;
    targetHalfWay *= 0.5;
    target += WSinfo::ball->pos;
    targetHalfWay += WSinfo::ball->pos;
    //TG08: avoid panic kicks straight towards my goal
    bool isDangerousClearance = false;
    if (    (    target.distance(MY_GOAL_CENTER) < 20.0
              &&   target.distance(MY_GOAL_CENTER) 
                 < WSinfo::ball->pos.distance(MY_GOAL_CENTER) )
         ||
            (    targetHalfWay.distance(MY_GOAL_CENTER) < 20.0
              &&   targetHalfWay.distance(MY_GOAL_CENTER) 
                 < WSinfo::ball->pos.distance(MY_GOAL_CENTER) )
       )
      isDangerousClearance = true;
    if ( isDangerousClearance == false )
    {
      intention.set_kicknrush(target,speed, WSinfo::ws->time);
      MYLOG_POL(0,<<"DEFAULT Clearance to "<<target<<" dir "<<RAD2DEG(dir)<<" w.speed "<<speed);
      return true;
    }
  }
  
  MYLOG_POL(0,<<"DEFAULT Move - Hold Turn is NOT Safe, but no other alternative, TRY");
  intention.set_holdturn((opgoalpos - WSinfo::me->pos).ARG(), WSinfo::ws->time);
  return true;
}


/************************************************************************************************************/

/*  Auxillary procedures                                                                                                                                                     */

/************************************************************************************************************/


void Wball06::display_intention(Intention & intention){
  if(intention.is_pass2teammate()){
    PPlayer p= WSinfo::get_teammate_by_number(intention.target_player);
    if(p){
      MYLOG_DRAW(0, VC2D(p->pos, 1.3, "red"));
    }


    if(intention.subtype == SUBTYPE_LAUFPASS){
      MYLOG_POL(0,<<"DISPLAY INTENTION. Intention is LAUFPASS (red): target_player: "<<intention.target_player
		<<" risky: "<<intention.risky_pass);
#if 1
      MYLOG_DRAW(0, VL2D(WSinfo::ball->pos,
			intention.kick_target,
			"red"));
      MYLOG_DRAW(0, VC2D(intention.kick_target,0.5, "red"));
#endif
      return;
    }
    else if(intention.subtype == SUBTYPE_SELFPASS){
      MYLOG_POL(0,<<"DISPLAY INTENTION. Intention is DREAM SELFPASS (red): target_player: "<<intention.target_player
		<<" risky: "<<intention.risky_pass);
#if 1
      MYLOG_DRAW(0, VL2D(WSinfo::ball->pos,
			intention.kick_target,
			"red"));
#endif
      MYLOG_DRAW(0, VC2D(intention.kick_target,0.5, "red"));
      return;
    }
    else{
      MYLOG_POL(0,<<"DISPLAY INTENTION. Intention is DIRECT PASS (red): target_player: "<<intention.target_player<<" risky: "
		<<intention.risky_pass);
#if 1
      MYLOG_DRAW(0, VL2D(WSinfo::ball->pos,
			intention.kick_target,
			"red"));
      MYLOG_DRAW(0, VC2D(intention.kick_target,0.5, "red"));
#endif
    }
  }
}

bool Wball06::is_my_passway_in_danger(Intention &pass_intention){
  if (pass_intention.valid_at()  != WSinfo::ws->time){
    MYLOG_POL(0, <<"check passing corridor: pass intention currently not valid !");
    return false;
  }

  float width = 10.0;
  if(DeltaPositioning::get_role(WSinfo::me->number) == 0)  //* defender
    width = 10.0;

  PlayerSet pset = WSinfo::valid_opponents;
  Vector respos = pass_intention.resultingpos;
  Quadrangle2d check_area = Quadrangle2d(WSinfo::me->pos, respos,width);
  pset.keep_players_in(check_area);
  if(pset.num>0){
     MYLOG_POL(0, <<"Someone enters passing corridor!");
    return true;
  }
  return false;
}



bool Wball06::am_I_attacked(const double factor, const double backExtraFactor)
{
  PlayerSet pset= WSinfo::valid_opponents;
  double radius_of_attacked_circle =  factor * ServerOptions::kickable_area;
  pset.keep_players_in_circle(WSinfo::me->pos,radius_of_attacked_circle);
  if (backExtraFactor > 0.0) // TG17
  {
    // a second / additional factor (for backwards) has been given
    // -> more sophisticated analysis!
    PlayerSet sophisticatedSet;
    for (int i=0; i<pset.num; i++)
      if ( pset[i]->pos.distance(HIS_GOAL_CENTER) <= WSinfo::me->pos.distance(HIS_GOAL_CENTER) )
        sophisticatedSet.append( pset[i] );
    PlayerSet backset= WSinfo::valid_opponents;
    backset.keep_players_in_circle( WSinfo::me->pos,
                                    backExtraFactor * ServerOptions::kickable_area);
    for (int i=0; i<backset.num; i++)
      if ( backset[i]->pos.distance(HIS_GOAL_CENTER) > WSinfo::me->pos.distance(HIS_GOAL_CENTER) )
        sophisticatedSet.append( backset[i] );
    pset = sophisticatedSet;
  }
  int tackleInactivityMin, tackleInactivityMax;

  // TG17: Do disregard players that have tackled!
  for (int i=0; i<pset.num; i++)
  {
    WSinfo::get_player_inactivity_interval( pset[ i ],
                                            tackleInactivityMin,
                                            tackleInactivityMax );
    if ( tackleInactivityMin == 0 )
      return true;
    else
    {
      MYLOG_POL(0,<<"am_I_attacked(): Ignore " << pset[i]->number << " as he "
        << "tackled at t=" << pset[i]->tackle_time);
    }
  }
  //return  (pset.num >0);
  return false;
}


void Wball06::aaction2intention(const AAction &aaction, Intention &intention){

  double speed = aaction.kick_velocity;
  double targetdir = aaction.kick_dir;
  Vector target = aaction.target_position;

  switch(aaction.action_type){
  case  AACTION_TYPE_PASS:
    intention.set_pass(target,speed, targetdir, WSinfo::ws->time, aaction.targetplayer_number, 0, 
		       aaction.actual_resulting_position, aaction.potential_position );
    intention.subtype = aaction.subtype;
    break;
  case  AACTION_TYPE_LAUFPASS:
    intention.set_laufpass(target,speed, targetdir, WSinfo::ws->time, aaction.targetplayer_number, 0, 
			   aaction.actual_resulting_position, aaction.risky_pass,  aaction.potential_position);
    break;
  default:
    MYLOG_POL(0,<<"aaction2intention: AActionType not known");
    MYLOG_ERR(0,<<"aaction2intention: AActionType not known");
  }
  intention.V = aaction.V;
}


void Wball06::get_onestepkick_params(double &speed, double &dir){
  double tmp_speed;
  Vector final;
  Vector ballpos = WSinfo::ball->pos;
  const int max_targets = 360;
  double testdir[max_targets];
  double testspeed[max_targets];

  int num_dirs = 0;

  for(float ang=0.;ang<PI/2.;ang+=5./180.*PI){ //ridi 05: allow only forward directions!
    for(int sign = -1; sign <= 1; sign +=2){
      ANGLE angle=ANGLE((float)(sign * ang));
      tmp_speed = onestepkick->get_max_vel_in_dir(angle);
      if(tmp_speed <0.1){
	final.init_polar(1.0, angle);
	final += ballpos;
      }
      else{
	testdir[num_dirs] = angle.get_value();
	testspeed[num_dirs] = pass_selection->adjust_speed2(WSinfo::ball->pos, angle.get_value(),tmp_speed);
	if(num_dirs < max_targets)
	  num_dirs ++;
	else {
	  MYLOG_POL(0,"test onestep_kicks: Warning: too many targets");
	}
	final.init_polar(tmp_speed, angle);
	final += ballpos;
      }
    }
  }


  int advantage;
  Vector ipos;
  int closest_teammate;
  Policy_Tools::get_best_kicknrush(WSinfo::me->pos,num_dirs,testdir,
				   testspeed,speed,dir,ipos,advantage, closest_teammate);

  if(WSinfo::me->pos.getX() >0){
    if(advantage<1 || closest_teammate == WSinfo::me->number){
      MYLOG_POL(0,<<"Check Panic Selfpasses ");
      if(get_best_panic_selfpass(testdir,num_dirs,speed,dir)) {
	MYLOG_POL(0,<<"found a panic selfpass speed "<<speed<<" dir "<<RAD2DEG(dir));
      }
    }
  }

  if(speed >0){
    MYLOG_POL(0,<<"found onestepkick with advantage "<<advantage<<" resulting pos : "
	      <<ipos<<" closest teammate "<<closest_teammate);
  }
  else{
    MYLOG_POL(0,<<"NO onestepkick found ");
  }
}

bool Wball06::get_best_panic_selfpass(const double testdir[],const int num_dirs,double &speed, double &dir){
  Vector ipos;
  double tmp_speed;
  double best_speed = 0;
  double best_dir = 2*PI;
  
  int stamina = Stamina::get_state();
  //MYLOG_POL(0,<<"Check selfpass: check stamina state "<<staminaa);
  if(stamina == STAMINA_STATE_RESERVE )
    return false;


  for(int i=0;i<num_dirs;i++)
  {
    if(fabs(testdir[i]) > 90/180.*PI)
      continue;
    onestepkick->reset_state();
    bool speed_ok = false;
    tmp_speed =.4;
    while (!speed_ok)
    {
      tmp_speed +=.1;
      onestepkick->kick_in_dir_with_initial_vel(tmp_speed,ANGLE(testdir[i]));
      speed_ok = onestepkick->get_vel(tmp_speed);
      if (tmp_speed >1.3)
        speed_ok = true;
    }
    if (tmp_speed >1.3)
      continue; // no acceptable speed possible in that dir; check next directions

    int myadvantage = Policy_Tools::get_selfpass_advantage( tmp_speed, testdir[i], ipos);
#if 0
    MYLOG_POL(0,<<"Test panic selfpass in dir "<<RAD2DEG(testdir[i])<<" speed "<<tmp_speed
	      <<" advantage "<<myadvantage);
#endif

    if(myadvantage<1)
      continue;
    if(fabs(testdir[i]) < fabs(best_dir)){
      best_speed = tmp_speed;
      best_dir = testdir[i];
    }     
  }
  if(best_speed >0){
    speed = best_speed;
    dir = best_dir;
    return true;
  }
  return false;
}


int Wball06::I_am_heavily_attacked_since(){
  // returns -1, if no attacking occured. Must be called at least once every cycle to be effective.
  PlayerSet pset= WSinfo::valid_opponents;
  double radius_of_attacked_circle =  2*ServerOptions::kickable_area;
  pset.keep_players_in_circle(WSinfo::me->pos,radius_of_attacked_circle); 

  if(pset.num ==0){ //* no attack
      last_heavy_attack_at = -1; // reset
    return -1;
  }
  // I am heavily attacked this cycle
  if(last_heavy_attack_at <0){
    last_heavy_attack_at = WSinfo::ws->time; // remember start of heavy attack
    return 1;
  }
  return ((WSinfo::ws->time - last_heavy_attack_at) +1);
}

void Wball06::check_adapt_pass_speed(Intention &pass_intention){
  
  Vector target;
  double pass_speed;
  double speed1, speed2;

  if (pass_intention.valid_at() != WSinfo::ws->time || pass_intention.is_pass2teammate() == false){
    return;
  }
  pass_intention.get_kick_info(pass_speed, target);
  oneortwo->kick_to_pos_with_initial_vel(pass_speed,target);
  oneortwo->get_vel(speed1,speed2);
  MYLOG_POL(0,"check adapt pass speed "<<target<<" pass_speed "<<pass_speed<<" : speed1 "<<speed1<<" speed2: "<<speed2);
  if(speed1>=pass_speed){
    MYLOG_POL(0,"check adapt pass speed. 1 step kick already ok. Nothing to do ");
    return;
  }
  if(speed1 >= pass_speed -.1){
    pass_intention.kick_speed = speed1;
    MYLOG_POL(0,"check adapt pass speed. 1 step kick is at least nearly ok. Adapt pass_speed to  "<<pass_intention.kick_speed);
    return;
  }
  if(speed2 >= pass_speed -.2){
    pass_intention.kick_speed = speed2;
    MYLOG_POL(0,"check adapt pass speed. 2 step kick is at least nearly ok. Adapt pass_speed to  "<<pass_intention.kick_speed);
    return;
  }
}



int Wball06::howmany_kicks2pass(Intention &pass_intention){
  if (pass_intention.valid_at() != WSinfo::ws->time){
    return 0;
  }
  int pass_type = pass_intention.get_type();
  if(pass_type == PASS || pass_type == LAUFPASS || pass_type == KICKNRUSH){
    Vector target;
    double speed;
    pass_intention.get_kick_info(speed, target);

    oneortwo->kick_to_pos_with_initial_vel(speed,target);
    double speed1, speed2;
    oneortwo->get_vel(speed1,speed2);
    MYLOG_POL(0,"how many steps to pass to "<<target<<" speed "<<speed<<" : speed1 "<<speed1<<" speed2: "<<speed2);
    if(fabs(speed1 -speed)<.1){
      return 1;
    }
    else{
      return 2;
    }	
  }
  return 0;
}


int Wball06::is_dribblebetween_possible(){
#define POL(XXX) MYLOG_POL(1,<<"isDribblePossible:"<<XXX)

  bool insecure=false;
  bool isPossible=true;


  bool isTargetVeryCool = false;
  if(DeltaPositioning::get_role(WSinfo::me->number)  == 0){// I'm a defender: 
    int stamina = Stamina::get_state();
    //DBLOG_POL(0,<<"Check selfpass: check stamina state "<<staminaa);
    if((stamina == STAMINA_STATE_ECONOMY || stamina == STAMINA_STATE_RESERVE )
       && WSinfo::me->pos.getX() >-35){// stamina low
      MYLOG_POL(0,<<"Check is dribble possible: I'm a defender and stamina level not so good -> dribble not possible");
      isPossible=false;
    }
    PlayerSet oppsAroundMe = WSinfo::valid_opponents;
    oppsAroundMe.keep_players_in_circle(WSinfo::me->pos,6.0);
    if(WSinfo::me->pos.getX() > 10. && oppsAroundMe.num > 0)
      isPossible = false; // Ridi 06: do not dribble in Ops. half
  }

  
  // migrated checks from DribbleBetween BEGIN
  // Check: My Offside Line
  if (WSinfo::me->pos.getX() - WSinfo::my_team_pos_of_offside_line() < 5.0 && !isTargetVeryCool)
    {
      POL("Dribblink not safe: too close to my offside line");
      insecure=true;
    }
  if(WSinfo::me->pos.getX() <-FIELD_BORDER_X+16 && fabs(WSinfo::me->pos.getY())<16){
    POL("Dribblink not safe: in own penalty-area");
    insecure=true;
  }
  // Check: My Stamina
  if (
      (WSinfo::me->stamina < (MIN_STAMINA_DRIBBLE) && WSinfo::me->pos.getX()>30)
      ||(WSinfo::me->stamina < MIN_STAMINA_DRIBBLE*1.0 )
      )
    {
		POL("Dribblink not safe: too weak (Stamina="<<WSinfo::me->stamina<<")");
		insecure=true;
	}
  // Check: My Role and Position
  if(OpponentAwarePositioning::getRole(WSinfo::me->number) == PT_DEFENDER
     && WSinfo::me->pos.getX()>0.25*FIELD_BORDER_X){
    POL("Dribblink not safe: I'm a defender, don't attack!");
    insecure=true;
  }
  if (WSinfo::me->pos.distance(HIS_GOAL_CENTER) < 15.0 ) {     //CHANGE BY TG
    if (WSinfo::his_team_pos_of_offside_line() - WSinfo::me->pos.getX() > 4.0 && !isTargetVeryCool){
      POL("Dribblink not safe: too close to opponent goal");
      insecure = true;
    }
  }
  // migrated checks from DribbleBetween END
  

#if 0 // activate when new dribbling06 is activated
  // shall i keep ball in my kickrange?
  PlayerSet oppsAroundMe = WSinfo::valid_opponents;
  oppsAroundMe.keep_players_in_circle(WSinfo::me->pos,15);
  dribble_between->set_keep_ball(false);
  if(OpponentAwarePositioning::getRole(WSinfo::me->number) == PT_DEFENDER && oppsAroundMe.num >0)
    dribble_between->set_keep_ball(true);
  oppsAroundMe.keep_players_in_circle(WSinfo::me->pos,10);
  if(OpponentAwarePositioning::getRole(WSinfo::me->number) == PT_MIDFIELD && oppsAroundMe.num >0)
    dribble_between->set_keep_ball(true);
#endif

	
  // now estimate the optimal target 
  Vector target;
  if(FIELD_BORDER_X-WSinfo::me->pos.getX() < 3*MIN(0,fabs(WSinfo::me->pos.getY())-12)){
    //MYLOG_POL(1,<<"Check is dribble possible: In 45 deg angle from goal.");
    target = Vector(50,FIELD_BORDER_X*0.8);
  } 
  else  
    target = Vector(50,WSinfo::me->pos.getY());
	
  dribble_between->set_target(target);
  bool isSafe = dribble_between->is_dribble_safe();
  bool isInsecure = dribble_between->is_dribble_insecure();
  isSafe = isSafe && isPossible;
  
  MYLOG_POL(0,<<"Check is dribble possible: is_dribble_safe() returns "<<isSafe<<", insecure="<<isInsecure);
  
  insecure = insecure || isInsecure;
  if(isSafe && !insecure) return 2;
  if(isSafe &&  insecure) return 1;
  if(!isSafe            ) return 0;
  return 0; // ridi05: default (to avoid compiler warnings)
#undef POL
}

double Wball06::get_prefered_tingletangle_dir(){
  if(WSinfo::me->pos.getX() > FIELD_BORDER_X -14)//TG08: increased from 12 to 14
    return get_prefered_tingletangle_dir_close2goalline();
  else
    return get_prefered_tingletangle_dir_infield();
}


bool Wball06::targetpos_keeps_dist2goalie(const double dir){
        //TG08: reduce MINDIST2GOALIE due to soccer server changes: 4->2.5
#define MINDIST2GOALIE 2.5 

  Vector targetpos;
  targetpos.init_polar(1.0, dir);
  targetpos += WSinfo::me->pos;

  if(WSinfo::his_goalie == NULL) // goalie not known yet.
    return true;

  if(targetpos.distance(WSinfo::his_goalie->pos) > MINDIST2GOALIE)
    return true; // everythings ok.

  if(WSinfo::me->pos.distance(WSinfo::his_goalie->pos) < MINDIST2GOALIE && // already too close
     WSinfo::me->pos.distance(WSinfo::his_goalie->pos) 
     < targetpos.distance(WSinfo::his_goalie->pos) -0.5){// targetpos is farther away
    return true;
  }

  MYLOG_POL(0,"target pos violates min dist 2 goalie");
  MYLOG_DRAW(0,VC2D(targetpos,0.3,"red"));
  
  return false;

}

bool Wball06::can_tingletangle2dir(double dir){
  Vector target;
  target.init_polar(30,dir);
  target += WSinfo::me->pos;

  tingletangle->setTarget(target);
  if(tingletangle->isSafe()== false)
    return false;
  return true;
}


double Wball06::get_prefered_tingletangle_dir_infield(){
  double y= WSinfo::me->pos.getY();
  //  double x= WSinfo::me->pos.x;
  double myangle = WSinfo::me->ang.get_value_mPI_pPI();
  int direction = 1;
  int num = 0;
  double testdir[10];

  // first, determine general direction (up +1 or down -1)
  if(y>0)
    direction = 1;  // up : go via wings
  else
    direction = -1;  // down : go via wings

  if(fabs(y) >FIELD_BORDER_Y - 3)//close 2 fieldborder->reverse direction
    direction *= -1;
  
  Vector whenGoingForwardsIWillCrossGoalline
    = Tools::point_on_line( Vector( WSinfo::me->ang ),
                            WSinfo::me->pos,
                            FIELD_BORDER_X );
  bool myBodyDirApproachesHisGoal 
    =      fabs(whenGoingForwardsIWillCrossGoalline.getY())
         < WSinfo::me->pos.distance(HIS_GOAL_CENTER)*0.5 //TG08
      && fabs(WSinfo::me->ang.get_value_mPI_pPI()) < PI/2.0 ;
  if (   (direction * myangle >0 && fabs(myangle) <45/180.*PI)
      || myBodyDirApproachesHisGoal //TG08
     )
  {
    // my body points in the general ok direction
    testdir[num ++] = myangle;
  }
  testdir[num ++] = 0.;
  testdir[num ++] = direction * 22.5/180.*PI;
  testdir[num ++] = direction * 45/180.*PI;
  
  for(int i= 0;i <num;i++){
    //    if(prefered_direction_is_free(ANGLE(testdir[i]),8.0,true) &&
    // targetpos_keeps_dist2goalie(testdir[i])){
    if(can_tingletangle2dir(testdir[i]) &&
       targetpos_keeps_dist2goalie(testdir[i])){
      return testdir[i];
    }
  }
  return 0.0; // general prefered direction.
}

bool Wball06::check2choose_tt_dir(double &dir){
  Vector target[30];
  int num = 0;
  double x = WSinfo::me->pos.getX();
  double y = WSinfo::me->pos.getY();
  double testdir;
  int side;

  if(y>0)
    side = 1; 
  else
    side = -1;

#if 0  // test this for a more flexible direction change
  if(tt_last_active_at == WSinfo::ws->time -1 && cycles_in_wait_and_see <3){
    return false;
  }
#endif

  if(tt_last_active_at == WSinfo::ws->time -1)
    return false;

  if(fabs(y) < 7)
    target[num ++] = Vector(FIELD_BORDER_X - 4,y); // go straight
  target[num ++] = Vector(FIELD_BORDER_X - 4,0); // go straight
  target[num ++] = Vector(FIELD_BORDER_X - 4,side * 5); // go straight
  target[num ++] = Vector(FIELD_BORDER_X - 4,-side * 5); // go diagonal
  target[num ++] = Vector(FIELD_BORDER_X - 10,0); // go straight
  target[num ++] = Vector(FIELD_BORDER_X - 10,side * 5); // go straight
  target[num ++] = Vector(FIELD_BORDER_X - 10,-side * 5); // go diagonal

  for(int i= 0; i< num ; i++){
    if(WSinfo::me->pos.distance(target[i]) < 2.)
      continue;
    if(target[i].getX() < x) // dont go back
      continue;
    testdir =(target[i]-WSinfo::me->pos).arg();
    if(prefered_direction_is_free(ANGLE(testdir),3.0,0.5, true)){
      MYLOG_DRAW(0,VC2D(target[i], .5,"green"));
      return testdir;
    }

  }

  return false;
}


double Wball06::get_prefered_tingletangle_dir_close2goalline(){
  double y= WSinfo::me->pos.getY();
  double x= WSinfo::me->pos.getX();
  double myangle = WSinfo::me->ang.get_value_mPI_pPI();
  int direction = 1;
  int num = 0;
  double testdir[10];
  double suggested_dir;

  #define MAX_X FIELD_BORDER_X -5.
  #define MIN_X FIELD_BORDER_X -12.

  // first, determine general direction (up +1 or down -1)
  if(y>3)
    direction = -1;  // down
  else if (y<-3)
    direction = +1;  // up
  else{ // hystereses: direction depends on current direction
    if(myangle >0)
      direction = +1; // still up
    else
      direction = -1; // still down
  }

  
  if(check2choose_tt_dir(suggested_dir) == true){ // check to set a new tt dir
    testdir[num ++] = suggested_dir;
    MYLOG_POL(0,"choose new tt dir in goal area: "<<RAD2DEG(suggested_dir));
  } 

  if(fabs(y) > 10){// outside the hot scoring area
    if(direction * myangle >0  && x < MAX_X && x > MIN_X ) 
      // my body points in the general ok direction
      testdir[num ++] = myangle;
    if(WSinfo::his_team_pos_of_offside_line()>FIELD_BORDER_X -10){// already overcame offside
      if(x < MAX_X && x > MIN_X ) 
	testdir[num ++] = direction * 92/180.*PI;
      if(x < MAX_X)
	testdir[num ++] = direction * 45/180.*PI;
      if(x > MIN_X)
	testdir[num ++] = direction * 135/180.*PI;
      if(x < MAX_X)
	testdir[num ++] = direction * 0/180.*PI;
//      if(x > MIN_X)
//	testdir[num ++] = direction * 179/180.*PI;
    }
    else{
      if(x < MAX_X)
	testdir[num ++] = direction * 0/180.*PI;
      if(x < MAX_X)
	testdir[num ++] = direction * 45/180.*PI;
      if(x < MAX_X && x > MIN_X ) 
	testdir[num ++] = direction * 92/180.*PI;
      if(x > MIN_X)
	testdir[num ++] = direction * 135/180.*PI;
//      if(x > MIN_X)
//	testdir[num ++] = direction * 179/180.*PI;
    }
  }
  else{ // inside the hot scoring area
    if(direction * myangle >0 // my body points in the general ok direction
       && x < MAX_X && x > MIN_X ) 
      testdir[num ++] = myangle;
    if(x < MAX_X)
      testdir[num ++] = direction * 45/180.*PI;
    if(x < MAX_X && x > MIN_X ) 
      testdir[num ++] = direction * 92/180.*PI;
    if(x > MIN_X)
      testdir[num ++] = direction * 135/180.*PI;
    if(x < MAX_X)
      testdir[num ++] = direction * 0/180.*PI;
//    if(x > MIN_X)
//      testdir[num ++] = direction * 179/180.*PI;
  }

  for(int i= 0;i <num;i++){
    if(can_tingletangle2dir(testdir[i]) &&
       targetpos_keeps_dist2goalie(testdir[i])){
    }
  }
  MYLOG_POL(0,"returning tt dir in goal area: "<<RAD2DEG(direction * 90/180.*PI));
  return direction * 90/180.*PI; // general prefered direction
}






/************************************************************************************************************/

/*  Set Neck and Communicate procedures                                                                                                                     */

/************************************************************************************************************/


bool Wball06::is_pass_announced(Intention &pass_intention){
  if(pass_intention.valid_at() != WSinfo::ws->time)
    return false;
  if(memory.communication.communicated_at != WSinfo::ws->time -1){
    MYLOG_POL(0,"check if pass is ANNOUNCED: No communication at previous time step. return false");
    return false;
  }
  
  int target_player_number;
  double speed;
  Vector target;
  pass_intention.get_kick_info(speed, target, target_player_number);
  ANGLE pass_dir= ANGLE(pass_intention.kick_dir);

  if(fabs(memory.communication.speed - speed) > 1.3){
    MYLOG_POL(0,"check if pass is ANNOUNCED: communicated speed "<<memory.communication.speed<<" and option pass speed "<<speed
	      <<" differ too much. return false" );
    return false;
  }

  if(fabs(pass_dir.diff(ANGLE(memory.communication.passdir)))>20./180.*PI){
    MYLOG_POL(0,"check if pass is ANNOUNCED: communicated targetdir "<<RAD2DEG(memory.communication.passdir)
	      <<" and option pass target "<<RAD2DEG(pass_dir.get_value())
	      <<" differ too much. return false" );
    return false;
  }

    MYLOG_POL(0,"check if pass is ANNOUNCED: communicated targetdir "<<RAD2DEG(memory.communication.passdir)
	      <<" and option pass target "<<RAD2DEG(pass_dir.get_value())
	      <<" are in accordance. return TRUE" );
  return true;  
}

void Wball06::set_communication(Intention &intention, Intention &pass_option, const Vector mypos, Cmd &cmd){

  MYLOG_POL(0, << "Set Communication: pass_option valid at: "<<pass_option.valid_at());
  
  PlayerSet oppsNearBall = WSinfo::valid_opponents;
  oppsNearBall.keep_players_in_circle(WSinfo::ball->pos, 1.5);
  oppsNearBall.keep_and_sort_closest_players_to_point(3,WSinfo::ball->pos);
  bool ballIsKickableForAnOpponent = false;
  for (int i=0; i<oppsNearBall.num; i++)
    if ( WSinfo::is_ball_kickable_for( oppsNearBall[i] ) == true )
      ballIsKickableForAnOpponent = true;
  
  if(intention.is_pass2teammate() == true){
    int cycles2go = 0; // indicates immediate pass; cycles2go = 1 would mean planned pass
    Vector ballvel;
    //  ballvel.init_polar(intention.kick_speed,intention.kick_dir);
    ballvel = intention.kick_target - mypos;
    ballvel.normalize(intention.kick_speed);
    if (    cmd.cmd_body.get_type() == Cmd_Body::TYPE_TACKLE
         && WSinfo::is_ball_kickable() == false
         && ballIsKickableForAnOpponent )
    {}
    else
      cmd.cmd_say.set_pass(mypos,ballvel,WSinfo::ws->time + cycles2go);
    // here we have the possibility to communicate more than the pass.

    memory.communication.communicated_at = WSinfo::ws->time;
    memory.communication.type = 0; // not used yet     
    memory.communication.speed = intention.kick_speed;
    memory.communication.passdir = intention.kick_dir;
    //    Blackboard::pass_intention = intention;  // previous mechanims
    MYLOG_POL(0, << "COMMUNICATION (magenta): intention is PASS: to "<<intention.target_player<<" target pos "
	      <<intention.kick_target<<" w. speed "<<intention.kick_speed<<" ballvel: "<<ballvel<<"-> Communicate");
#if 0
    MYLOG_DRAW(0, L2D(mypos.x, mypos.y, tmp_target.x, tmp_target.y, "magenta"));
#endif
    MYLOG_POL(0, << "COMMUNICATON: Set attention to pass_receiver "<<intention.target_player);
    cmd.cmd_att.set_attentionto(intention.target_player);
  }
  else if(pass_option.valid_at() == WSinfo::ws->time){
    MYLOG_POL(0, << "COMMUNICATION TO: Set attention *REQUEST* [TG08] to OPTIONAL pass_receiver "<<pass_option.target_player);
    //cmd.cmd_att.set_attentionto(pass_option.target_player);
    Tools::set_attention_to_request( pass_option.target_player ); //TG08
    if(intention.type == SELFPASS || am_I_attacked() == false){
      MYLOG_POL(0, << "COMMUNICATION: I either play a selfpass, or I am not attacked -> do NOT COMMUNICATE");
      return;
    }
    MYLOG_POL(0, << "COMMUNICATION (grey): communicate OPTION pass (NOT intented) "<<pass_option.target_player<<" target pos "
	    <<pass_option.kick_target<<" w. speed "<<pass_option.kick_speed<<" -> Communicate");

    MYLOG_DRAW(0, VL2D(mypos,
		      pass_option.kick_target,
		      "grey"));

    int cycles2go = 1; // indicates immediate pass; cycles2go = 1 would mean planned pass
    Vector ballvel;
    ballvel.init_polar(pass_option.kick_speed,pass_option.kick_dir);
    if (    cmd.cmd_body.get_type() == Cmd_Body::TYPE_TACKLE 
         && WSinfo::is_ball_kickable() == false
         && ballIsKickableForAnOpponent )
    {}
    else
      cmd.cmd_say.set_pass(mypos,ballvel,WSinfo::ws->time + cycles2go);
    memory.communication.communicated_at = WSinfo::ws->time;
    memory.communication.type = 0; // not used yet     
    memory.communication.speed = pass_option.kick_speed;
    memory.communication.passdir = pass_option.kick_dir;
  }
  return;
}

/*void Wball06::set_communication_tackle( Cmd & cmd )
{
  double tackleParamater;
  cmd.cmd_main.get_tackle( tackleParamater );
  if (ClientOptions::server_version >= 12.0)
  {
    //tackle parameter is an angle
    double leftHandTackleParameter = -tackleParamater;
    Angle tackleAngle = DEG2RAD( leftHandTackleParameter );
    //calculate effective tackle power
    double effectiveTacklePower
      =   ServerOptions::max_back_tackle_power
        +   (ServerOptions::max_tackle_power - ServerOptions::max_back_tackle_power)
          * (fabs(leftHandTackleParameter)/180.0);
    Vector player_2_ball = WSinfo::ball->pos - WSinfo::me->pos;
    player_2_ball.rotate( - WSinfo::me->ang.get_value_mPI_pPI() );
    ANGLE player_2_ball_ANGLE = player_2_ball.ARG();
    effectiveTacklePower
      *= 1.0 - 0.5 * ( fabs( player_2_ball_ANGLE.get_value_mPI_pPI() ) / PI );
    Vector tackle_dir = Vector( WSinfo::me->ang + ANGLE(tackleAngle) );
    tackle_dir.normalize(   ServerOptions::tackle_power_rate 
                          * effectiveTacklePower);    
    tackle_dir += WSinfo::ball->vel;
    if ( tackle_dir.norm() > ServerOptions::ball_speed_max )
      tackle_dir.normalize( ServerOptions::ball_speed_max );
  }
  else
  {
    //just set no pass communication
  }
}*/


void Wball06::set_neck(Intention &intention, Intention &pass_option, Intention & selfpass_option){
  MYLOG_POL(0,"******** SET NECK AND COMMUNICATE.");
  if(intention.is_pass2teammate() == true){
    MYLOG_POL(0,"Neck pass.");
    set_neck_pass(intention);
  }
  else if(intention.is_selfpass() == true){
    set_neck_selfpass(intention);
  }
  else if(intention.is_dribble() == true){
    set_neck_dribble();
  }
  set_neck_default(intention, pass_option, selfpass_option);  // this has only an influence, if neck not already set.
}


bool Wball06::I_am_in_left_corner(){
  if (WSinfo::me->pos.getX() < FIELD_BORDER_X- 16.0)
    return false;
  if (WSinfo::me->pos.getY() < 10.0)
    return false;
  return true;
}

bool Wball06::I_am_in_right_corner(){
  if (WSinfo::me->pos.getX() < FIELD_BORDER_X- 16.0)
    return false;
  if (WSinfo::me->pos.getY() >- 10.0)
    return false;
  return true;
}


void Wball06::set_neck_dribble(){
  if(status.dribble_neckreq.is_set() == true)
    Tools::set_neck_request(status.dribble_neckreq.get_type(),status.dribble_neckreq.get_param());
}

void Wball06::set_neck_default(Intention &intention, Intention &pass_option, Intention & selfpass_option){

  //check for looking to the goalie (without forcing)
  int goalieAgeThreshold = 1;
  if (    WSinfo::his_goalie 
       && WSinfo::his_goalie->pos.distance(WSinfo::me->pos) < 6.0 ) 
    goalieAgeThreshold = 0;
  if (   WSinfo::me->pos.distance(HIS_GOAL_CENTER) < 22.0
      && WSinfo::his_goalie
      && WSinfo::his_goalie->pos.distance(WSinfo::me->pos) < 12.0
      && WSinfo::his_goalie->age > goalieAgeThreshold)
  {
    ANGLE toGoalie = (WSinfo::his_goalie->pos - WSinfo::me->pos).ARG();
    if (Tools::could_see_in_direction( toGoalie ) )
    {
      MYLOG_POL(0,<<"TG@WBALL06: Set neck to goalie: look to dir "
        <<RAD2DEG(toGoalie.get_value_mPI_pPI()));
      Tools::set_neck_request(NECK_REQ_LOOKINDIRECTION, toGoalie);
      return;
    }
  }

  double targetdir[10];
  int num_dirs = 0;

  if(pass_option.valid_at()!= WSinfo::ws->time){
    Vector target;
    double speed;
    pass_option.get_kick_info(speed, target);
    targetdir[num_dirs ++] = (target - WSinfo::ball->pos).arg(); // direction of the pass target
  }
  if(I_am_in_left_corner()){
    targetdir[num_dirs ++] = -90./180. *PI;
    targetdir[num_dirs ++] = -170./180. *PI;
  }
  else  if(I_am_in_right_corner()){
    targetdir[num_dirs ++] = 90./180. *PI;
    targetdir[num_dirs ++] = 170./180. *PI;
  }
  else{
    targetdir[num_dirs ++] = 0./180. *PI;
    targetdir[num_dirs ++] = 45./180. *PI;
    targetdir[num_dirs ++] = -45./180. *PI;
    targetdir[num_dirs ++] = 90./180. *PI;
    targetdir[num_dirs ++] = -90./180. *PI;
  }

  for(int i= 0; i<num_dirs; i++){
    if(WSmemory::last_seen_in_dir(ANGLE(targetdir[i])) >1 && Tools::could_see_in_direction(targetdir[i])){ 
      MYLOG_POL(0,<<"set neck default : look to dir "<<RAD2DEG(targetdir[i]));
      Tools::set_neck_request(NECK_REQ_LOOKINDIRECTION, targetdir[i]);
      return;
    }
  }
  
  Tools::set_neck_request(NECK_REQ_LOOKINDIRECTION, 0.0);
}

void Wball06::set_neck_pass(Intention &pass_intention){
  Vector target;
  double speed;
  pass_intention.get_kick_info(speed, target);
  ANGLE ball2targetdir = (target - WSinfo::ball->pos).ARG(); // direction of the target
  if(WSmemory::last_seen_in_dir(ball2targetdir) >=0){ // probably change this to > 0 -> sometimes look!!!
    MYLOG_POL(0,<<"Set Neck Pass: Intention is to play pass-> look 2 receiver, direction: "<<RAD2DEG(ball2targetdir.get_value()));
    Tools::set_neck_request(NECK_REQ_LOOKINDIRECTION, ball2targetdir.get_value(), true); // forced
  }
  else{
    MYLOG_POL(0,<<"Set Neck Pass: no neck set, information fresh ");
  }
}



void Wball06::set_neck_selfpass(Intention & selfpass_intention){
  if (selfpass_intention.valid_at() != WSinfo::ws->time)
    return;
#if LOGGING && BASIC_LOGGING && WBALL06LOGGING
  int opage = -1;
#endif
  ANGLE opdir = selfpass_intention.target_body_dir;  // reasonable default (just in case...)
  bool can_look2_op = false;

  PlayerSet opset= WSinfo::valid_opponents;
  PPlayer attacker = opset.get_player_by_number(selfpass_intention.attacker_num);
  if(attacker != NULL){
    opdir = (attacker->pos  - WSinfo::me->pos).ARG();
#if LOGGING && BASIC_LOGGING && WBALL06LOGGING
    opage = attacker->age;
#endif
    can_look2_op = Tools::could_see_in_direction(opdir) ;
    MYLOG_DRAW(0, VC2D(attacker->pos, 1.3, "orange"));
    Tools::display_direction(attacker->pos, attacker->ang, 5.0,1);
    MYLOG_POL(0,<<"SET NECK selfpass: Op age: "<<opage<<" angle age: "<<attacker->age_ang
	      <<" attacer age vel: "<<attacker->age_vel);
  }
  else{ // no attacker known
    MYLOG_POL(0,<<"set neck selfpass: no attacker known. ");
    can_look2_op = false ;
  }

  bool can_look2_target =Tools::could_see_in_direction(selfpass_intention.target_body_dir.get_value());

  if(can_look2_target == false && can_look2_op ==false){
    MYLOG_POL(0,<<"SET NECK selfpass: Cannot look2 op. and cannot look2 target dir. DO NOT SET NECK ");
    return;
  }

  if(can_look2_op == false){
    MYLOG_POL(0,<<"SET NECK selfpass: Cannot look2 op. look in target dir. ");
    Tools::set_neck_request(NECK_REQ_LOOKINDIRECTION,selfpass_intention.target_body_dir.get_value(),true);    // force
    return;
  }
  if(can_look2_target == false){
    MYLOG_POL(0,<<"SET NECK selfpass: Cannot look2 target dir. Look to opponent. ");
    Tools::set_neck_request(NECK_REQ_LOOKINDIRECTION,opdir.get_value(),true);    // force
    return;
  }
  // from here on, I can look in both directions.
  if(WSmemory::last_seen_in_dir(selfpass_intention.target_body_dir) >= 1){ // must be consistent with threshold in test_selfpass()
    MYLOG_POL(0,<<"set neck selfpass. Turn 2 targetdir ");
    Tools::set_neck_request(NECK_REQ_LOOKINDIRECTION,selfpass_intention.target_body_dir.get_value(),true);    // force
    return;
  }

  MYLOG_POL(0,<<"set neck selfpass. Age of opponent: "<<opage<<" Turn 2 opdir ");
  Tools::set_neck_request(NECK_REQ_LOOKINDIRECTION,opdir.get_value(),true);    // force
}

BodyBehavior *
Wball06::getScoreBehavior()
{
  return 
    (BodyBehavior*) score05_sequence;
}

BodyBehavior *
Wball06::getOneStepScoreBehavior()
{
  return
    (BodyBehavior*) onestepscore;
}

bool 
Wball06::test_tingletangle_in_scoring_area(Intention& intention)
{
  const int MAX_TT_DIRS = 10;
  Vector testTarget[MAX_TT_DIRS];
  int numTTTarget = 0;
    
  //TG08: after oot passes we may have to use tingle tangle anyway
  PlayerSet opps = WSinfo::valid_opponents;
  if (WSinfo::his_goalie) opps.remove(WSinfo::his_goalie);
  opps.keep_players_in_circle( WSinfo::me->pos, 4.5);
  PlayerSet tmms = WSinfo::valid_teammates;
  tmms.keep_players_in_circle( WSinfo::me->pos, 25.0);
  tmms.keep_and_sort_closest_players_to_point( 1, HIS_GOAL_CENTER );
  Quadrangle2d wayToGoal( WSinfo::me->pos, HIS_GOAL_CENTER, 5.0, 15.0 );
  PlayerSet playersInFrontOfMe = WSinfo::valid_opponents;
  if (WSinfo::his_goalie) playersInFrontOfMe.remove(WSinfo::his_goalie);
  playersInFrontOfMe.keep_players_in( wayToGoal );
  bool iNecessarilyShouldAdvanceByUsingTingleTangle = false;
  if (   WSinfo::his_goalie != NULL
      && WSinfo::me->stamina < (ServerOptions::stamina_max*ServerOptions::effort_dec_thr+150.0) //TG09: alt: 1400 //cannot advance by self pass
      && WSinfo::his_goalie->pos.distance(WSinfo::me->pos) < 15.0
      && WSinfo::me->pos.getX() > WSinfo::his_team_pos_of_offside_line() - 2.0
      && opps.num == 0
      && (    tmms.num == 0 //potential pass receiver distant from goal
           ||   tmms[0]->pos.distance(HIS_GOAL_CENTER)
              > WSinfo::me->pos.distance(HIS_GOAL_CENTER) + 10.0 )
      && playersInFrontOfMe.num == 0
     )
  {
    LOG_POL(0,"Wball06: goalarea: Tingletangle: I necessarily should "
      <<"advance using TT.");
    iNecessarilyShouldAdvanceByUsingTingleTangle = true;
  } 
  if (    WSinfo::his_goalie == NULL
       || (   WSinfo::his_goalie 
           && WSinfo::his_goalie->pos.distance(WSinfo::me->pos) > 5.25
           && iNecessarilyShouldAdvanceByUsingTingleTangle == false )
     )
  {
    LOG_POL(0,"Wball06: goalarea: Tingletangle: Too distant from goalie.");
    return false;
  }
  opps = WSinfo::valid_opponents;
  if (WSinfo::his_goalie) opps.remove(WSinfo::his_goalie);
  opps.keep_players_in_circle( WSinfo::me->pos, 2.5);
  if (opps.num > 0)
  {
    LOG_POL(0,"Wball06: goalarea: Tingletangle: An opponent approaches me.");
    return false;
  }
    
  //general variables
  double myDistToHisGoalie
    = (WSinfo::his_goalie) ? WSinfo::me->pos.distance(WSinfo::his_goalie->pos)
                           : 100.0; 
  ANGLE myANGLEToHisGoal = (HIS_GOAL_CENTER-WSinfo::me->pos).ARG();
  Vector vectorFromMeToHisGoalie
    = (WSinfo::his_goalie) ? WSinfo::his_goalie->pos - WSinfo::me->pos
                           : Vector(0.0,0.0);
  bool myBodyANGLEPointsLeftwards 
    = (WSinfo::me->ang.get_value_0_p2PI() < PI);
  
  //TT to his goal  
  if (   fabs(myANGLEToHisGoal.get_value_mPI_pPI()) < (PI*(45./180.))
      && myDistToHisGoalie > 4.0)
    testTarget[ numTTTarget ++ ] = HIS_GOAL_CENTER;
  //TT 70 degree deviation
  if ( vectorFromMeToHisGoalie.norm() > 2.5 )
  {
    Vector deviation = vectorFromMeToHisGoalie;
    if (fabs(WSinfo::me->pos.getY()) > 7.0)
    {
      if (WSinfo::me->pos.getY() > 0.0)
        deviation.rotate( - (PI*(70./180.)) );
      else
        deviation.rotate( (PI*(70./180.)) );
    }
    else //i am quite central
    {
      if (myBodyANGLEPointsLeftwards)
        deviation.rotate( (PI*(70./180.)) );
      else
        deviation.rotate( - (PI*(70./180.)) );
    }
    deviation.normalize(5.0);
    deviation += WSinfo::me->pos;  
    testTarget[ numTTTarget ++ ] = deviation;
  }
  //TT 80 degree deviation
  if ( vectorFromMeToHisGoalie.norm() > 2.0 )
  {
    Vector deviation = vectorFromMeToHisGoalie;
    if (fabs(WSinfo::me->pos.getY()) > 7.0)
    {
      if (WSinfo::me->pos.getY() > 0.0)
        deviation.rotate( - (PI*(80./180.)) );
      else
        deviation.rotate( (PI*(80./180.)) );
    }
    else //i am quite central
    {
      if (myBodyANGLEPointsLeftwards)
        deviation.rotate( (PI*(80./180.)) );
      else
        deviation.rotate( - (PI*(80./180.)) );
    }
    deviation.normalize(5.0);
    deviation += WSinfo::me->pos;  
    testTarget[ numTTTarget ++ ] = deviation;
  }
  //TT 90 degree deviation
  if ( vectorFromMeToHisGoalie.norm() > 2.0 )
  {
    Vector deviation = vectorFromMeToHisGoalie;
    if (fabs(WSinfo::me->pos.getY()) > 7.0)
    {
      if (WSinfo::me->pos.getY() > 0.0 )
        deviation.rotate( - (PI*(90./180.)) );
      else
        deviation.rotate( (PI*(90./180.)) );
    }
    else //i am quite central
    {
      if (myBodyANGLEPointsLeftwards)
        deviation.rotate( (PI*(90./180.)) );
      else
        deviation.rotate( - (PI*(90./180.)) );
    }
    deviation.normalize(5.0);
    deviation += WSinfo::me->pos;  
    testTarget[ numTTTarget ++ ] = deviation;
  }
  //TT 100 degree deviation
  if ( vectorFromMeToHisGoalie.norm() > 1.5 )
  {
    Vector deviation = vectorFromMeToHisGoalie;
    if (fabs(WSinfo::me->pos.getY()) > 7.0)
    {
      if (WSinfo::me->pos.getY() > 0.0)
        deviation.rotate( - (PI*(100./180.)) );
      else
        deviation.rotate( (PI*(100./180.)) );
    }
    else //i am quite central
    {
      if (myBodyANGLEPointsLeftwards)
        deviation.rotate( (PI*(100./180.)) );
      else
        deviation.rotate( - (PI*(100./180.)) );
    }
    deviation.normalize(5.0);
    deviation += WSinfo::me->pos;  
    testTarget[ numTTTarget ++ ] = deviation;
  }
  
  for (int i=0; i<numTTTarget; i++)
  {
    tingletangle->setTarget( testTarget[i] );
    if ( tingletangle->isSafe() == true )
    {
      LOG_POL(0,"goalarea: tingle tangle target: "<<testTarget[i]);
      intention.set_tingletangle(testTarget[i], WSinfo::ws->time);
      return true;
    }
  }
  LOG_POL(0,"Wball06: goalarea: Tingletangle: No suitable target found.");
  return false;
}

