#include "attention_to_bma.h"
#include "intentiontype.h"
#include "blackboard.h"
#include "ws_info.h"
#include "ws_memory.h"
#include "../policy/positioning.h"

#if 0
#define MYLOG_POL(LLL,XXX) LOG_POL(LLL,<<"ATTENTION TO: "XXX)
#define MYLOG_DRAW(LLL,XXX) LOG_POL(LLL,<<_2D<<XXX)
#else
#define MYLOG_POL(LLL,XXX) 
#define MYLOG_DRAW(LLL,XXX)
#endif

#define COMMUNICATE_PLAYERS_DISTANCE_THRESHOLD    37.5

bool AttentionTo::initialized = false;

AttentionTo::AttentionTo() {
}

bool AttentionTo::get_cmd(Cmd &cmd) {

  construct_say_message(cmd);
  set_attention_to(cmd);

  return true;
}

bool AttentionTo::init(char const * conf_file, int argc, char const* const* argv) {
  if(initialized) return initialized;
  initialized = true;
  if(!initialized) { std::cout << "\nAttentionTo behavior not initialized."; }
  return initialized;
}

void AttentionTo::set_attention_to(Cmd & cmd){

  if (cmd.cmd_att.is_cmd_set()){
    LOG_POL(0, << "ATTENTION TO:  CMD FORM ALREADY SET !!! ");
    return;
  }
  
  
  LOG_POL(0, << "ATTENTION TO:  No attention set yet -> computing new attention ");


  Vector target;
  double speed;
  int target_player_number;

  // attention of PASS PLAYER
  if(Blackboard::get_pass_info(WSinfo::ws->time, speed, target, target_player_number) == true){
    if ( target_player_number > 0 ){
      LOG_POL(0, << "ATTENTION TO: Set attention to pass_receiver "<<target_player_number);
      cmd.cmd_att.set_attentionto(target_player_number);
    }
    else {
      LOG_POL(0, << "ATTENTION TO: pass info set, but no pass_receiver "<<target_player_number);
    }
    return;
  } // pass player

  int  new_att= 0;
  Vector ballholder_pos;
  int ballholder;
  if(teammate_controls_ball(ballholder, ballholder_pos)){// I do not control the ball myself and a teammate has the ball
    if(ballholder >0 && ballholder != WSinfo::me->number)
      new_att = ballholder;
    LOG_POL(0, << "ATTENTION TO: Set attention to ball holder "<<ballholder);
  }

  // if someone plays a pass, keep attention to him!
  PPlayer p= WSinfo::get_teammate_with_newest_pass_info();
  if ( p ) {
    new_att = p->number;
    LOG_POL(0, << "ATTENTION TO: Set attention to pass giver "<<new_att);
  }

  //take AttentionToRequest into consideration
  int currentAttentionToRequest = Tools::get_attention_to_request();
  if (currentAttentionToRequest != ATT_REQ_NONE)
  {
    new_att = currentAttentionToRequest;
    LOG_POL(0, << "ATTENTION TO: Set attention to teammate "<<new_att<<" (potential pass receiver / one_step_pass' foresee, via Tools::get_attention_to_request())");
  }

  LOG_POL(0,<<"ATTENTION TO: Current Attention to is set to player "<< WSinfo::ws->my_attentionto);

  if(new_att == 0){// not set yet

    PPlayer teammate;

    for(int i=0; i<WSinfo::num_relevant_teammates;i++)
    {
      if (WSinfo::get_teammate(WSinfo::relevant_teammate[i],teammate))
      {
#if 1
	LOG_POL(0,<<"ATTENTION 2: Checking relevant teammate "
		<<teammate->number<<" age "<<teammate->age);
#endif
	if(new_att == 0 && teammate->age >1){ // not yet found and too old
	  new_att = teammate->number;
	}
        if (   WSinfo::me->pos.distance(MY_GOAL_CENTER) < 30.0
            && WSinfo::ball->pos.distance(MY_GOAL_CENTER) < 20.0
            && WSinfo::ball->age > 2 )
        {
          LOG_POL(0,<<"ATTENTION 2: ALERT. Have lost ball during defending.");
          new_att = 1; //high danger: unknown ball during defending -> listen to goalie
          break;
        }
	if(teammate->age >3){ // much too old -> emergency set attention
	  new_att = teammate->number;
	  break;
	}
      }
      else{ // strange: relevant teammate not found -> set attention to
	LOG_POL(0,<<"ATTENTION 2: Teammate "<<WSinfo::relevant_teammate[i]<<" not found. Set attention to that player");
	new_att =WSinfo::relevant_teammate[i];
	break; // quit loop
      }
    }
  }

  LOG_POL(0,<<"ATTENTION TO: computed new attention: "<<new_att);


  int old_att= WSinfo::ws->my_attentionto;
  if (old_att <= 0)
    old_att= 0;
  if (new_att <= 0)
    new_att= 0;

  if ( old_att == new_att ) {
    LOG_POL(0,<<"ATTENTION TO: old attention = new attention: "<<new_att);
    return;
  }

  //  LOG_POL(0,<<"ATTENTION TO: computed new attention2: "<<new_att);

  if ( new_att > 0 ) 
  {
      cmd.cmd_att.set_attentionto(new_att);
      LOG_POL(0,<<"ATTENTION TO: Set Attention to player "<<new_att);
  }
  else
  { // new_att <= 0;  this should not happen
    cmd.cmd_att.set_attentionto_none();
    LOG_POL(0,<<"ATTENTION TO: set NO attention -> CHECK!!!");
    for(int i=0; i<WSinfo::num_relevant_teammates;i++)
    {
      PPlayer teammate = NULL;
      if (WSinfo::get_teammate(WSinfo::relevant_teammate[i],teammate))
      {
        if(new_att == 0 && teammate->age > 0)
        { // not yet found and too old
          LOG_POL(0,<<"ATTENTION TO: set attention to player "
            <<teammate->number<<" with age=1.");
          new_att = teammate->number;
          cmd.cmd_att.set_attentionto(new_att);
          break;
        }
      }
    }
  }
    
  LOG_POL(0,<<"ATTENTION TO: finished: with new attention: "<<new_att);
}

void AttentionTo::construct_say_message(Cmd & cmd){
  Vector target;
  double speed;
  int target_player_number;

  if (cmd.cmd_say.is_cmd_set()){
    // Ridi 06: added to avoid communicating something else, if say string is already set (in wball06)
    LOG_POL(0, << "Attentionto CONSTRUCT SAY:  CMD FORM ALREADY SET !!! ");
    LOG_ERR(0, << "Attentionto CONSTRUCT SAY:  CMD FORM ALREADY SET (should occur only when pass is communicated by wball06. please check)!!! "<<std::flush);
    MYLOG_POL(0, << "Attentionto CONSTRUCT SAY:  CMD FORM ALREADY SET (should occur only when pass is communicated by wball06. please check)!!! "<<std::flush);
    generate_players4communication(cmd); /*TG-BREMEN*/
    return;
  }
  else
  {
    MYLOG_POL(0, << "Attentionto CONSTRUCT SAY:  CMD FORM IS STILL UNSET ["<<(&cmd.cmd_say)<<"]!"<<std::flush);
  }

  //TGdoa: begin
  //always communicate the direct opponent assignment
  int assignment;
  if (Blackboard::get_direct_opponent_assignment_accepted( WSinfo::ws->time, assignment ) )
  {
    MYLOG_POL(0,<<"COMMUNICATION: I am communicating the new assignment "<<assignment);
    cmd.cmd_say.set_direct_opponent_assignment( assignment );
  }
  else
  {
    MYLOG_POL(0,<<"COMMUNICATION: I am communicating the old assignment "<<WSinfo::ws->current_direct_opponent_assignment);
    cmd.cmd_say.set_direct_opponent_assignment( WSinfo::ws->current_direct_opponent_assignment );
  }
  //TGdoa: end

  // Communication for GOALIE
  if (ClientOptions::consider_goalie) 
  {
    if ((WSinfo::ball->age <= 4) && (WSinfo::ws->play_mode == PM_PlayOn)) 
    {
      LOG_POL(0,<<"AttentionTo: I am the goalie: First communicate ball, then some players.");
      //first: ball information
      cmd.cmd_say.set_ball(WSinfo::ball->pos, WSinfo::ball->vel, WSinfo::ball->age, 4);
      //second: some players who are too distant from their direct opponent
      PlayerSet pl4comm
        = generate_players4communication4OpponentAwareDefense();
      //third: my next pos
      //       -> compute my next position
      Vector my_next_pos, my_next_vel;
      Angle my_next_ang;
      Tools::model_player_movement(WSinfo::me->pos,WSinfo::me->vel,WSinfo::me->ang.get_value(),
                                   cmd.cmd_body, my_next_pos,
                                   my_next_vel, my_next_ang);
      Player next_me = *WSinfo::me;
      next_me.pos = my_next_pos;
      pl4comm.prepend(&next_me);
      //finally: set players
      LOG_POL(0,<<"AttentionTo: I am the goalie: I communicate "
        <<pl4comm.num<<" players.");
      cmd.cmd_say.set_players( pl4comm );
      return;
    } 
  } // Goalie


  // Communication for PASS PLAYER
  if(Blackboard::get_pass_info(WSinfo::ws->time, speed, target, target_player_number) == true){
    MYLOG_POL(0, << "COMMUNICATION: Blackboard intention is PASS: to "<<target_player_number<<" target pos "
	    <<target<<" w. speed "<<speed<<" -> Communicate");
    int cycles2go = 0; // communicate for one cycle only
    if (Blackboard::pass_intention.immediatePass == false) cycles2go = 1;
    Vector ballvel;
    ballvel.init_polar(speed,(target - WSinfo::me->pos).arg());
    cmd.cmd_say.set_pass(WSinfo::me->pos,ballvel,WSinfo::ws->time + cycles2go);
    return;
  } // pass info was set
  else
  {
    MYLOG_POL(0,<<"COMMUNICATE: Blackboard contains no pass info. Probably already communicated (see also Wball06)."<<std::flush);
  }

  // Communication for PASS PLAYER, WHEN BALL JUST LEFT
  MYLOG_POL(0, << "Communicate: INFO: is_ball_kickable() == "<<WSinfo::is_ball_kickable()<<" ball_was_kickable4me_at == "<<WSmemory::ball_was_kickable4me_at<<" t=="<<WSinfo::ws->time);
  if(WSinfo::is_ball_kickable() == false && WSmemory::ball_was_kickable4me_at == WSinfo::ws->time -1)
  {
    // I had the ball last time, so communicate where the ball is now
    MYLOG_POL(0, << "Communicate: I had the ball last cycle, now I say where it goes ["<<WSinfo::ball->pos+WSinfo::ball->vel<<","<<ServerOptions::ball_decay*WSinfo::ball->vel<<"]");
    cmd.cmd_say.set_ball(WSinfo::ball->pos, WSinfo::ball->vel, WSinfo::ball->age, WSinfo::ball->age );
  }
  else
  // Communication for PLAYER near ball who OBSERVED an opponent's pass //TG
  if (   WSinfo::is_ball_kickable() == false 
      && WSinfo::ball->age == 0
      && WSmemory::get_opponent_last_at_ball_time() == WSinfo::ws->time - 1
      && WSinfo::ball->vel.norm() > 0.8 )
  {
    MYLOG_POL(0, << "Communicate: I saw an opponent's pass (by opp "
      <<WSmemory::get_opponent_last_at_ball_number()<<"). - So, now I say where it goes ["<<WSinfo::ball->pos+WSinfo::ball->vel<<","<<ServerOptions::ball_decay*WSinfo::ball->vel<<"]");
    cmd.cmd_say.set_ball(WSinfo::ball->pos, WSinfo::ball->vel, WSinfo::ball->age, WSinfo::ball->age );
  }
  
  // Communication for BALL HOLDER
  if(WSinfo::is_ball_pos_valid() && WSinfo::is_ball_kickable()){ // ball is kickable, but no pass info was set
    MYLOG_POL(0, << "COMMUNICATE: I have the ball and my Position. Use set me as Ballholder"<<std::flush);
    cmd.cmd_say.set_me_as_ball_holder(WSinfo::me->pos);
    generate_players4communication(cmd); /*TG-BREMEN*/
    return;
  }

  //TG08
  //communication of a freely flowing ball
  PlayerSet playersNearBall = WSinfo::valid_teammates;
  playersNearBall.join( WSinfo::valid_opponents );
  playersNearBall.keep_players_in_circle( WSinfo::ball->pos, 5.0 );
  if (    WSinfo::ball->age == 0
       && playersNearBall.num == 0
       && WSinfo::ball->pos.distance(WSinfo::me->pos) < 20.0
       && WSinfo::ball->vel.norm() > 1.5 )
  {
    MYLOG_POL(0, << "Communicate: I see the ball now, say where it goes ["<<WSinfo::ball->pos+WSinfo::ball->vel<<","<<ServerOptions::ball_decay*WSinfo::ball->vel<<"]");
    cmd.cmd_say.set_ball(WSinfo::ball->pos, WSinfo::ball->vel, WSinfo::ball->age, WSinfo::ball->age );
  }

  //Communication for player without ball REQUESTING A PASS //TG08
  int passRequestedInNSteps, passRequestedInAngle;
  if ( Blackboard::get_pass_request_request( passRequestedInNSteps,
                                             passRequestedInAngle) )
  {
    //first convert the requested pass angle to a value that
    //team communication can handle!
    passRequestedInAngle = passRequestedInAngle / 10 + 8;
    if ( passRequestedInAngle < 0 || passRequestedInAngle > 15 )
    {
      MYLOG_POL(0, << "COMMUNICATE: Want to set a PASS REQUEST in cmd_say "
        <<" but parameter is out of range: "<<passRequestedInAngle);
    }
    else
    {
      cmd.cmd_say.set_pass_request( passRequestedInNSteps,
                                    passRequestedInAngle );
      MYLOG_POL(0, << "COMMUNICATE: Set a PASS REQUEST in cmd_say to the ball"
        <<" leading player in "
        <<passRequestedInNSteps<<" time steps with parameter "
        <<passRequestedInAngle<<std::flush);
    }
  }
  else
  {
    MYLOG_POL(0, << "COMMUNICATE: No valid PASS REQUEST in Blackboard."<<std::flush);
  }

  // Communication for PLAYER WO BALL
  generate_players4communication(cmd);
}

#define MAX_SAY_CAPACITY 4 // no more than 4 players

void AttentionTo::generate_players4communication(Cmd & cmd_form)
{
  Vector target;
  int target_player_number;
  
  PlayerSet players4communication;
  PPlayer ballholder = WSinfo::teammate_closest2ball();

  // compute my next position
  Vector my_next_pos, my_next_vel;
  Angle my_next_ang;
  Tools::model_player_movement(WSinfo::me->pos,WSinfo::me->vel,WSinfo::me->ang.get_value(),
               cmd_form.cmd_body, my_next_pos, 
               my_next_vel, my_next_ang);

  //communication if pass announced
  PlayerSet ballHolderCommunicationSet;
  ballHolderCommunicationSet = WSinfo::valid_opponents;
  PPlayer potentialFeelRangePlayer = ballHolderCommunicationSet.get_player_by_number(0);
  if (potentialFeelRangePlayer) ballHolderCommunicationSet.remove(potentialFeelRangePlayer);

  ballHolderCommunicationSet.join(WSinfo::valid_teammates_without_me);
  potentialFeelRangePlayer = ballHolderCommunicationSet.get_player_by_number(0);
 if (potentialFeelRangePlayer) ballHolderCommunicationSet.remove(potentialFeelRangePlayer);


  ballHolderCommunicationSet.keep_players_with_max_age(1);
  if (    (   Blackboard::main_intention.get_type() == PASS
           || Blackboard::main_intention.get_type() == LAUFPASS )
       && WSinfo::ws->time - Blackboard::main_intention.valid_at_cycle < 5 
     )
  {
    target_player_number = Blackboard::main_intention.target_player;
    if (target_player_number > 1 && target_player_number < 12)
    {
        ballHolderCommunicationSet = WSinfo::valid_opponents;
        potentialFeelRangePlayer = ballHolderCommunicationSet.get_player_by_number(0);
        if (potentialFeelRangePlayer) ballHolderCommunicationSet.remove(potentialFeelRangePlayer);
        ballHolderCommunicationSet.keep_players_with_max_age(2);
        PPlayer passTargetPlayer
          = WSinfo::alive_teammates.get_player_by_number( target_player_number );
        if ( passTargetPlayer)
        {  //should always be found
          ballHolderCommunicationSet.
            keep_and_sort_closest_players_to_point(4,passTargetPlayer->pos);
          PlayerSet tooNearOpponents = WSinfo::valid_opponents;
          tooNearOpponents.keep_players_in_circle( passTargetPlayer->pos, 2.5 );
          ballHolderCommunicationSet.remove( tooNearOpponents );
          MYLOG_POL(0,<<"COMMUNICATE: A pass to "<<target_player_number
            <<" has been announced. There are "
            <<ballHolderCommunicationSet.num<<" opponents near to him. "
            <<"Pass valid since "<<Blackboard::main_intention.valid_since_cycle<<", valid at "<<Blackboard::main_intention.valid_at_cycle
            <<std::flush);
            if (   players4communication.num < MAX_SAY_CAPACITY
                && ballHolderCommunicationSet.num > 0)
            {
               MYLOG_POL(0,<<"COMMUNICATE: Found player close2e pass receiver: "
                 <<ballHolderCommunicationSet.num<<std::flush);
               players4communication.join(ballHolderCommunicationSet);
            }
        }
        else
        {
          MYLOG_POL(0,<<"COMMUNICATE: Could not determine pass target player."<<std::flush);
        }
    }
    else
    {
       MYLOG_POL(0,<<"COMMUNICATE: Could not determine pass target player (found number "<<target_player_number<<")."<<std::flush);
    }
  }
  else
  {
    if (WSinfo::is_ball_kickable())
    {
      ballHolderCommunicationSet = WSinfo::valid_opponents;
      potentialFeelRangePlayer = ballHolderCommunicationSet.get_player_by_number(0);
      if (potentialFeelRangePlayer) ballHolderCommunicationSet.remove(potentialFeelRangePlayer);
      ballHolderCommunicationSet.join(WSinfo::valid_teammates_without_me);
      potentialFeelRangePlayer = ballHolderCommunicationSet.get_player_by_number(0);
      if (potentialFeelRangePlayer) ballHolderCommunicationSet.remove(potentialFeelRangePlayer);


      ballHolderCommunicationSet.keep_players_with_max_age(1);
      MYLOG_POL(0,<<"COMMUNICATE: No pass info in Blackboard (type="
        <<Blackboard::main_intention.get_type()<<")."<<std::flush);
      ballHolderCommunicationSet.keep_players_in_circle(my_next_pos,40.0); // opponents close 2 me
      ballHolderCommunicationSet.keep_and_sort_closest_players_to_point(4,my_next_pos);
      if (   players4communication.num < MAX_SAY_CAPACITY
          && ballHolderCommunicationSet.num > 0)
      {
        MYLOG_POL(0,<<"COMMUNICATE: Found player close2e my next pos: "
          <<ballHolderCommunicationSet.num<<std::flush);
        players4communication.join(ballHolderCommunicationSet);
      }
    }
  }

  // I do not control the ball myself
  if( !WSinfo::is_ball_kickable()) 
  {

    MYLOG_POL(0,<<"I am not holding the ball, so communicate my next position "
	    << my_next_pos<<" (current pos "<<WSinfo::me->pos);      
    MYLOG_DRAW(0,C2D(my_next_pos.x, my_next_pos.y, 1.1, "grey"));

    // construct communication set :
    Player next_me = *WSinfo::me;

    next_me.pos = my_next_pos;
    players4communication.join(&next_me);
      
    if(ballholder != NULL)
    {
      PlayerSet pset = WSinfo::valid_opponents;
      if (    WSinfo::ball->pos.distance(HIS_GOAL_CENTER) < 22.0
	   && WSinfo::his_goalie
           && WSinfo::his_goalie->pos.distance(WSinfo::me->pos) < COMMUNICATE_PLAYERS_DISTANCE_THRESHOLD )
      {
	pset.remove( WSinfo::his_goalie );
        if (   WSinfo::his_goalie->age <= 1
            && WSmemory::last_seen_to_point( WSinfo::his_goalie->pos ) <= 1 )
        {
          players4communication.join( WSinfo::his_goalie );
          MYLOG_POL(0,<<"COMMUNICATE: Added goalie with age="<<WSinfo::his_goalie->age
            <<" and last seen to his pos at "<<WSmemory::last_seen_to_point( WSinfo::his_goalie->pos ));
        }
        else
        {
          MYLOG_POL(0,<<"COMMUNICATE: Did not add his goalie, age and last seen to his pos: "
            <<WSinfo::his_goalie->age<<","<<WSmemory::last_seen_to_point( WSinfo::his_goalie->pos ));
        }
      }
	    
      PPlayer potentialFeelRangePlayer = pset.get_player_by_number(0);
      if (potentialFeelRangePlayer) pset.remove(potentialFeelRangePlayer);

      pset.keep_players_with_max_age(1);

      float width1 =  1.0 * 2* ((ballholder->pos-WSinfo::me->pos).norm()/2.5);
      float width2 = 4; // at ball be  a little smaller
      Quadrangle2d check_area = Quadrangle2d(ballholder->pos, WSinfo::me->pos , width2, width1);
      MYLOG_DRAW(0, check_area );
      pset.keep_players_in(check_area);
      pset.keep_players_in_circle( WSinfo::me->pos, COMMUNICATE_PLAYERS_DISTANCE_THRESHOLD );
      //pset.keep_players_in_quadrangle(WSinfo::me->pos, ballholder->pos, 10 ); // players between me and ballholder

      pset.keep_and_sort_closest_players_to_point(3,WSinfo::me->pos);  //ballholder->pos);
    
      if(pset.num>0)
      {
        MYLOG_POL(0,<<"COMMUNICATE: Found Opponents in passway: "<<pset.num<<std::flush);
        players4communication.join(pset);
      }
      pset= WSinfo::valid_opponents;
      potentialFeelRangePlayer = pset.get_player_by_number(0);
      if (potentialFeelRangePlayer) pset.remove(potentialFeelRangePlayer);

      pset.keep_players_with_max_age(0);
      pset.keep_players_in_circle(ballholder->pos,5.); // opponents close 2 me
      pset.keep_and_sort_closest_players_to_point(2,ballholder->pos);
      pset.keep_players_in_circle( WSinfo::me->pos, COMMUNICATE_PLAYERS_DISTANCE_THRESHOLD );
      if(pset.num>0)
      {
        MYLOG_POL(0,<<"COMMUNICATE: Found Opponents close2e ballholder: "<<pset.num<<std::flush);
        players4communication.join(pset);
      }
    }

    if(players4communication.num < MAX_SAY_CAPACITY)
    {
      PlayerSet pset= WSinfo::valid_opponents;
      PPlayer potentialFeelRangePlayer = pset.get_player_by_number(0);
      if (potentialFeelRangePlayer) pset.remove(potentialFeelRangePlayer);

      pset.keep_players_with_max_age(0);
      pset.keep_players_in_circle(my_next_pos,COMMUNICATE_PLAYERS_DISTANCE_THRESHOLD); // opponents close 2 me
      pset.keep_and_sort_closest_players_to_point(10,my_next_pos);
      if(pset.num>0)
      {
        MYLOG_POL(0,<<"COMMUNICATE: Found Opponents close2e my next pos: "<<pset.num<<std::flush);
        players4communication.join(pset);
      }
    }

    if(players4communication.num < MAX_SAY_CAPACITY)
    {
      PlayerSet pset= WSinfo::valid_teammates_without_me;
      PPlayer potentialFeelRangePlayer = pset.get_player_by_number(0);
      if (potentialFeelRangePlayer) pset.remove(potentialFeelRangePlayer);

      pset.keep_players_with_max_age(0);
      pset.keep_players_in_circle(my_next_pos,COMMUNICATE_PLAYERS_DISTANCE_THRESHOLD); // opponents close 2 me
      pset.keep_and_sort_closest_players_to_point(10,my_next_pos);
      if(pset.num>0)
      {
        MYLOG_POL(0,<<"COMMUNICATE: Found Teammates close2e my next pos: "<<pset.num<<std::flush);
        players4communication.join(pset);
      }
    }

  } // ball is not kickable for me and teammate controls the ball

  if (players4communication.num < MAX_SAY_CAPACITY)
  {
     players4communication.
       join( generate_players4communication4OpponentAwareDefense() );
  }



  for(int i=0; i<players4communication.num;i++){
    MYLOG_POL(0,<<"My communication set "<<i<<" number "
	    <<players4communication[i]->number<<" pos "<<players4communication[i]->pos
        <<" team "<<players4communication[i]->team
        <<" age "<<players4communication[i]->age);
    
    MYLOG_DRAW(0,C2D(players4communication[i]->pos.x, players4communication[i]->pos.y, 1.3, "magenta"));
  }
  cmd_form.cmd_say.set_players( players4communication );

}

bool AttentionTo::teammate_controls_ball(int &number, Vector & pos){
  number = 0; // default
  if(WSinfo::is_ball_kickable() != false){// I control the ball myself!
    return false;
  }
  if (WSinfo::is_ball_pos_valid() == false)
    return false; //cannot do intercept estimations

  int steps2go;
  Vector ipos;
    
  PlayerSet teammates = WSinfo::valid_teammates_without_me;
  InterceptResult ires[1];
  teammates.keep_and_sort_best_interceptors_with_intercept_behavior_to_WSinfoBallPos
            ( 1, ires );
  if (teammates.num > 0 && ires[0].time <= 2)
  {
    number = teammates[0]->number;
    pos    = teammates[0]->pos;
    MYLOG_POL(2,<<"AttentionTo: /EXPENSIVE INTERCEPT/ To my way of thinking, the fastest intercepting teammate, to whom I ought to pay attention, is #"<<number<<", standing at "<<pos);
    return true;
  }

  PPlayer teammate=Tools::get_our_fastest_player2ball(ipos, steps2go);
  if(steps2go <= 2 && teammate && teammate != WSinfo::me){ // check: steps2go <= 1 oder steps2go <= 2 besser?
    number=teammate->number;
    pos = teammate->pos;
    MYLOG_POL(2,<<"AttentionTo: /CHEAP INTERCEPT/ To my way of thinking, the fastest intercepting teammate, to whom I ought to pay attention, is #"<<number<<", standing at "<<pos);
    return true;
  }
  return false;
}


PlayerSet
AttentionTo::generate_players4communication4OpponentAwareDefense()
{
     PlayerSet surveilledOpponents
       = OpponentAwarePositioning::
           getViableSurveillanceOpponentsForPlayer( WSinfo::me );
     MYLOG_POL(0,<<"COMMUNICATE: Found direct opponents too far away from their responsible teammate: "<<surveilledOpponents.num<<std::flush);
     for (int j=0; j<surveilledOpponents.num; j++)
       MYLOG_POL(0,<<"COMMUNICATE: ... these are: "<<surveilledOpponents[j]->number<<std::flush);
     return surveilledOpponents;
}


