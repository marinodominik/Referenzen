#include "formation.h"
#include "macro_msg.h"
#include "ws_info.h"
#include "valueparser.h"
#include "log_macros.h"
#include "ws_memory.h"
#include "mdp_info.h"
#include "blackboard.h"
#include <stdio.h>

bool Formation433::init(char const * conf_file, int argc, char const* const* argv) {
  if (NUM_PLAYERS < 11) {
    ERROR_OUT << "\nwrong number of players: " << NUM_PLAYERS; 
    return false;
  }
  //the following entries are (usually) in [0,1]x[-1,1]
  //x values smaller then 0 violate our defence line
  //x values greater then 1 violate our offence line
  //y values > 1 or < -1 violate the top or bottom field borders
  //goalie doesn't retrieve his pos from a Formation, but just to be consistent: he would run out of the field!
  home[1].pos = Vector(0, 0); 
  home[2].pos = Vector(0, 0.6);
  home[3].pos = Vector(0, 0.2);
  home[4].pos = Vector(0,-0.2);
  home[5].pos = Vector(0,-0.6);
  home[6].pos = Vector(0.5, 0.4);
  home[7].pos = Vector(0.5, 0.0);
  home[8].pos = Vector(0.5,-0.4);
  home[9].pos = Vector(1.0, 0.4);
  home[10].pos= Vector(1.0, 0.0);
  home[11].pos= Vector(1.0,-0.4);

  home[1].stretch_pos_x = 0.0;   home[1].stretch_neg_x = 0.0;   home[1].stretch_y = 0.0; 
  home[2].stretch_pos_x = 0.0;   home[2].stretch_neg_x = 0.0;   home[2].stretch_y = 0.0; 
  home[3].stretch_pos_x = 0.0;   home[3].stretch_neg_x = 0.0;   home[3].stretch_y = 0.0; 
  home[4].stretch_pos_x = 0.0;   home[4].stretch_neg_x = 0.0;   home[4].stretch_y = 0.0; 
  home[5].stretch_pos_x = 0.0;   home[5].stretch_neg_x = 0.0;   home[5].stretch_y = 0.0; 
  home[6].stretch_pos_x = 0.5;   home[6].stretch_neg_x = 0.0;   home[6].stretch_y = 0.5; 
  home[7].stretch_pos_x = 0.2;   home[7].stretch_neg_x = 0.0;   home[7].stretch_y = 0.5; 
  home[8].stretch_pos_x = 0.5;   home[8].stretch_neg_x = 0.0;   home[8].stretch_y = 0.5; 
  home[9].stretch_pos_x = 0.0;   home[9].stretch_neg_x = 0.0;   home[9].stretch_y = 0.25; 
  home[10].stretch_pos_x= 0.0;   home[10].stretch_neg_x= 0.0;   home[10].stretch_y= 0.25; 
  home[11].stretch_pos_x= 0.0;   home[11].stretch_neg_x= 0.0;   home[11].stretch_y= 0.25; 

  home[1].role = PT_GOALIE;
  home[2].role = PT_DEFENDER;
  home[3].role = PT_DEFENDER;
  home[4].role = PT_DEFENDER;
  home[5].role = PT_DEFENDER;
  home[6].role = PT_MIDFIELD;
  home[7].role = PT_MIDFIELD;
  home[8].role = PT_MIDFIELD;
  home[9].role = PT_ATTACKER;
  home[10].role= PT_ATTACKER;
  home[11].role= PT_ATTACKER;


  defence_line_ball_offset= 5.0;

  if ( ! conf_file ) {
    WARNING_OUT << " using hard wired formation";
    return true;
  }

  ValueParser vp(conf_file,"Formation433");
  INFO_OUT << "\nconf_file= " << conf_file;
  //vp.set_verbose(true);
  char key_pos[10];

#ifdef TRAINING
  double values[6];
#else
  double values[5];
#endif
  for(int i=1;i<=11;i++){
    sprintf(key_pos, "player_%d",i);
#ifdef TRAINING
    int num= vp.get(key_pos, values,6);
    if (num != 6) {
      ERROR_OUT << "\n wrong number of arguments (" << num << ") for [" << key_pos << "]";
      return false;
    }
#else
    int num= vp.get(key_pos, values,5);
    if (num != 5) {
      ERROR_OUT << "\n wrong number of arguments (" << num << ") for [" << key_pos << "]";
      return false;
    }
#endif

    home[i].pos.setX( values[0] );
    home[i].pos.setY( values[1] );
    home[i].stretch_pos_x= values[2];
    home[i].stretch_neg_x= values[3];
    home[i].stretch_y= values[4];
#ifdef TRAINING
    home[i].role= static_cast<int>(values[5]);
#endif
//    std::cout << "\n" << i << " pos= " << home[i].pos << " stretch= " << home[i].stretch_pos_x << ", " << home[i].stretch_neg_x << ", " << home[i].stretch_y << " role= " << home[i].role;
  }

  vp.get("defence_line_ball_offset",defence_line_ball_offset);
  //INFO_OUT << "\ndefence_line_ball_offset= " << defence_line_ball_offset;
  return true;
}

int Formation433::get_role(int number) {
  /*
  if ((WSinfo::ball->pos.x < -10.0) && (number == 7)) {
    return PT_DEFENDER;
    }*/
  if(WSinfo::ws->play_mode == PM_my_PenaltyKick)
    return 2;  

  return home[number].role;
}

Vector Formation433::get_grid_pos(int number) {
  //LOG_DEB(0,"play_mode= " << PLAYMODE_STR[WSinfo::ws->play_mode] << " ball_pos= " << WSinfo::ball->pos << " ball_age= " << WSinfo::ball->age);

  Vector res;
  
  Home & h = home[number]; 

  if (number == 7) {
    if ( (WSinfo::ball->pos.getX() < -10.0) && !mdpInfo::is_my_team_attacking() ) h.pos = Vector(0.2, 0.0);
    else if ( (WSinfo::ball->pos.getX() > -3.0) ||
	      ((WSinfo::ball->pos.getX() > -10.0) && mdpInfo::is_my_team_attacking()) )
      h.pos = Vector(0.5, 0.0);
  } else if (number == 6) {
    if ( (WSinfo::ball->pos.getX() < -25.0) && !mdpInfo::is_my_team_attacking() ) h.pos = Vector(0.3, 0.4);
    else if ( (WSinfo::ball->pos.getX() > -20.0) ||
	      ((WSinfo::ball->pos.getX() > -25.0) && mdpInfo::is_my_team_attacking()) )
      h.pos = Vector(0.5, 0.4);
  } else if (number == 8) {
    if ( (WSinfo::ball->pos.getX() < -25.0) && !mdpInfo::is_my_team_attacking() ) h.pos = Vector(0.3, -0.4);
    else if ( (WSinfo::ball->pos.getX() > -20.0) ||
	      ((WSinfo::ball->pos.getX() > -25.0) && mdpInfo::is_my_team_attacking()) )
	      h.pos = Vector(0.5, -0.4);
  }

  get_boundary(defence_line, offence_line); 

  double x_stretch= offence_line-defence_line;
  double y_stretch= FIELD_BORDER_Y;
  if ( get_role(number) == PT_ATTACKER && offence_line < FIELD_BORDER_X - 16.0 )
    y_stretch *= 1.5;

  switch(WSinfo::ws->play_mode) {
  case PM_my_GoalieFreeKick:
    if(get_role(number)==PT_DEFENDER) { 
      //y_stretch*=1.3;
    } else if(get_role(number)==PT_MIDFIELD) {
      //y_stretch*=1.5;
      x_stretch/=1.5;
    }
    break;
  case PM_my_GoalKick:
    if(get_role(number)==PT_DEFENDER) { 
      //y_stretch*=1.3;
    } else if(get_role(number)==PT_MIDFIELD) {
      //y_stretch*=1.5;
      x_stretch/=1.8;
    }
    break;
  }
  res.setX( defence_line + h.pos.getX() * x_stretch );
  res.setY( h.pos.getY() * y_stretch );

	// CIJAT_OSAKA
	// Reminder: This whole file is pretty unflexible and should be completely redone.
	switch ( WSinfo::ws->play_mode ) {
					case PM_my_BeforeKickOff:
					case PM_his_BeforeKickOff:
					case PM_my_KickOff:
					case PM_his_KickOff:
					case PM_Half_Time:
					case PM_my_AfterGoal:
					case PM_his_AfterGoal:
									if(number == 11 || number == 9)
													res.subFromX(3);
									break;
					default:
									break;
	}
	// CIJAT END
	
  switch ( WSinfo::ws->play_mode ) {
  case PM_his_BeforeKickOff:
  case PM_his_KickOff:
  case PM_my_AfterGoal:
    if ( number == 10 ) {
      res.subFromX(9.5);
      //LOG_DEB(0,"play_mode2= " << PLAYMODE_STR[WSinfo::ws->play_mode] << " res_pos= " << res);
    }
    break;
#if 0
  case PM_my_GoalieFreeKick:
    switch(number) {
    case 7: res.x = -16.5; break;
    }
    break;
  case PM_my_GoalKick:
    switch(number) {
    case 3: res.y = 0; break;
    case 7: res.x = -16.5; break;
    }
    break;
#endif
  }
#if 1
  if ( (defence_line < -33) && (get_role(number) == PT_DEFENDER) )
    res.setY( h.pos.getY() * y_stretch / 1.8 );
#endif

  Vector attr_to_ball= WSinfo::ball->pos - WSinfo::me->pos;
  double xxx= fabs(WSinfo::ball->pos.getY()/ FIELD_BORDER_Y)*1.5;
  if ( xxx > 1.0 )
    xxx= 1.0;
  attr_to_ball.mulYby(xxx);
  //attr_to_ball.y= WSinfo::ball->pos.y; //go2003

  switch ( WSinfo::ws->play_mode ) {
  case PM_my_BeforeKickOff:
  case PM_his_BeforeKickOff:
  case PM_my_KickOff:
  case PM_his_KickOff:
  case PM_my_AfterGoal:
  case PM_his_AfterGoal:
  case PM_his_GoalieFreeKick:
  case PM_his_GoalKick:
    attr_to_ball= Vector(0,0);
  }

  if ( attr_to_ball.getX() > 0 )
    attr_to_ball.mulXby(h.stretch_pos_x);
  else
    attr_to_ball.mulXby(h.stretch_neg_x);

  //Ball is more attractive if near the opponents goal!
  double reinforce_towards_x;
  if ( WSinfo::ball->pos.getX() < 0.0 )
    //reinforce_towards_x= 0.0;
    reinforce_towards_x= 1.0;
  else
    reinforce_towards_x= 1.0 +  WSinfo::ball->pos.getX()/ FIELD_BORDER_X;

  attr_to_ball.mulYby(h.stretch_y);
  attr_to_ball.mulYby(reinforce_towards_x);

  res += attr_to_ball;

#if 0
  if ( defence_line < -5.0 ) {
    double factor= double(-defence_line)/FIELD_BORDER_X;
    if ( number == 2 ) 
      res.y -= 7* factor;

    if ( number == 3 ) 
      res.y -= 2* factor;

    if ( number == 4 ) 
      res.y += 2* factor;

    if ( number == 5 ) 
      res.y += 7* factor;
  }
#endif

  //GO2003
  if ((number == 7) && (WSinfo::ws->play_mode == PM_PlayOn) && (defence_line < -FIELD_BORDER_X + 15.0)) {
    res.setX(defence_line + 6.0);
    res.setY(0.0);
  }

  LOG_DEB(0, << _2D  << VC2D(res,1.5,"ff0000") << STRING2D(res.getX()+0.4,res.getY(), number,"ff0000"));
  return res;
}

bool Formation433::need_fine_positioning(int number) {
  Vector grid_pos= get_grid_pos(number);

  if ( WSinfo::me->pos.sqr_distance( grid_pos ) > SQUARE(3.0) )
    return true;
  return false;
}

Vector Formation433::get_fine_pos(int number) {
  Vector res= get_grid_pos(number);

  //don't go into offside
  if ( WSinfo::me->pos.getX() > offence_line )
    return res;

  //don't be lazy if you are a defender, and should be closer to your own goal
  if ( get_role(number) == PT_DEFENDER && 
       WSinfo::me->pos.getX() > intercept_ball_pos.getX()  - defence_line_ball_offset )
    return res;

  double sqr_ball_dist= WSinfo::me->pos.sqr_distance( WSinfo::ball->pos);
  double lazyness;
  if ( sqr_ball_dist < SQUARE(30) )
    lazyness= 5.0;
  else
    lazyness= 9.0;
  
  if ( WSinfo::me->stamina > 3500 ) 
    lazyness /= 2;

  //if ( get_role(number) == PT_ATTACKER  && WSinfo::me->pos.x < offence_line -2.0 )
    
  //be lazy
  if ( res.sqr_distance( WSinfo::me->pos) < SQUARE(lazyness) )
    return WSinfo::me->pos;

  return res;
}

void Formation433::get_boundary(double & defence, double & offence) {
  if ( boundary_update_cycle == WSinfo::ws->time ) {
    defence= defence_line;
    offence= offence_line;
  }

  boundary_update_cycle= WSinfo::ws->time;

  switch ( WSinfo::ws->play_mode ) {
  case PM_my_BeforeKickOff:
  case PM_his_BeforeKickOff:
  case PM_my_KickOff:
  case PM_his_KickOff:
  case PM_my_AfterGoal:
  case PM_his_AfterGoal:
    defence_line= -20.0;
    offence_line= -0.1;
    defence= defence_line;
    offence= offence_line;
    return;
  case PM_my_GoalieFreeKick:
  case PM_my_GoalKick:
    defence_line= -33.0;
    offence_line= 5.0;
    defence= defence_line;
    offence= offence_line;
    return;
  case PM_his_GoalieFreeKick:
  case PM_his_GoalKick:
    defence_line= -5.0;
    offence_line= 33.0;
    defence= defence_line;
    offence= offence_line;
    return;
  }

  Vector ball_pos= WSinfo::ball->pos;
  Vector ball_vel= WSinfo::ball->vel;
  if ( ! WSinfo::is_ball_pos_valid() ) //don't use the velocity, it the ball is too old!
    ball_vel= Vector(0,0);

  PlayerSet pset= WSinfo::valid_teammates;
  pset+= WSinfo::valid_opponents;
  InterceptResult ires[1];
  
  pset.keep_and_sort_best_interceptors(1, ball_pos, ball_vel, ires);
  intercept_ball_pos= ires[0].pos;

  //go back defence_line_ball_offset meters behind the point, where the ball will be intercepted
  if ( (WSinfo::ws->play_mode == PM_his_KickIn) ||
       (WSinfo::ws->play_mode == PM_his_FreeKick) ) {
    defence_line= ires[0].pos.getX() - 15.0;
  } else {
    //    if(WSmemory::team_last_at_ball() == MY_TEAM){
    if(WSmemory::team_last_at_ball() == 0){
      defence_line= ires[0].pos.getX() ;
    }
    else
      defence_line= ires[0].pos.getX() - defence_line_ball_offset;
  }

  //don't go into the opponent's half
  if ( defence_line > -4.0 )
    defence_line= -4.0;

  //don't go out of the field
  if ( defence_line < -FIELD_BORDER_X + 10 )
    defence_line= -FIELD_BORDER_X + 10;

  offence_line= WSinfo::his_team_pos_of_offside_line();
  if ( offence_line > FIELD_BORDER_X - 5.0 )
    offence_line= FIELD_BORDER_X- 5.0;
  
  if ( intercept_ball_pos.getX() < -10.0) //usopen2003 hack, to get the attackers nearer to the midfield
    offence_line -= 5.0;

  if ( offence_line - defence_line > 50.0 )
    offence_line= defence_line+ 50.0;
  
  defence= defence_line;
  offence= offence_line;
}

//////////////////////////////////////////////////////////////////////////////
// CLASS Formation05 by tga
//////////////////////////////////////////////////////////////////////////////
#define TGA_DUMMY_FORMATION_C

//STATIC VARIABLES
int Formation05::cvMyCurrentDirectOpponentAssignment = -1;

Formation05::DistanceCollection::DistanceCollection()
{
    myPlayer = NULL;
    hisPlayer = NULL;
    distance = 0.0;
}

Formation05::CoachAssignmentUpdateInformation::CoachAssignmentUpdateInformation()
{
    time[ 0 ] = -2;
    time[ 1 ] = -1;
    numberOfAssignments[ 0 ] = 0;
    numberOfAssignments[ 1 ] = 0;
}

void Formation05::CoachAssignmentUpdateInformation::update()
{
    if( WSinfo::ws->number_of_direct_opponent_updates_from_coach > numberOfAssignments[ 1 ] )
    {
        time[ 0 ] = time[ 1 ];
        time[ 1 ] = WSinfo::ws->time;
        numberOfAssignments[ 0 ] = numberOfAssignments[ 1 ];
        numberOfAssignments[ 1 ] = WSinfo::ws->number_of_direct_opponent_updates_from_coach;
    }
}

/**
 * Initialisation Method
 * 
 */
bool 
Formation05::init(char const *       conf_file, 
                  int                argc, 
                  char const* const* argv) {
  if (NUM_PLAYERS < 11) 
  {
    ERROR_OUT << "\nWrong number of players: " << NUM_PLAYERS; 
    return false;
  }
  //the following entries are (usually) in [0,1]x[-1,1]
  //x values smaller then 0 violate our defence line
  //x values greater then 1 violate our offence line
  //y values > 1 or < -1 violate the top or bottom field borders
  //goalie doesn't retrieve his pos from a Formation, but just to 
  //be consistent: he would run out of the field!
  ivHomePositions[1].pos = Vector(0, 0); 
  ivHomePositions[2].pos = Vector(0.1, 0.5);
  ivHomePositions[3].pos = Vector(0.1, 0.0);
  ivHomePositions[4].pos = Vector(0.0, 0.0);
  ivHomePositions[5].pos = Vector(0.1,-0.5);
  ivHomePositions[6].pos = Vector(0.5, 0.4);
  ivHomePositions[7].pos = Vector(0.5, 0.0);
  ivHomePositions[8].pos = Vector(0.5,-0.4);
  ivHomePositions[9].pos = Vector(1.0, 0.4);
  ivHomePositions[10].pos= Vector(1.0, 0.0);
  ivHomePositions[11].pos= Vector(1.0,-0.4);

  ivHomePositions[1].stretch_pos_x = 0.0;   ivHomePositions[1].stretch_neg_x = 0.0;   ivHomePositions[1].stretch_y = 0.0; 
  ivHomePositions[2].stretch_pos_x = 0.0;   ivHomePositions[2].stretch_neg_x = 0.0;   ivHomePositions[2].stretch_y = 0.0; 
  ivHomePositions[3].stretch_pos_x = 0.0;   ivHomePositions[3].stretch_neg_x = 0.0;   ivHomePositions[3].stretch_y = 0.0; 
  ivHomePositions[4].stretch_pos_x = 0.0;   ivHomePositions[4].stretch_neg_x = 0.0;   ivHomePositions[4].stretch_y = 0.0; 
  ivHomePositions[5].stretch_pos_x = 0.0;   ivHomePositions[5].stretch_neg_x = 0.0;   ivHomePositions[5].stretch_y = 0.0; 
  ivHomePositions[6].stretch_pos_x = 0.5;   ivHomePositions[6].stretch_neg_x = 0.0;   ivHomePositions[6].stretch_y = 0.5; 
  ivHomePositions[7].stretch_pos_x = 0.2;   ivHomePositions[7].stretch_neg_x = 0.0;   ivHomePositions[7].stretch_y = 0.5; 
  ivHomePositions[8].stretch_pos_x = 0.5;   ivHomePositions[8].stretch_neg_x = 0.0;   ivHomePositions[8].stretch_y = 0.5; 
  ivHomePositions[9].stretch_pos_x = 0.0;   ivHomePositions[9].stretch_neg_x = 0.0;   ivHomePositions[9].stretch_y = 0.25; 
  ivHomePositions[10].stretch_pos_x= 0.0;   ivHomePositions[10].stretch_neg_x= 0.0;   ivHomePositions[10].stretch_y= 0.25; 
  ivHomePositions[11].stretch_pos_x= 0.0;   ivHomePositions[11].stretch_neg_x= 0.0;   ivHomePositions[11].stretch_y= 0.25; 

  ivHomePositions[1].role = PT_GOALIE;
  ivHomePositions[2].role = PT_DEFENDER;
  ivHomePositions[3].role = PT_DEFENDER;
  ivHomePositions[4].role = PT_DEFENDER;
  ivHomePositions[5].role = PT_DEFENDER;
  ivHomePositions[6].role = PT_MIDFIELD;
  ivHomePositions[7].role = PT_MIDFIELD;
  ivHomePositions[8].role = PT_MIDFIELD;
  ivHomePositions[9].role = PT_ATTACKER;
  ivHomePositions[10].role= PT_ATTACKER;
  ivHomePositions[11].role= PT_ATTACKER;

  ivDirectOpponentAssignments[1].directOpponent = -1;
  ivDirectOpponentAssignments[2].directOpponent = -1;
  ivDirectOpponentAssignments[3].directOpponent = -1;
  ivDirectOpponentAssignments[4].directOpponent = -1;
  ivDirectOpponentAssignments[5].directOpponent = -1;
  ivDirectOpponentAssignments[6].directOpponent = -1;
  ivDirectOpponentAssignments[7].directOpponent = -1;
  ivDirectOpponentAssignments[8].directOpponent = -1;
  ivDirectOpponentAssignments[9].directOpponent = -1;
  ivDirectOpponentAssignments[10].directOpponent= -1;
  ivDirectOpponentAssignments[11].directOpponent= -1;

  ivDirectOpponentAssignments[1].previousDirectOpponent = -1;
  ivDirectOpponentAssignments[2].previousDirectOpponent = -1;
  ivDirectOpponentAssignments[3].previousDirectOpponent = -1;
  ivDirectOpponentAssignments[4].previousDirectOpponent = -1;
  ivDirectOpponentAssignments[5].previousDirectOpponent = -1;
  ivDirectOpponentAssignments[6].previousDirectOpponent = -1;
  ivDirectOpponentAssignments[7].previousDirectOpponent = -1;
  ivDirectOpponentAssignments[8].previousDirectOpponent = -1;
  ivDirectOpponentAssignments[9].previousDirectOpponent = -1;
  ivDirectOpponentAssignments[10].previousDirectOpponent= -1;
  ivDirectOpponentAssignments[11].previousDirectOpponent= -1;

return true;

  if ( ! conf_file ) {
    WARNING_OUT << " using hard wired formation";
    return true;
  }

  ValueParser vp(conf_file,"Formation05");
  INFO_OUT << "\nconf_file= " << conf_file;

  char key_pos[10];

  double values[5];

  for(int i=1; i<=NUM_PLAYERS; i++)
  {
    sprintf(key_pos, "player_%d",i);
    int num = vp.get(key_pos, values, 5);
    if (num != 5) 
    {
      ERROR_OUT << "\n wrong number of arguments (" << num << ") for [" << key_pos << "]";
      return false;
    }

    ivHomePositions[i].pos.setX( values[0] );
    ivHomePositions[i].pos.setY( values[1] );
    ivHomePositions[i].stretch_pos_x= values[2];
    ivHomePositions[i].stretch_neg_x= values[3];
    ivHomePositions[i].stretch_y= values[4];
    //std::cout << "\n" << i << " pos= " << ivHomePositions[i].pos 
    //          << " stretch= " << ivHomePositions[i].stretch_pos_x 
    //          << ", " << ivHomePositions[i].stretch_neg_x << ", " 
    //          << ivHomePositions[i].stretch_y << " role= " 
    //          << ivHomePositions[i].role;
  }
  return true;
}

/**
 * Each player is assigned a role, which may be one of GOALIE,
 * DEFENDER, MIDFIELDER, ATTACKER. This method returns the corresponding
 * integer value (cf. globaldef.h).
 */
int 
Formation05::get_role(int number) 
{
/*  //PENALTY MODE: I AM ATTACKER!
  if(WSinfo::ws->play_mode == PM_my_PenaltyKick)
    return 2;  
  PPlayer directOpponent;
  //START OF MATCH: center midfielder starts defensively
  if (WSinfo::me->number == 7 && WSinfo::ws->time < 500
      && getDirectOpponent(7,directOpponent) && directOpponent->pos.x<0.0 )
    return PT_DEFENDER;
  //CONSIDER HIS ATTACKERS
  WSpset opps = WSinfo::valid_opponents;
  opps.keep_and_sort_players_by_x_from_left( 3 );
  if (    WSinfo::me->number >= 6 && WSinfo::me->number <= 8 //actually midfielder
       && getDirectOpponent(WSinfo::me->number,directOpponent) 
       && opps.get_player_by_number( directOpponent->number ) )
    return PT_DEFENDER;
  return ivHomePositions[number].role;*/

  //PENALTY MODE: I AM ATTACKER!
  if(WSinfo::ws->play_mode == PM_my_PenaltyKick)
    return 2;  
  PPlayer directOpponent;

  //TG09: BEGIN ... special mode vs. WE09 exploits

  if (    WSinfo::get_current_opponent_identifier() == TEAM_IDENTIFIER_WRIGHTEAGLE
       || WSinfo::get_current_opponent_identifier() == TEAM_IDENTIFIER_OXSY
       ||
          ( 0 && WSinfo::ws->my_team_score == 0 && WSinfo::ws->his_team_score == 1 )        
     )
  {
    PlayerSet allHisPlayers = WSinfo::valid_opponents;
    //try to discover his defenders
    PlayerSet hisDefenders = allHisPlayers;
    hisDefenders.keep_and_sort_players_by_x_from_right( 5 );//assume 4 defenders + goalie
    if (WSinfo::his_goalie)
      hisDefenders.remove( WSinfo::his_goalie );
    //try to discover his attackers
    PlayerSet hisAttackers = allHisPlayers;
    hisAttackers.keep_and_sort_players_by_x_from_left( 3 );
    //paint them
    for (int i=0; i<allHisPlayers.num; i++)
    {
      if (hisDefenders.get_player_by_number( allHisPlayers[i]->number ) )
      {
        LOG_POL(4,<<_2D<<VC2D( allHisPlayers[i]->pos, 2.5, "999999"));
      }
      else
      if (hisAttackers.get_player_by_number( allHisPlayers[i]->number ) )
      {
        LOG_POL(4,<<_2D<<VC2D( allHisPlayers[i]->pos, 2.5, "55ff55"));
      }
      else
      if ( allHisPlayers[i] == WSinfo::his_goalie )
      {
        LOG_POL(4,<<_2D<<VC2D( allHisPlayers[i]->pos, 2.5, "ffff00"));
      }
      else
      {
        LOG_POL(4,<<_2D<<VC2D( allHisPlayers[i]->pos, 2.5, "ff0000"));
      }
    }
    //who is my direct opponent
    PPlayer theDirectOpponent = NULL;
    getDirectOpponent( number, theDirectOpponent );
    if ( theDirectOpponent != NULL )
    {
      if (    hisDefenders.get_player_by_number( theDirectOpponent->number ) 
           && (   fabs( WSinfo::his_team_pos_of_offside_line() - theDirectOpponent->pos.getX())
                < fabs( WSinfo::my_team_pos_of_offside_line() - theDirectOpponent->pos.getX()) )
         )
      {
        PPlayer thePlayer = WSinfo::alive_teammates.get_player_by_number(number);
        if (thePlayer)
        { LOG_POL(3,<<_2D<<VC2D( thePlayer->pos, 2.5, "55ff55")); }
        LOG_POL(0,"I ["<<WSinfo::me->number<<"] see teammate "<<number
          <<" as ATTACKER");
        return PT_ATTACKER;
      }
      else
      if (    hisAttackers.get_player_by_number( theDirectOpponent->number )
           && (   fabs( WSinfo::his_team_pos_of_offside_line() - theDirectOpponent->pos.getX())
                > fabs( WSinfo::my_team_pos_of_offside_line() - theDirectOpponent->pos.getX()) )
         )
      {
        PPlayer thePlayer = WSinfo::alive_teammates.get_player_by_number(number);
        if (thePlayer)
        { LOG_POL(3,<<_2D<<VC2D( thePlayer->pos, 2.5, "999999")); }
        LOG_POL(0,"I ["<<WSinfo::me->number<<"] see teammate "<<number
          <<" as DEFENDER");
        return PT_DEFENDER;
      }
      else
      {
        PPlayer thePlayer = WSinfo::alive_teammates.get_player_by_number(number);
        if (thePlayer)
        { LOG_POL(3,<<_2D<<VC2D( thePlayer->pos, 2.5, "ff0000")); }
        LOG_POL(0,"I ["<<WSinfo::me->number<<"] see teammate "<<number
          <<" as MIDFIELDER");
        return PT_MIDFIELD;
      }
    }
  }
  //TG09: END
    
  //START OF MATCH: center midfielder starts defensively
  if (number == 7 && WSinfo::ws->time < 500
      && getDirectOpponent(7,directOpponent) && directOpponent->pos.getX()<-15.0 )//TG08:0->-15
    return PT_DEFENDER;
    
  //CONSIDER HIS ATTACKERS
  PlayerSet opps = WSinfo::valid_opponents;
  opps.keep_and_sort_players_by_x_from_left( 3 );
  if (    number >= 6 && number <= 8 //actually midfielder
       && getDirectOpponent(number, directOpponent) 
       && opps.get_player_by_number( directOpponent->number )
       && directOpponent->pos.getX() < 10.0 //ZUI: for testing changed from 10 to 0, please re-change before atlanta! //TG08
       && (WSinfo::ball->pos.getX() < 0.5*FIELD_BORDER_X || WSinfo::ball->vel.getX() < 0.0 ))
    return PT_DEFENDER;

  //TG17: SPECIAL ANTI HELIOS MODE, IMPLEMENTED BEFORE FINAL DAY
  if ( WSinfo::get_current_opponent_identifier() == TEAM_IDENTIFIER_HELIOS )
  {
    PlayerSet heliosAttacker334 = WSinfo::valid_opponents;
    heliosAttacker334.keep_and_sort_closest_players_to_point( heliosAttacker334.num, HIS_GOAL_CENTER );
    // reduce to only the 4 most _distant_ players by removing the first ones
    while (heliosAttacker334.num > 4)
      heliosAttacker334.remove( heliosAttacker334[0] );
    // the 4 players most distant to his goal are his attackers
    if (   number >= 6 && number <= 8                // I'm actually midfielder)
        && getDirectOpponent(number, directOpponent) // I do have a DO
        && (   (   heliosAttacker334.get_player_by_number( directOpponent->number ) // my do is within the set of his foremost players
                && directOpponent->pos.getX() < 0.0          // my DO is quite advanced
               )
            || directOpponent->pos.distance(MY_GOAL_CENTER) < 20
            || directOpponent->pos.getX() < -FIELD_BORDER_X + PENALTY_AREA_LENGTH
           )
        && WSmemory::cvCurrentInterceptPeople.num > 0
        && (   WSmemory::cvCurrentInterceptPeople[0]->team == HIS_TEAM
            || WSinfo::ball->pos.getX() < 0.5*FIELD_BORDER_X )
       )
    {
      LOG_POL(0,<<"ANTI HELIOS MODE: consider that my #"<<number<<" is DEFENDER");
      return PT_DEFENDER;
    }
  }

  return ivHomePositions[number].role;
}

bool
Formation05::getDirectOpponent(int number, PPlayer & opponent)
{
  opponent = WSinfo::alive_opponents.get_player_by_number
      (
        this->getDirectOpponentNumber(number)
      );
  LOG_POL(5, << "Formation05: Retrieval of direct opponent("
             << this->getDirectOpponentNumber(number)<<") for me("<<number<<") "
             << (opponent?" ok.":" failed.") << flush);
  if (opponent != NULL)
    return true;
  return false;
}

bool
Formation05::getResponsiblePlayerForOpponent(int number, PPlayer & respTeammate)
{
  PPlayer opponent = WSinfo::alive_opponents.get_player_by_number(number);
  respTeammate = NULL;
  if ( ! opponent)
    return false;
  int dirOppNr = opponent->number;
  for (int i=0; i<WSinfo::alive_teammates.num; i++)
  {
    if (  this->getDirectOpponentNumber( WSinfo::alive_teammates[i]->number ) 
          == dirOppNr )
    {
      respTeammate = WSinfo::alive_teammates[i];
      return true;
    }
  }
  return false;
}

/**
 * The Formation05 assign dynamically a direct opponent player
 * to each player. This direct opponent's number can be retrieved
 * with this method.
 */
int
Formation05::getDirectOpponentNumber(int number)
{
  return this->ivDirectOpponentAssignments[number].directOpponent;
}

bool    
Formation05::getDirectOpponentPosition(int number, Vector & pos)
{
  LOG_POL(4, << "Noball05: My direct opponent is "<<getDirectOpponentNumber(number)<<flush);
  for (int i=0; i<WSinfo::alive_opponents.num; i++)
    LOG_POL(4,<<"          Alive opponent is "<<WSinfo::alive_opponents[i]->number<<" having age "<<WSinfo::alive_opponents[i]->age<<flush);
  for (int i=0; i<WSinfo::valid_opponents.num; i++)
    LOG_POL(4,<<"          Valid opponent is "<<WSinfo::valid_opponents[i]->number<<flush);
  PPlayer opponent
    = WSinfo::alive_opponents.get_player_by_number
      (
        this->getDirectOpponentNumber(number)
      );
  if (opponent != NULL)
  {
    pos = opponent->pos;
    return true;
  }
  return false;
}

  
Vector  
Formation05::getHomePosition(int number, bool initialHomePosition)
{
  XYRectangle2d hisPlayerRect;
  if (WSinfo::alive_opponents.num >= 3  &&  initialHomePosition ==false )
    hisPlayerRect = this->getCurrentFieldPlayerRectangle(HIS_TEAM);
  else{
    hisPlayerRect = XYRectangle2d(Vector(-30,FIELD_BORDER_Y),Vector(-5,-FIELD_BORDER_Y));
  }
  return getHomePosition(number,hisPlayerRect,initialHomePosition);
}
Vector  
Formation05::getHomePosition(int number, XYRectangle2d rect, bool initialHomePosition)
{
  Vector returnValue;
  float minX = rect.center.getX() - (rect.size_x/2.0);
  float maxX = rect.center.getX() + (rect.size_x/2.0);
  returnValue.setX( minX + ivHomePositions[number].pos.getX()*(maxX-minX) );
  returnValue.setY( rect.center.getY()
    + ivHomePositions[number].pos.getY() * (rect.size_y) );
  return returnValue;
}

  
// ... TBD ...
Vector  
Formation05::get_grid_pos(int number)
{
  return Vector(0.0,0.0);
}
bool    
Formation05::need_fine_positioning(int number)
{
  return true;
}
Vector  
Formation05::get_fine_pos(int number)
{
  return Vector(0.0,0.0);
}
void 
Formation05::get_boundary(double & defence, double & offence)
{
  return;
}
// ... TBD ...

/**
 * This method returns the current backward offside line of a team.
 * Important: The term "Backward" here refers to the global coordinate
 * system. Hence, also for the opponent team we use OUR coordinate system.
 * It excludes the respective goalie.
 */
double
Formation05::getCurrentRearmostPlayerLine(int team)
{
  PlayerSet playerSet;
  if (team==MY_TEAM)
    playerSet = WSinfo::valid_teammates;
  else
    playerSet = WSinfo::valid_opponents;
  double offsideLine = - FIELD_BORDER_X;
  playerSet.keep_and_sort_players_by_x_from_left(2);
  if (playerSet.num > 0 )
  {
    if ( (team==MY_TEAM && playerSet[0]->number == WSinfo::ws->my_goalie_number)
        ||
         (team==HIS_TEAM && playerSet[0]->number == WSinfo::ws->his_goalie_number)
       )
    {
      if (playerSet.num > 1)
        offsideLine = playerSet[1]->pos.getX();
    }
    else
    {
      offsideLine = playerSet[0]->pos.getX();
    }
  }
  return offsideLine;
}

/**
 * This method returns the current forward offside line of a team.
 * Important: The term "forward" here refers to the global coordinate
 * system. Hence, also for the opponent team we use OUR coordinate system.
 * It excludes the respective goalie.
 */
double
Formation05::getCurrentForemostPlayerLine(int team)
{
  PlayerSet playerSet;
  if (team==MY_TEAM)
    playerSet = WSinfo::valid_teammates;
  else
    playerSet = WSinfo::valid_opponents;
  double offsideLine = FIELD_BORDER_X;
  playerSet.keep_and_sort_players_by_x_from_right(2);
  if (playerSet.num > 0 )
  {
    if ( (team==MY_TEAM && playerSet[0]->number == WSinfo::ws->my_goalie_number)
        ||
         (team==HIS_TEAM && playerSet[0]->number == WSinfo::ws->his_goalie_number)
       )
    {
      if (playerSet.num > 1)
        offsideLine = playerSet[1]->pos.getX();
    }
    else
    {
      offsideLine = playerSet[0]->pos.getX();
    }
  }
  return offsideLine;
}


/**
 * This method provides us with the y-coordinate of our currently
 * leftmost player.
 */
double
Formation05::getCurrentLeftmostPlayerLine(int team)
{
  PlayerSet playerSet; 
  if (team == MY_TEAM)
    playerSet = WSinfo::valid_teammates;
  else
    playerSet = WSinfo::valid_opponents;
  double leftLine = - FIELD_BORDER_Y;
  playerSet.keep_and_sort_players_by_y_from_left(2);
  if (playerSet.num > 0 )
  {
    if ( (team==MY_TEAM && playerSet[0]->number == WSinfo::ws->my_goalie_number)
         ||
         (team==HIS_TEAM && playerSet[0]->number == WSinfo::ws->his_goalie_number)
       )
    {
      if (playerSet.num > 1 )
        leftLine = playerSet[1]->pos.getY();
    }
    else
    {
      leftLine = playerSet[0]->pos.getY();
    }
  }
  return leftLine;
}

/**
 * This method provides us with the y-coordinate of our currently
 * rightmost player.
 */
double
Formation05::getCurrentRightmostPlayerLine(int team)
{
  PlayerSet playerSet; 
  if (team == MY_TEAM)
    playerSet = WSinfo::valid_teammates;
  else
    playerSet = WSinfo::valid_opponents;
  double rightLine = FIELD_BORDER_Y;
  playerSet.keep_and_sort_players_by_y_from_right(2);
  if (playerSet.num > 0 )
  {
    if ( (team==MY_TEAM && playerSet[0]->number == WSinfo::ws->my_goalie_number)
         ||
         (team==HIS_TEAM && playerSet[0]->number == WSinfo::ws->his_goalie_number)
       )
    {
      if (playerSet.num > 1 )
        rightLine = playerSet[1]->pos.getY();
    }
    else
    {
      rightLine = playerSet[0]->pos.getY();
    }
  }
  return rightLine;
}

/**
 * 
 */
XYRectangle2d
Formation05::getCurrentFieldPlayerRectangle(int team)
{
  return
    XYRectangle2d(
                   Vector( this->getCurrentRearmostPlayerLine(team),
                           this->getCurrentRightmostPlayerLine(team) ),
                   Vector( this->getCurrentForemostPlayerLine(team),
                           this->getCurrentLeftmostPlayerLine(team) )
                 );
}

/**
 * This is a helping method for method checkDirectOpponentAssignments().
 * It searches for the nearest opponent to a specific position.
 */
PPlayer
Formation05::findNearestOpponentTo( Vector consideredPos,
                                    bool * assignableOpps)
{
  PPlayer returnValue = NULL;
  float minDist = 1000.0;
  for (int i=0; i<NUM_PLAYERS; i++)
    if (i < WSinfo::alive_opponents.num)
      if (assignableOpps[ WSinfo::alive_opponents[i]->number ])
      {
        LOG_POL(3,<<"We have "<<WSinfo::alive_opponents.num<<" opponents alive"<<flush);
        LOG_POL(3,<<"We have "<<WSinfo::alive_teammates.num<<" teammates alive"<<flush);
        if (WSinfo::alive_opponents[i]->number == 0)
        {
          LOG_POL(2,<<"I cannot handle an opponent with number #0 (feel range player).");
          continue;
        }
        Vector oPos = WSinfo::alive_opponents[i]->pos;
        if ( oPos.norm() < 10.0 && WSinfo::ws->time<10) //player near anstosskreis
        {  //zoom position to centre
           LOG_POL(4,<<"player "<<WSinfo::alive_opponents[i]->number<<" is near abstosskreis!"<<flush);
           oPos.setX( -1.0+(0.1*oPos.getX()) );
        } 
        //////////////////////////////////////////////////////////////////////
        if (0 && WSinfo::ws->time < 2)
        {
          PlayerSet opps = WSinfo::valid_opponents;
          opps.keep_and_sort_players_by_x_from_left(opps.num);
          if (opps.num > 0)
          {
            double foremostX = opps[0]->pos.getX();
            if ( WSinfo::ws->play_mode == PM_his_KickOff )
              if (opps.num > 1) //ignore his kickoff player
                foremostX = opps[1]->pos.getX();
            int playersNearlyOnOneLine = 1;
            for (int a=0; a<opps.num; a++)
            {
              if ( opps[a]->pos.getX() < foremostX + 7.0 )
                playersNearlyOnOneLine = a+1;
            }
            if (playersNearlyOnOneLine > 5) //special handling needed
            {
              for (int a=0; a<opps.num; a++)
              {
                if (    opps[a]->pos.getX() < foremostX + 7.0
                     && opps[a]->number == WSinfo::alive_opponents[i]->number )
                {
                  int consideredPlayerNumber = WSinfo::alive_opponents[i]->number;
                  if (   WSinfo::ws->his_goalie_number > 0
                      && WSinfo::ws->his_goalie_number < 12 )
                  {
                    if ( WSinfo::ws->his_goalie_number > consideredPlayerNumber )
                      consideredPlayerNumber += 1;
                    else
                    if ( WSinfo::ws->his_goalie_number < consideredPlayerNumber )
                      consideredPlayerNumber += 0; //no change
                    else // ==
                      consideredPlayerNumber = 1; //goalie
                  }   
                                          
                  switch (consideredPlayerNumber)
                  {
                    case 1:  oPos.addToX(14.0); break;
                    case 2:  oPos.addToX(14.0); break;
                    case 3:  oPos.addToX(14.0); break;
                    case 4:  oPos.addToX(14.0); break;
                    case 5:  oPos.addToX(14.0); break;
                    case 6:  oPos.addToX( 7.0); break;
                    case 7:  oPos.addToX( 7.0); break;
                    case 8:  oPos.addToX( 7.0); break;
                    case 9:  oPos.addToX( 0.0); break;
                    case 10: oPos.addToX( 0.0); break;
                    case 11: oPos.addToX( 0.0); break;
                    default: break;
                  }
                }
              }
            }
          }
        }
        //////////////////////////////////////////////////////////////////////
        float delta 
          = sqrt( (consideredPos.getX()-oPos.getX())*(consideredPos.getX()-oPos.getX())
                 +1.5*(consideredPos.getY()-oPos.getY())*(consideredPos.getY()-oPos.getY()) );
        if (delta<minDist)
        {
          minDist = delta;
          returnValue = WSinfo::alive_opponents[i];
        }
      }
  LOG_POL(3,<<"I JUST COMPUTED: best opp for me ("<<WSinfo::me->number<<") ["<<consideredPos.getX()<<","<<consideredPos.getY()<<"] is #"<<((returnValue)?(returnValue->number):0)<<" because dist="<<minDist<<flush);
  return returnValue;
}

//============================================================================
// checkDirectOpponentAssignments()
//============================================================================
/**
 * This method checks if direct opponents have been assigned correctly.
 * If not, a simple assignment (fallback solution) is provided here.
 * 
 * Returns true only, if everything with coach assignments is ok.
 */
bool
Formation05::checkDirectOpponentAssignments()
{
  LOG_POL(3,<<"Formation05: I am checking my direct opponent assignments."<<std::flush);
  int sum = 0, sumPrev = 0;
  for (int i=0; i<WSinfo::alive_teammates.num; i++)
    if (WSinfo::alive_teammates[i]->direct_opponent_number < 1)
      sum --;
    else
      sum ++;
  if (sum>0) 
  {
    LOG_POL(3,<<"Formation05: Direct opponent assignments (obtained from coach) are ok.");
    return true;
  }
  LOG_POL(3,<<"Formation05: WARNING: Coach has not assigned direct opponents!"<<flush);
  //less than half of the players of my team have an assigned
  //direct opponent ==> we need a fallback solution
  sum = 0;
  for (int i=0; i<WSinfo::alive_teammates.num; i++)
  {
    if (ivDirectOpponentAssignments[ WSinfo::alive_teammates[i]->number ].
          directOpponent < 1)
      sum --; else sum ++;
    if (ivDirectOpponentAssignments[ WSinfo::alive_teammates[i]->number ].
          previousDirectOpponent < 1)
      sumPrev --; else sumPrev ++;
  }
  if (sum>=7) 
  {
    LOG_POL(3,<<"Formation05: WARNING: But we have our own assignments (sum="<<sum<<")!"<<flush);
    return true;
  }
  if (sumPrev > 7)
  {
    for (int i=0; i<NUM_PLAYERS; i++)
      ivDirectOpponentAssignments[i+1].directOpponent
        = ivDirectOpponentAssignments[i+1].previousDirectOpponent;
    LOG_POL(3,<<"Formation05: WARNING: We have no assigned direct opponents, but we had them a cycle ago; we use the old assignments as fallback solution (sumprev="<<sumPrev<<")!"<<flush);
    return true;
  }
  LOG_POL(3,<<"Formation05: WARNING: We have no assigned direct opponents, compute them now!"<<flush);
  //we must create our fallback solution now!
  bool assignableOpponents[NUM_PLAYERS+1];
  for (int i=0; i<NUM_PLAYERS+1; i++) 
  {
    assignableOpponents[i] = true;
    if (WSinfo::ws->his_goalie_number != 0 && WSinfo::ws->his_goalie_number == i)
      assignableOpponents[i] = false;
  }
  assignableOpponents[0] = false; //feel range player: not considered!!!
  for (int i=0; i<WSinfo::alive_teammates.num; i++)
  {
    PPlayer consideredPlayer = WSinfo::alive_teammates[i];
    int consideredPlayerNr = consideredPlayer->number;
    //Tormann ignorieren
    if (consideredPlayer->number == WSinfo::ws->my_goalie_number) 
    {
      ivDirectOpponentAssignments[consideredPlayerNr].directOpponent = -1;
      continue;
    }
    //Unsere 4 ist "Ausputzer".
    if (consideredPlayer->number == 4) 
    {
      ivDirectOpponentAssignments[consideredPlayerNr].directOpponent = -1;
      continue;
    }
    Vector consideredPos = consideredPlayer->pos;
    if (WSinfo::ws->time > 2) consideredPos = getHomePosition(consideredPlayerNr,true);
    PPlayer nearestOpponent 
        = this->findNearestOpponentTo( consideredPos,
                                       assignableOpponents );
    if ( nearestOpponent )
    {
      ivDirectOpponentAssignments[consideredPlayerNr].directOpponent
        = nearestOpponent->number;
      assignableOpponents[nearestOpponent->number] = false;
    }
  }
  return false;
}

/**
 * This methods set/computes a direct opponent for each player.
 * Actually, this assignment ought to be realized by our online coach.
 * However, we need a fallback solution in case the assignments from
 * the coach do not get through. 
 */
void
Formation05::computeDirectOpponents()
{
  ivCoachAssignmentUpdateInformation.update();
  //determine if new assignments are available
  bool aTeammateHasCommunicatedThatHeAcceptedNewCoachAssignments = false,
       thereAreAssignmentsChanges = false;
  for (int i=0; i<NUM_PLAYERS; i++)
  {
    if (WSinfo::alive_teammates.get_player_by_number( i+1 ))
    {
      if (    ivDirectOpponentAssignments[i+1].directOpponent
           != WSinfo::alive_teammates.get_player_by_number( i+1 )
                                      ->direct_opponent_number )
          thereAreAssignmentsChanges = true; 
    }    
  }  
  if (    thereAreAssignmentsChanges
       && WSinfo::alive_teammates.num == NUM_PLAYERS )
  {
    int diff =   WSinfo::ws->current_direct_opponent_assignment
               - ( ( WSinfo::ws->number_of_direct_opponent_updates_from_coach
                     - 1 ) % 4);
    if (    (    diff == 0
              && WSinfo::ws->current_direct_opponent_assignment != -1 )
         || (    WSinfo::ws->current_direct_opponent_assignment == 0 //init value
              && WSinfo::ws->number_of_direct_opponent_updates_from_coach == 1 )
       ) 
    {
      aTeammateHasCommunicatedThatHeAcceptedNewCoachAssignments = true;
      LOG_POL(0,"Formation05: A teammate has accepted the new assignment, I will"
        <<" accept it as well (currentDOA="
        <<WSinfo::ws->current_direct_opponent_assignment<<",nrOfAss="
        <<WSinfo::ws->number_of_direct_opponent_updates_from_coach<<").");
    }
  }
  
  //compute the actual direct opponents
  bool standardAcception = false,
       teammateCommunicationInvokedAcception = false;
  for (int i=0; i<NUM_PLAYERS; i++)
  {
    ivDirectOpponentAssignments[i+1].previousDirectOpponent
      = ivDirectOpponentAssignments[i+1].directOpponent;
    if (WSinfo::alive_teammates.get_player_by_number( i+1 ))
    {
      if (    ivDirectOpponentAssignments[i+1].directOpponent
           != WSinfo::alive_teammates.get_player_by_number( i+1 )
                                                    ->direct_opponent_number )
      {
        LOG_POL(0,"WARNING: New direct opponent assignments from coach are available ("<<i+1<<"):"
                  << " oldDO: " << WSinfo::alive_teammates.get_player_by_number( i+1 )->direct_opponent_number
                  << " newDO: " << ivDirectOpponentAssignments[i+1].directOpponent );
        if (    WSinfo::ws->play_mode != PM_PlayOn
             || (   (   (   WSinfo::ball->pos.getX() > 30 //TG16:10->30
                         || WSinfo::ball->pos.getX() - WSinfo::my_team_pos_of_offside_line() > 40) //TG16:20->40
                     && WSmemory::team_in_attack()==MY_TEAM)
                 || ivDirectOpponentAssignments[i+1].directOpponent == -1 
                 || relevantTeammateHasAcceptedTheNewAssignment() ) )
        {
          ivDirectOpponentAssignments[i+1].directOpponent
            = WSinfo::alive_teammates.get_player_by_number( i+1 )
                                                ->direct_opponent_number;
          if ( ivDirectOpponentAssignments[i+1].directOpponent == -1 )
          {LOG_POL(0,"No standard acception, just -1 overriding.");}
          else
          {
            LOG_POL(0,"Setting standardAcception to true ("<<i+1<<").");
            standardAcception = true;
          }
        }
        else
        if ( aTeammateHasCommunicatedThatHeAcceptedNewCoachAssignments )
        {
          ivDirectOpponentAssignments[i+1].directOpponent
            = WSinfo::alive_teammates.get_player_by_number( i+1 )
                                                ->direct_opponent_number;
          teammateCommunicationInvokedAcception = true;
        }
      }
      
      if (WSinfo::alive_opponents.get_player_by_number( ivDirectOpponentAssignments[i+1].directOpponent ))
      {
        LOG_POL(3, 
         << _2D  
         << VL2D( WSinfo::alive_teammates.get_player_by_number( i+1 )->pos,
          WSinfo::alive_opponents.get_player_by_number( ivDirectOpponentAssignments[i+1].directOpponent )->pos,
          "44ff44")<<flush);
      }
    }
  }

  //The coach has probably not announced its assignments, yet. So, we 
  //now check this and eventually create a fallback assignment.
  bool doaCheckOk = this->checkDirectOpponentAssignments();

  if ( standardAcception && doaCheckOk)
  {
    cvMyCurrentDirectOpponentAssignment 
      = ( WSinfo::ws->number_of_direct_opponent_updates_from_coach - 1 ) % 4;
    Blackboard::set_direct_opponent_assignment_accepted
                (cvMyCurrentDirectOpponentAssignment);
    LOG_POL(0,"Formation05: I accepted the new assigment the normal way,"
      <<" cvMyCurrentDirectOpponentAssignment="<<cvMyCurrentDirectOpponentAssignment);
  }
  else
  if ( teammateCommunicationInvokedAcception && doaCheckOk)
  {
    cvMyCurrentDirectOpponentAssignment 
      = WSinfo::ws->current_direct_opponent_assignment;
    Blackboard::set_direct_opponent_assignment_accepted
                (cvMyCurrentDirectOpponentAssignment);
    LOG_POL(0,"Formation05: A teammate has accepted and I heard it, "
      <<" accepted it, too, and communicate that, "
      <<"cvMyCurrentDirectOpponentAssignment="<<cvMyCurrentDirectOpponentAssignment);
  }


/*  WSpset myTeamFromLeftToRight = WSinfo::valid_teammates;
  myTeamFromLeftToRight.keep_and_sort_players_by_x_from_left(NUM_PLAYERS);
  PPlayer myGoalie = WSinfo::alive_teammates.
                     get_player_by_number(WSinfo::ws->my_goalie_number);
  PPlayer hisGoalie = WSinfo::alive_opponents.
                     get_player_by_number(WSinfo::ws->his_goalie_number);
  myTeamFromLeftToRight.remove(myGoalie);
  WSpset alreadyAssignedOpponents;
  for (int i=0; i<myTeamFromLeftToRight.num; i++)
  {
    PPlayer consideredTeammate = myTeamFromLeftToRight[i];
    WSpset consideredOpponents = WSinfo::alive_opponents;
    consideredOpponents.remove( alreadyAssignedOpponents );
    consideredOpponents.remove( hisGoalie );
    if (consideredOpponents.num > 0)
    {
      //standard: naechster spieler
      PPlayer bestMatchingOpponent
        = consideredOpponents.
            closest_player_to_point( consideredTeammate->pos );
      //sonderfall: ein gegenspieler ist durchgekommen ...
      WSpset copyOfConsideredOpponents(consideredOpponents);
      copyOfConsideredOpponents.keep_and_sort_players_by_x_from_left(NUM_PLAYERS);
      for (int j=0; j<copyOfConsideredOpponents.num; j++)
      {
        if ( (copyOfConsideredOpponents[j]->pos.x < consideredTeammate->pos.x)
             &&
             (WSinfo::alive_teammates.closest_player_to_point(
              copyOfConsideredOpponents[j]->pos)->number == consideredTeammate->number))
        {
          bestMatchingOpponent = copyOfConsideredOpponents[j];
          break;
        }
      }
      if (bestMatchingOpponent != NULL)
      {
        ivDirectOpponentAssignments[consideredTeammate->number].directOpponent 
          = bestMatchingOpponent->number;
        alreadyAssignedOpponents.append(bestMatchingOpponent);
        //cout<<"Formation05::computeDirectOpponents(): "<<i<<"->"<<bestMatchingOpponent->number <<endl;
LOG_POL(0, 
  << _2D  
  << L2D(bestMatchingOpponent->pos.x,bestMatchingOpponent->pos.y,
         consideredTeammate->pos.x, consideredTeammate->pos.y,"44ff44"));
      }
    }
  }*/
  
  
/*  
  XYRectangle2d ourPlayerRect = this->getCurrentFieldPlayerRectangle(MY_TEAM);
int team=MY_TEAM;
LOG_POL(0,<<_2D
        <<L2D(this->getCurrentRearmostPlayerLine(team),this->getCurrentRightmostPlayerLine(team),this->getCurrentForemostPlayerLine(team),this->getCurrentLeftmostPlayerLine(team),"4444ff"));
  XYRectangle2d hisPlayerRect = this->getCurrentFieldPlayerRectangle(HIS_TEAM);
team=HIS_TEAM;
LOG_POL(0,<<_2D
        <<L2D(this->getCurrentRearmostPlayerLine(team),this->getCurrentRightmostPlayerLine(team),this->getCurrentForemostPlayerLine(team),this->getCurrentLeftmostPlayerLine(team),"ff4444"));
  Vector delta = hisPlayerRect.center - ourPlayerRect.center;
  WSpset alreadyAssignedOpponents;

  DistanceCollection distances[3*(NUM_PLAYERS-1)];
  for (int i=2; i<=NUM_PLAYERS; i++)
  {
    PPlayer consideredPlayer
      = WSinfo::valid_teammates.get_player_by_number(i);
    if (consideredPlayer != NULL)
    {
      Vector myPlayerBasePoint = consideredPlayer->pos + delta;
      WSpset consideredOpponentPlayers = WSinfo::valid_opponents;
      consideredOpponentPlayers.
          keep_and_sort_closest_players_to_point(3, myPlayerBasePoint);
      for (int j=0; j<3; j++)
      {
        distances[(i-2)*3+j].myPlayer  = consideredPlayer;
        distances[(i-2)*3+j].hisPlayer = consideredOpponentPlayers[j];
        distances[(i-2)*3+j].distance  = 
          (myPlayerBasePoint - consideredOpponentPlayers[j]->pos).norm();
      }
    }
  }
  for (int i=0; i<(3*(NUM_PLAYERS-1))-1; i++)
    for (int j=i+1; j<(3*(NUM_PLAYERS-1)); j++)
       if (  (distances[i].distance > distances[j].distance
              && !(distances[j].myPlayer==NULL || distances[j].hisPlayer==NULL)) 
           ||
             (distances[i].myPlayer==NULL || distances[i].hisPlayer==NULL)
          )
       { //swap
         DistanceCollection d = distances[j];
         distances[j] = distances[i];
         distances[i] = d;       
       }
  int maxIndex = 0;
  for (int i=0; i<(3*(NUM_PLAYERS-1)); i++)
    if (distances[i].myPlayer!=NULL && distances[i].hisPlayer!=NULL)   
      maxIndex = i;
  WSpset myPlayers  = WSinfo::valid_teammates;
  WSpset hisPlayers = WSinfo::valid_opponents;
  for (int i=0; i<maxIndex; i++)
    if (
         myPlayers.get_player_by_number( distances[i].myPlayer->number ) != NULL
         &&
         hisPlayers.get_player_by_number(distances[i].hisPlayer->number) != NULL
       )
    {
      ivHomePositions[distances[i].myPlayer->number].directOpponent 
        = distances[i].hisPlayer->number;
      //cout<<"Formation05::computeDirectOpponents(): "<<distances[i].myPlayer->number<<"->"<<distances[i].hisPlayer->number <<endl;
LOG_POL(0, 
  << _2D  
  << L2D(distances[i].myPlayer->pos.x,distances[i].myPlayer->pos.y,
         distances[i].hisPlayer->pos.x, distances[i].hisPlayer->pos.y,"44ff44"));
      //found a direct opponent for distances[i].myPlayer
      myPlayers.remove( distances[i].myPlayer );
      hisPlayers.remove( distances[i].hisPlayer );
    }
*/
  //loop starts at 2, as there is no need to assign an opponent to our goalie
/*  for (int i=2; i<=NUM_PLAYERS; i++)
  {
    PPlayer consideredPlayer
      = WSinfo::valid_teammates.get_player_by_number(i);
    if (consideredPlayer != NULL)
    {
      Vector myPlayerBasePoint 
        = consideredPlayer->pos + delta;
      WSpset consideredOpponentPlayers = WSinfo::valid_opponents;
      consideredOpponentPlayers.remove( alreadyAssignedOpponents );
      if (consideredOpponentPlayers.num > 0)
      {
        PPlayer bestMatchingOpponent
          = consideredOpponentPlayers.closest_player_to_point( myPlayerBasePoint );
        if (bestMatchingOpponent != NULL)
        {
          alreadyAssignedOpponents.append(bestMatchingOpponent);
cout<<"Formation05::computeDirectOpponents(): "<<i<<"->"<<bestMatchingOpponent->number <<endl;
LOG_POL(0, 
  << _2D  
  << L2D(consideredPlayer->pos.x,consideredPlayer->pos.y,
         bestMatchingOpponent->pos.x, bestMatchingOpponent->pos.y,"44ff44"));
        }
      }
    }
  }*/
  
  
}

#include "positioning.h"

bool 
Formation05::relevantTeammateHasAcceptedTheNewAssignment()
{
    
  //note that this method should be called only when new direct
  //opponent assignments from the coach are available!
  
  int myOLDdirectOpponentNumber 
    = ivDirectOpponentAssignments[WSinfo::me->number].directOpponent;
  int myNEWdirectOpponentNumber = WSinfo::me->direct_opponent_number;
  
  //i don't care, if nothing has changed for me
  if ( myOLDdirectOpponentNumber == myNEWdirectOpponentNumber )
    return false;
    
  //find out who is responsible for my OLD direct opponent 
  //w.r.t. the NEW assignment
  int oldDirectOpponentOfTeammateNOWresponsibleForMyOLDdirectOpponent = -1,
      newDirectOpponentOfTeammateNOWresponsibleForMyOLDdirectOpponent = -1;
  PPlayer teammatePPlayerNOWresponsibleForMyOLDdirectOpponent = NULL;
  for (int i=0; i<NUM_PLAYERS; i++)
  {
    PPlayer teammate = WSinfo::alive_teammates.get_player_by_number( i+1 );
    if (    teammate
         &&    myOLDdirectOpponentNumber 
            == teammate->direct_opponent_number )
    {
      teammatePPlayerNOWresponsibleForMyOLDdirectOpponent
        = teammate;
      oldDirectOpponentOfTeammateNOWresponsibleForMyOLDdirectOpponent 
        = ivDirectOpponentAssignments[i+1].directOpponent;
      newDirectOpponentOfTeammateNOWresponsibleForMyOLDdirectOpponent
        = teammate->direct_opponent_number;
    }
  }
  
  //i found the relevant teammate
  //=> now check, if he has already moved to his NEW direct opponent
  if ( teammatePPlayerNOWresponsibleForMyOLDdirectOpponent != NULL )
  {
    PPlayer oldDirectOppOfTeammate = NULL,
            newDirectOppOfTeammate = NULL;
    oldDirectOppOfTeammate 
      = WSinfo::alive_opponents.get_player_by_number
                ( oldDirectOpponentOfTeammateNOWresponsibleForMyOLDdirectOpponent );
    newDirectOppOfTeammate 
      = WSinfo::alive_opponents.get_player_by_number
                ( newDirectOpponentOfTeammateNOWresponsibleForMyOLDdirectOpponent );

    if (oldDirectOppOfTeammate && newDirectOppOfTeammate)
    {
      Vector coverPosOfOldOpp 
        = OpponentAwarePositioning::getCoverPosition( oldDirectOppOfTeammate );
      Vector coverPosOfNewOpp
        = OpponentAwarePositioning::getCoverPosition( newDirectOppOfTeammate );
      double dist2OldDirectOpponentCoverPos
        = teammatePPlayerNOWresponsibleForMyOLDdirectOpponent
          ->pos.distance( coverPosOfOldOpp );
      double dist2NewDirectOpponentCoverPos
        = teammatePPlayerNOWresponsibleForMyOLDdirectOpponent
          ->pos.distance( coverPosOfNewOpp );
      if (    teammatePPlayerNOWresponsibleForMyOLDdirectOpponent->age < 2
           && oldDirectOppOfTeammate->age < 2
           && newDirectOppOfTeammate->age < 2  )
      {
        if ( dist2NewDirectOpponentCoverPos < 0.8*dist2OldDirectOpponentCoverPos )
        {
          LOG_POL(0,"DOA: I ACCEPT THE NEW ASSIGNMENT!");
          LOG_POL(0,"     myNumber="<<WSinfo::me->number
                  <<" myOLDdirOppNum="<<myOLDdirectOpponentNumber
                  <<" myNEWdirOppNum="<<myNEWdirectOpponentNumber
                  <<" dist2OldDirOppCoverPos="<<dist2OldDirectOpponentCoverPos
                  <<" dist2NewDirOppCoverPos="<<dist2NewDirectOpponentCoverPos<<std::flush);

          return true;
        }
      }
    }
  }
  
  LOG_POL(0,"DOA: I DISREGARD THE NEW ASSIGNMENT!"<<std::flush);
  LOG_POL(0,"     myNumber="<<WSinfo::me->number
          <<" myOLDdirOppNum="<<myOLDdirectOpponentNumber
          <<" myNEWdirOppNum="<<myNEWdirectOpponentNumber
          <<" oldDirOppOfTmmNOWresp4MyOLDdirOpp="<<oldDirectOpponentOfTeammateNOWresponsibleForMyOLDdirectOpponent
          <<" newDirOppOfTmmNOWresp4MyOLDdirOpp="<<newDirectOpponentOfTeammateNOWresponsibleForMyOLDdirectOpponent<<std::flush);
  return false;
}

