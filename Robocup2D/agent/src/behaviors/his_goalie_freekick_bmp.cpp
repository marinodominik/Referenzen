#include "his_goalie_freekick_bmp.h"

#include "tools.h"
//#include "options.h"
//#include "ws_info.h"
//#include "mdp_info.h"
//#include "blackboard.h"
#include "log_macros.h"
//#include "valueparser.h"
#include "../policy/positioning.h"
//#include "../policy/policy_tools.h"
//#include "../policy/planning.h"
//#include <math.h>
#include "intention.h"


bool HisGoalieFreeKick::initialized=false;

HisGoalieFreeKick::HisGoalieFreeKick() 
{
  ivpMyDirectOpponent = NULL;
  ivpGo2PosBehavior   = new NeuroGo2Pos();
  ivpBasicCmdBehavior = new BasicCmd();
  ivpSearchBallBehavior = new SearchBall();
}

bool 
HisGoalieFreeKick::init(char const * conf_file, int argc, char const* const* argv) 
{
  if (initialized) return true;
  initialized = true;
  bool res = NeuroGo2Pos::init(conf_file,argc,argv) &&
             BasicCmd::init(conf_file,argc,argv) &&
             SearchBall::init(conf_file,argc,argv);
  //std::cout<<"HisGoalieFreeKick: INIT -> "<<res;
  return res;    
}

bool 
HisGoalieFreeKick::get_cmd(Cmd & cmd)
{
  bool res;
  if (ClientOptions::consider_goalie) 
  {
    res = get_goalie_cmd(cmd);
  } 
  else 
  {
    res = get_player_cmd(cmd);
    double desiredPower, allowedPower = 100.0;
    //consider stamina savings ...
    if ( cmd.cmd_body.get_type() == Cmd_Body::TYPE_DASH )
    {
      cmd.cmd_body.get_dash( desiredPower );
      //standard considerations
      if ( WSinfo::me->stamina < 0.45*ServerOptions::stamina_max )  //TG09: alt: 2500       
        allowedPower = 75.0 - WSinfo::me->stamina_inc_max;
      else
      if ( WSinfo::me->stamina < 0.375*ServerOptions::stamina_max ) //TG09: alt: 1750
        allowedPower = (75.0 - WSinfo::me->stamina_inc_max) * 0.5;
      //direct opponent consideration for defenders
      double dirOppHeadStart = 0.0;
      if ( ivpMyDirectOpponent )
        dirOppHeadStart =   WSinfo::me->pos.distance(MY_GOAL_CENTER)
                          - ivpMyDirectOpponent->pos.distance(MY_GOAL_CENTER);
      //hin-und-herlauf-considerations
      if (   ivCurrentTargetPosition.distance(WSinfo::me->pos) < 4.0//TG08:6->4
          && WSinfo::ball->vel.norm() < 1.0
          && WSinfo::me->stamina < 0.9*ServerOptions::stamina_max //TG09: alt: 3750
          && ! ( dirOppHeadStart > 2.0 )
         )
        allowedPower = WSinfo::me->stamina_inc_max * 0.45;  

      if ( desiredPower > allowedPower )
      {
        cmd.cmd_body.unset_lock();
        cmd.cmd_body.unset_cmd();
        cmd.cmd_body.set_dash( allowedPower );
      }
    }
  }
  return res;
}

bool 
HisGoalieFreeKick::get_goalie_cmd(Cmd &cmd) 
{
  return get_player_cmd_fallback(cmd);
}


/*bool 
HisGoalieFreeKick::get_player_cmd(Cmd &cmd) 
{
  PPlayer myDirectOpponent
      = OpponentAwarePositioning::getDirectOpponent(WSinfo::me->number);
    
  //INFO
  LOG_POL(0,<<_2D<<C2D(WSinfo::me->pos.x,WSinfo::me->pos.y,0.3,"66ff66"));
  LOG_POL(0,<<_2D<<C2D(WSinfo::ball->pos.x,WSinfo::ball->pos.y,0.3,"ff8c00"));
  LOG_POL(0,<<_2D<<C2D(WSinfo::ball->pos.x,WSinfo::ball->pos.y,0.2,"ff8c00"));
  LOG_POL(0,<<"HisGoalieFreeKick: ball->age  : "<<WSinfo::ball->age);
  if (myDirectOpponent)
  {
    LOG_POL(0,<<"HisGoalieFreeKick: dirOp->age : "<<myDirectOpponent->age);
    LOG_POL(0,<<_2D<<C2D(myDirectOpponent->pos.x,myDirectOpponent->pos.y,0.3,"ff6666"));
    LOG_POL(0,<<_2D<<L2D(myDirectOpponent->pos.x,myDirectOpponent->pos.y,
                         WSinfo::me->pos.x,WSinfo::me->pos.y,"66ff66"));
  }
    
  if ( myDirectOpponent && myDirectOpponent->age > 4 
                        && myDirectOpponent->age < 5)
  {
    LOG_POL(0,<<"HisGoalieFreeKick: myDirectOpponent->age="<<myDirectOpponent->age<<" ==> TURN TO DIROPP");
    return turn_to_direct_opponent(cmd, myDirectOpponent);
  }
  if (myDirectOpponent && myDirectOpponent->age < 4)
  {
    LOG_POL(0,"HisGoalieFreeKick: Cover my direct opponent");
    bool returnValue = cover_direct_opponent(cmd, myDirectOpponent);
    if (   WSinfo::me->stamina < 3000 
        && cmd.cmd_main.get_type() == Cmd_Main::TYPE_DASH
        && WSinfo::me->pos.distance(myDirectOpponent->pos) < 10.0)
    {
      double dashPower;
      cmd.cmd_main.get_dash( dashPower );
      dashPower *= 0.5;
      cmd.cmd_main.unset_lock();
      cmd.cmd_main.set_dash( dashPower );
    }
    return returnValue;
  } 
  if (myDirectOpponent && myDirectOpponent->age < 4)
  {
    Tools::set_neck_request(NECK_REQ_LOOKINDIRECTION, 
                            (myDirectOpponent->pos-WSinfo::me->pos).ARG());
  }
  if (WSinfo::ball->age > 5 && WSinfo::ball->age < 20)
  {
    LOG_POL(0,"HisGoalieFreeKick: ball->age="<<WSinfo::ball->age<<" ==> SEARCH BALL");
    return search_ball(cmd);
  }
  if (!myDirectOpponent)
    return get_player_cmd_fallback(cmd);
  return scan_field(cmd);
}*/

bool
HisGoalieFreeKick::cover_direct_opponent(Cmd & cmd, PPlayer dirOpp)
{
  Vector targetPosition 
    = OpponentAwarePositioning::getAttackOrientedPosition(0.0);
    
  LOG_POL(0,"HisGoalieFreeKick: INFO: WSinfo::me->pos = "<<WSinfo::me->pos
    <<" target x/y = "<<targetPosition);
   
  while ( RIGHT_PENALTY_AREA.inside( targetPosition ) )
  {
    if (targetPosition.getX() >= FIELD_BORDER_X - PENALTY_AREA_LENGTH - 0.3)
      targetPosition.subFromX( 0.5 );
    if (    targetPosition.getY() >= 0.0
         && targetPosition.getY() < PENALTY_AREA_WIDTH / 2.0 + 0.3)
      targetPosition.addToY( 0.15 );
    if (    targetPosition.getY() < 0.0
         && targetPosition.getY() > -PENALTY_AREA_WIDTH / 2.0 - 0.3)
      targetPosition.subFromY( 0.15 );
    LOG_POL(0,"HisGoalieFreeKick: target x/y position has been reduced to "
      <<targetPosition.getX()<<" / "<<targetPosition.getY());
  }
  while (    WSinfo::me->pos.getX() > FIELD_BORDER_X - PENALTY_AREA_LENGTH - 0.3
          && WSinfo::me->pos.getY() >= PENALTY_AREA_WIDTH / 2.0 - 1.0
          && targetPosition.getY() < PENALTY_AREA_WIDTH / 2.0 + 0.3 )
  {
    LOG_POL(0,"HisGoalieFreeKick: Change target y position from "
      <<targetPosition.getY()<<" to "<<targetPosition.getY()+0.5);
    targetPosition.addToY( 0.5 );
  }
  while (    WSinfo::me->pos.getX() > FIELD_BORDER_X - PENALTY_AREA_LENGTH - 0.3
          && WSinfo::me->pos.getY() <= - PENALTY_AREA_WIDTH / 2.0 + 1.0
          && targetPosition.getY() > - PENALTY_AREA_WIDTH / 2.0 - 0.3 )
  {
    LOG_POL(0,"HisGoalieFreeKick: Change target y position from "
      <<targetPosition.getY()<<" to "<<targetPosition.getY()-0.5);
    targetPosition.subFromY( 0.5 );
  }
  while (    targetPosition.getX() > FIELD_BORDER_X - PENALTY_AREA_LENGTH - 0.3
          && targetPosition.getY() >= PENALTY_AREA_WIDTH / 2.0 - 1.0
          && WSinfo::me->pos.getY() < PENALTY_AREA_WIDTH / 2.0 + 0.3 )
  {
    LOG_POL(0,"HisGoalieFreeKick: Change target x position from "
      <<targetPosition.getX()<<" to "<<targetPosition.getX()-0.5);
    targetPosition.subFromX( 0.5 );
  }
  while (    targetPosition.getX() > FIELD_BORDER_X - PENALTY_AREA_LENGTH - 0.3
          && targetPosition.getY() <= - PENALTY_AREA_WIDTH / 2.0 + 1.0
          && WSinfo::me->pos.getY() > - PENALTY_AREA_WIDTH / 2.0 - 0.3 )
  {
    LOG_POL(0,"HisGoalieFreeKick: Change target x position from "
      <<targetPosition.getX()<<" to "<<targetPosition.getX()-0.5);
    targetPosition.subFromX( 0.5 );
  }
    
  while (    WSinfo::ball->pos.distance( targetPosition ) < 9.5
          && (    WSinfo::ws->play_mode == PM_his_GoalieFreeKick
               || (    WSinfo::ws->play_mode == PM_his_GoalKick
                    && WSinfo::ball->pos.getX() > FIELD_BORDER_X - 6.0 )
             )
        )
  {
    LOG_POL(0,"HisGoalieFreeKick: Reduce target x position from "
      <<targetPosition.getX()<<" to "<<targetPosition.getX()-0.5
      <<" ballvel="<<WSinfo::ball->vel.norm()<<" (1), pm="<<WSinfo::ws->play_mode);
    targetPosition.subFromX( 0.5 );
  }
  
  LOG_POL(0,<<_2D<<VL2D(WSinfo::me->pos,
    targetPosition,"ffff55"));
  
  if (WSinfo::me->pos.distance(targetPosition) < 1.0)
  {
    LOG_POL(0,"HisGoalieFreeKick: I am near enough to direct opponent ==> SCAN FIELD"<<std::flush);
    return scan_field(cmd);
  }
  ivpGo2PosBehavior->set_target(targetPosition);
  ivCurrentTargetPosition = targetPosition;
  return ivpGo2PosBehavior->get_cmd(cmd);
}

bool
HisGoalieFreeKick::turn_to_direct_opponent(Cmd & cmd, PPlayer dirOpp)
{
  ANGLE myAngleToDirOpp 
    = (dirOpp->pos - WSinfo::me->pos).ARG() - WSinfo::me->ang;
  Tools::set_neck_request(NECK_REQ_SCANFORBALL); //lock neck
  ivpBasicCmdBehavior->set_turn_inertia( myAngleToDirOpp.get_value() );
  return ivpBasicCmdBehavior->get_cmd(cmd);
}

bool
HisGoalieFreeKick::search_ball(Cmd & cmd)
{
  if (!ivpSearchBallBehavior->is_searching())
    ivpSearchBallBehavior->start_search();
  return ivpSearchBallBehavior->get_cmd(cmd);
}

bool
HisGoalieFreeKick::get_player_cmd_fallback(Cmd & cmd)
{
  bool res = false;
  LOG_POL(0,<<"HisGoalieFreeKick: Try fallback cmd ...");
  //ivpGo2PosBehavior->set_target(DeltaPositioning::get_position(WSinfo::me->number));
  //res = ivpGo2PosBehavior->get_cmd(cmd);
  //LOG_POL(0,<<"HisGoalieFreeKick: Go to default pos: "<<((res==false)?"NOPE":"YEP"));
  if (!res) //target pos already reached
    res = scan_field(cmd);
  return res;
}

bool 
HisGoalieFreeKick::scan_field(Cmd &cmd) 
{
  //standard scanning of the field
  LOG_POL(0,<<"HisGoalieFreeKick: Scan field!");
  Tools::set_neck_request(NECK_REQ_SCANFORBALL);
  if (WSinfo::ws->view_quality == Cmd_View::VIEW_QUALITY_LOW) 
  {
    cmd.cmd_body.set_turn(PI/18.0);
    return true;
  } 
  Angle turn = .5*(Tools::next_view_angle_width()+Tools::cur_view_angle_width()).get_value();
  ivpBasicCmdBehavior->set_turn_inertia(turn);
  return ivpBasicCmdBehavior->get_cmd(cmd);
}

///////////////////////////////////////////////////////////////////////////////

bool 
HisGoalieFreeKick::get_player_cmd(Cmd &cmd) 
{
  ivpMyDirectOpponent
      = OpponentAwarePositioning::getDirectOpponent(WSinfo::me->number);
    
  //sweeper
  if (WSinfo::me->number == 4 ) //MY_SWEEPER_NUMBER)
  {
    Vector targetPosition(7.0,9.0);//(7.0,8.0);
    targetPosition.setX( WSinfo::my_team_pos_of_offside_line() - 4.0 );
    if (targetPosition.getX() < 3.0) targetPosition.setX( 3.0 );
    
    /*
    WSpset myDefenders;
    if (WSinfo::get_teammate_by_number(2))
      myDefenders.append( WSinfo::get_teammate_by_number(2) );
    if (WSinfo::get_teammate_by_number(3))
      myDefenders.append( WSinfo::get_teammate_by_number(3) );
    if (WSinfo::get_teammate_by_number(5))
      myDefenders.append( WSinfo::get_teammate_by_number(5) );
    myDefenders.keep_and_sort_players_by_y_from_left(myDefenders.num);
    if ( myDefenders.num < 3 )
      targetPosition.y = 5.0;
    else
    {
      double maxHole = myDefenders[0]->pos.y - (-FIELD_BORDER_Y);
      LOG_POL(0,"HisGoalieFreeKick: maxHole = "<<maxHole<<" for left side"
        <<" (my def 0 at "<<myDefenders[0]->pos.y<<")");
      targetPosition.y = (-FIELD_BORDER_Y + myDefenders[0]->pos.y) * 0.5;
      if (   FIELD_BORDER_Y - myDefenders[myDefenders.num-1]->pos.y  
           > maxHole )
      {
        maxHole = FIELD_BORDER_Y - myDefenders[myDefenders.num-1]->pos.y;
        targetPosition.y = (myDefenders[myDefenders.num-1]->pos.y + FIELD_BORDER_Y)*0.5;
        LOG_POL(0,"HisGoalieFreeKick: maxHole = "<<maxHole<<" for right side");
        }
      for (int i=0; i < myDefenders.num - 1; i++)
        if (   myDefenders[ i+1 ]->pos.y - myDefenders[ i ]->pos.y 
             > maxHole*0.5 )
        {
          maxHole = 2.0*(myDefenders[ i+1 ]->pos.y - myDefenders[ i ]->pos.y);
          targetPosition.y = (myDefenders[ i ]->pos.y + myDefenders[ i+1 ]->pos.y ) * 0.5;
          LOG_POL(0,"HisGoalieFreeKick: maxHole = "<<maxHole<<" for i="<<i);
        }
      LOG_POL(0,"HisGoalieFreeKick: maxHole = "<<maxHole);
    }
    LOG_POL(0,"HisGoalieFreeKick: targetPosition = "<<targetPosition);
    */
    
    if ( WSinfo::me->pos.distance(targetPosition) < 2.5)
    {
        ANGLE myAngToBall = (WSinfo::ball->pos - WSinfo::me->pos).ARG();
        ANGLE diffBetweenNeckAndBallAng = WSinfo::me->neck_ang - myAngToBall;
        if ( fabs(diffBetweenNeckAndBallAng.get_value_mPI_pPI()) < PI/8.0)
        {
          LOG_POL(0,"HisGoalieFreeKick: tryToOptimizeCovering: NOPE (i "
            <<"am near enough to the cleverPosition and am oriented towards the ball)"<<std::flush);
          return false;
        }
        LOG_POL(0,"HisGoalieFreeKick: tryToOptimizeCovering: YEP (i will now "
          <<"look into the ball's direction, fixing my neck to my body)"<<std::flush);
        ANGLE turnAngle = (WSinfo::ball->pos - WSinfo::me->pos).ARG() - WSinfo::me->ang;
        Tools::set_neck_request(NECK_REQ_SCANFORBALL);
        ivpBasicCmdBehavior->set_turn_inertia(turnAngle.get_value());
        return ivpBasicCmdBehavior->get_cmd(cmd);
    }
    
    ivpGo2PosBehavior->set_target(targetPosition);
    ivCurrentTargetPosition = targetPosition;
    LOG_POL(0,"HisGoalieFreeKick: Special mode for sweeper!"<<std::flush);
    return ivpGo2PosBehavior->get_cmd(cmd);
  }
    
  //INFO
  LOG_POL(0,<<_2D<<VC2D(WSinfo::me->pos,0.3,"66ff66"));
  LOG_POL(0,<<_2D<<VC2D(WSinfo::ball->pos,0.3,"ff8c00"));
  LOG_POL(0,<<_2D<<VC2D(WSinfo::ball->pos,0.2,"ff8c00"));
  LOG_POL(0,<<"HisGoalieFreeKick: ball->age  : "<<WSinfo::ball->age);
  if (ivpMyDirectOpponent)
  {
    LOG_POL(0,<<"HisGoalieFreeKick: dirOp->age : "<<ivpMyDirectOpponent->age);
    LOG_POL(0,<<_2D<<VC2D(ivpMyDirectOpponent->pos,0.3,"ff6666"));
    LOG_POL(0,<<_2D<<VL2D(ivpMyDirectOpponent->pos,
                         WSinfo::me->pos,"66ff66"));
  }

  if (
          tryToSearchDirectOpponent(cmd)
       || tryToCoverDirectOpponent(cmd)
       || tryToSearchBall(cmd)
       || tryToOptimizeCovering(cmd)
     )
    return true;
  return false;
}

bool
HisGoalieFreeKick::tryToSearchDirectOpponent(Cmd &cmd)
{
  if (ivpMyDirectOpponent == NULL)
  {
    LOG_POL(0,<<"HisGoalieFreeKick: tryToSearchDirectOpponent: NOPE (have no direct opponent)");
    return false;
  }
  if (ivpMyDirectOpponent->age < 5)
  {
    LOG_POL(0,<<"HisGoalieFreeKick: tryToSearchDirectOpponent: NOPE (direct"
      <<" opponent is not outdated (age="<<ivpMyDirectOpponent->age<<"))");
    return false;
  }
  bool returnValue = scan_field(cmd);
  if (returnValue == false)
  {
    LOG_POL(0,<<"HisGoalieFreeKick: tryToSearchDirectOpponent: NOPE (scanning"
      <<" the field is not possible)");
    return false;
  }
  LOG_POL(0,<<"HisGoalieFreeKick: tryToSearchDirectOpponent: YEP (search"
    <<" him by scanning the field)");
  return true;
}

bool 
HisGoalieFreeKick::tryToCoverDirectOpponent(Cmd &cmd)
{
  if (ivpMyDirectOpponent == NULL)
  {
    LOG_POL(0,<<"HisGoalieFreeKick: tryToCoverDirectOpponent: NOPE (have no direct opponent)");
    return false;
  }

  Vector targetPosition 
    = OpponentAwarePositioning::getAttackOrientedPosition(0.0);
  targetPosition.subFromX( 0.2 );
    
  correctTargetPosition( targetPosition );
  
  LOG_POL(0,<<_2D<<VL2D(WSinfo::me->pos,
    targetPosition,"ffff55"));
  LOG_POL(0,<<_2D<<VC2D(WSinfo::ball->pos,9.15,"555500"));
  
  if (WSinfo::me->pos.distance(targetPosition) < 1.0)
  {
    LOG_POL(0,"HisGoalieFreeKick: tryToCoverDirectOpponent: NOPE (i "
      <<"am near enough to direct opponent)"<<std::flush);
    return false;
  }
  
  ANGLE myAngToDO = (ivpMyDirectOpponent->pos - WSinfo::me->pos).ARG();
  if ( ivpMyDirectOpponent->age > 2 && Tools::could_see_in_direction( myAngToDO ) )
    Tools::set_neck_request(NECK_REQ_LOOKINDIRECTION, myAngToDO);
  else
  {
    ANGLE myAngToBall = (WSinfo::ball->pos - WSinfo::me->pos).ARG();
    if ( WSinfo::ball->age > 2 && Tools::could_see_in_direction( myAngToBall ) )
      Tools::set_neck_request(NECK_REQ_LOOKINDIRECTION, myAngToBall);
  }
  ivpGo2PosBehavior->set_target(targetPosition);
  ivCurrentTargetPosition = targetPosition;
  LOG_POL(0,"HisGoalieFreeKick: tryToCoverDirectOpponent: YEP (i move"
    <<"towards direct opponent and also try to look to the ball)"<<std::flush);
  return ivpGo2PosBehavior->get_cmd(cmd);
}

bool 
HisGoalieFreeKick::tryToSearchBall(Cmd &cmd)
{
  if (WSinfo::ball->age < 2)
  {
    LOG_POL(0,"HisGoalieFreeKick: tryToSearchBall: NOPE (ball "
      <<"is not outdated (age="<<WSinfo::ball->age<<"))"<<std::flush);
    return false;
  }
  
  ANGLE myAngToBall = (WSinfo::ball->pos - WSinfo::me->pos).ARG();
  ANGLE diffBetweenNeckAndBallAng = WSinfo::me->neck_ang - myAngToBall;
  if (    fabs(diffBetweenNeckAndBallAng.get_value_mPI_pPI()) < PI/8.0 
       && WSinfo::ball->age > 20)
  {
    bool returnValue = scan_field(cmd);
    if (returnValue == false)
    {
      LOG_POL(0,"HisGoalieFreeKick: tryToSearchBall: NOPE (i have "
        <<"looked into the ball's direction and it is not there any more, field scan is not possible)"<<std::flush);
      return false;
    }
    LOG_POL(0,"HisGoalieFreeKick: tryToSearchBall: YEP (i have "
      <<"looked into the ball's direction and it is not there any more, i scan the field)"<<std::flush);
    return true;
  }
  
  LOG_POL(0,"HisGoalieFreeKick: tryToSearchBall: YEP (i will now "
    <<"look into the ball's direction, fixing my neck to my body)"<<std::flush);
  ANGLE turnAngle = (WSinfo::ball->pos - WSinfo::me->pos).ARG() - WSinfo::me->ang;
  Tools::set_neck_request(NECK_REQ_SCANFORBALL);
  ivpBasicCmdBehavior->set_turn_inertia(turnAngle.get_value());
  return ivpBasicCmdBehavior->get_cmd(cmd);
}

bool
HisGoalieFreeKick::tryToOptimizeCovering(Cmd &cmd)
{
  if (ivpMyDirectOpponent == NULL)
  {
    LOG_POL(0,<<"HisGoalieFreeKick: tryToOptimizeCovering: NOPE (have no direct opponent)");
    return false;
  }
  
  Vector offset = WSinfo::ball->pos - ivpMyDirectOpponent->pos;
  offset.normalize(0.5);
  Vector cleverPosition = ivpMyDirectOpponent->pos - offset;
  if (WSinfo::me->pos.distance(cleverPosition) < 0.5)
  {
    ANGLE myAngToBall = (WSinfo::ball->pos - WSinfo::me->pos).ARG();
    ANGLE diffBetweenNeckAndBallAng = WSinfo::me->neck_ang - myAngToBall;
    if ( fabs(diffBetweenNeckAndBallAng.get_value_mPI_pPI()) < PI/8.0)
    {
      LOG_POL(0,"HisGoalieFreeKick: tryToOptimizeCovering: NOPE (i "
        <<"am near enough to the cleverPosition and am oriented towards the ball)"<<std::flush);
      return false;
    }
    LOG_POL(0,"HisGoalieFreeKick: tryToOptimizeCovering: YEP (i will now "
      <<"look into the ball's direction, fixing my neck to my body)"<<std::flush);
    ANGLE turnAngle = (WSinfo::ball->pos - WSinfo::me->pos).ARG() - WSinfo::me->ang;
    Tools::set_neck_request(NECK_REQ_SCANFORBALL);
    ivpBasicCmdBehavior->set_turn_inertia(turnAngle.get_value());
    return ivpBasicCmdBehavior->get_cmd(cmd);
  }
  correctTargetPosition( cleverPosition );
  LOG_POL(0,<<_2D<<VL2D(WSinfo::me->pos,
    cleverPosition,"ffff55"));
  LOG_POL(0,<<_2D<<VC2D(WSinfo::ball->pos,9.15,"555500"));
  ivpGo2PosBehavior->set_target(cleverPosition);
  ivCurrentTargetPosition = cleverPosition;
  LOG_POL(0,"HisGoalieFreeKick: tryToOptimizeCovering: YEP (i move"
    <<"towards the cleverPosition)"<<std::flush);
  LOG_POL(0,<<_2D<<VC2D(cleverPosition,0.5,"red"));
  return ivpGo2PosBehavior->get_cmd(cmd);
}

bool
HisGoalieFreeKick::tryToScanField(Cmd &cmd)
{
  LOG_POL(0,<<"HisGoalieFreeKick: tryToScanField: YEP (have found no other action)");
  return scan_field(cmd);  
}

void 
HisGoalieFreeKick::correctTargetPosition( Vector & targetPosition )
{
  LOG_POL(1,"HisGoalieFreeKick: INFO: WSinfo::me->pos = "<<WSinfo::me->pos
    <<" target x/y = "<<targetPosition);
   
  while (    RIGHT_PENALTY_AREA.inside( targetPosition )      )
  {
    if (targetPosition.getX() >= FIELD_BORDER_X - PENALTY_AREA_LENGTH - 0.3)
      targetPosition.subFromX( 0.5 );
    if (    targetPosition.getY() >= 0.0
         && targetPosition.getY() < PENALTY_AREA_WIDTH / 2.0 + 0.3)
      targetPosition.addToY( 0.15 );
    if (    targetPosition.getY() < 0.0
         && targetPosition.getY() > -PENALTY_AREA_WIDTH / 2.0 - 0.3)
      targetPosition.subFromY( 0.15 );
    LOG_POL(2,"HisGoalieFreeKick: target x/y position has been reduced to "
      <<targetPosition.getX()<<" / "<<targetPosition.getY());
  }
  while (    WSinfo::me->pos.getX() > FIELD_BORDER_X - PENALTY_AREA_LENGTH - 0.3
          && WSinfo::me->pos.getY() >= PENALTY_AREA_WIDTH / 2.0 - 1.0
          && targetPosition.getY() < PENALTY_AREA_WIDTH / 2.0 + 0.3 )
  {
    LOG_POL(2,"HisGoalieFreeKick: Change target y position from "
      <<targetPosition.getY()<<" to "<<targetPosition.getY()+0.5);
    targetPosition.addToY( 0.5 );
  }
  while (    WSinfo::me->pos.getX() > FIELD_BORDER_X - PENALTY_AREA_LENGTH - 0.3
          && WSinfo::me->pos.getY() <= - PENALTY_AREA_WIDTH / 2.0 + 1.0
          && targetPosition.getY() > - PENALTY_AREA_WIDTH / 2.0 - 0.3 )
  {
    LOG_POL(2,"HisGoalieFreeKick: Change target y position from "
      <<targetPosition.getY()<<" to "<<targetPosition.getY()-0.5);
    targetPosition.subFromY( 0.5 );
  }
  while (    targetPosition.getX() > FIELD_BORDER_X - PENALTY_AREA_LENGTH - 0.3
          && targetPosition.getY() >= PENALTY_AREA_WIDTH / 2.0 - 1.0
          && WSinfo::me->pos.getY() < PENALTY_AREA_WIDTH / 2.0 + 0.3 )
  {
    LOG_POL(2,"HisGoalieFreeKick: Change target x position from "
      <<targetPosition.getX()<<" to "<<targetPosition.getX()-0.5);
    targetPosition.subFromX( 0.5 );
  }
  while (    targetPosition.getX() > FIELD_BORDER_X - PENALTY_AREA_LENGTH - 0.3
          && targetPosition.getY() <= - PENALTY_AREA_WIDTH / 2.0 + 1.0
          && WSinfo::me->pos.getY() > - PENALTY_AREA_WIDTH / 2.0 - 0.3 )
  {
    LOG_POL(2,"HisGoalieFreeKick: Change target x position from "
      <<targetPosition.getX()<<" to "<<targetPosition.getX()-0.5);
    targetPosition.subFromX( 0.5 );
  }
    
  while (    WSinfo::ball->pos.distance( targetPosition ) < 10.0 
          && (    WSinfo::ws->play_mode == PM_his_GoalieFreeKick
               || (    WSinfo::ws->play_mode == PM_his_GoalKick
                    && WSinfo::ball->pos.getX() > FIELD_BORDER_X - 6.0 )
             )
        )
  {
    LOG_POL(2,"HisGoalieFreeKick: Reduce target x position from "
      <<targetPosition.getX()<<" to "<<targetPosition.getX()-0.5
      <<" ballvel="<<WSinfo::ball->vel.norm()<<" (2), pm="<<WSinfo::ws->play_mode);
    targetPosition.subFromX( 0.5 );
  }
  
  PPlayer hisGoalie = NULL;
  hisGoalie 
    = WSinfo::get_opponent_by_number( WSinfo::ws->his_goalie_number );
  if (    hisGoalie
       && WSinfo::is_ball_pos_valid()
       && WSinfo::me->pos.getX() > hisGoalie->pos.getX()
       && WSinfo::me->pos.getX() > WSinfo::ball->pos.getX()
       && fabs( hisGoalie->pos.getY() - WSinfo::me->pos.getY() ) < 7.0
       && hisGoalie->pos.distance(WSinfo::me->pos) < 10.0 )
  {
    LOG_POL(2,"HisGoalieFreeKick: Der bloede Tormann steht mir im Weg."
      <<" Tut mir Leid, aber ich sehe es echt nicht ein, jetzt um den"
      <<" drumherum zu laufen und so meine Stamina zu verschwenden.");
    targetPosition = WSinfo::me->pos;
  }
  if (   hisGoalie
      &&   WSinfo::me->pos.distance( targetPosition )
         > hisGoalie->pos.distance( targetPosition ) )
  {
    Vector lotfuss
      = Tools::get_Lotfuss( WSinfo::me->pos, targetPosition, hisGoalie->pos);
    if ( lotfuss.distance( hisGoalie->pos ) < 9.5 )
    {
      targetPosition = lotfuss - hisGoalie->pos;
      targetPosition.normalize( 9.5 );
      targetPosition += hisGoalie->pos;
      lotfuss
        = Tools::get_Lotfuss( WSinfo::me->pos, targetPosition, hisGoalie->pos);
      if ( lotfuss.distance( hisGoalie->pos ) < 9.5 )
      {
        targetPosition = lotfuss - hisGoalie->pos;
        targetPosition.normalize( 9.5 );
        targetPosition += hisGoalie->pos;
      }
    }
  }

  //TG17
  if (   WSinfo::ws->play_mode == PM_his_GoalKick
      && ivpMyDirectOpponent
      && ivpMyDirectOpponent->pos.distance(WSinfo::ball->pos) < 1.5 )
  {
    // my direct opponent is going to execute the goal kick :-)
    targetPosition.setY( 0.3 * ivpMyDirectOpponent->pos.getY() );
  }

}

