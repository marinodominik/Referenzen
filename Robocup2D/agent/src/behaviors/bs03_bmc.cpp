#include "bs03_bmc.h"

#include "ws_info.h"
#include "ws_memory.h"
#include "log_macros.h"
#include "tools.h"
#include "blackboard.h"

bool Bs03::initialized = false;
int  Bs03::cvHisOffsideCounter = 0;

Bs03::Bs03() {
  ivLastOffSideTime = -999;
  noball03 = new Noball03;
  noball03_attack = new Noball03_Attack;
  wball06 = new Wball06;
  onesteppass = new OneStepPass;
  if ( wball06 && onesteppass )
  {
    onesteppass->setScoreBehaviors( wball06->getScoreBehavior(),
                                    wball06->getOneStepScoreBehavior()
                                  );
  }
  interceptball = new InterceptBall;
  goalkick = new GoalKick;
  faceball = new FaceBall;
  standardSit07 = StandardSituation::getInstance();
  goalie_bs03 = new Goalie_Bs03;
  go2pos = new NeuroGo2Pos;
  line_up = new LineUp();
  hisgoaliefreekick = new HisGoalieFreeKick();
  overcomeOffside08wball = new OvercomeOffside08Wball();
  foul2010 = new Foul2010();
  myGoalKick = new MyGoalKick2016();
  avengers_Player = new Penalty();

  do_standard_kick = false;
}

Bs03::~Bs03() {
  delete noball03;
  delete noball03_attack;
  delete wball06;
  delete onesteppass;
  delete goalkick;
  delete faceball;
  delete goalie_bs03;
  delete go2pos;
  delete hisgoaliefreekick;
  delete line_up;
  delete overcomeOffside08wball;
  delete interceptball;
  delete foul2010;
  delete myGoalKick;
  delete avengers_Player;
}

bool Bs03::init(char const * conf_file, int argc, char const* const* argv)
{
  if ( initialized )
    return true;

  bool returnValue = true;

  returnValue &= Wball06::init(                   conf_file, argc, argv );
  returnValue &= OneStepPass::init(               conf_file, argc, argv );
  returnValue &= Noball03::init(                  conf_file, argc, argv );
  returnValue &= Noball03_Attack::init(                  conf_file, argc, argv );
  returnValue &= GoalKick::init(                  conf_file, argc, argv );
  returnValue &= HisGoalieFreeKick::init(         conf_file, argc, argv );
  returnValue &= FaceBall::init(                  conf_file, argc, argv );
  returnValue &= StandardSituation::init(       conf_file, argc, argv );
  returnValue &= Goalie_Bs03::init(               conf_file, argc, argv );
  returnValue &= NeuroGo2Pos::init(               conf_file, argc, argv );
  returnValue &= LineUp::init(                    conf_file, argc, argv );
  returnValue &= OvercomeOffside08Wball::init(    conf_file, argc, argv );
  returnValue &= Foul2010::init(                  conf_file, argc, argv );
  returnValue &= MyGoalKick2016::init(            conf_file, argc, argv );
  returnValue &= Penalty::init(        conf_file, argc, argv );

  return initialized = returnValue;
}

void Bs03::reset_intention() {
  noball03->reset_intention();
  wball06->reset_intention();

  Blackboard::init();
}

bool Bs03::get_cmd(Cmd & cmd)
{
  bool cmd_set = false; // return value

#if LOGGING && BASIC_LOGGING
  long start_bs03 = Tools::get_current_ms_time();
#endif

  LOG_POL( 0, << "Entered BS03 " << start_bs03 - WSinfo::ws->ms_time_of_sb << " ms after sense body" );
  LOG_POL( 0, << "CONTENTS OF WSinfo::ws->play_mode is " << WSinfo::ws->play_mode );

  LOG_POL( 0, << "BLACKBOARD::main_intention.get_type()==" << Blackboard::main_intention.get_type() );
  LOG_POL( 0, << "BLACKBOARD::pass_intention.get_type()==" << Blackboard::pass_intention.get_type() );

  select_relevant_teammates();

  if( WSinfo::ws->time_of_last_update != WSinfo::ws->time )
  {
    LOG_POL( 0, << "WARNING - did NOT get SEE update!" );
  }

#if LOGGING && WORLD_MODEL_LOGGING
  WSinfo::visualize_state();
#endif



  bool playModeIsAMyPenaltyPM = ( WSinfo::ws->play_mode == PM_my_BeforePenaltyKick || WSinfo::ws->play_mode == PM_my_PenaltyKick );
  bool playModeIsAHisPenaltyPM = ( WSinfo::ws->play_mode == PM_his_BeforePenaltyKick || WSinfo::ws->play_mode == PM_his_PenaltyKick );
  bool playModeIsAAnyPenaltyPM = ( playModeIsAMyPenaltyPM || playModeIsAHisPenaltyPM );

  int resp_player = 0;
  if( playModeIsAMyPenaltyPM )
  {
    resp_player = 12 - ( WSinfo::ws->penalty_count % NUM_PLAYERS );
    if( resp_player == 12 )
    {
      resp_player = 1;
    }
  }

  bool itsMyPenaltyTurn = ( resp_player == WSinfo::me->number );



  if( !ClientOptions::consider_goalie || ( playModeIsAMyPenaltyPM && itsMyPenaltyTurn ) )
  {
    switch( determineCurrentPlayMode() )
    {
      case PM_his_BeforePenaltyKick:
      case PM_his_PenaltyKick:
      {
        Vector dir = Vector( 6, 0 );
        dir.rotate( 2 * PI / 10 * ( WSinfo::me->number - 1 ) );
        LOG_POL( 0, << "PM_his_(Before)PenaltyKick: I am passive" << dir );
        go2pos->set_target( Vector( 0, 0 ) + dir );
        cmd_set = go2pos->get_cmd( cmd );
        break;
      }

      case PM_my_BeforePenaltyKick: 
      {
        if( !itsMyPenaltyTurn )
        {
          Vector dir = Vector( 6, 0 );
          dir.rotate( 2 * PI / 10 * ( WSinfo::me->number - 1 ) );
          LOG_POL( 0, << "PM_my_BeforePenaltyKick: I am passive" << dir );
          go2pos->set_target( Vector( 0, 0 ) + dir );
          cmd_set = go2pos->get_cmd( cmd );
        } 
        else 
        {
          LOG_POL( 0, << "PM_my_BeforePenaltyKick: I am ACTIVE!" );
          if( !WSinfo::is_ball_pos_valid() )
          {
            faceball->turn_to_ball();
            cmd_set = faceball->get_cmd( cmd );
          } 
          else
          {
            if( WSinfo::me->pos.distance( WSinfo::ball->pos ) >= WSinfo::me->kick_radius - 0.1 )
            {
              go2pos->set_target( WSinfo::ball->pos );
              cmd_set = go2pos->get_cmd( cmd );
            } 
            else 
            {
              cmd.cmd_body.set_turn( -WSinfo::me->ang.get_value() );
              cmd_set = cmd.cmd_body.is_cmd_set();
            }
          }
        }
        break;
      }
      case PM_my_PenaltyKick: 
      {
        Blackboard::need_goal_kick = false; // just to be sure!
        do_standard_kick = false;
        if( !itsMyPenaltyTurn )
        {
          LOG_POL( 0, << "PM_my_PenaltyKick: I am passive" );
          cmd.cmd_body.set_turn( 0 );
          cmd_set = cmd.cmd_body.is_cmd_set();
          break;
        }
        else
        {
          LOG_POL( 0, << "PM_my_PenaltyKick: I am ACTIVE!" );
          cmd_set = avengers_Player->get_cmd(cmd);
          break;
        }
      }
      case PM_PlayOn:
        if( Blackboard::need_goal_kick )
        {
          cmd_set = goalkick->get_cmd( cmd );
        }
        if( !cmd_set )
        {
          Blackboard::need_goal_kick = false;

          if( WSinfo::is_ball_pos_valid() && WSinfo::is_ball_kickable() )
          { // Ball is kickable
            if( do_standard_kick ) //Ball is kickable and I started a standardsit
            {
              cmd_set = standardSit07->get_cmd( cmd );
            }
            else
            {
              if (onesteppass->isSuitableOneStepPassSituation() || onesteppass->isSuitableOneStepPassSituationForExtremePanic() )
                cmd_set = onesteppass->get_cmd( cmd );
              if ( !cmd_set && overcomeOffside08wball->isOOTPassPossible() )
              {
                LOG_POL( 0, << "Bs03: CALL TO NEW OOT" );
                cmd_set = overcomeOffside08wball->get_cmd( cmd );
              }
              if ( !cmd_set )
              {
                cmd_set =  wball06->get_cmd( cmd );
              }
            }
          } // end ball is kickable
          else // Ball is not kickable
          {    
            do_standard_kick = false; // just to be sure...
            if( foul2010->foul_situation() ) //TG16: Adapted for Leipzig after lots of experiments (see inside!).
            {
              cmd_set = foul2010->get_cmd( cmd );
            }
            else 
              {
                if( Noball03_Attack::am_I_attacker() )
                {
                  cmd_set = noball03_attack->get_cmd( cmd );
                }
                else 
                {
                    cmd_set = noball03->get_cmd( cmd );
                }
              }
          } // end of: ball is not kickable
      } // end cmd is not set
      break;
    case PM_my_GoalieFreeKick:
      cmd_set = goalkick->get_cmd( cmd );
      break;
    case PM_his_GoalKick:
    case PM_his_GoalieFreeKick: 
    {
      cmd_set = hisgoaliefreekick->get_cmd( cmd );
      break;
    }
    case PM_my_GoalKick:
      cmd_set = myGoalKick->get_cmd( cmd );
      break;
    case PM_my_KickIn:
    case PM_his_KickIn:
    case PM_my_CornerKick:
    case PM_his_CornerKick:
    case PM_my_FreeKick:
    case PM_his_FreeKick:
    case PM_my_OffSideKick:
    case PM_his_OffSideKick:
    case PM_my_KickOff:

      if( WSinfo::ws->play_mode == PM_his_OffSideKick
        && ivLastOffSideTime != WSinfo::ws->time
        && ivLastOffSideTime + 200 < WSinfo::ws->time
        )
      { 
        cvHisOffsideCounter++;
        ivLastOffSideTime = WSinfo::ws->time;
        LOG_POL( 0, << "WE ARE IN OFFSIDE" );
      }

      cmd_set = standardSit07->get_cmd( cmd );

      if( WSinfo::is_ball_pos_valid() && WSinfo::is_ball_kickable() )
      { // Ball is kickable
        do_standard_kick = true;
      }

      break;

    case PM_my_BeforeKickOff:
    case PM_his_BeforeKickOff:
    case PM_my_AfterGoal:
    case PM_his_AfterGoal:
    case PM_Half_Time:
      cmd_set = line_up->get_cmd( cmd, true );//true==sector-based scanning
      break;
    default:
      ERROR_OUT << "time " << WSinfo::ws->time << " player nr. " << WSinfo::me->number << " play_mode is " << WSinfo::ws->play_mode << " no command was set by behavior";
      return false;  // behaviour is currently not responsible for that case
    }
  } 
  else 
  {
    LOG_POL( 0, << "In BS03_BMC [goalie mode]: " );
    if( WSinfo::ws->play_mode == PM_my_GoalKick )
    {
      cmd_set = myGoalKick->get_cmd( cmd );
      //cmd_set = standardSit07->get_cmd( cmd );
    }
    if( !cmd_set )
    {
      if( WSinfo::ws->play_mode == PM_my_GoalKick )
      {
        LOG_POL( 0, << "MyGoalKick2016 => Goalie_Bs03 (Weiterleitung)" );
      }
      cmd_set = goalie_bs03->get_cmd( cmd );
    }
  }


  LOG_POL( 0, << "BS03. Decision needed " << Tools::get_current_ms_time() - start_bs03 << " ms" << std::flush );

  log_cmd_main( cmd );

  if( playModeIsAAnyPenaltyPM || ( playModeIsAMyPenaltyPM && itsMyPenaltyTurn ) )
  {
    if( playModeIsAMyPenaltyPM && onesteppass->foresee( cmd ) == true )
    {
      LOG_POL( 0, <<"BS03: Foresse for OneStepPass was successful!" );
    }
    else // check wheter ball is kickable in next cycle and what to do if yes.
    {
      wball06->foresee( cmd );
    }
  }
  
  long nowtime = Tools::get_current_ms_time();

  if(nowtime - WSinfo::ws->ms_time_of_sb > 90)
  {
    LOG_ERR( 0, << "TIME WARNING! BS03. Alltogehter needed " << nowtime - start_bs03 << " ms Finishing " << nowtime - WSinfo::ws->ms_time_of_sb << " ms after sense body" << std::flush );
  }
  else
  {
    LOG_POL( 0, << "BS03. Alltogehter needed " << nowtime - start_bs03 << " ms Finishing " << nowtime - WSinfo::ws->ms_time_of_sb << " ms after sense body" << std::flush );
  }

  LOG_POL( 0, << "Out BS03_BMC : intention was set " << cmd_set << std::flush );
  return cmd_set;
}

void Bs03::log_cmd_main( Cmd &cmd )
{
    Angle angle;
    double power, x, y, foul;

    switch( cmd.cmd_body.get_type() )
    {
    case Cmd_Body::TYPE_KICK:
      cmd.cmd_body.get_kick( power, angle );
      LOG_POL( 0, << "bs03_bmc: cmd KICK, power " << power << ", angle " << RAD2DEG( angle ) );
      break;
    case Cmd_Body::TYPE_TURN:
      cmd.cmd_body.get_turn( angle );
      LOG_POL( 0, << "bs03_bmc: cmd Turn, angle " << RAD2DEG( angle ) );
      break;
    case Cmd_Body::TYPE_DASH:
      cmd.cmd_body.get_dash( power, angle );
      LOG_POL( 0, << "bs03_bmc: cmd DASH, power " << power << "[>=SSS13: angle " << RAD2DEG( angle ) << "]");
      break;
    case Cmd_Body::TYPE_CATCH:
      cmd.cmd_body.get_catch( angle );
      LOG_POL( 0, << "bs03_bmc: cmd Catch, angle " << RAD2DEG( angle ) );
      break;
    case Cmd_Body::TYPE_TACKLE:
      cmd.cmd_body.get_tackle( power, foul );
      LOG_POL( 0, << "bs03_bmc: cmd Tackle, power " << power << " fouling? " << foul);
      break;
    case Cmd_Body::TYPE_MOVETO:
      cmd.cmd_body.get_moveto( x, y );
      LOG_POL( 0, << "bs03_bmc: cmd Moveto, target " << x << " " << y );
      break;
    default:
      LOG_POL( 0, << "bs03_bmc: No CMD was set " << std::flush );
    }
}

void Bs03::select_relevant_teammates() // player dependend
{
  switch( WSinfo::me->number )
  {
  case 2:
    if( WSmemory::team_last_at_ball() != 0 ) // we defend!
    {
      WSinfo::set_relevant_teammates( 4, 3, 6, 7, 1 ); //TG: old:  3 4 7 6 1
    }
    else
    {
      WSinfo::set_relevant_teammates( 6, 7, 3, 9);
    }
    break;

  case 3:
    if( WSmemory::team_last_at_ball() != 0 ) // we defend!
    {
      WSinfo::set_relevant_teammates( 4, 2, 5, 7, 1 );  //TG: old: 2 4 7 1
    }
    else
    {
      WSinfo::set_relevant_teammates( 6, 7, 8, 2, 4 );
    }
    break;

  case 4:
    if( WSmemory::team_last_at_ball() != 0 ) // we defend!
    {
      WSinfo::set_relevant_teammates( 1, 3, 2, 5, 7 );  //TG: old: 3 5 7 1
    }
    WSinfo::set_relevant_teammates( 7, 6, 8, 2, 5, 3 );
    break;

  case 5:
    if( WSmemory::team_last_at_ball() != 0 ) // we defend!
    {
      WSinfo::set_relevant_teammates( 4, 3, 8, 7, 1 ); //TG: old: 4 3 7 8 1
    }
    WSinfo::set_relevant_teammates( 8, 7, 4, 11 );
    break;

  case 6:
    if( WSinfo::me->pos.getX() > 20 )
    {
      WSinfo::set_relevant_teammates( 9, 10, 7);
    }
    else
    {
      WSinfo::set_relevant_teammates( 9, 10, 7, 4, 3, 2 ); //TG: old: 9 10 7 3 2
    }
    break;

  case 7:
    if( WSinfo::me->pos.getX() > 20 )
    {
      WSinfo::set_relevant_teammates( 10, 6, 8, 9, 11 ); // care only for attack players
    }
    else // default attention
    {
      WSinfo::set_relevant_teammates( 10, 6, 8, 9, 11, 4, 3 ); //TG: old: 10 6 8 9 11 3 4
    }
    break;

  case 8:
    if( WSinfo::me->pos.getX() > 20 )
    {
      WSinfo::set_relevant_teammates( 11, 10, 7 );
    }
    else
    {
      WSinfo::set_relevant_teammates( 11, 10, 7, 4, 5, 3 );  //TG: old: 11 10 7 5 4
    }
    break;

  case 9:
    WSinfo::set_relevant_teammates( 10, 6, 7, 11, 8 );
    break;

  case 10:
    WSinfo::set_relevant_teammates( 11, 9, 7, 6, 8 );
    break;

  case 11:
    WSinfo::set_relevant_teammates( 10, 8, 7, 9, 6 );
    break;
  }

  //TG: We want to rely on our teammates' help in case we do not know any more where our direct opponent is.

  LOG_POL( 0, << "BS03: Check for having lost direct opponents ... [t=" << WSinfo::ws->time << "]");

  PPlayer myGoalie = WSinfo::get_teammate_by_number( WSinfo::ws->my_goalie_number );

  if( !WSinfo::is_ball_pos_valid() && myGoalie && WSinfo::me->pos.distance( myGoalie->pos ) < 45.0 )
  {
    LOG_POL( 0, << "BS03: Have lost the ball very near to my goal. => I SET ATTENTION TO GOALIE!" );
    WSinfo::set_relevant_teammates( WSinfo::ws->my_goalie_number );
  }

}

int Bs03::determineCurrentPlayMode()
{
  if( WSinfo::ws->play_mode == PM_his_GoalKick )
  {
    const double fuenfmeterraumX = FIELD_BORDER_X - 5.5;
    const double fuenfmeterraumYAbs = 9.15;
    const double abstossAusgefuehrtSchwellwert = 2.0;
    bool  avoidEnteringPenaltyArea = false;

    if(  (fabs( WSinfo::me->pos.getY() ) < PENALTY_AREA_WIDTH * 0.5 )
       && ( WSinfo::me->pos.getX() > FIELD_BORDER_X - PENALTY_AREA_LENGTH - 0.1 )
       && ( fabs( WSinfo::me->ang.get_value_mPI_pPI() ) < PI * 0.5 ) // VORSICHT!!! Kann zu unvorhergesehenen Problemen führen!!!
      )
    {
      avoidEnteringPenaltyArea = true;
    }

    if(  (  fabs( WSinfo::ball->pos.getX() - fuenfmeterraumX ) > abstossAusgefuehrtSchwellwert
         || fabs( fabs(WSinfo::ball->pos.getY()) - fuenfmeterraumYAbs ) > abstossAusgefuehrtSchwellwert )
      && !avoidEnteringPenaltyArea
      && WSinfo::ball->vel.norm() > 1.0
      && WSinfo::me->stamina > 0.8 * ServerOptions::stamina_max //TG09: alt: 3500
      && ! (   WSmemory::cvCurrentInterceptPeople.num > 0 //TG17
            && WSmemory::cvCurrentInterceptPeople[0]->team == HIS_TEAM
            && RIGHT_PENALTY_AREA.inside( WSmemory::cvCurrentInterceptResult[0].pos ) )
      )
    {
      return PM_PlayOn;
    }

  }

  return WSinfo::ws->play_mode;
}
