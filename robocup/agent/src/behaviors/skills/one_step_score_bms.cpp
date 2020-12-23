/*
 * one_step_score_bms.cpp
 *
 *  Created on: 13.06.2016
 *      Author: tgabel
 */

#include "one_step_score_bms.h"
#include "ws_info.h"
#include "log_macros.h"
#include "base_bm.h"
#include "angle.h"
#include "Vector.h"
#include "intention.h"
#include "one_step_kick_bms.h"



bool OneStepScore::cvInitialized = false;
const double OneStepScore::cvcMinimalGoalDistance   = 23.0;
const double OneStepScore::cvcGoalPostDistance      = 0.35;
const double OneStepScore::cvcGoalPostDistanceExtra = 0.75;
const double OneStepScore::cvcAggressivenes         = 1.5;

bool
OneStepScore::init(char const * conf_file, int argc, char const* const* argv)
{
  if(cvInitialized) return true;
  cvInitialized = true;
  bool returnValue = OneStepKick::init(conf_file,argc,argv);
  return returnValue;

}

OneStepScore::OneStepScore()
{
  ivpOneStepKick = new OneStepKick();
  // statistics
  ivShotCnt = 0;
  ivAverageShotDist = 0.0;
}

OneStepScore::~OneStepScore()
{
  delete ivpOneStepKick;
}

bool
OneStepScore::get_cmd(Cmd &cmd)
{
  ivpOneStepKick->kick_in_dir_with_initial_vel(ServerOptions::ball_speed_max, ivShotAngle );
  LOG_POL(0,"OneStepScore: I do shoot now!");
  ivShotCnt++;
  double rate = 1.0/ivShotCnt;
  ivAverageShotDist = (1.0-rate)*ivAverageShotDist + (rate)*(FIELD_BORDER_X-WSinfo::ball->pos.getX());
  LOG_POL(4,"OneStepScore: HISTORY: ivShotCnt="<<ivShotCnt<<" ivAverageShotDist="<<ivAverageShotDist
            <<" myScore="<<WSinfo::ws->my_team_score);
  //cout<<"OneStepScore: HISTORY: ivShotCnt="<<ivShotCnt<<" ivAverageShotDist="<<ivAverageShotDist
  //<<" myScore="<<WSinfo::ws->my_team_score<<endl;
  ivpOneStepKick->get_cmd( cmd );
  return true;
}

bool
OneStepScore::test_score_now( Intention& intention )
{
  if ( isSuitableOneStepScoreSituation() )
  {
    intention.type = SCORE;
    intention.kick_speed = ServerOptions::ball_speed_max;
    intention.kick_target = ivShotTarget;
    intention.immediatePass = true;
    intention.valid_at_cycle = WSinfo::ws->time;
    LOG_POL(0,"OneStepScore: test_score_now() ==> YEP (setting intention)");
    /*Cmd cmd; // just for verification, that the cmd will be fine
    get_cmd(cmd);
    LOG_POL(0,"OneStepScore: VERIFY: " << cmd.cmd_main );*/
    return true;
  }
  return false;
}

bool
OneStepScore::isSuitableOneStepScoreSituation()
{
  /*TG17: activate
  if (   WSinfo::get_current_opponent_identifier() != TEAM_IDENTIFIER_OXSY
      && WSinfo::get_current_opponent_identifier() != TEAM_IDENTIFIER_GLIDERS )
  {
    LOG_POL(0,"OneStepScore: isSuitableOneStepScoreSituation() ==> NOPE (not against this opponent)");
    return false;
  }*/

  if (   WSinfo::me->pos.distance(HIS_GOAL_CENTER) > cvcMinimalGoalDistance
      && WSinfo::me->pos.distance(HIS_GOAL_LEFT_CORNER) > cvcMinimalGoalDistance
      && WSinfo::me->pos.distance(HIS_GOAL_RIGHT_CORNER) > cvcMinimalGoalDistance )
  {
    LOG_POL(0,"OneStepScore: isSuitableOneStepScoreSituation() ==> NOPE (too distant)");
    return false;
  }
  if (   WSinfo::me->pos.distance(HIS_GOAL_CENTER) > 0.8*cvcMinimalGoalDistance
      && WSinfo::his_team_pos_of_offside_line() - WSinfo::me->pos.getX() < 4.0 )
  {
    PlayerSet frOpps = WSinfo::valid_opponents;
    frOpps.keep_players_in_circle(WSinfo::me->pos, 4.0);
    PlayerSet goOpps = WSinfo::valid_opponents;
    goOpps.keep_players_in_quadrangle( WSinfo::me->pos, HIS_GOAL_CENTER, 8.0, 16.0 );
    if (   frOpps.num == 0
        && (   goOpps.num == 0
            || (goOpps.num==1 && goOpps[0] == WSinfo::his_goalie ) )
       )
    {
      LOG_POL(0,"OneStepScore: isSuitableOneStepScoreSituation() ==> NOPE (I should really go myself [A])");
      return false;
    }
  }
  if ( WSinfo::his_team_pos_of_offside_line() - WSinfo::me->pos.getX() < 4.0 )
  {
    PlayerSet frOpps = WSinfo::valid_opponents;
    frOpps.keep_players_in_circle(WSinfo::me->pos, 5.0);
    PlayerSet goOpps = WSinfo::valid_opponents;
    goOpps.keep_players_in_quadrangle( WSinfo::me->pos, HIS_GOAL_CENTER, 10.0, 18.0 );
    if (   frOpps.num == 0
        && (   goOpps.num == 0
            || (goOpps.num==1 && goOpps[0] == WSinfo::his_goalie ) )
       )
    {
      LOG_POL(0,"OneStepScore: isSuitableOneStepScoreSituation() ==> NOPE (I should really go myself [B])");
      return false;
    }
  }
  if ( 1 )
  {
    PlayerSet frOpps = WSinfo::valid_opponents;
    frOpps.keep_players_in_circle(WSinfo::me->pos, 8.0);
    PlayerSet goOpps = WSinfo::valid_opponents;
    goOpps.keep_players_in_quadrangle( WSinfo::me->pos, HIS_GOAL_CENTER, 12.0, 20.0 );
    if (   frOpps.num == 0
        && (   goOpps.num == 0
            || (goOpps.num==1 && goOpps[0] == WSinfo::his_goalie ) )
       )
    {
      LOG_POL(0,"OneStepScore: isSuitableOneStepScoreSituation() ==> NOPE (I should really go myself [C])");
      return false;
    }
  }
  Vector targetLeft = HIS_GOAL_LEFT_CORNER, targetRight = HIS_GOAL_RIGHT_CORNER;
  targetLeft.subFromY( cvcGoalPostDistance );
  targetRight.addToY ( cvcGoalPostDistance );
  Vector targetLeftExtra = HIS_GOAL_LEFT_CORNER, targetRightExtra = HIS_GOAL_RIGHT_CORNER;
  targetLeftExtra.subFromY( cvcGoalPostDistanceExtra );
  if ( fabs((targetLeftExtra-WSinfo::ball->pos).ARG().get_value_mPI_pPI()) > DEG2RAD(70.0) )
    targetLeftExtra.subFromY( 0.75*cvcGoalPostDistanceExtra );
  targetRightExtra.addToY ( cvcGoalPostDistanceExtra );
  if ( fabs((targetRightExtra-WSinfo::ball->pos).ARG().get_value_mPI_pPI()) > DEG2RAD(70.0) )
    targetRightExtra.addToY ( 0.75*cvcGoalPostDistanceExtra );
  double scoreLeftCorner  = evaluateTarget( targetLeft );
  double scoreRightCorner = evaluateTarget( targetRight );
  double scoreLeftCornerExtra  = evaluateTarget( targetLeftExtra, true );
  double scoreRightCornerExtra = evaluateTarget( targetRightExtra, true );
  LOG_POL(0,"OneStepScore: SCORES: LP "<<scoreLeftCorner<<" nLP "<<scoreLeftCornerExtra
    <<" RP "<<scoreRightCorner<<" nRP "<<scoreRightCornerExtra);
  Vector bestTarget;
  double bestScore;
  if (scoreLeftCorner>scoreRightCorner)
  {
    bestTarget = targetLeft;
    bestScore = scoreLeftCorner;
  }
  else
  {
    bestTarget = targetRight;
    bestScore = scoreRightCorner;
  }
  if (scoreLeftCornerExtra > bestScore)
  {
    bestTarget = targetLeftExtra;
    bestScore = scoreLeftCornerExtra;
  }
  if (scoreRightCornerExtra > bestScore)
  {
    bestTarget = targetRightExtra;
    bestScore = scoreRightCornerExtra;
  }

  if (bestScore < 1.0)
  {
    LOG_POL(0,"OneStepScore: isSuitableOneStepScoreSituation() ==> NOPE (too little power)");
    return false;
  }
  // Success:
  ivShotAngle  = (bestTarget - WSinfo::ball->pos).ARG();
  ivShotTarget = bestTarget;
  LOG_POL(0,"OneStepScore: isSuitableOneStepScoreSituation() ==> YEP (no exclusion criteria found, best score:"
    <<bestScore<<", bestTarget="<<bestTarget<<")");
  LOG_POL(0,<<_2D<<VL2D(WSinfo::ball->pos,ivShotTarget,"99ff99"));
  return true;
}

double
OneStepScore::evaluateTarget( Vector target, bool extra )
{
  LOG_POL(0,<<_2D<<VL2D(WSinfo::ball->pos,target,"55ffff"));
  // 1.) Velocity to target
  double startVel, finalVel, evaluation;
  int steps;
  calculateVelocityToTarget( target, startVel, finalVel, steps );
  evaluation = finalVel;
  if (extra)
    if (!WSinfo::his_goalie || WSinfo::his_goalie->age <= 2)
      evaluation += 0.5;
  ANGLE kickAngle = (target - WSinfo::ball->pos).ARG();
  Vector ballVel(kickAngle);
  ballVel.normalize(startVel);
  if (!WSinfo::his_goalie || WSinfo::his_goalie->age > 10)
  {
    LOG_POL(0,"OneStepScore: Warning! His goalie is extremely outdated!");
  }
  if ( startVel < 0.1 || finalVel < 0.1 )
  {
    LOG_POL(0,"OneStepScore: Exclude this target, achievable vel is too low (start="
            <<startVel<<", final="<<finalVel<<").");
    return -1.0;
  }
  // 2.) Number of steps to target
  double aggressiveness = cvcAggressivenes;
  if (WSinfo::get_current_opponent_identifier() == TEAM_IDENTIFIER_HELIOS)
    aggressiveness -= 0.5;
  if (WSinfo::his_goalie && WSinfo::his_goalie->age > 0)
    aggressiveness -= sqrt(WSinfo::his_goalie->age);
  if (extra) aggressiveness -= cvcGoalPostDistanceExtra;
  if (   WSinfo::ball->pos.distance(target) > cvcMinimalGoalDistance - 3.0
      && fabs((target - WSinfo::ball->pos).ARG().get_value_mPI_pPI()) > DEG2RAD(40.0) )
    aggressiveness -= 0.5;
  Vector assumedGoaliePos;
  if ( WSinfo::his_goalie )
  {
    assumedGoaliePos = WSinfo::his_goalie->pos;
    if (WSinfo::his_goalie->age_vel == 0) assumedGoaliePos += WSinfo::his_goalie->vel;
    if ( steps - aggressiveness > assumedGoaliePos.distance(target) )
    {
      LOG_POL(0,"OneStepScore: Exclude this target, too many steps required: steps="<<steps
              <<" -"<<aggressiveness<<" vs. distG2tgt="<<WSinfo::his_goalie->pos.distance(target));
      return -1.0;
    }
  }
  if (WSinfo::his_goalie)
  {
    LOG_POL(0,<<"OneStepScore: Target good, advantage="<<(steps - aggressiveness - WSinfo::his_goalie->pos.distance(target))
      <<", where steps="<<steps<<" aggressiveness="<<aggressiveness<<" dist="<<WSinfo::his_goalie->pos.distance(target));
  }
  // 3.) Goalie almost directly on shot way
  if (   WSinfo::his_goalie
      && WSinfo::his_goalie->pos.getX() > WSinfo::ball->pos.getX() )
  {
    Vector lotfuss = Tools::get_Lotfuss( WSinfo::ball->pos, target, WSinfo::his_goalie->pos );
    double stepsToLot = WSinfo::ball->pos.distance(lotfuss) / startVel;
    double goalieToLotfuss = WSinfo::his_goalie->pos.distance(lotfuss);
    if (    goalieToLotfuss < stepsToLot
         && stepsToLot > 2.0 )
    {
      LOG_POL(0,"OneStepScore: Exclude this target, goalie reaches lotfuss in "<<stepsToLot);
      return -1.0;
    }
    else
    {
      LOG_POL(0,"OneStepScore: Goalie lotfuss check ok: stepsToLot="<<stepsToLot<<" goalieToLotfuss="<<goalieToLotfuss);
    }
  }
  // 4a.) Too much danger in first two steps
  Vector ballPos1 = WSinfo::ball->pos + ballVel;
  if (WSinfo::his_goalie)
  {
    Vector ballPos2 = ballPos1 + (ServerOptions::ball_decay*ballVel);
    if (     WSinfo::his_goalie->pos.distance(ballPos1)
           < ServerOptions::catchable_area_l + ServerOptions::ball_size
        ||   WSinfo::his_goalie->pos.distance(ballPos2)
           < ServerOptions::catchable_area_l + ServerOptions::ball_size + 0.6 ) //0.6 for first dash
    {
      LOG_POL(0,"OneStepScore: Exclude this target, goalie reaches ball at step 1 or 2.");
      return -1.0;
    }
  }
  // 4b.) Too much danger in first steps due to opponent //TG17
  PlayerSet opps = WSinfo::valid_opponents;
  opps.keep_players_in_circle( ballPos1, 2.0 );
  if (opps.num > 0)
  {
    for ( int o = 0; o < opps.num; o++ )
    {
      if (   opps[o]->age <= 1
          && opps[o]->pos.distance(ballPos1) < opps[o]->kick_radius )
      {
        LOG_POL(0,"OneStepScore: Exclude this target, opp #"<<opps[o]->number
          <<" reaches ball at step 1 very likely (pos).");
        return -1.0;
      }
      if (   opps[o]->age_vel == 0
          && (opps[o]->pos+opps[o]->vel).distance(ballPos1) < opps[o]->kick_radius )
      {
        LOG_POL(0,"OneStepScore: Exclude this target, opp #"<<opps[o]->number
          <<" reaches ball at step 1 very likely (pos+vel).");
        return -1.0;
      }
      Vector angVec( opps[o]->ang );
      angVec.normalize(1.0);
      if (   opps[o]->age_ang == 0
          && (opps[o]->pos + opps[o]->vel + angVec).distance(ballPos1) < opps[o]->kick_radius )
      {
        LOG_POL(0,"OneStepScore: Exclude this target, opp #"<<opps[o]->number
          <<" reaches ball at step 1 very likely (pos+vel+ang).");
        return -1.0;
      }
    }
  }
  // 5.) Aribtrary opponents on shot line
  opps = WSinfo::valid_opponents;
  opps.keep_players_in_quadrangle( WSinfo::ball->pos + ballVel, target, 1.5, steps );
  LOG_POL(0,<<_2D<<Quadrangle2d(WSinfo::ball->pos + ballVel, target, 1.5, steps));
  if (opps.num > 0)
  {
    LOG_POL(0,"OneStepScore: Exclude this target, at least one enemy is on shot line.");
    return -1.0;
  }
  // 5.) Prefer goalie's back side
  if ( WSinfo::his_goalie )
  {
    PlayerSet justGoalie;
    justGoalie.append(WSinfo::his_goalie);
    justGoalie.keep_players_in_cone( WSinfo::ball->pos,
                                     HIS_GOAL_RIGHT_CORNER-WSinfo::ball->pos,
                                     HIS_GOAL_LEFT_CORNER-WSinfo::ball->pos );
    if (justGoalie.num > 0)
    {
      if (justGoalie[0]->ang.get_value_mPI_pPI() > 0.0 )
      {
        // adjusted to north -> prefer south post
        LOG_POL(0,"OneStepScore: Goalie adjusted to north.");
        if ( target.getY() > 0.0 )
        {
          evaluation *= 0.7;
          LOG_POL(0,"OneStepScore: Goalie adjusted to north. Impair north post ("<<finalVel<<"->"<<evaluation<<").");
        }
      }
      else
      {
        // adjusted to south -> prefer north post
        LOG_POL(0,"OneStepScore: Goalie adjusted to south. ");
        if ( target.getY() < 0.0 )
        {
          evaluation *= 0.7;
          LOG_POL(0,"OneStepScore: Goalie adjusted to south. Impair south post ("<<finalVel<<"->"<<evaluation<<").");
        }
      }
    }
    else
    {
        LOG_POL(0,"OneStepScore: Goalie not in cone!");
    }
  }
  return evaluation;
}

void
OneStepScore::calculateVelocityToTarget( Vector target, double& velAtStart, double& velAtTarget, int& steps )
{
  ANGLE kickAngle = (target - WSinfo::ball->pos).ARG();
  ivpOneStepKick->kick_in_dir_with_initial_vel(ServerOptions::ball_speed_max, kickAngle );
  ivpOneStepKick->get_vel(velAtStart);
  LOG_POL(0,"OneStepScore: Kicking to "<<target<< " I can achieve v="<< velAtStart<< " with one step.");
  steps = 1000;
  if ( velAtStart < 0.1 )
    velAtTarget = 0.0;
  else
  if ( WSinfo::ball->pos.distance(target) / velAtStart > 10.0 )
    velAtTarget = 0.0;
  else
  {
    Vector ballPos = WSinfo::ball->pos, ballVel(kickAngle);
    ballVel.normalize(velAtStart);
    steps = 0;
    while ( ballPos.getX() < FIELD_BORDER_X )
    {
      ballPos += ballVel;
      if (steps==0)
      {LOG_POL(0,<<_2D<<VSTRING2D(ballPos,ballVel.norm(),"000000"));}
      LOG_POL(0,<<_2D<<VC2D(ballPos,0.3,"55ffff"));
      ballVel *= ServerOptions::ball_decay;
      steps++;
    }
    velAtTarget = ballVel.norm();
  }
  LOG_POL(0,"OneStepScore: Then, the final vel will be "<<velAtTarget<<" after "<<steps<<" steps.");
}
