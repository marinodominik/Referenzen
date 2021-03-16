/* author: thomas huber, 2010
 *
 * foul behavior
 *
 */
 
#include "foul2010_bmp.h"

#if 1   /* 1: debugging; 0: no debugging */
#include "log_macros.h"
#define DRAW(XXX)  LOG_POL(0,<<_2D<<XXX)
#else
#define DRAW(XXX)
#endif
// Drawing macros >>>1
#define MARK_POS(P,C) DRAW(C2D((P).x,(P).y,0.3,#C));
#define DRAW_LINE(P,Q,C) DRAW(VL2D((P),(Q),#C))

#define DEFAULT_MIN_FOUL_PROBABILITY 0.9
#define DEFAULT_BALLVEL_AFTER_FOUL 0.5
#define FOUL_LOG_LEVEL  1

bool Foul2010::initialized=false;

Foul2010::Foul2010() {
    basiccmd = new BasicCmd;
    ivMaxBallVelAfterFoul = DEFAULT_BALLVEL_AFTER_FOUL;
    ivLastFoulExecuted = -1;
    ivLastExecutedDecideForSituation = -1;
}

Foul2010::~Foul2010() {
    delete basiccmd;
}

bool
Foul2010::init(char const * conf_file, int argc, char const* const* argv) {
  if(initialized) return true;
  initialized = true;
  bool returnValue = true;
  returnValue &= BasicCmd::init(conf_file,argc,argv);
  return returnValue;
}

//============================================================================
// public: get_cmd
//============================================================================
/**
 *  foul get_cmd function. decides whether or not a foul is possible
 *  if a foul is possible it will be executed otherwise false is returned
 */
bool
Foul2010::get_cmd(Cmd & cmd)
{
  bool returnValue = false;
  if(ivLastExecutedDecideForSituation != WSinfo::ws->time)
  {
    if(decide_situation())
    {
      cmd.cmd_body.set_tackle(RAD2DEG(-ivFoulAngle.get_value_mPI_pPI()),1);
      ivLastFoulExecuted = WSinfo::ws->time;
      LOG_POL(FOUL_LOG_LEVEL,<<"Foul2010: foul executed by "<<WSinfo::me->number<<" in direction: "<<-ivFoulAngle.get_value_mPI_pPI()<<" R2D: "<<RAD2DEG(ivFoulAngle.get_value_mPI_pPI()));
      returnValue = true;
    }
  }
  else
  {
    cmd.cmd_body.set_tackle(RAD2DEG(-ivFoulAngle.get_value_mPI_pPI()),1);
    ivLastFoulExecuted = WSinfo::ws->time;
    LOG_POL(FOUL_LOG_LEVEL,<<"Foul2010: foul executed by "<<WSinfo::me->number<<" in direction: "<<-ivFoulAngle.get_value_mPI_pPI()<<" R2D: "<<RAD2DEG(ivFoulAngle.get_value_mPI_pPI()));
    returnValue = true;
  }
  return returnValue;
}

//============================================================================
// public: foul_possible
//============================================================================
/**
 *  this function decides if a foul is currently possible
 */
bool
Foul2010::decide_situation()
{
  bool returnValue = false;

  if(ivLastFoulExecuted < WSinfo::ws->time-2)
  {
    if(0)
    {
    }
      else
       if(get_ball())
        {
          returnValue = true;
        }
      else
       if(stop_ball())
        {
          returnValue = true;
        }
    ivLastExecutedDecideForSituation = WSinfo::ws->time;
  }
  return returnValue;
}

bool
Foul2010::foul_situation()
{
  //TG16: Added restrictions
  if (   WSinfo::get_current_opponent_identifier() == TEAM_IDENTIFIER_HELIOS
      || WSinfo::get_current_opponent_identifier() == TEAM_IDENTIFIER_GLIDERS)
  {
    LOG_POL(FOUL_LOG_LEVEL,<<"Foul2010: Do not play foul: Do never play foul against this opponent.");
    return false;
  }
  if (WSinfo::his_team_pos_of_offside_line() - WSinfo::ball->pos.getX() > 4.0)
  {
    LOG_POL(FOUL_LOG_LEVEL,<<"Foul2010: Do not play foul: Too distant from offside line.");
    return false;
  }
  if (WSinfo::ball->pos.distance(HIS_GOAL_CENTER) > 25.0)
  {
    LOG_POL(FOUL_LOG_LEVEL,<<"Foul2010: Do not play foul: Too distant from his goal.");
    return false;
  }
  if (   WSinfo::his_goalie
      && (   WSmemory::cvCurrentInterceptPeople.num > 0
          && WSmemory::cvCurrentInterceptPeople[0] == WSinfo::his_goalie
          && WSmemory::cvCurrentInterceptResult[0].time == 0)
     )
  {
    // don't risk a freekick
    LOG_POL(FOUL_LOG_LEVEL,<<"Foul2010: Do not play foul: Goalie is malfunctioning, does not catch.");
    return false;
  }
  Vector toGoal = WSinfo::me->pos + (HIS_GOAL_CENTER - WSinfo::me->pos).normalize(5.0);
  PlayerSet opps = WSinfo::valid_opponents;
  if (   WSmemory::cvCurrentInterceptPeople.num > 0
      && WSmemory::cvCurrentInterceptPeople[0]->team == HIS_TEAM)
    opps.remove( WSmemory::cvCurrentInterceptPeople[0] );
  opps.keep_players_in_quadrangle(WSinfo::me->pos, toGoal, 5.0, 5.0);
  if (opps.num > 0)
  {
    // don't risk a freekick
    LOG_POL(FOUL_LOG_LEVEL,<<"Foul2010: Do not play foul: Meaningless, since many opps in way to goal.");
    return false;
  }

  return foul_situation(DEFAULT_MIN_FOUL_PROBABILITY);
}

bool
Foul2010::foul_situation(double minProbability)
{
  // we don't allow a minProbability below 0.5
  if(minProbability < 0.5)
  {
    minProbability = 0.5;
  }
  
  double scaleProbability = 0.7;
  bool returnValue = false;
  bool foulSuccessProbabilityGiven = false;
  PlayerSet interferingOpps = WSinfo::valid_opponents;
  PlayerSet kickableOpps;
  
  for(int i=0; i<interferingOpps.num; i++)
  {
    Vector oppDistToBall = interferingOpps[i]->pos - WSinfo::ball->pos;
    if(oppDistToBall.norm() <= interferingOpps[i]->kick_radius)
    {
      kickableOpps.join(interferingOpps[i]);
      LOG_POL(FOUL_LOG_LEVEL,<<"Foul2010: Opp: "<<interferingOpps[i]->number<<" has the ball in his kickrange of " << interferingOpps[i]->kick_radius);
      
    }
  }
  // Check if all kickableOpps are already fouled
/*  for(int i=0; i<kickableOpps.num; i++)
  {
    if(!kickableOpps[i]->fouled)
    {
      kickableOppsNotAlreadyFouled = true;
    }
  }
*/

  // Check if the current success probability is high enough
  double currentFoulSuccessProbability =
     Tools::get_foul_success_probability(WSinfo::me->pos,
                                         WSinfo::ball->pos,
                                         WSinfo::me->ang.get_value_mPI_pPI());
  LOG_POL(FOUL_LOG_LEVEL,<<"Foul2010: Current foul success probability is: "<<
      currentFoulSuccessProbability);
      
  // we have scored more goals then the opponent team
  if(WSinfo::ws->my_team_score > WSinfo::ws->his_team_score)
  {
    if(0)
    {
    }
      else
        if(WSinfo::ball->pos.getX() > (FIELD_BORDER_X - PENALTY_AREA_LENGTH) &&
           (WSinfo::ball->pos.getY() > (-PENALTY_AREA_WIDTH/2) &&
          WSinfo::ball->pos.getY() < (PENALTY_AREA_WIDTH/2)))
        {
          if(currentFoulSuccessProbability > 0.5)
          {
            foulSuccessProbabilityGiven = true;
          }      
        }
      else
        if(WSinfo::ball->pos.getX() > (FIELD_BORDER_X - PENALTY_AREA_LENGTH))
        {
          if(currentFoulSuccessProbability > scaleProbability)
          {
            foulSuccessProbabilityGiven = true;
          }      
        }
      else
        if(WSinfo::ball->pos.getX() > 0)
        {
          double neededSuccessProbability = minProbability;
          if(minProbability>0.6)
          {
            // calculate the required foul success probability
            double positionPercentage = (WSinfo::ball->pos.getX() / (FIELD_BORDER_X - PENALTY_AREA_LENGTH));
            neededSuccessProbability = (minProbability - (positionPercentage * (minProbability - scaleProbability)));
          }
          if(currentFoulSuccessProbability > neededSuccessProbability)
          {
            foulSuccessProbabilityGiven = true;
          }
        }
      // If we the ball is in our side of the field we have to have at least the
      // given minimum foul probability to be able to execute the foul behavior
      else
        if(WSinfo::ball->pos.getX() < 0)
        {
          if(currentFoulSuccessProbability > minProbability)
          {
            foulSuccessProbabilityGiven = true;
          }
        }
  }
  // we havn't scored more goals then the oppoent team
  else
  {
    if(0)
    {
    }
      else
        if(WSinfo::ball->pos.getX() > (FIELD_BORDER_X - PENALTY_AREA_LENGTH) &&
           (WSinfo::ball->pos.getY() > (-PENALTY_AREA_WIDTH/2) &&
          WSinfo::ball->pos.getY() < (PENALTY_AREA_WIDTH/2)))
        {
          if(currentFoulSuccessProbability > 0.7)
          {
            foulSuccessProbabilityGiven = true;
          }      
        }
      else
        if(WSinfo::ball->pos.getX() > (FIELD_BORDER_X - 30) )
        {
          if( currentFoulSuccessProbability >= 0.9 )
          {
            foulSuccessProbabilityGiven = true;
          }
        }
  }
  /* basic requirement for a foul:  
   *    1. at least one opponent is able to kick the ball
   *    2. we require a certain foul success probability
   *    3. we don't have a yellow card yet
   */
  if(kickableOpps.num > 0 &&
//   kickableOppsNotAlreadyFouled &&
     foulSuccessProbabilityGiven &&
     !WSinfo::me->yellow_card)
  {
    // basic requirements are true
    returnValue = true;
    // forbid special foul situations
    // we don't want to execute a foul if we are close to our penalty area
    double DIST_TO_PENALTY_AREA = 6.0;
    if(WSinfo::ball->pos.getX() < (-FIELD_BORDER_X + PENALTY_AREA_LENGTH + DIST_TO_PENALTY_AREA)
    /*&&
       (WSinfo::ball->pos.y > (-PENALTY_AREA_WIDTH/2 - DIST_TO_PENALTY_AREA) &&
        WSinfo::ball->pos.y < (PENALTY_AREA_WIDTH/2 + DIST_TO_PENALTY_AREA))*/
        )
    {
      returnValue = false;
    }
  }
  if(returnValue == true && !decide_situation())
  {
    returnValue = false;
  }
  return returnValue;
}

//============================================================================
// protected: stop_ball
//============================================================================
/**
 *  this function tries to find a suitable foul angle such that the
 *  ball has at maximum maxBallVelAfterFoul velocity
 */
bool
Foul2010::stop_ball()
{
  bool returnValue = false;
  if(getMinimalBallVelAfterFoul(false))
  {
    LOG_POL(FOUL_LOG_LEVEL,<<"Foul2010: STOP_BALL foul angle found in direction "<<ivFoulAngle.get_value_mPI_pPI());
    returnValue = true;
  }
  return returnValue;
}

//============================================================================
// protected: get_ball
//============================================================================
/**
 *  this function tries to find a suitable foul angle such that the
 *  ball has at maximum maxBallVelAfterFoul velocity
 */
bool
Foul2010::get_ball()
{
  bool returnValue = false;
  if(getMinimalBallVelAfterFoul(true))
  {
    LOG_POL(FOUL_LOG_LEVEL,<<"Foul2010: GET_BALL foul angle found in direction "<<ivFoulAngle.get_value_mPI_pPI());
    returnValue = true;
  }
  return returnValue;
}

//============================================================================
// public: getMinimalBallVelAfterFoul
//============================================================================
/**
 *  This function decides if an angle can be found to foul and
 *  stop the ball such that the ball has at maximum a specified velocity
 *  param ballHasToBeFree: does the ball has to be free after the foul
 *  return value: true if a suitable angle was found. If not false
 */
bool
Foul2010::getMinimalBallVelAfterFoul(bool ballHasToBeFree)
{
  bool returnValue = false;
  Vector bestEvaluation = Vector(100,100), currentEvaluation;
  Angle   bestCheckedAngle  = 0;
  ANGLE myAngle   = WSinfo::me->ang;
  Vector myPos    = WSinfo::me->pos;
  Vector ballPos  = WSinfo::ball->pos;
  Vector ballVel  = WSinfo::ball->vel;
  PlayerSet interferingOpps = WSinfo::valid_opponents;
  PlayerSet kickableOpps;
  for(int i=0; i<interferingOpps.num; i++)
  {
    Vector oppDistToBall = interferingOpps[i]->pos - ballPos;
    if(oppDistToBall.norm() <= interferingOpps[i]->kick_radius)
      kickableOpps.join(interferingOpps[i]);
  }
  for ( int checkAngle=0; checkAngle<360; checkAngle += 5 )
  {
    currentEvaluation = evaluateFoulDirection( (double)checkAngle,
                                                        myAngle,
                                                        myPos,
                                                        ballPos,
                                                        ballVel,
                                                        0 );       
    if ((currentEvaluation.norm() < bestEvaluation.norm()) && ballHasToBeFree)
    {
      Vector newBallPos = WSinfo::ball->pos + currentEvaluation;
      bool freeBall = true;      
      for(int j=0;j<2;j++)
      {
        for(int i=0; i<kickableOpps.num; i++)
        {
          if((kickableOpps[i]->pos - newBallPos).norm() <= kickableOpps[i]->kick_radius)
          {
            freeBall = false;
            LOG_POL(FOUL_LOG_LEVEL,<<"Foul2010: No FreeBall if I ("<<WSinfo::me->number<<") foul at angle "<<checkAngle<<"! Player "<<kickableOpps[i]->number<< " can kick the ball");
          }
        }
        newBallPos = newBallPos + currentEvaluation;
      }
      if(freeBall)
      {
    DRAW_LINE(WSinfo::ball->pos,(WSinfo::ball->pos + currentEvaluation),0000FF);
        if(bestEvaluation.getX() == 100)
        {
          bestEvaluation = currentEvaluation;
          bestCheckedAngle = checkAngle;
          LOG_POL(FOUL_LOG_LEVEL,<<"Foul2010: getMinimalBallVelAfterFoul (Need FreeBall): New bestEvaluation found: "
                  <<checkAngle<<" is "<<currentEvaluation.norm());        
        }
        else if(currentEvaluation.getX() > bestEvaluation.getX() &&
           currentEvaluation.norm() < ivMaxBallVelAfterFoul)
        {
          bestEvaluation = currentEvaluation;
          bestCheckedAngle = checkAngle;
          LOG_POL(FOUL_LOG_LEVEL,<<"Foul2010: getMinimalBallVelAfterFoul (Need FreeBall): New bestEvaluation found: "
                  <<checkAngle<<" is "<<currentEvaluation.norm());
        }
      }
    }
    else if(currentEvaluation.norm() < bestEvaluation.norm())
    {
      bestEvaluation = currentEvaluation;
      bestCheckedAngle = checkAngle;
      LOG_POL(FOUL_LOG_LEVEL,<<"Foul2010: getMinimalBallVelAfterFoul (Don't need FreeBall): Evaluation of "
              <<checkAngle<<" is "<<currentEvaluation.norm());
    }                         
  }
  if ( bestEvaluation.norm() <= ivMaxBallVelAfterFoul )
  {
    ivFoulAngle = ANGLE(DEG2RAD(bestCheckedAngle));
    DRAW_LINE(WSinfo::ball->pos,(WSinfo::ball->pos + 5*bestEvaluation),FF0000);
    LOG_POL(FOUL_LOG_LEVEL,"Foul2010: "<<WSinfo::me->number<<" getMinimalBallVelAfterFoul successfull: BEST ANGLE IS "
      <<bestCheckedAngle<<" R2D: "<<RAD2DEG(ivFoulAngle.get_value_mPI_pPI())<<" with minBallVel "<<bestEvaluation.norm());
    returnValue = true;
  }
  else
  {
    LOG_POL(FOUL_LOG_LEVEL,<<"Foul2010: getMinimalBallVelAfterFoul: No suitable "
      <<"foul angle found.");
    returnValue = false;
  }
  return returnValue;
}

//============================================================================
// evaluateAngularTackleDirection (last update 28. march 2010 by thomas huber)
//============================================================================
/**
 *  This function evaluates the prefered tackle angle for a foul.
 *  foulGoals:  0 = stop the ball
 *              1 = pass the ball into the direction of a teammate
 */
Vector
Foul2010::evaluateFoulDirection( double checkAngle,
                                 ANGLE  myAngle,
                                 Vector myPos,
                                 Vector ballPos,
                                 Vector ballVel,
                                 int    foulGoal)
{
    // foul power depends on the foul angle
    double effectiveTacklePower
      =   ServerOptions::max_back_tackle_power
        +   (ServerOptions::max_tackle_power - ServerOptions::max_back_tackle_power)
          * (fabs(180.0 - checkAngle)/180.0);
    Vector player_2_ball = ballPos - myPos;
    player_2_ball.rotate( - myAngle.get_value_mPI_pPI() );
    ANGLE player_2_ball_ANGLE = player_2_ball.ARG();
    effectiveTacklePower
      *= 1.0 - 0.5 * ( fabs( player_2_ball_ANGLE.get_value_mPI_pPI() ) / PI );
    
LOG_POL(FOUL_LOG_LEVEL,<<"Foul2010: [TACKLE-INFO] checkAngle="<<checkAngle
<<" player2ball="<<player_2_ball.arg()
<<" effPowDecay="<<(1.0 - 0.5 * ( fabs( player_2_ball_ANGLE.get_value_mPI_pPI() ) / PI ))
<<" effPower="<<effectiveTacklePower);

    // calculate the estimated ball velocity after the foul was executed
    Vector tackle_dir = Vector( myAngle + ANGLE(DEG2RAD(checkAngle)) );
    tackle_dir.normalize(   ServerOptions::tackle_power_rate 
                          * effectiveTacklePower);
    tackle_dir += ballVel;
    if ( tackle_dir.norm() > ServerOptions::ball_speed_max )
      tackle_dir.normalize( ServerOptions::ball_speed_max );
    LOG_POL(FOUL_LOG_LEVEL,<<"Foul2010: [TD] checkAngle="<<checkAngle<<" resulting ball speed "<<tackle_dir.norm());
  return tackle_dir;
}

double
Foul2010::get_maxBallVelAfterFoul()
{
  return ivMaxBallVelAfterFoul;
}

void
Foul2010::set_maxBallVelAfterFoul(double maxBallVelAfterFoul)
{
  ivMaxBallVelAfterFoul = maxBallVelAfterFoul;
}
