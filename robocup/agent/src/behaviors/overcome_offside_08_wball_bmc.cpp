#include "overcome_offside_08_wball_bmc.h"

#include <limits.h>

bool OvercomeOffside08Wball::initialized = false;

/* initialize constants */
const int OvercomeOffside08Wball::cvc_PREP_KICK = 1;
const int OvercomeOffside08Wball::cvc_TWO_STEP = 2;
const int OvercomeOffside08Wball::cvc_PASS_ANGLE_SEARCH_DELTA = 5;
const double OvercomeOffside08Wball::cvc_THREE_STEP_KEEPAWAY_CIRCLE = 3.;
const int OvercomeOffside08Wball::cvc_PASS_ANGLE_MAX_DEVIATION = 5;
const double OvercomeOffside08Wball
      ::cvc_MAX_PASS_VELOCITY_RELAXATION_UNDER_PRESSURE = 0.8;
const double OvercomeOffside08Wball::cvc_INACCAPTABLE_OOT_PASS  = 1000.0;
Vector 
  OvercomeOffside08Wball::cvPassBallPositionArray
                          [OvercomeOffside08Wball::cvc_OOT_FUTURE_BALL_POSITIONS];

OvercomeOffside08Wball::OvercomeOffside08Wball()
{
  ivpOneStepKick      = new OneStepKick();
	ivpTwoStepKick      = new OneOrTwoStepKick;
	ivpRaumPass         = new RaumPassInfo;
  ivpNeuroKickWrapper = new NeuroKickWrapper();
  ivpNeuroKick05      = new NeuroKick05();
	offside_line = 0;
	safety_margin = 0;
	offside_line_age = 1000;
}

OvercomeOffside08Wball::~OvercomeOffside08Wball()
{
  if ( ivpOneStepKick )      delete ivpOneStepKick;
	if ( ivpTwoStepKick )      delete ivpTwoStepKick;
	if ( ivpRaumPass )         delete ivpRaumPass;
  if ( ivpNeuroKickWrapper ) delete ivpNeuroKickWrapper;
  if ( ivpNeuroKick05 )      delete ivpNeuroKick05;
}

bool
OvercomeOffside08Wball::announcePass(Cmd &cmd, RaumPassInfo *lvRaumPass )
{
  Vector vel;
  if ( lvRaumPass->first_kick.set )
  {
    vel.init_polar(lvRaumPass->first_kick.vel, lvRaumPass->first_kick.ang);
  }
  else if ( lvRaumPass->second_kick.set )
  {
    vel.init_polar(lvRaumPass->second_kick.vel, lvRaumPass->second_kick.ang);
  }
  else
  {
    JTSPOL2("Error i should have announced pass but there was no pass present!");
    return false;
  }
	JTSPOL2("SETTING CMD_SAY ["<<(&cmd.cmd_say)<<"]"<< WSinfo::me->time 
    << " pass at " << WSinfo::me->time);
	cmd.cmd_say.set_pass(WSinfo::ball->pos, vel, WSinfo::me->time );
	return true;
}

double
OvercomeOffside08Wball
  ::evaluateOOTPassCorridor( PPlayer passRequester,
                             int    passAngle, //degree
                             double  offsideLine,
                             int  & icptStepGuess)
{
  if (passRequester == NULL)
  {
    TGoowLOGPOL(2,<<"OO08Wball: ERROR: Pass requester is NULL.");
    return cvc_INACCAPTABLE_OOT_PASS;
  }
  //large values are bad, small values are good
  double returnValue = 0.0;
  Vector passStartPosition = WSinfo::me->pos,
         passEndPoint;
  passEndPoint.init_polar(20.0, DEG2RAD(passAngle));
  passEndPoint += passStartPosition;
  //exclude passes with direct opps in passway
  Quadrangle2d directPassWay( passStartPosition, passEndPoint, 0.1, 8.0);
  PlayerSet opps = WSinfo::valid_opponents;
  opps.keep_players_in( directPassWay );
  if ( opps.num > 0 )
  {
    JTSPOL2("Must exclude pass direction "<<passAngle
     <<" due to "<<opps.num<<" opponents in DIRECT pass way.");
   return cvc_INACCAPTABLE_OOT_PASS;
  }  
  //ok, the pass has not been excluded ... we can continue
  TGoowLOGPOL(2,<<"OO08Wball: Evaluate pass starting at "
    <<passStartPosition<<" with angle "<<passAngle);
  //fill up future ball positions
  fillOOTPassBallPositionsArray( passStartPosition, passAngle );
  //evaluate the earliest possible interception time for me
  int smallestNumberOfIcptSteps
    = getSmallestNumberOfInterceptionSteps( passStartPosition, 
                                            passAngle,
                                            offsideLine,
                                            passRequester );
  if ( smallestNumberOfIcptSteps >= cvc_OOT_FUTURE_BALL_POSITIONS-1 )
  {
    TGoowLOGPOL(2,<<"OO08Wball: Must exclude pass direction "<<passAngle
     <<" since I would need too many steps to intercept it.");
    return cvc_INACCAPTABLE_OOT_PASS;
  }  

  //evaluate the danger produced by opponents nearby
  Quadrangle2d extendedPassWay( passStartPosition, passEndPoint, 4.0, 30.0);
  PlayerSet icptOpps = WSinfo::valid_opponents;
  icptOpps.keep_players_in( extendedPassWay );
  if (icptOpps.num > 0)
  {
    TGooLOGPOL(3,<<"OO08Noball: Pass is dangerous due to "
      <<icptOpps.num<<" opponents in the *EXTENDED* passway.");
  }
  for (int i=0; i<icptOpps.num; i++)
  {
    Vector icptOppLotfuss = Tools::get_Lotfuss( passStartPosition,
                                                passEndPoint,
                                                icptOpps[i]->pos );
    double stepsToIcptOppLotfuss
       =   icptOppLotfuss.distance(passStartPosition)
         / (OvercomeOffside08Noball::cvc_ASSUMED_OOT_PASS_SPEED*0.85);
    double icptOppWay
       = Tools::get_dist2_line( passStartPosition,
                                passEndPoint,
                                icptOpps[i]->pos );
    double estimatedOppIcptSteps
       =   (icptOppWay - icptOpps[i]->kick_radius) //subtract kick radius 
         / icptOpps[i]->speed_max; //est.speed
    if (estimatedOppIcptSteps < 0.0)
    {
      TGoowLOGPOL(2,<<"OO08Wball: Must exclude pass direction "<<passAngle
        <<" since opp "<<icptOpps[i]->number<<" is in passway.");
      return cvc_INACCAPTABLE_OOT_PASS;
    }
    double impairmentFactor = 10.0;
    double impairment
         = Tools::max( 0.0, stepsToIcptOppLotfuss - estimatedOppIcptSteps);
    if ( stepsToIcptOppLotfuss / estimatedOppIcptSteps > impairment )
      impairment = stepsToIcptOppLotfuss / estimatedOppIcptSteps;
    impairment *= impairmentFactor;
    TGooLOGPOL(3,<<"OO08Noball: Opponent "<<icptOpps[i]->number
      <<" may intercept the ball. ==> Impair pass score by "
      <<impairment<<" points (estimatedOppIcptSteps="
      <<estimatedOppIcptSteps<<", stepsToIcptOppLotfuss="
      <<stepsToIcptOppLotfuss<<").");
    returnValue += impairment;
  }
  return returnValue;
}

double
OvercomeOffside08Wball::evaluatePassRequestImportance
                        ( PPlayer passRequester, 
                          int & angleDelta,
                          double & passRelaxationSuggested )
{
  angleDelta = 0; 
  passRelaxationSuggested = OvercomeOffside08Noball::cvc_ASSUMED_OOT_PASS_SPEED;
  double returnValue = OvercomeOffside08Noball::cvc_INACCAPTABLE_OOT_PASS;
  if ( passRequester == NULL )
  {
    JTSPOL("SEVERE ERROR: NULL player as pass requester.");
    return OvercomeOffside08Noball::cvc_INACCAPTABLE_OOT_PASS;
  }
  //1. check the pass corridor
  int passAngle
    = passRequester->pass_request.pass_param_as_angle;
  OvercomeOffside08Noball::setPassReceivingPlayer( passRequester );
  bool requestedPassIsInacceptable
    = OvercomeOffside08Noball
      ::isPassTotallyInacceptable( WSinfo::me->pos,   
                                   passRequester, 
                                   offside_line,
                                   passAngle );
  int dummyIcptInfo;
  angleDelta = 0;
  returnValue 
    = evaluateOOTPassCorridor( passRequester, 
                               passAngle, 
                               offside_line, 
                               dummyIcptInfo);
  JTSPOL("Pass to "<<passRequester->number<<" with angle "<<passAngle
    <<" yields a value of "<<returnValue);
  if (    requestedPassIsInacceptable
       || returnValue >= cvc_INACCAPTABLE_OOT_PASS
       || 1  ) //##################### ZUI TEST!
  {
    JTSPOL("Pass to "<<passRequester->number<<" with angle "<<passAngle
      <<" is totally inacceptable. I try to vary the angle ");
    for ( int delta = -cvc_PASS_ANGLE_MAX_DEVIATION;
          delta <= cvc_PASS_ANGLE_MAX_DEVIATION;
          delta += cvc_PASS_ANGLE_SEARCH_DELTA )
    {
      int dummyIcptInfo;
      double adaptedPassValue
        = evaluateOOTPassCorridor( passRequester, 
                                   passAngle + delta, 
                                   offside_line, 
                                   dummyIcptInfo );
      adaptedPassValue += abs(delta); //impair value
      if (adaptedPassValue < returnValue)
      {
        returnValue = adaptedPassValue;
        angleDelta = delta;
        JTSPOL("By adapting the requested pass by "<<delta
          <<" degree, I get an improvement to "<<returnValue);
      }
    }
  }

  if (   returnValue >= cvc_INACCAPTABLE_OOT_PASS
      || returnValue >= 30.0 )
  {
    JTSPOL("Pass to "<<passRequester->number<<" with angle "<<passAngle
      <<" must be disregarded. Its value is too high: "<<returnValue);
    return cvc_INACCAPTABLE_OOT_PASS;
  }
  //check the pass time and satisfiability
  int passMustBePlayedAtGlobalTime
    =   (passRequester->pass_request.received_at-1)
      + passRequester->pass_request.pass_in_n_steps;
  //how many steps (kicks) would i need to play that pass
  double kickVelAfter1stKick, kickVelAfter2ndKick;
  int numberOfKicksINeedToPlayThatPass 
    = this->getNumberOfKicksToPlayPass
            ( OvercomeOffside08Noball::cvc_ASSUMED_OOT_PASS_SPEED,
              passAngle + angleDelta,
              kickVelAfter1stKick,
              kickVelAfter2ndKick );
  JTSPOL("The pass request from player "<<passRequester->number
    <<" requests to play that pass at t="
    <<passMustBePlayedAtGlobalTime<<". I need "<<numberOfKicksINeedToPlayThatPass
    <<" to play that pass.");
  int numberOfStepsIStillHaveToPlayThatPass
    = passMustBePlayedAtGlobalTime - WSinfo::ws->time;
  //differentiate a number of cases
  int desiredPassAngle = passAngle + angleDelta;
  double oneStepKickMaxVel  //false -> ignore is_pos_ok
    = OneStepKick::get_max_vel_in_dir( ANGLE(DEG2RAD(desiredPassAngle)), false );
  int passLatenessStandard 
    = numberOfKicksINeedToPlayThatPass - numberOfStepsIStillHaveToPlayThatPass;
  if ( passLatenessStandard < 0 )
  {
    JTSPOL2("I have more time to play the pass under consideration than I "
      <<"need to perform the kicking. => Impair value by "
      <<abs(passLatenessStandard));
    returnValue += 5.0*abs(passLatenessStandard);
  }
  else if ( passLatenessStandard == 0 )
  {
    JTSPOL2("Great: I need as many kicks for this pass as I have time.");
  }
  else
  {
    JTSPOL2("Well, when kicking I am actually "
      <<passLatenessStandard<<" step(s) too late, try a pass "
      <<"relaxation.");
    if (    numberOfKicksINeedToPlayThatPass == 1
         && passLatenessStandard == 1 )
    {
      JTSPOL2("NOPE, I don't have to relax the pass velocity because playing"
        <<" the pass requires just a single kick. => Impair pass by 1.");
      returnValue += 1.0;
    }
    else
    {
      //determine number of kicks necessary after relaxation
      int numKicksAfterRelax = numberOfKicksINeedToPlayThatPass;
      if (   kickVelAfter2ndKick
           >   cvc_MAX_PASS_VELOCITY_RELAXATION_UNDER_PRESSURE
             * OvercomeOffside08Noball::cvc_ASSUMED_OOT_PASS_SPEED )
      {
        numKicksAfterRelax = 2;
        JTSPOL2("Well, a relaxed TWO STEP kick brings about a velocity of "
          <<kickVelAfter2ndKick<<". => NICE!");
      }
      else
      { //just debug
        JTSPOL2("Well, a relaxed TWO STEP kick brings about a velocity of "
          <<kickVelAfter2ndKick<<". => NOT ENOUGH!");
      }
      int relaxedLateness 
        = numKicksAfterRelax - numberOfStepsIStillHaveToPlayThatPass;
      //check a one-step kick, too
      if ( relaxedLateness > 0 )
      {
        if (   oneStepKickMaxVel
             >   cvc_MAX_PASS_VELOCITY_RELAXATION_UNDER_PRESSURE
               * OvercomeOffside08Noball::cvc_ASSUMED_OOT_PASS_SPEED )
        { 
          numKicksAfterRelax = 1;
          JTSPOL2("Well, a relaxed ONE STEP kick brings about a velocity of "
            <<oneStepKickMaxVel<<". => NICE!");
        }
        else
        {
          JTSPOL2("Well, a relaxed ONE STEP kick brings about a velocity of "
            <<oneStepKickMaxVel<<". => NOT ENOUGH!");
        }
      }
      //check if relaxation is useful
      relaxedLateness 
        = numKicksAfterRelax - numberOfStepsIStillHaveToPlayThatPass;
      if ( relaxedLateness <= 1 )
      {
        if (numKicksAfterRelax == 2)
        { 
          passRelaxationSuggested = kickVelAfter2ndKick;
          returnValue
            *= pow(  OvercomeOffside08Noball::cvc_ASSUMED_OOT_PASS_SPEED
                   / passRelaxationSuggested,
                   3.0 );
          JTSPOL2("After relaxing the pass (2 kicks) my lateness is "
            <<relaxedLateness<<" step(s). Pass relaxation to: "
            <<passRelaxationSuggested<<". Score increases to"
            <<returnValue);
        }
        else 
        if (numKicksAfterRelax == 1)
        {
          passRelaxationSuggested = oneStepKickMaxVel;
          returnValue
            *= pow(   OvercomeOffside08Noball::cvc_ASSUMED_OOT_PASS_SPEED
                    / passRelaxationSuggested,
                    3.0 );
          JTSPOL2("After relaxing the pass (1 kick, only) my lateness is "
            <<relaxedLateness<<" step(s). Pass relaxation: "
            <<passRelaxationSuggested<<". Score increases to"
            <<returnValue);
        }
        else
        {
          JTSPOL2("Sorry, I did not manage to relax the pass.");
          passRelaxationSuggested
            = OvercomeOffside08Noball::cvc_ASSUMED_OOT_PASS_SPEED;
        }
      }
      else
      {
        JTSPOL("TOO LATE. This pass request should have been played "
          <<"until t="<<passMustBePlayedAtGlobalTime<<"! Ball would leave"
          <<" my KR in "<<numberOfKicksINeedToPlayThatPass
          <<". Even a relaxation does not help.");
        return OvercomeOffside08Noball::cvc_INACCAPTABLE_OOT_PASS;
      }
    }      
  }
  //check for too bad passes
  if (   returnValue >= cvc_INACCAPTABLE_OOT_PASS
      || returnValue >= 30.0 )
  {
    JTSPOL("Pass to "<<passRequester->number<<" with angle "<<passAngle
      <<" must be disregarded. Its value is too high: "<<returnValue);
    return cvc_INACCAPTABLE_OOT_PASS;
  }
  //final value / return value
  JTSPOL("The pass request from player "<<passRequester->number
    <<" is evaluated with "<<returnValue<<" points.");
  return returnValue;
}

void
OvercomeOffside08Wball
  ::fillOOTPassBallPositionsArray( Vector passStartPosition,
                                   int    passAngle )
{
  Vector currentBallPos = passStartPosition,
         ballVel;
  ballVel.init_polar( OvercomeOffside08Noball::cvc_ASSUMED_OOT_PASS_SPEED, 
                      DEG2RAD(passAngle) );
  for ( int i=0; i < cvc_OOT_FUTURE_BALL_POSITIONS; i++ )
  {
    cvPassBallPositionArray[i] = currentBallPos;
    currentBallPos += ballVel;
    ballVel *= ServerOptions::ball_decay;
  }
}

bool
OvercomeOffside08Wball::get_cmd(Cmd & cmd)
{
  bool returnValue;
  //update offside line
	if ( offside_line_age != WSinfo::me->time )
	{
		update_offside_line(offside_line, safety_margin);
		offside_line_age = WSinfo::me->time;
	}

  if ( ivpRaumPass->receiver == NULL )
  {
    // if we arrive here there actually was no urgent pass 
    // that should never happen though  
    JTSPOL("ERROR: There was no urgent pass but i was in get_cmd. WHY???");
    return false;
  }

  //set the chosen pass
  ivpRaumPass->isPassChosen = true;
  
  //check for a relaxed (one step) kick
  if (   ivpRaumPass->first_kick.vel 
       < OvercomeOffside08Noball::cvc_ASSUMED_OOT_PASS_SPEED )
  {
    JTSPOL2("EXEC: Play a relaxed pass (v="
      <<ivpRaumPass->first_kick.vel<<") using OneOrTwoStepKick.");
  }
  else
  {
    JTSPOL2("EXEC: Play a NON-relaxed pass (v="
      <<ivpRaumPass->first_kick.vel<<") using OneOrTwoStepKick or NeuroKick.");
  }

  ivpOneStepKick->reset_state();
  ivpOneStepKick->kick_in_dir_with_initial_vel(ivpRaumPass->first_kick.vel, 
                                               ivpRaumPass->first_kick.ang);
  JTSPOL2("EXEC: Call OneStepKick with vel="<<ivpRaumPass->first_kick.vel
    <<" and ang="<<ivpRaumPass->first_kick.ang.get_value());
  double resultingVelocityOneStepKick;
  ivpOneStepKick->get_vel( resultingVelocityOneStepKick );
  JTSPOL("EXEC: INFO: The result of OneStepKick is a pass with "
    <<" with resultingVelocityOneStepKick="<<resultingVelocityOneStepKick);

  ivpTwoStepKick->reset_state();
  ivpTwoStepKick->kick_in_dir_with_initial_vel(ivpRaumPass->first_kick.vel, 
                                               ivpRaumPass->first_kick.ang);
  JTSPOL2("EXEC: Call TwoStepKick with vel="<<ivpRaumPass->first_kick.vel
    <<" and ang="<<ivpRaumPass->first_kick.ang.get_value());
  double resultingVelocityOfKick1, resultingVelocityOfKick2;
  ivpTwoStepKick->get_vel( resultingVelocityOfKick1, resultingVelocityOfKick2 );
  JTSPOL("EXEC: INFO: The result of OneOrTwoStepKick is a pass with "
    <<" with resultingVelocityOfKick1="<<resultingVelocityOfKick1
    <<" and resultingVelocityOfKick2="<<resultingVelocityOfKick2);
  Vector drawingShotVector( ivpRaumPass->first_kick.ang );
  if ( fabs( resultingVelocityOneStepKick - ivpRaumPass->first_kick.vel ) < 0.1  )
  {
    //one step kick manages it :-)))
    returnValue = ivpOneStepKick->get_cmd( cmd );
    JTSPOL("EXEC: INFO: The result of OneStepKick satisfying. Do it."
      <<" Result of get_cmd is: "<<returnValue);
    drawingShotVector.normalize( 10.0*resultingVelocityOneStepKick );
    DRAW_LINE( WSinfo::me->pos,
               (WSinfo::me->pos+drawingShotVector),
               "ffbbbb"  );
    if ( returnValue == false )
    {
      JTSPOL("EXEC: OneStepKick returns false, which means that the resulting"
        <<" ball pos max be NOT ok. We override that and return true.")
      returnValue = true;
    }
    //announce pass
      //it is the final kick
      announcePass( cmd, ivpRaumPass );
  }
  else
  if (    fabs( resultingVelocityOfKick1 - ivpRaumPass->first_kick.vel ) < 0.1
       || fabs( resultingVelocityOfKick2 - ivpRaumPass->first_kick.vel ) < 0.1 )
  {
    //two step kicks manages it :-)
    returnValue = ivpTwoStepKick->get_cmd( cmd );
    JTSPOL("EXEC: INFO: The result of OneOrTwoStepKick satisfying. Do it."
      <<" Result of get_cmd is: "<<returnValue);
    drawingShotVector.normalize( 10.0*resultingVelocityOfKick2 );
    DRAW_LINE( WSinfo::me->pos,
               (WSinfo::me->pos+drawingShotVector),
               "ff7777"  );
    //announce pass
    if (fabs( resultingVelocityOfKick1 - ivpRaumPass->first_kick.vel ) < 0.2) 
      //it is the final kick
      announcePass( cmd, ivpRaumPass );
  }
  else
  {
    ivpNeuroKick05->kick_in_dir_with_initial_vel(ivpRaumPass->first_kick.vel, 
                                               ivpRaumPass->first_kick.ang);
    returnValue = ivpNeuroKick05->get_cmd( cmd );
    JTSPOL("EXEC: NEURO KICK MUST BE USED ("<<returnValue<<")!");
    drawingShotVector.normalize( 10.0*ivpRaumPass->first_kick.vel );
    DRAW_LINE( WSinfo::me->pos,
               (WSinfo::me->pos+drawingShotVector),
               "ff3333"  );
  }
  return returnValue;  
}

int
OvercomeOffside08Wball::getNumberOfKicksToPlayPass
                        ( double   targetVel,
                          int      angInDegree,
                          double & resultingVelocityOfKick1,
                          double & resultingVelocityOfKick2 )
{
  int returnValue;
  ivpTwoStepKick->reset_state();
  ivpTwoStepKick->kick_in_dir_with_initial_vel( targetVel, 
                                                ANGLE( DEG2RAD(angInDegree) ) );
  JTSPOL2("Call TwoStepKick with vel="<<targetVel
    <<" and ang="<<angInDegree);
  ivpTwoStepKick->get_vel( resultingVelocityOfKick1, resultingVelocityOfKick2 );

  if ( fabs( targetVel - resultingVelocityOfKick1 ) < 0.05 )
    returnValue = 1;
  else
  if ( fabs( targetVel - resultingVelocityOfKick2 ) < 0.05 )
    returnValue = 2;
  else
    returnValue = 3;
  JTSPOL2("INFO: ivpTwoStepKick needs "<<returnValue
    <<" kicks to accelarate the ball to"
    <<" v="<<targetVel<<" into ang="<<angInDegree<<" (v1="
    <<resultingVelocityOfKick1<<", v2="<<resultingVelocityOfKick2<<")");
  return returnValue;
}



int
OvercomeOffside08Wball
  ::getSmallestNumberOfInterceptionSteps( Vector  passStartPosition,
                                          int     passAngle,
                                          double  offsideLine,
                                          PPlayer passRequester )
{
  int returnValue = cvc_OOT_FUTURE_BALL_POSITIONS-1;
  if (passRequester==NULL) return returnValue;   
  for (int i=0; i<cvc_OOT_FUTURE_BALL_POSITIONS; i++)
  {
    //intercept passes only behind the offside line
    if ( cvPassBallPositionArray[i].getX() < offsideLine )
      continue;
    //intercept passes only using a way that goes forward
    Vector runWayVector =   cvPassBallPositionArray[i] 
                          - passRequester->pos;
    if (   fabs(runWayVector.ARG().get_value_mPI_pPI()) 
         > 50.0*(PI/180.0) )
      continue;
    double sprintingPlayerWayToGo = passRequester->pos
                                   .distance( cvPassBallPositionArray[i] );
    double sprintingPlayerRequiredStepsToGo
      =   sprintingPlayerWayToGo 
        / passRequester->speed_max;
    if (   sprintingPlayerRequiredStepsToGo < (double)returnValue
        && sprintingPlayerRequiredStepsToGo < i)
    {
      TGoowLOGPOL(3,<<"DEBUG: i="<<i<<": from passSt="<<passStartPosition
      <<" to ball in "<<i<<" steps at thenPos="<<cvPassBallPositionArray[i]
      <<" where i need "<<sprintingPlayerRequiredStepsToGo<<" for a distance"
      <<" of "<<sprintingPlayerWayToGo<<"m");
      returnValue = (int)sprintingPlayerRequiredStepsToGo + 1;
    }
  }
  return returnValue;
}

bool
OvercomeOffside08Wball::init(char const * conf_file, int argc, char const* const* argv)
{
	if ( initialized )
		return true;
	initialized 
    = 1 && OneOrTwoStepKick::init(conf_file, argc, argv)
        && OneStepKick::init(conf_file, argc, argv)
        && NeuroKick05::init(conf_file, argc, argv);
	return initialized;
}

bool
OvercomeOffside08Wball::isOOTPassPossible() 
{
  //set standard var values
  ivpRaumPass->receiver = NULL;
  //update offside line
	if ( offside_line_age != WSinfo::me->time )
	{
		update_offside_line(offside_line, safety_margin);
		offside_line_age = WSinfo::me->time;
	}
  //update potential pass receivers
  ivPassPlayers = WSinfo::valid_teammates_without_me;
  //first clear out all very old pass requests from our possible pass receivers
  ivPassPlayers.keep_players_with_recent_pass_requests(7);
  JTSPOL("I got "<<ivPassPlayers.num<<" players with recent"
    <<" pass requests.");
  //debug
  for (int i=0; i<ivPassPlayers.num; i++)
  {
    JTSPOL2("Pass request from player "<<ivPassPlayers[i]->number
      <<": received_at="<<ivPassPlayers[i]->pass_request.received_at
      <<", in_n_steps="<<ivPassPlayers[i]->pass_request.pass_in_n_steps
      <<", angle="<<ivPassPlayers[i]->pass_request.pass_param_as_angle
      <<", valid="<<ivPassPlayers[i]->pass_request.valid);
    char printString[30];
    sprintf(printString,"%d:%d(%d)",ivPassPlayers[i]->pass_request.received_at,
     ivPassPlayers[i]->pass_request.pass_in_n_steps,
     ivPassPlayers[i]->pass_request.pass_param_as_angle);
    TGoowLOGPOL(0,<<_2D<<VL2D(ivPassPlayers[i]->pos,
      WSinfo::me->pos,"005500"));
    if (ivPassPlayers[i]->pass_request.valid)
    {
      TGoowLOGPOL(0,<<_2D<<VSTRING2D(0.5*(ivPassPlayers[i]->pos+WSinfo::me->pos),
        printString,"55aa55"));
    }
    else
    {
      TGoowLOGPOL(0,<<_2D<<VSTRING2D(0.5*(ivPassPlayers[i]->pos+WSinfo::me->pos),
        printString,"005500"));
    }
  }
  //now, clear out pass requests that are outdated insofar as they have been
  //received, but that - at a later point of time - no further request from
  //the same teammate has come in
  ivPassPlayers.keep_players_with_valid_pass_requests();
  ivPassPlayers.keep_players_with_urgent_pass_requests(5);
  JTSPOL("Out of them, "<<ivPassPlayers.num<<" ought to be played"
    <<" within the next few (5) time steps.");
  //general conditions
  bool ootGen_1 = WSinfo::is_ball_kickable(),
       ootGen_2 = WSinfo::me->pos.getX() > -5.0,
       ootGen_3 = offside_line - WSinfo::me->pos.getX() < 15.0,
       ootGen_4 = WSinfo::me->pos.getX() < FIELD_BORDER_X - PENALTY_AREA_LENGTH - 3.0;
  if ( ootGen_1 && ootGen_2 && ootGen_3 && ootGen_4 )
  {
    JTSPOL("General conditions might allow for an OOT pass.");
  }
  else
  {
    JTSPOL("General conditions DO NOT allow for an OOT pass (ol="
      <<offside_line<<"): "
      <<ootGen_1 << ootGen_2 << ootGen_3 << ootGen_4);
    return false;
  }

  //find most important pass request
  double valueOfMostImportantPassRequest
          = OvercomeOffside08Noball::cvc_INACCAPTABLE_OOT_PASS,
        currentImportanceOfPassRequest;
  PPlayer mostImportantPassRequestingTeammate = NULL;
  int   angleDelta = 0;
  double passRelaxationSuggested
          = OvercomeOffside08Noball::cvc_ASSUMED_OOT_PASS_SPEED,
        shallBestPassRequestBeRelaxed 
          = OvercomeOffside08Noball::cvc_ASSUMED_OOT_PASS_SPEED;
  for (int i=0; i<ivPassPlayers.num; i++)
  {
    currentImportanceOfPassRequest
      = this->evaluatePassRequestImportance( ivPassPlayers[i],
                                             angleDelta,
                                             passRelaxationSuggested );
    if (currentImportanceOfPassRequest < valueOfMostImportantPassRequest)
    {
      valueOfMostImportantPassRequest = currentImportanceOfPassRequest;
      mostImportantPassRequestingTeammate = ivPassPlayers[i];
      shallBestPassRequestBeRelaxed = passRelaxationSuggested;
    }
  }

  //do we have to act now?
  if ( mostImportantPassRequestingTeammate )
  {
    JTSPOL("DECISION: The most important pass requesting teammate "
      <<"has number "<<mostImportantPassRequestingTeammate->number<<".");
    ivPassPlayers.keep_players_with_urgent_pass_requests(2);
    JTSPOL("DECISION: Well, there are " << ivPassPlayers.num 
      << " urgent pass requests.");
    if ( ivPassPlayers
         .get_player_by_number(mostImportantPassRequestingTeammate->number) )
    {
      JTSPOL("DECISION: The most important pass requester is among the"
        <<" urgent pass requesters. => Serve it.");
      ivpRaumPass->first_kick.vel 
        = OvercomeOffside08Noball::cvc_ASSUMED_OOT_PASS_SPEED;   
      if (   fabs(  shallBestPassRequestBeRelaxed
                  - OvercomeOffside08Noball::cvc_ASSUMED_OOT_PASS_SPEED )
           > 0.001 )
      {
        JTSPOL("DECISION: NOTE that a relaxed pass (v=)"
          <<shallBestPassRequestBeRelaxed<<" will be played.");
        ivpRaumPass->first_kick.vel = shallBestPassRequestBeRelaxed;
      }
      double resultingPassAngle = (double)mostImportantPassRequestingTeammate
                                        ->pass_request.pass_param_as_angle
                                 + (double)angleDelta;
      ivpRaumPass->first_kick.ang
        = ANGLE( DEG2RAD( resultingPassAngle ) );
      JTSPOL2("Setting a kick with angle "
        <<resultingPassAngle<<" (="
        <<ivpRaumPass->first_kick.ang.get_value()
        <<") and vel="<<ivpRaumPass->first_kick.vel);
      ivpRaumPass->first_kick.set = true;
      ivpRaumPass->receiver = mostImportantPassRequestingTeammate;
      JTSPOL("DECISION: YEP, I decide for serving the request issued by"
        <<" player number"<<mostImportantPassRequestingTeammate->number);
      TGoowLOGPOL(0,<<_2D<<L2D(mostImportantPassRequestingTeammate->pos.getX(),
        mostImportantPassRequestingTeammate->pos.getY(),
        WSinfo::me->pos.getX(),WSinfo::me->pos.getY(),"005500"));
      //request to remain attention to the pass requester, FORCING MODE!
      Tools::set_attention_to_request( ivpRaumPass->receiver->number, true );
      return true;
    }
    else
    {
      JTSPOL("DECISION: NOPE: The most important pass requester is NOT"
        <<"among the urgent pass requesters. => Don't serve it, but keep"
        <<" an EAR on it: Set attention to request "
        <<mostImportantPassRequestingTeammate->number);
      Tools::set_attention_to_request
             (mostImportantPassRequestingTeammate->number);
      return false;
    }
  }
	else
  {
    //there is none, recheck next cycle :)
    JTSPOL("DECISION: NOPE, I found no possible OOT pass and no"
      <<" most important pass requesting teammate.");
    int suggestedAttentionToTeammateNumber
      = this->suggestAttentionToTeammateNumber( offside_line );
    if ( suggestedAttentionToTeammateNumber != -1 )
    {
      JTSPOL("DECISION: Summarizing, it may be a good idea to set attention"
        <<" to teammate "<<suggestedAttentionToTeammateNumber);
      Tools::set_attention_to_request(suggestedAttentionToTeammateNumber);
    }
 
    return false;
  }
  	
	JTSPOL("WARNING: This point of code should not be reached.");
	return false;
}

int
OvercomeOffside08Wball::suggestAttentionToTeammateNumber
                        ( double   offsideLine )
{
  //att check 1: if there is an important teammate who we have just seen
  //             and who may be worth listening to
  PlayerSet possAttentTeammates = WSinfo::valid_teammates_without_me;
  possAttentTeammates.keep_players_with_max_age( 0 );
  possAttentTeammates.keep_and_sort_players_by_x_from_right(4);
  possAttentTeammates.remove( possAttentTeammates );
  for (int i=0; i<possAttentTeammates.num; i++)
  {
    JTSPOL("ATT-CHECK: INFO: check tmm "<<possAttentTeammates[i]->number
      <<" for attention.");
    if (    fabs( possAttentTeammates[i]->ang.get_value_mPI_pPI() ) < 0.25*PI
         && possAttentTeammates[i]->vel.getX() > 0.3
         && offsideLine - possAttentTeammates[i]->pos.getX() < 10.0)
    {
      JTSPOL("ATT-CHECK: Well, it may be a good idea to set attention"
        <<" to teammate "<<possAttentTeammates[i]->number);
      return possAttentTeammates[i]->number;
    }
  }

  //att check 2: if there is an important teammate to listen to
  possAttentTeammates = WSinfo::valid_teammates_without_me;
  possAttentTeammates.keep_and_sort_players_by_x_from_right(4);
  possAttentTeammates.keep_players_in_circle(WSinfo::me->pos, 25.0);
  possAttentTeammates.keep_players_in_halfplane
                      ( Vector( offsideLine-10.0, 0.0 ), Vector(1.0, 0.0) );
  int bestValue = INT_MAX, bestTeammate = -1, currValue = 0;
  for (int i=0; i<possAttentTeammates.num; i++)
  { //minimize
    currValue = (int)(offsideLine - possAttentTeammates[i]->pos.getX());
    currValue -= WSmemory::last_attentionto_to(possAttentTeammates[i]->number);
    currValue += (int)(possAttentTeammates[i]->pos
                          .distance(WSinfo::me->pos) / 3.0);
#if LOGGING && BASIC_LOGGING
      bool doContinue = false;
      if (   (int)(offsideLine - possAttentTeammates[i]->pos.getX())
           > WSmemory::last_attentionto_to(possAttentTeammates[i]->number ) + 3 )
        doContinue = true;
#endif
      JTSPOL2("ATT-CHECK: Attention to teammate "
        <<possAttentTeammates[i]->number<<" eval "<<currValue<<"  "
        <<(doContinue?"[DISREGARD]":"[OK]"));
      if ( currValue < bestValue )
      {
        bestValue = currValue;
        bestTeammate = possAttentTeammates[i]->number;
      }
  }
  return bestTeammate;
}

#define DUMMY_OOT_NOBALL_SEPARATOR
//............................................................................
//----------------------------------------------------------------------------
//============================================================================
//############################################################################
//############################################################################
//============================================================================
//----------------------------------------------------------------------------
//............................................................................

double
OvercomeOffside08Wball::evaluatePass( PPlayer passRequester,
                                      int     passAngle )
{
  if ( passRequester == NULL )
  {
    JTSPOL("SEVERE ERROR: NULL player as pass requester.");
    return OvercomeOffside08Noball::cvc_INACCAPTABLE_OOT_PASS;
  }
  OvercomeOffside08Noball::setPassReceivingPlayer( passRequester );
  int dummyIcptInfo;
  return 
    OvercomeOffside08Noball
      ::evaluateOOTPassCorridor( WSinfo::me->pos, 
                                 passAngle, 
                                 offside_line, 
                                 dummyIcptInfo );
} 
 


bool
OvercomeOffside08Wball::getOptPassSequence( PPlayer       bestPlayer, 
                                            RaumPassInfo *lvPassSequence, 
                                            double         angle_diff)
{
	// first check if the originally wanted pass can be played
	ANGLE param_as_ANGLE = ANGLE(double(bestPlayer->pass_request.pass_param_as_angle));
	if ( getPassSequence(bestPlayer, lvPassSequence, param_as_ANGLE) )
		return true;
	// otherwise scan for alternatives
	ANGLE ang_dec = param_as_ANGLE;
	ANGLE ang_inc = param_as_ANGLE;
	JTSPOL2("now searching for alternatives to requested angle");
	for (int i = 0; i < angle_diff; i += cvc_PASS_ANGLE_SEARCH_DELTA)
	{
		ang_dec -= ANGLE(angle_diff);
		ang_inc += ANGLE(angle_diff);
		if (   getPassSequence(bestPlayer, lvPassSequence, ang_dec) 
			  || getPassSequence(bestPlayer, lvPassSequence, ang_inc)
		)
		{
			return true;
		} 
	}
	return false;
}

bool
OvercomeOffside08Wball::getPassSequence( PPlayer       bestPlayer, 
                                         RaumPassInfo *lvPassSequence, 
                                         const ANGLE  &passDir )
{
	if( ! bestPlayer || ! lvPassSequence )
	{
		return false;
	}
	ivpTwoStepKick->reset_state();
	double best_vel = OvercomeOffside08Noball::cvc_ASSUMED_OOT_PASS_SPEED;
  if (   ivpTwoStepKick->is_kick_possible(best_vel, passDir) == 2
      && lookAheadPassWayStillFree(bestPlayer, WSinfo::ball->pos,  true)  
     )
	{ // if a twostep kick reaches the goal in exactly 2 steps we can execute it
		JTSPOL2("I can execute the pass to player " << bestPlayer->number 
				<< " using 2stepPass");
		lvPassSequence->first_kick.type = cvc_TWO_STEP;
		lvPassSequence->first_kick.ang = passDir;
		lvPassSequence->first_kick.vel 
      = OvercomeOffside08Noball::cvc_ASSUMED_OOT_PASS_SPEED; 
		lvPassSequence->first_kick.set = true;
		lvPassSequence->second_kick.set = false; // just for security reasons
		return true;
	}
	// otherwise check if we can make it with an additional onestep kick
	// but then we want that there is no opponent near us
	PlayerSet oppNear = WSinfo::alive_opponents;
	oppNear.keep_players_in_circle( WSinfo::me->pos, 
                                  cvc_THREE_STEP_KEEPAWAY_CIRCLE);
	if (   WSinfo::is_ball_kickable()
      && oppNear.num == 0
     )
	{
		return getPassSequenceWithOneAdditionalKick(bestPlayer, lvPassSequence, passDir);
	} 
	// if we arrive here it is simply not possible to play the pass
	return false;
}

bool
OvercomeOffside08Wball::getPassSequenceWithOneAdditionalKick
                          ( PPlayer bestPlayer, 
                            RaumPassInfo *lvPassSequence, 
                            const ANGLE &passDir)
{
	Cmd test_cmd;
	Vector next_my_pos;
	Vector next_my_vel;
	ANGLE  next_my_ang;
	Vector next_ball_pos;
	Vector next_ball_vel;
  double  best_vel = OvercomeOffside08Noball
                    ::cvc_ASSUMED_OOT_PASS_SPEED;
  ivpNeuroKick05->kick_in_dir_with_initial_vel( best_vel, passDir );
  if ( ivpNeuroKick05->get_cmd( test_cmd ) == false )
  {
    JTSPOL("NeuroKick05 fails.");
    return false;
  }
                    
  Tools::model_cmd_main( WSinfo::me->pos,
                         WSinfo::me->vel, 
                         WSinfo::me->ang,  
                         WSinfo::ball->pos, 
                         WSinfo::ball->vel, 
                         test_cmd.cmd_body, 
                         next_my_pos, 
                         next_my_vel, 
			  					       next_my_ang, 
                         next_ball_pos,
                         next_ball_vel);
                         
	if ( next_ball_pos.distance(WSinfo::me->pos) >= WSinfo::me->kick_radius )
  {
    double kickP, kickA;
    test_cmd.cmd_body.get_kick(kickP,kickA);
    JTSPOL("SEVER ERROR: NeuroKick05 fails, lets ball get out of KR"
      <<" (command=KICK:"<<kickP<<"@"<<kickA<<").");
    return false;
  }
  else
  { // ball is still in kick range
		ivpTwoStepKick->reset_state();
		ivpTwoStepKick->set_state( next_my_pos, 
                               next_my_vel, 
                               next_my_ang, 
                               next_ball_pos, 
                               next_ball_vel);
		if (   ivpTwoStepKick->is_kick_possible(best_vel, passDir) > 0 
				&& lookAheadPassWayStillFree(bestPlayer, next_ball_pos, true) 
       )
    {	// if a kick is possible we compute ball look ahead
			// and check if the resulting corridor is completely free
			// if so we can safely execute the pass
			JTSPOL2("I can execute the pass to player " << bestPlayer->number 
		    << " using 3 passes");
			lvPassSequence->first_kick.type = cvc_PREP_KICK;
			lvPassSequence->first_kick.ang = passDir;
			lvPassSequence->first_kick.vel = best_vel;
			lvPassSequence->first_kick.set = true;
			// second kick -> the real pass
			lvPassSequence->second_kick.type = cvc_TWO_STEP;
			lvPassSequence->second_kick.ang = passDir;
			lvPassSequence->second_kick.vel 
        = OvercomeOffside08Noball::cvc_ASSUMED_OOT_PASS_SPEED;
			lvPassSequence->second_kick.set = true;
			return true;																
		}
    else
    {
      JTSPOL("SEVER ERROR: NeuroKick05 succeeds, but kicking requires"
        <<" more than 3 steps. -> return false");
      return false;
    }
	}
}

bool 
OvercomeOffside08Wball::lookAheadPassWayStillFree( PPlayer       bestPlayer,
                                                   const Vector &ballPos, 
                                                   bool          twoStepSet)
{
	if ( twoStepSet )
	{
		double first_kick_vel;
		double second_kick_vel;
		ivpTwoStepKick->get_vel(first_kick_vel, second_kick_vel);
		if (   ivpTwoStepKick->need_two_steps() 
			&& passWayCompletelyFree(bestPlayer, ballPos, 1.5) 
		)
		{
				return true;
		}
		else if (  passWayCompletelyFree(bestPlayer, ballPos, 1.5) 
		)
		{
			return true;
		}
	}
	return false;
}

#if 0 
// do not use right now!
bool 
OvercomeOffside08::will_selected_player_be_best_interceptor( const PPlayer selected_player, 
															 RaumPassInfo *lvRaumPassInfo,
															 const Vector &intended_pos, // specify seperately because we might decide to play a slightly altered pass 
															 const Vector &additional_vel,
															 int time_to_go )
{
	if(! selected_player ) 
		return false; // check that we are not passed NULL 
	/* initialize variables */
	PlayerSet pset_tmp = WSinfo::valid_opponents;
	pset_tmp.keep_players_in_rectangle(Vector(WSinfo::me->pos.x , FIELD_BORDER_Y),Vector(FIELD_BORDER_X, -FIELD_BORDER_Y));
	WS::Player offside_killer_p;
	offside_killer_p = *selected_player; // JTS check this! does copying the whole struct work ?
	PPlayer offside_killer = &offside_killer_p; // get the pointer for more convenient usage
	pset_tmp.join(offside_killer);
	int numberOfConsideredInterceptors = pset_tmp.num;
	InterceptResult intercept_res[numberOfConsideredInterceptors];
	lvRaumPassInfo->best_opp_num = -1;
	lvRaumPassInfo->offside_killer_steps = 1000;
	lvRaumPassInfo->best_opp_steps = 1000;
  	
  	// now alter offside_killer informations
  	if ( ! get_estimated_position(offside_killer, intended_pos , time_to_go) )
  	{
  		return false; // if we cannot get any estimation something went wrong  
  	}
  	//JTSPOL2("intercept info requested assuming ball_vel " 
  	//        << WSinfo::ball->vel + additional_vel);
  	// JTS for now assume the ball pos stays the same and the ball has desired vel
  	pset_tmp.keep_and_sort_best_interceptors_with_intercept_behavior
  			(numberOfConsideredInterceptors, WSinfo::ball->pos, 
  			 additional_vel, intercept_res);
  	for (int i=0; i<numberOfConsideredInterceptors; i++)
  	{
    	if (    lvRaumPassInfo->best_opp_num == -1
       		 && pset_tmp[i]->team == HIS_TEAM
       		 && pset_tmp[i]->age <= 3 )
    	{
      		lvRaumPassInfo->best_opp_num = pset_tmp[i]->number;
      		lvRaumPassInfo->best_opp_steps = intercept_res[i].time;
    	}
    	if (	lvRaumPassInfo->best_opp_num != -1
    		 && pset_tmp[i]->team == HIS_TEAM
    		 && pset_tmp[i]->number != 0 // JTS do we want to consider goalie as interceptor ?
    		 && intercept_res[i].time <= lvRaumPassInfo->best_opp_steps)
    	{
    		lvRaumPassInfo->best_opp_num = pset_tmp[i]->number;
      		lvRaumPassInfo->best_opp_steps = intercept_res[i].time;
    	}
    	if (    pset_tmp[i]->team == MY_TEAM
    		 && pset_tmp[i]->number == offside_killer->number)
    	{	// only consider the offside killer
    		lvRaumPassInfo->offside_killer_steps = intercept_res[i].time;
    		// maybe we want to assume that our teammate is actually faster
    		// because he knows about the pass
    		if ( 1 || !lvRaumPassInfo->is_pass_communicated ) //JTS HACK
    		{
    			lvRaumPassInfo->offside_killer_steps -= 3;
    		}
		}
    }
    // if the best opponent needs very few steps it is a good idea
    // not to assume that the offside killer is 3 steps faster
    if (  lvRaumPassInfo->best_opp_steps <= 4 )
    	lvRaumPassInfo->offside_killer_steps += 3;
    JTSPOL2("Intercept Info offside_killer_steps " << lvRaumPassInfo->offside_killer_steps 
    	    << " opp " << lvRaumPassInfo->best_opp_num << " opp_steps " << lvRaumPassInfo->best_opp_steps);
    if ( lvRaumPassInfo->offside_killer_steps <= lvRaumPassInfo->best_opp_steps )
    {
    	lvRaumPassInfo->offside_killer = WSinfo::get_teammate_by_number(offside_killer->number);
    	return true;
    }
    else
    {
    	return false;
    }
}

#endif

bool
OvercomeOffside08Wball::passIsAbsolutelyDeadly(PPlayer bestPlayer)
{
	// JTS TG says we need to do this in order to have a variable in Noball behavior set
	if ( OvercomeOffside08Noball::isPassTotallyInacceptable(WSinfo::me, bestPlayer, offside_line, bestPlayer->pass_request.pass_param_as_angle) )
		return false;
	int pass_receiver_step_guess = 1000;
	double spaceWonBestPlayer
    = OvercomeOffside08Noball
      ::evaluateOOTPassCorridor( WSinfo::me->pos, 
                                 bestPlayer->pass_request.pass_param, 
                                 offside_line, pass_receiver_step_guess);
	if (   WSinfo::me->time - ( bestPlayer->pass_request.received_at - 1 
						  + bestPlayer->pass_request.pass_in_n_steps )  
		   <= 3
		&& WSinfo::me->time - bestPlayer->pass_request.received_at <= 4
		// && bestPlayer->pos.x > cvc_DEADLY_POS // JTS maybe this is not a good criteria
		&& passWayCompletelyFree(bestPlayer, WSinfo::ball->pos,  2.)
		&& spaceWonBestPlayer < 1000.0 //TODO: schwellwert einstellen
	)
	{
		return true;	
	}
	else
	{
		return false;
	}
}

bool
OvercomeOffside08Wball::passWayCompletelyFree(PPlayer bestPlayer, const Vector &ball_pos, double corrSize)
{
	Vector steigung;
	PlayerSet oppInCorr = WSinfo::valid_opponents;
	steigung.init_polar(1., DEG2RAD(bestPlayer->pass_request.pass_param_as_angle));
	Quadrangle2d passCorr(ball_pos, Vector(offside_line, Tools::point_on_line( steigung , ball_pos, offside_line).getY()), corrSize, corrSize * 1.5);
	oppInCorr.keep_players_in(passCorr);
	JTSPOL2("Check if passWayCompletelyFree num opp: " << oppInCorr.num);
	if( oppInCorr.num > 0 )
		return false;
	else
		return true;
}
bool
OvercomeOffside08Wball::thereIsNoBetterPassComing(PPlayer bestPlayer, 
                                                  int     bestPlayerPositionInIvPassPlayers)
{
	if (    (    bestPlayer 
            && ivPassPlayers.num - 1 == bestPlayerPositionInIvPassPlayers ) 
       || passIsAbsolutelyDeadly(bestPlayer)
	)
	{ // if there is no other pass or the pass is very good
		return true;
	}

	int pass_receiver_step_guess = 1000;
	double spaceWonBestPlayer
    = OvercomeOffside08Noball
      ::evaluateOOTPassCorridor( WSinfo::me->pos, 
                                 bestPlayer->pass_request.pass_param, 
                                 offside_line, 
                                 pass_receiver_step_guess);
	// otherwise check all passes in our WSpset
	for (int i = bestPlayerPositionInIvPassPlayers; i < ivPassPlayers.num; i++)
	{
		if (   ivPassPlayers[i] 
        && ! OvercomeOffside08Noball
               ::isPassTotallyInacceptable( WSinfo::me, 
                                            ivPassPlayers[i], 
                                            offside_line, 
                                            bestPlayer->pass_request.pass_param_as_angle)
		    && OvercomeOffside08Noball
             ::evaluateOOTPassCorridor( WSinfo::me->pos, 
                                        ivPassPlayers[i]->pass_request.pass_param, 
                                        offside_line, 
                                        pass_receiver_step_guess)
		    	< spaceWonBestPlayer
		    // JTS we definitely need more finegrained rules here!
		)
		{
			return false;
		}
	}
	// if we arrive here there should not be a better pass coming
	return true;
} 

bool
OvercomeOffside08Wball::followPassSequence(RaumPassInfo *passSequence, Cmd & cmd)
{
	if ( !passSequence )
		return false;
	
	announcePass(cmd, passSequence );

	
	if (   passSequence->first_kick.set 
      && passSequence->first_kick.type == cvc_TWO_STEP
      && !passSequence->second_kick.set
	   )//TODO: check ob passkorridor frei
	{	// if only one pass is set it should be a real twoStepKick
		ivpTwoStepKick->reset_state();
		ivpTwoStepKick->kick_in_dir_with_initial_vel(passSequence->first_kick.vel , 
                                                 passSequence->first_kick.ang);
		JTSPOL2("executing final pass sequence!")
#if 1
		Vector test;
		test.init_polar( 1., passSequence->first_kick.ang);
		DRAW_LINE(WSinfo::ball->pos, WSinfo::ball->pos + test , "fff600");
#endif
		return ivpTwoStepKick->get_cmd(cmd);
	}	
	if (    passSequence->first_kick.set 
		   && passSequence->first_kick.type == cvc_PREP_KICK
		   && passSequence->second_kick.set
	   )
	{
    ivpNeuroKick05
      ->kick_in_dir_with_initial_vel( passSequence->first_kick.vel, 
                                      passSequence->first_kick.ang );
		JTSPOL2("executing preparation pass using NeuroKick05");
#if 1
		Vector test;
		test.init_polar(1., passSequence->first_kick.ang);
		DRAW_LINE(WSinfo::ball->pos, WSinfo::ball->pos + test , "fff600");
#endif
		passSequence->first_kick = passSequence->second_kick;
		passSequence->second_kick.set = false;
    return  ivpNeuroKick05->get_cmd( cmd );
	}
		
	// if we arrive here something was wrong and we cannot do anything
  JTSPOL2("ERROR: point of code should not be reached");
	return false;
}

void
OvercomeOffside08Wball::update_offside_line(double &offside_line, double &safetyMargin)
{
	/* NOTE: WE DO NOT consider safety margin in an absolutely correct manner here
	 * 	     because the pass receiving players should check that
	 * 		 code is taken from NoBall05
	 */
	double hisTeamOffsideLine = WSinfo::his_team_pos_of_offside_line(),
        hisTeamOffsideLineMovement = WSmemory::get_his_offsideline_movement();
    safetyMargin = 0.25;
    if ( hisTeamOffsideLineMovement < 0.0 )
    	safetyMargin = Tools::max(0.5, -hisTeamOffsideLineMovement);
    // get all opponents to consider
    PlayerSet pset_tmp= WSinfo::valid_opponents;
    // remove goalie because he is not important for offside measurements
  	if ( WSinfo::ws->his_goalie_number > 0 )
  	{
    	PPlayer hisGoalie = WSinfo::get_opponent_by_number(WSinfo::ws->his_goalie_number);
    	if (hisGoalie) pset_tmp.remove(hisGoalie);
  	}
  	// keep only 4rerkette
  	pset_tmp.keep_and_sort_players_by_x_from_right(4);
  	// check if information is too old
  	if (pset_tmp.num>0 && pset_tmp[0]->age > 7)
  	{
  		// if so remove that player and move offside_line
  		if (pset_tmp[0]->pos.getX() < WSinfo::ball->pos.getX())
      		safetyMargin += 0.5;
  		if (pset_tmp.num>1) 
  			hisTeamOffsideLine = pset_tmp[1]->pos.getX();
    	pset_tmp.remove( pset_tmp[0] );
    }
    // now check for viererkette if all 4 players are valid
    PlayerSet viererkette; 
    bool viererketteConsidered = false;
  	if ( pset_tmp.num == 4 )
  	{
   	 	//aufrueckende viererkette!
    	viererkette = pset_tmp;
    	viererkette.keep_and_sort_players_by_age( viererkette.num );
    	int minAge = viererkette[0]->age;
    	viererkette.keep_players_with_max_age( minAge );
    	if (    viererkette.num >= 3
         	 && minAge <= 2 )
      	if (    fabs( viererkette[0]->pos.getX() - viererkette[1]->pos.getX() ) < 1.0
           	 && fabs( viererkette[0]->pos.getX() - viererkette[2]->pos.getX() ) < 1.0
           	 && fabs( viererkette[1]->pos.getX() - viererkette[2]->pos.getX() ) < 1.0 )
      	{
        	hisTeamOffsideLine
          		=   (1.0/3.0) * viererkette[0]->pos.getX()
            	  + (1.0/3.0) * viererkette[1]->pos.getX()
            	  + (1.0/3.0) * viererkette[2]->pos.getX();
           	safetyMargin += minAge * 0.2;
        	viererketteConsidered = true;
        	JTSPOL2("Viererkette considered!");
      	}
   	}
   	if (pset_tmp.num > 0 && viererketteConsidered == false)
  	{  
    	pset_tmp.keep_and_sort_players_by_x_from_right(pset_tmp.num);
    	safetyMargin += Tools::min( 0.25 * (pset_tmp[0]->age), 1.5);
    }
    offside_line = hisTeamOffsideLine; //- safetyMargin;
    //"torauslinie" considerations
  	if ( offside_line > FIELD_BORDER_X )
    	offside_line = FIELD_BORDER_X;
  
  	//"mittellinie" considerations
  	if ( offside_line < 0.0 - safetyMargin )
    	offside_line = -0.25;
    DRAW_LINE(Vector(offside_line, FIELD_BORDER_Y), Vector(offside_line, -FIELD_BORDER_Y), "fff600");
    DRAW_LINE(Vector(offside_line - safety_margin, FIELD_BORDER_Y), Vector(offside_line - safety_margin, -FIELD_BORDER_Y), "ff0b0b");
}
