#include "intercept_ball_bms.h"

#define BASELEVEL 3

#define INTERCEPT_ANGLE_TOLERANCE 15.
#define MAX_FIELD_DIST 120.
#define INITPOWER -1000.

#define kickable_tolerance .1
#define kickable_tolerance_risky .03

/***********************************************************************************/
/* Routines for hand coded intercept by Alex Sung, Feb. 2002 - imported by Ridi    */
/* Converted into behavior framework (and gotten rid of that mdpInfo crap) by Sput */
/***********************************************************************************/
double InterceptBall::cvBallPlayerDistanceAtInterception = 1000.0;

bool InterceptBall::initialized=false;

InterceptBall::InterceptBall()
{
  ivRequestedTurnAngle = 0.0;
  ivValidityOfRequestedTurnAngle = -1;
}

InterceptBall::~InterceptBall(){

}

bool InterceptBall::get_cmd(Cmd & cmd){
  return get_cmd(cmd,WSinfo::me->pos,WSinfo::me->vel,WSinfo::me->ang,
		 WSinfo::ball->pos, WSinfo::ball->vel);

}

bool InterceptBall::get_cmd(Cmd & cmd, int & num_cycles){
  return get_cmd(cmd,WSinfo::me->pos,WSinfo::me->vel,WSinfo::me->ang,
		 WSinfo::ball->pos, WSinfo::ball->vel, num_cycles);

}

bool InterceptBall::get_cmd(  Cmd          & cmd, 
                              const Vector & my_pos,
                              const Vector & my_vel, 
                    			    const ANGLE    my_ang, 
                    			    Vector         ball_pos, 
                              Vector         ball_vel)
{
  int num_cycles;
  return get_cmd(cmd, my_pos, my_vel, my_ang, ball_pos, ball_vel, num_cycles);
}

bool InterceptBall::get_cmd(Cmd & cmd, const Vector & my_pos,const Vector & my_vel, 
			    const ANGLE my_ang, 
			    Vector ball_pos, Vector ball_vel, int &num_cycles, bool riskyIntercept){

Vector destination;
#if 0
  // alex' code starts here
  //  if (!WSinfo::is_ball_kickable()) {
  if(my_pos.sqr_distance(ball_pos)>SQUARE(WSinfo::me->kick_radius)){ 
    // if ball isn't kickable calculate when it is
    int i = 0;
    do {
      ball_pos+=ball_vel;
      ball_vel*=ServerOptions::ball_decay;
    } while (!is_destination_reachable(ball_pos,my_pos,my_vel,my_ang,++i));
    //LOG_MOV(0,<<"Intercept: estimated cycles2go: "<<i);
    num_cycles = i;
  }

  destination = ball_pos;
  LOG_MOV(0,<< _2D << C2D(destination.x,destination.y,1,"#00ff00"));
#endif
  Vector startBallPos = ball_pos, startBallVel = ball_vel;
  // ridis code starts here
  //  if (!WSinfo::is_ball_kickable()) {
  if(my_pos.sqr_distance(ball_pos)>SQUARE(WSinfo::me->kick_radius)){ 
    // if ball isn't kickable calculate when it is
    int i = 0;
    do {
      ball_pos+=ball_vel;
      ball_vel*=ServerOptions::ball_decay;
    } while (!is_destination_reachable2(ball_pos,my_pos,my_vel,my_ang,++i));
    //LOG_MOV(0,<<"Intercept: reach 2 estimated cycles2go: "<<i);
    num_cycles = i;
    if (riskyIntercept)
    {
      int riskyCycles = 0;
      i = 0;
      Vector riskyBallPos = startBallPos, riskyBallVel = startBallVel;
      do 
      {
        riskyBallPos += riskyBallVel;
        riskyBallVel *= ServerOptions::ball_decay;
      } 
      while (!is_destination_reachable2(riskyBallPos,my_pos,my_vel,my_ang,++i,true));//riskyness
      riskyCycles = i;
      if ( riskyCycles < num_cycles/5 )
      {
        LOG_MOV(0,"Intercept: I ("<<WSinfo::me->number<<") reach this goal RISKY ("<<riskyCycles<<" vs. "<<num_cycles<<")!");
        num_cycles = riskyCycles;
        ball_pos = riskyBallPos;
      }
    }
  }
  else num_cycles = 0; //this line was added by TG 06/05

  destination = ball_pos;
  LOG_MOV(0,<< _2D << VC2D(destination,1,"#0000ff"));
  LOG_MOV(0,"Intercept: I reach this goal in cycle: "<<num_cycles);

  // check body angle to ball destination
  Angle my_angle_to_destination = my_angle_to(my_pos, my_ang,destination).get_value_mPI_pPI();
  Angle deviation_threshold = calculate_deviation_threshold((destination-my_pos).norm());
  
  if (    num_cycles == 1 
       && checkForOptimzed1StepIntercept( WSinfo::me,cmd,destination,
                                          my_pos,my_vel,my_ang ) )
  {
    //ok, cmd has been set appropriately
  }
  else
  if (fabs(my_angle_to_destination)>deviation_threshold) {
    // this is code from turn inertia:
    double p_moment = my_angle_to_destination;
    p_moment = Tools::get_angle_between_mPI_pPI(p_moment);
    p_moment=p_moment*(1.0+(WSinfo::me->inertia_moment*(my_vel.norm())));
    if (p_moment > 3.14) p_moment = 3.14;
    if (p_moment < -3.14) p_moment = -3.14;
    p_moment = Tools::get_angle_between_null_2PI(p_moment);
    cmd.cmd_body.set_turn(p_moment);
  }
  else {
    cmd.cmd_body.set_dash(100);
  }
  return true;
}

/* returns deviation_threshold for determining the next action
 */
Angle InterceptBall::calculate_deviation_threshold(double distance) {
  // deviation_threshold determines when the player should turn instead of dash
  Angle deviation_threshold = 0.3;
  if (distance<10) {
    // when near destination only turn if player will miss the destination by more than 1 meter
    Angle deviation_max = asin(1/distance);
    if (deviation_max>deviation_threshold) {
      deviation_threshold=deviation_max;
    }
  }
  return deviation_threshold;
} 

bool InterceptBall::init(char const * conf_file, int argc, char const* const* argv) {
  if(initialized) return true;
  initialized = true;
  //cout << "\nInterceptBall behavior initialized.";
  return true;
}

/* returns if destination is reachable (in kick range) within the given number of turns
 */
bool InterceptBall::is_destination_reachable(const Vector& destination, Vector my_pos,
					     Vector my_vel, ANGLE my_ang, int turns) {
  while (turns-->0) {
    Vector me_to_destination = destination-my_pos;
    Angle my_angle_to_destination = me_to_destination.angle()-my_ang.get_value_mPI_pPI();
    if (my_angle_to_destination<-PI)
      my_angle_to_destination+=2*PI;
    if (my_angle_to_destination>PI)
      my_angle_to_destination-=2*PI;
    Angle deviation_threshold = calculate_deviation_threshold(me_to_destination.norm());

    if (fabs(my_angle_to_destination)>deviation_threshold) {
      double inertia_factor = my_vel.norm()*WSinfo::me->inertia_moment+1.0;
      Angle turn_angle = my_angle_to_destination*inertia_factor;
      if (turn_angle>3.14159) turn_angle=3.14159;
      if (turn_angle<-3.14159) turn_angle=-3.14159;

      my_ang+=ANGLE(turn_angle/inertia_factor);
      //my_ang=Tools::get_angle_between_mPI_pPI(my_ang);
    }
    else {
      my_vel+=100.0*WSinfo::me->dash_power_rate*WSinfo::me->effort
	* Vector(cos(my_ang.get_value()),sin(my_ang.get_value()));
      double my_vel_norm = my_vel.norm();
      if (my_vel_norm>ServerOptions::player_speed_max)
	my_vel*=ServerOptions::player_speed_max/my_vel_norm;
    }

    my_pos+=my_vel;
    my_vel*=WSinfo::me->decay;
    if ((destination-my_pos).norm()<=WSinfo::me->kick_radius) {
      return true;
    }
  }
  return false;
}


/* returns if destination is reachable (in kick range) within the given number of cycles */
bool InterceptBall::is_destination_reachable2(const Vector& destination, Vector my_pos,
					     Vector my_vel, ANGLE my_ang, int maxcycles, bool riskyIntercept) 
{
  double localKickableTolerance = kickable_tolerance;
  if (riskyIntercept) localKickableTolerance = kickable_tolerance_risky;
  Vector dummy1,dummy2;
  Vector my_new_pos,my_new_vel;
  ANGLE my_new_angle;
  Cmd_Body cmd;

  while (maxcycles-->0) {
    Vector me_to_destination = destination-my_pos;
    Angle my_angle_to_destination = me_to_destination.angle()-my_ang.get_value_mPI_pPI();
    if (my_angle_to_destination<-PI)
      my_angle_to_destination+=2*PI;
    if (my_angle_to_destination>PI)
      my_angle_to_destination-=2*PI;
    Angle deviation_threshold = calculate_deviation_threshold(me_to_destination.norm());

    if (fabs(my_angle_to_destination)>deviation_threshold) {
      double inertia_factor = my_vel.norm()*WSinfo::me->inertia_moment+1.0;
      Angle turn_angle = my_angle_to_destination*inertia_factor;
      if (turn_angle>3.14159) turn_angle=3.14159;
      if (turn_angle<-3.14159) turn_angle=-3.14159;

      cmd.unset_lock();
      cmd.unset_cmd();
      cmd.set_turn(turn_angle);
    }
    else {
      cmd.unset_lock();
      cmd.unset_cmd();
      cmd.set_dash(100);
    }

    Tools::model_cmd_main(my_pos,my_vel,my_ang,Vector(0), Vector(0),
			  cmd, my_new_pos, my_new_vel, my_new_angle, dummy1, dummy2);
    
    my_vel= my_new_vel;
    my_pos= my_new_pos;
    my_ang = my_new_angle;

    if ((destination-my_pos).norm()<=WSinfo::me->kick_radius - localKickableTolerance) 
    {
      cvBallPlayerDistanceAtInterception = (destination-my_pos).norm();
      return true;
    }
  }
  return false;
}


ANGLE InterceptBall::my_angle_to(const Vector & my_pos, const ANGLE &my_angle, 
				 Vector target){
  ANGLE result;
#if 0
  target -= WSinfo::me->pos;
  result = target.angle() - WSinfo::me->ang;
#endif
  target -= my_pos;
  result = target.ARG() - my_angle;
  return result;
}

////////////////////////// TG ////////////////////////////////////////////////
bool 
InterceptBall::get_cmd_arbitraryPlayer( PPlayer player,
                                        Cmd & cmd, 
                                        const Vector & my_pos,
                                        const Vector & my_vel, 
                                        const ANGLE my_ang, 
                                        Vector ball_pos, 
                                        Vector ball_vel, 
                                        int &num_cycles,
                                        int maxcycles,
                                        bool riskyIntercept)
{
  if (!player) return false;
  
  Vector destination;

  num_cycles = 0; //do not let this variable be uninitialize!
  Vector startBallPos = ball_pos, startBallVel = ball_vel;
  if(my_pos.sqr_distance(ball_pos) > SQUARE(player->kick_radius))
  { 
    // if ball isn't kickable calculate when it is
    int i = 0;
    do 
    {
      ball_pos+=ball_vel;
      ball_vel*=ServerOptions::ball_decay;
      if (i>maxcycles)
      {num_cycles=maxcycles; return false;}
    } 
    while (!is_destination_reachable2_arbitraryPlayer( player,
                                                       ball_pos,
                                                       my_pos,
                                                       my_vel,
                                                       my_ang,
                                                       ++i));
    //LOG_MOV(0,<<"Intercept: reach 2 estimated cycles2go: "<<i);
    num_cycles = i;
    
    if (riskyIntercept)
    {
      int riskyCycles = 0;
      i = 0;
      Vector riskyBallPos = startBallPos, riskyBallVel = startBallVel;
      do 
      {
        riskyBallPos += riskyBallVel;
        riskyBallVel *= ServerOptions::ball_decay;
      } 
      while ( ! is_destination_reachable2_arbitraryPlayer( player,
                                                           riskyBallPos,
                                                           my_pos,
                                                           my_vel,
                                                           my_ang,
                                                           ++i,
                                                           true));//riskyness
      riskyCycles = i;
      if ( riskyCycles < num_cycles/5 )
      {
        LOG_MOV(0,"Intercept: I ("<<player->number<<") reach this goal RISKY ("<<riskyCycles<<" vs. "<<num_cycles<<")!");
        num_cycles = riskyCycles;
        ball_pos = riskyBallPos;
      }
    }
  }

  destination = ball_pos;
  //  LOG_MOV(0,<< _2D << C2D(destination.x,destination.y,1,"#00ffff"));

  // check body angle to ball destination
  Angle my_angle_to_destination 
    = my_angle_to(my_pos, my_ang,destination).get_value_mPI_pPI();
  Angle deviation_threshold 
    = calculate_deviation_threshold((destination-my_pos).norm());
  
  if (    num_cycles == 1 
       && checkForOptimzed1StepIntercept( player,
                                          cmd,
                                          destination,
                                          my_pos,
                                          my_vel,
                                          my_ang ) )
  {
    //ok, cmd has been set appropriately
  }
  else
  if (fabs(my_angle_to_destination)>deviation_threshold) 
  {
    // this is code from turn inertia:
    double p_moment = my_angle_to_destination;
    p_moment = Tools::get_angle_between_mPI_pPI(p_moment);
    p_moment=p_moment*(1.0+(player->inertia_moment*(my_vel.norm())));
    if (p_moment > 3.14) p_moment = 3.14;
    if (p_moment < -3.14) p_moment = -3.14;
    p_moment = Tools::get_angle_between_null_2PI(p_moment);
    cmd.cmd_body.set_turn(p_moment);
  }
  else 
  {
    cmd.cmd_body.set_dash(100);
  }
  return true;
}

bool InterceptBall::checkForOptimzed1StepIntercept( PPlayer player,
                                                    Cmd & cmd,
                                                    Vector ballDestination,
                                                    Vector myPos,
                                                    Vector myVel,
                                                    ANGLE  myAng )
{
  if ( WSinfo::ws->time != ivValidityOfRequestedTurnAngle )
    return false;
  double p_moment = ivRequestedTurnAngle - myAng.get_value_0_p2PI();
  p_moment = Tools::get_angle_between_mPI_pPI(p_moment);
  p_moment=p_moment*(1.0+(player->inertia_moment*(myVel.norm())));
  if (p_moment > 3.14) p_moment = 3.14;
  if (p_moment < -3.14) p_moment = -3.14;
  p_moment = Tools::get_angle_between_null_2PI(p_moment);
  cmd.cmd_body.set_turn(p_moment);

  Vector myNewPos, myNewVel, dummy1, dummy2;
  ANGLE  myNewAng;
  Tools::setModelledPlayer(player);
  Tools::model_cmd_main( myPos,
                         myVel,
                         myAng,
                         Vector(0), 
                         Vector(0),
                         cmd.cmd_body, 
                         myNewPos, 
                         myNewVel, 
                         myNewAng, 
                         dummy1, 
                         dummy2);
  Tools::setModelledPlayer( WSinfo::me );
  if (    (ballDestination - myNewPos).norm() 
       <= player->kick_radius - kickable_tolerance ) 
    return true;
  else
    return false;
}

bool InterceptBall::is_destination_reachable2_arbitraryPlayer
                               (PPlayer player,
                                const Vector& destination, 
                                Vector my_pos,
                                Vector my_vel, 
                                ANGLE my_ang, 
                                int maxcycles,
                                bool riskyIntercept) 
{
  if (!player) return false;
  
  double localKickableTolerance = kickable_tolerance;
  if (riskyIntercept) localKickableTolerance = kickable_tolerance_risky;
  Vector dummy1,dummy2;
  Vector my_new_pos,my_new_vel;
  ANGLE my_new_angle;
  Cmd_Body cmd;

  while (maxcycles-->0) 
  {
    Vector me_to_destination = destination-my_pos;
    Angle my_angle_to_destination 
      = me_to_destination.angle()-my_ang.get_value_mPI_pPI();
    if (my_angle_to_destination<-PI)
      my_angle_to_destination+=2*PI;
    if (my_angle_to_destination>PI)
      my_angle_to_destination-=2*PI;
    Angle deviation_threshold 
      = calculate_deviation_threshold(me_to_destination.norm());

    if (fabs(my_angle_to_destination)>deviation_threshold) 
    {
      double inertia_factor = my_vel.norm() * player->inertia_moment+1.0;
      Angle turn_angle = my_angle_to_destination*inertia_factor;
      if (turn_angle>3.14159) turn_angle=3.14159;
      if (turn_angle<-3.14159) turn_angle=-3.14159;

      cmd.unset_lock();
      cmd.unset_cmd();
      cmd.unset_lock();
      cmd.set_turn(turn_angle);
    }
    else {
      cmd.unset_lock();
      cmd.unset_cmd();
      cmd.unset_lock();
      cmd.set_dash(100);
    }

    Tools::setModelledPlayer(player);
    Tools::model_cmd_main(my_pos,my_vel,my_ang,Vector(0), Vector(0),
        cmd, my_new_pos, my_new_vel, my_new_angle, dummy1, dummy2);
    Tools::setModelledPlayer(WSinfo::me);
    
    my_vel= my_new_vel;
    my_pos= my_new_pos;
    my_ang = my_new_angle;

    if ((destination-my_pos).norm() <= player->kick_radius - localKickableTolerance) 
    {
      return true;
    }
  }
  return false;
}

void
InterceptBall::setRequestForTurnAngle(Angle turnAngle, long timeOfValidity)
{
  ivRequestedTurnAngle = turnAngle;
  ivValidityOfRequestedTurnAngle = timeOfValidity;
}

bool 
InterceptBall::get_cmd(Cmd &cmd, int &num_cycles, double &resBallDist)//TG09
{
  cvBallPlayerDistanceAtInterception = 1000.0;
  if ( get_cmd( cmd, num_cycles) )
  {
    //INFO: cvBallPlayerDistanceAtInterception wird in is_destination_reachable2 gesetzt
    resBallDist = cvBallPlayerDistanceAtInterception;
    return true;
  }
  return false;
}

double
InterceptBall::getBallPlayerInterceptDistanceForRecentRequest()
{
  return cvBallPlayerDistanceAtInterception;
}


