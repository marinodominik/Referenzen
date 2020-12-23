#ifndef _INTERCEPT_H_
#define _INTERCEPT_H_

#include "Vector.h"
#include "ws.h"

/*****************************************************************************/

struct InterceptResult {
  int time;
  Vector pos;
};
  
class Intercept {
  const static int MAX_STEPS= 14; //2**(14-2)= 4096 > 3000
  bool ball_in_player_radius_at_time( double ball_vel_decay_to_the_power_of_time, int time );
 public:
  /** ball_pos[t]= ball_pos + 1.0*ball_vel + ball_vel_decay*ball_vel + ... + ball_vel_decay**(t-1) * ball_vel
      || player_pos[t] - player_pos || =  t * player_vel_max
      
      || p || denotes the euclidean norm of the vector p  ;-)
      We seek the first time t, so that ball_pos[t] is in reach of the player.
      
  */
  Intercept(): player(NULL) {};
  Vector  ball_pos;
  Vector ball_vel;
  double  ball_vel_decay;
  Vector  player_pos;
  double  player_vel_max;
  /** player_radius is the radius of the player, in which he can catch the ball at time 0.
      Negative Values are allowed, and can be interpreted as additional safety distance 
      (this feature will be probably never used, but anyway :-)*/
  double  player_radius;
  PPlayer player;

 public:
  /** if the needed time to intercept ball is > 2**(MAX_STEPS-2)  then 
      2**(MAX_STEPS-2) is the returned time and pos ist the ball position  
      after 2**(MAX_STEPS-2) steps */
  void minimal_time_player_reaches_ball_at_pos( int & time, Vector & pos );
  void minimal_time_player_reaches_ball_at_pos( InterceptResult & res );
};


#endif
