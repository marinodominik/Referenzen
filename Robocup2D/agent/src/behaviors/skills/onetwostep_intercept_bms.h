#ifndef ONEOR2STEP_INTERCEPT_BMS_H_
#define ONEOR2STEP_INTERCEPT_BMS_H_

#include "../../basics/Cmd.h"
#include "../base_bm.h"
#include "mystate.h"
#include "log_macros.h"

class OneTwoStepInterceptItrActions {
  /* set params / initialization in .c file */
  
  /* neuro_intercept uses dashs */
  static const double dash_power_min;
  static const double dash_power_inc;
  static const double dash_power_max;
  static const double dash_power_steps;
  
  /* neuro_intercept uses turns */
  static const double turn_min;  // important: start with strongest turn to the right
  static const double turn_max;  // and end with strongest turn to the left!
  //  static const double turn_inc= 2*M_PI/16.0; // this gives 16 increments
  //static const double turn_inc= 2*M_PI/72.0;
  static const double turn_inc;
  //static const double turn_inc= 2*M_PI/180.0;
  /* non integral constant -> INITIALIZATION BELOW */
  static const int turn_steps;
  
  /*new for soccer server v13 and beyond*/
  static const double side_dash_power_min;
  static const double side_dash_power_inc;
  static const double side_dash_power_max;
  static const double side_dash_angle_min;
  static const double side_dash_angle_max;
  static const double side_dash_angle_inc;
  static const int side_dash_power_steps,
                   side_dash_angle_steps,
                   side_dash_steps;

  Cmd_Body action;
  Angle turn;
  int turn_counter;
  double dash;
  int dash_counter;
  double side_dash_power;//new for soccer server v13 and beyond
  Angle side_dash_angle;//new for soccer server v13 and beyond
  int side_dash_counter;//new for soccer server v13 and beyond
  double move;
 public:
  //ItrActions();
  static bool init();

  void reset() 
  {
    turn = turn_min;
    dash = dash_power_min;
    turn_counter= 0;
    dash_counter= 0;
    side_dash_power   = side_dash_power_min;//new for soccer server v13 and beyond
    side_dash_angle   = side_dash_angle_min;//new for soccer server v13 and beyond
    side_dash_counter = 0;//new for soccer server v13 and beyond
  }

  Cmd_Body * next() 
  {
    if ( turn_counter < turn_steps ) 
    {
      action.unset_lock();
      action.unset_cmd();
      // a bit tricky: make sure that maximum turn angles are contained in action set !!!!
      if (turn <= -(M_PI - 0.01))  // angle is smaller or approx. equal than max neg. turn 
        action.set_turn(-(M_PI - 0.00001/180. * PI));
      //action.set_turn(-M_PI);
      else 
      if (turn >=(M_PI - 0.01))// angle is larger or approx. equal than max pos. turn 
        action.set_turn((M_PI - 0.00001/180. * PI));
      //action.set_turn(M_PI);
      else
        action.set_turn( turn );
      turn += turn_inc;
      turn_counter++;
      return &action;
    }
    
    if ( dash_counter < dash_power_steps ) 
    {
      action.unset_lock();
      action.unset_cmd();
      if (dash>dash_power_max)
        dash=dash_power_max;
      action.set_dash( dash );
      dash += dash_power_inc;
      dash_counter++;
      return &action;
    }
    
    //new for soccer server v13 and beyond
    if ( side_dash_counter < side_dash_steps ) 
    {
      action.unset_lock();
      action.unset_cmd();
      if ( side_dash_angle > side_dash_angle_max )
      {
        //switch to the next side dash power level 
        side_dash_power += side_dash_power_inc;
        side_dash_angle = side_dash_angle_min;
      }
      action.set_dash( side_dash_power, side_dash_angle );
      //switch to the next side dash angle
      side_dash_angle += side_dash_angle_inc;
      side_dash_counter ++ ;
      return &action;
    }    
    
    return 0;
  }
};



class OneTwoStep_Intercept : public BodyBehavior {
  static bool initialized;
  ANGLE my_angle_to(const Vector & my_pos, const ANGLE &my_angle, Vector target);
 private:
  bool get_1step_cmd(Cmd & cmd, double &sqrdist, const MyState &state);
  bool get_2step_cmd(Cmd & cmd, double &sqrdist, int &steps, const MyState &state);
  bool get_1step_cmd_virtual(double &step2sqrdist, double &step3sqrdist, const MyState &state);


  double step1success_sqrdist;
  double step2success_sqrdist;
  double currentBestBallDist;

 public:
  bool get_cmd(Cmd &cmd, const MyState &state);
  bool get_cmd(Cmd &cmd, const MyState &state, int &steps);
  bool get_cmd(Cmd &cmd, int &steps);
  bool get_cmd(Cmd &cmd, int &steps, double &resBallDist);
  bool get_cmd(Cmd &cmd);
 
  OneTwoStep_Intercept();
  static bool init(char const * conf_file, int argc, char const* const* argv); 
};

#endif
