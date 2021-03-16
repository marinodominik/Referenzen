#include "onetwostep_intercept_bms.h"
#include "tools.h"
#include "options.h"
#include "ws_info.h"
#include "log_macros.h"

#define BASELEVEL 3

bool OneTwoStep_Intercept::initialized=false;

//double OneTwoStep_Intercept::step1success_sqrdist;
//double OneTwoStep_Intercept::step2success_sqrdist;

/* initialize constants */
  
  /* neuro_intercept uses dashs */
const double OneTwoStepInterceptItrActions::dash_power_min = -100;
const double OneTwoStepInterceptItrActions::dash_power_inc = 20;
const double OneTwoStepInterceptItrActions::dash_power_max= 100;
const double OneTwoStepInterceptItrActions::dash_power_steps = (int)((dash_power_max - dash_power_min)/dash_power_inc) +1;

  /* neuro_intercept uses turns */
const double OneTwoStepInterceptItrActions::turn_min= -M_PI;  // important: start with strongest turn to the right
const double OneTwoStepInterceptItrActions::turn_max= M_PI;  // and end with strongest turn to the left!
const double OneTwoStepInterceptItrActions::turn_inc= 2*M_PI/72.0;
const int OneTwoStepInterceptItrActions::turn_steps=(int)((turn_max - turn_min)/turn_inc) +1;

/*new for soccer server v13 and beyond*/
const double OneTwoStepInterceptItrActions::side_dash_power_min = 100.0;
const double OneTwoStepInterceptItrActions::side_dash_power_inc = 0.0;
const double OneTwoStepInterceptItrActions::side_dash_power_max = 100.0;
const double OneTwoStepInterceptItrActions::side_dash_angle_min = -M_PI*0.75; //-M_PI*0.5; //TG16
const double OneTwoStepInterceptItrActions::side_dash_angle_max =  M_PI*1.0;  //M_PI*0.5; <- old, erroneous
const double OneTwoStepInterceptItrActions::side_dash_angle_inc =  M_PI*0.25; //M_PI;        values
const int OneTwoStepInterceptItrActions::side_dash_power_steps
          =   OneTwoStepInterceptItrActions::side_dash_power_inc > 0.0001
            ? (int)((side_dash_power_max - side_dash_power_min) / side_dash_power_inc) + 1
            : 0 + 1;
const int OneTwoStepInterceptItrActions::side_dash_angle_steps
          = (int)((side_dash_angle_max - side_dash_angle_min) / side_dash_angle_inc) + 1;
const int OneTwoStepInterceptItrActions::side_dash_steps 
          =  OneTwoStepInterceptItrActions::side_dash_power_steps
           * OneTwoStepInterceptItrActions::side_dash_angle_steps;

#if 0
#define PROT(XXX)(std::cout<<XXX)
#define MYLOG_POL(XXX,YYY) LOG_POL(XXX,YYY);
#else
#define MYLOG_POL(XXX,YYY)
#define PROT(XXX)
#endif

#if 0 // ridi: considerd harmful before Lisbon, probably rechange
#define onestep_safety_margin  0.
#define twostep_safety_margin  0.
#endif

#define onestep_safety_margin  0.04
#define twostep_safety_margin  0.15
#define outdated_ball_extra_safety_margin 0.03


ANGLE OneTwoStep_Intercept::my_angle_to(const Vector & my_pos, const ANGLE &my_angle, 
				 Vector target){
  ANGLE result;
  target -= my_pos;
  result = target.ARG() - my_angle;
  return result;
}


OneTwoStep_Intercept::OneTwoStep_Intercept(){
    //    step1success_sqrdist = SQUARE(WSinfo::me->kick_radius - onestep_safety_margin);
    //step2success_sqrdist = SQUARE(WSinfo::me->kick_radius - twostep_safety_margin);
    step1success_sqrdist = SQUARE(ServerOptions::kickable_area - onestep_safety_margin);
    step2success_sqrdist = SQUARE(ServerOptions::kickable_area - twostep_safety_margin);
    currentBestBallDist = -1;
}


bool OneTwoStep_Intercept::init(char const * conf_file, int argc, char const* const* argv){
    if(initialized) return true;
    initialized = true;
    INFO_OUT << "\n1or2step_Intercept behavior initialized.";
   return true;
}

bool OneTwoStep_Intercept::get_1step_cmd(Cmd & cmd, double &sqrdist, const MyState &state){
  MyState next_state;
  Cmd_Body stepone_action;
  OneTwoStepInterceptItrActions itr_actions;
  double best1step_sqrdist = step1success_sqrdist;
  double best1step_dashpower = 10000;
  double surely_save = state.me->kick_radius - 0.3;
  if (WSinfo::ball->age > 0) surely_save = state.me->kick_radius - 0.4; //TG17
  double power, angle;

  itr_actions.reset();
  while ( Cmd_Body * action = itr_actions.next() ) 
  {
    Tools::get_successor_state(state,*action,next_state);
    //MYLOG_POL(0,<<"1Icpt: Test action "<<*action<<" successor "<<next_state);    
    double sqrdist = next_state.my_pos.sqr_distance(next_state.ball_pos);

    if (sqrdist < best1step_sqrdist) // this action improves current best
    {
      if (best1step_sqrdist > SQUARE(surely_save))// optimum not reached yet, so take action!
      {
        best1step_sqrdist = sqrdist;
        stepone_action = *action;
        PROT("1Icpt: Wow! One step action improves dist "<<stepone_action
           <<" new square dist: "<<best1step_sqrdist<<std::endl);
        MYLOG_POL(2,"1Icpt: Wow! One step action improves dist "<<stepone_action
           <<" new square dist: "<<best1step_sqrdist);
      }
      if (best1step_sqrdist <= SQUARE(surely_save)) // do finetuning of actions
      {
        if ((action->get_type(power,angle) == Cmd_Body::TYPE_DASH))
        {
          if ((fabs(power) < best1step_dashpower))
          {
            best1step_sqrdist = sqrdist;
            stepone_action = *action;
            best1step_dashpower = fabs(power);
            PROT("1Icpt: Fine Improvement SUCCESS! Found: "<<stepone_action
               <<" new square dist: "<<best1step_sqrdist<<std::endl);
            MYLOG_POL(2,"1Icpt: Fine Improvement SUCCESS! Found: "<<stepone_action
               <<" new square dist: "<<best1step_sqrdist);
          }
        }
        else 
        if ((action->get_type(power,angle) == Cmd_Body::TYPE_TURN))
        {
          // Hey, I can get the Ball without dashing, so let's turn toward opponent goal
          Vector const opponent_goalpos(52.5,0.);
          Angle turn2goal_angle = (opponent_goalpos-state.my_pos).arg()-state.my_angle.get_value();
          turn2goal_angle = Tools::get_angle_between_mPI_pPI(turn2goal_angle);
          turn2goal_angle=turn2goal_angle*(1.0+(state.me->inertia_moment*
                           (state.my_vel.norm())));
          if (turn2goal_angle > 3.14) turn2goal_angle = 3.14;
          if (turn2goal_angle < -3.14) turn2goal_angle = -3.14;
          turn2goal_angle = Tools::get_angle_between_null_2PI(turn2goal_angle);
          stepone_action.unset_lock();
          stepone_action.set_turn(turn2goal_angle);
          PROT("1Icpt: Turn is safe! "<<stepone_action
          <<" new square dist: "<<best1step_sqrdist<<" -> Turn2goal "<<RAD2DEG(turn2goal_angle)<<std::endl);
          MYLOG_POL(2,"1Icpt: Turn is safe! "<<stepone_action
          <<" new square dist: "<<best1step_sqrdist<<" -> Turn2goal "<<RAD2DEG(turn2goal_angle));
          break;
        }
      }
      if (best1step_sqrdist <= SQUARE(surely_save))// action is safe, give other actions a chance
      {
        PROT("1Icpt: YEAH, current best action is surely save!"<<std::endl);
        MYLOG_POL(2,"1Icpt: YEAH, current best action is surely save!");
        best1step_sqrdist = SQUARE(surely_save-0.01);
      }
    }
  }//end of while
  //  PROT(endl);

  if (best1step_sqrdist < step1success_sqrdist) // successful!
  {
    cmd.cmd_body.unset_lock();
    cmd.cmd_body.clone( stepone_action );
    sqrdist = best1step_sqrdist;
    return true;
  }
  return false;
}

bool OneTwoStep_Intercept::get_1step_cmd_virtual(double &step2sqrdist, double &step3sqrdist, const MyState &state){
  MyState next_state;
  OneTwoStepInterceptItrActions itr_actions;
  double best2step_sqrdist = step1success_sqrdist;
  double best3step_sqrdist = step1success_sqrdist;

  itr_actions.reset();
  while ( Cmd_Body * action = itr_actions.next() ) {
    Tools::get_successor_state(state,*action,next_state);
    double sqrdist = next_state.my_pos.sqr_distance(next_state.ball_pos);
    if(sqrdist >step1success_sqrdist){ // second chance -> what happens, if I just wait for ball???
      Vector new_my_pos,  new_ball_pos;      
      new_my_pos = next_state.my_pos + next_state.my_vel;     //me
      new_ball_pos = next_state.ball_pos + next_state.ball_vel; //ball

      sqrdist =new_my_pos.sqr_distance(new_ball_pos);
      if(sqrdist <= best3step_sqrdist){ // this action improves current best
	best3step_sqrdist = sqrdist;	
      }
      continue;
    }

    if(sqrdist < best2step_sqrdist){ // this action improves current best
      best2step_sqrdist = sqrdist;
    }
  }
  
  if (best2step_sqrdist < step1success_sqrdist || best3step_sqrdist < step1success_sqrdist ){ // successful!
    step2sqrdist = best2step_sqrdist;
    step3sqrdist = best3step_sqrdist;
    return true;
  }
  return false;
}



bool OneTwoStep_Intercept::get_2step_cmd(Cmd & cmd, double &sqrdist,int & steps, const MyState &state){

  MyState next_state;
  Cmd_Body steptwo_action,stepthree_action;
  OneTwoStepInterceptItrActions itr_actions;
  double best2step_sqrdist = step2success_sqrdist;
  double best3step_sqrdist = step2success_sqrdist;
  double step1sqrdist, step2sqrdist, step3sqrdist;
  Cmd cmd_tmp;

  if(get_1step_cmd(cmd_tmp, step1sqrdist, state) == true){
    LOG_MOV(0,<<"OneTwoIntercept: 1StepCmd Found -> Done.");
    steps =1;
    sqrdist = step1sqrdist;
    cmd.cmd_body.clone(cmd_tmp.cmd_body);
    return true;
  }

  itr_actions.reset();
  while ( Cmd_Body * action = itr_actions.next() ) {
    Tools::get_successor_state(state,*action,next_state);
    //PROT("2Icpt: Test action "<<*action<<" successor "<<next_state<<std::endl);
    //MYLOG_POL(2, "2Icpt: Test action "<<*action<<" successor "<<next_state<<std::endl);
    if(get_1step_cmd_virtual(step2sqrdist, step3sqrdist, next_state) == true)
    {
      PROT("2Icpt: There is a one step command from the successor state"<<std::endl);
      MYLOG_POL(2,"2Icpt: There is a one step command from the successor state");
      if(step2sqrdist < best2step_sqrdist)
      {
        best2step_sqrdist = step2sqrdist;
        steptwo_action = *action;
        PROT("2Icpt: Wow! Two step action improves dist "<<steptwo_action
             <<" new square dist: "<<best2step_sqrdist<<std::endl);
        MYLOG_POL(2,"2Icpt: Wow! Two step action improves dist "<<steptwo_action
             <<" new square dist: "<<best2step_sqrdist);
      }
      if(step3sqrdist < best3step_sqrdist)
      {
        best3step_sqrdist = step3sqrdist;
        stepthree_action = *action;
        PROT("2Icpt: Wow! Three step action improves dist "<<steptwo_action
              <<" new square dist: "<<best2step_sqrdist<<std::endl);
        MYLOG_POL(2,"2Icpt: Wow! Three step action improves dist "<<steptwo_action
              <<" new square dist: "<<best2step_sqrdist);
      }
    }
  }
  //  PROT(endl);

  if (best2step_sqrdist <step2success_sqrdist){ // successful!
    LOG_MOV(0,<<"OneTwoIntercept: 2StepCmd Found -> 2 steps 2 go.");
    cmd.cmd_body.clone( steptwo_action );
    sqrdist = best2step_sqrdist;
    steps = 2;
    return true;
  }
  if (best3step_sqrdist <step2success_sqrdist){ // successful!
    LOG_MOV(0,<<"OneTwoIntercept:  3StepCmd Found -> 3 steps 2 go.");
    cmd.cmd_body.clone( stepthree_action );
    sqrdist = best3step_sqrdist;
    steps = 3;
    return true;
  }
  return false;
}



bool OneTwoStep_Intercept::get_cmd(Cmd & cmd, const MyState &state){
  int steps;

  return(get_cmd(cmd,state,steps));
}

bool OneTwoStep_Intercept::get_cmd(Cmd & cmd, const MyState &state, int &steps){
  double dist;

  step1success_sqrdist = SQUARE(state.me->kick_radius - onestep_safety_margin);
  step2success_sqrdist = SQUARE(state.me->kick_radius - twostep_safety_margin);

  if (WSinfo::ball->age > 0)
    step1success_sqrdist = SQUARE(state.me->kick_radius - onestep_safety_margin
                                                        - outdated_ball_extra_safety_margin);


  if(  state.my_pos.sqr_distance(state.ball_pos)
     > SQUARE(   3.0 * (state.ball_vel.norm() + state.me->speed_max)
               + state.me->kick_radius ) )
  {
    PROT("Check intercept in "<<2<<" steps "<<" Ball too far "<<
    state.my_pos.distance(state.ball_pos)<<std::endl);
    MYLOG_POL(2,"Check intercept in "<<2<<" steps "<<" Ball too far "<<
    state.my_pos.distance(state.ball_pos));
    return false;
  }

  return(get_2step_cmd(cmd, dist,steps, state));
}

bool OneTwoStep_Intercept::get_cmd(Cmd & cmd){
  int steps;
  return get_cmd(cmd,steps);
}

bool OneTwoStep_Intercept::get_cmd(Cmd & cmd, int &steps){
  MyState state;
  
  state.my_pos = WSinfo::me->pos;
  state.my_vel = WSinfo::me->vel;
  state.ball_pos = WSinfo::ball->pos;
  state.ball_vel = WSinfo::ball->vel;
  state.my_angle = WSinfo::me->ang;

  return get_cmd(cmd,state,steps);

}

//NEU: TG09
bool OneTwoStep_Intercept::get_cmd(Cmd & cmd, int &steps, double &resBallDist)
{
  currentBestBallDist = 1000.0;
  MyState state;
  
  state.my_pos = WSinfo::me->pos;
  state.my_vel = WSinfo::me->vel;
  state.ball_pos = WSinfo::ball->pos;
  state.ball_vel = WSinfo::ball->vel;
  state.my_angle = WSinfo::me->ang;

  double squareDist;

  step1success_sqrdist = SQUARE(WSinfo::me->kick_radius - onestep_safety_margin);
  step2success_sqrdist = SQUARE(WSinfo::me->kick_radius - twostep_safety_margin);

  if(state.my_pos.sqr_distance(state.ball_pos) > SQUARE(2*(state.ball_vel.norm() +
              WSinfo::me->speed_max) +
              WSinfo::me->kick_radius))
  {
    MYLOG_POL(2,"Check intercept in "<<2<<" steps "<<" Ball too far "<<
    state.my_pos.distance(state.ball_pos));
    return false;
  }

  if (get_2step_cmd(cmd, squareDist, steps, state) == true)
  {
    resBallDist = sqrt( squareDist );
    return true;
  }

  return false;
}
