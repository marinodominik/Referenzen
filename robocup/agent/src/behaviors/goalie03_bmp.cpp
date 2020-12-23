/* Vorsicht bei goalie_earlie_intercept: wenn der Goalie draussen ist und dann ein Gegner auf das Tor schiesst,
   kann es sein, dass der Goalie versucht, den Ball ausserhalb des Strafraums zu intercepten. Dann haette er
   wohl ein Problem, weil er nicht weiss, was er mit dem Ball machen soll.
*/
#include "goalie03_bmp.h"
#include "goalie_bs03_bmc.h"
#include "../policy/positioning.h"
//#include "mdp_info.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "tools.h"
#include "options.h"
#include "globaldef.h"
#include "log_macros.h"
#include "valueparser.h"
#include "ws_memory.h"

#define BASELEVEL 0

//#define LOG_DAN(YYY,XXX) LOG_POL(YYY,XXX)
#define LOG_DAN(YYY,XXX)


bool Goalie03::initialized = false;

const Vector TOP_LEFT = Vector(-47.0, 9.0);
const Vector TOP_RIGHT = Vector(-40.0, 5.0);
const Vector BOTTOM_LEFT = Vector(-47.0, -9.0);
const Vector BOTTOM_RIGHT = Vector(-40.0, -5.0);
const double START_MOVE_TRAPEZ = -20.0;
const double normal_xA = -50.5-0.8;//-49.0 //-0.8 wg. verkl. Catch-Rad. TG09
const double Goalie03::DEG2RAD = M_PI/180;
const double Goalie03::RAD2DEG = 180/M_PI;
double Goalie03::xA = normal_xA;
double Goalie03::TOP_TRAPEZ_ANGLE = (TOP_LEFT - TOP_RIGHT).arg();  
double Goalie03::BOTTOM_TRAPEZ_ANGLE = (BOTTOM_LEFT - BOTTOM_RIGHT).arg();

Goalie03::Goalie03() {
	
  ivLastInterceptDecisionBeyondPenaltyArea = -1;
	
  left_corner = ServerOptions::own_goal_pos + Vector(0, ServerOptions::goal_width/2.0);
  right_corner = ServerOptions::own_goal_pos + Vector(0, -ServerOptions::goal_width/2.0);
  direction = TOP;
  direction_t = NORTH;
  use_trapez = 1;//Fuers Training geandert!(Auch im Spiel sinnvoll?)
  break_through_nr = -1;
  intercept_out_of_penalty_area = 0;
  last_time_ball_pos_invalid = -10;
  last_go2pos = -10;
  less_than_two_steps = false;
  saved_go2pos = Vector(-50.5, 0.0);

  goalside_tolerance = 1.0;
  normal_tolerance = 1.5;
  intercept_out_of_penalty_area = 0;

  ValueParser parser( CommandLineOptions::policy_conf, "Goalie03" );
  //parser.get("look_for_ball_timer", look_for_ball_timer_value);
  parser.get("ball_near_treshold", ball_near_treshold);
  //parser.get("ball_very_near_treshold", ball_very_near_treshold);
  //parser.get("intercept_radius", intercept_radius);
  //parser.get("nearest_opponent_safety", nearest_opponent_safety);
  //parser.get("nearest_teammate_safety", nearest_teammate_safety);
  //parser.get("maxdist_to_home", maxdist_to_home);

  parser.get("goalside_tolerance", goalside_tolerance);
  parser.get("normal_tolerance", normal_tolerance);
  parser.get("intercept_out_of_penalty_area", intercept_out_of_penalty_area);
  
  is_ball_catchable = false;
  last_intercept = -24;
  last_intercept_outside_p = -24;
  Goalie_Bs03::catch_ban = 0;

  go2pos = new GoToPos2016; //NeuroGo2Pos;
  intercept = new NeuroIntercept;
  basic_cmd = new BasicCmd;
  face_ball = new FaceBall;
  neuro_kick = new NeuroKickWrapper;
  oneortwo = new OneOrTwoStepKick;
  go2pos_backwards = new Go2PosBackwards;
  hold_turn = new OneTwoHoldTurn;
  ivpGoalieTackle08 = new GoalieTackle08();
  interceptball = new InterceptBall(); 
}


Goalie03::~Goalie03() {
  delete go2pos;
  delete intercept;
  delete basic_cmd;
  delete face_ball;
  delete neuro_kick;
  delete oneortwo;
  delete go2pos_backwards;
  delete hold_turn;
  delete ivpGoalieTackle08;
  delete interceptball;
}


void Goalie03::reset_intention() {
  direction = TOP;
  direction_t = NORTH;
  use_trapez = 1;//Fuers Training geandert!(Auch im Spiel sinnvoll?)
  break_through_nr = -1;
  intercept_out_of_penalty_area = 0;
  last_time_ball_pos_invalid = -10;
  last_go2pos = -10;
  less_than_two_steps = false;
  saved_go2pos = Vector(-50.5, 0.0);

  goalside_tolerance = 1.0;
  normal_tolerance = 1.5;
  intercept_out_of_penalty_area = 0;

  is_ball_catchable = false;
  last_intercept = -24;
  last_intercept_outside_p = -24;
  Goalie_Bs03::catch_ban = 0;
}

/*
bool Goalie03::reconsider_move(Main_Move *current_move) {
  IntentionType intention;
  mdpInfo::get_my_intention(intention);

  set_vars();

  if ((intention.type == DECISION_TYPE_RUNTO) && (intention.time == mdpInfo::time_current()-1)) {
    if (use_trapez) {
      //taken from test_goalshot_trapez()
      double steigung2 = -(ball_vel.y / ball_vel.x);
      
      Vector i_pos_w_shot = Vector(my_pos.x, (ball_pos.x - my_pos.x) * steigung2 + ball_pos.y);
      
      if (!is_ball_heading_for_goal() ||
	  (ball_steps_to(i_pos_w_shot) - 2 > goalie_steps_to(i_pos_w_shot)) ||
	  (ball_steps_to(i_pos_w_shot) < 0)) {
      } else {
	LOG_DAN(0, << "reconsider move is active!!!");
	current_move->terminate(); // terminate current move and decide from new
	mdpInfo::clear_my_intention();
	LOG_DAN(BASELEVEL+0, << "clearing my intention because of goalshot!!!");
	return playon();
      }
    }
    if (mdpInfo::is_object_in_my_penalty_area(my_pos) &&
	is_ball_catchable) {
      LOG_DAN(0, << "reconsider move is active!!!");
      current_move->terminate(); // terminate current move and decide from new
      mdpInfo::clear_my_intention();
      LOG_DAN(BASELEVEL+0, << "clearing my intention because ball is catchable!!!");
      return playon();
    }
    if (!use_trapez) {
      //taken from test_goal_shot()
      double steigung = -(ball_vel.y / ball_vel.x);
      
      id_pos_when_goalshot = Vector(my_pos.x, (ball_pos.x - my_pos.x) * steigung + ball_pos.y);
      
      // Abfrage, ob der Ball aufs Tor geschossen wurde und ob er jetzt abgefangen werden muss
      if (is_ball_heading_for_goal()
	  && (ball_steps_to(id_pos_when_goalshot) - 2 <= goalie_steps_to(id_pos_when_goalshot)) 
	  && (ball_steps_to(id_pos_when_goalshot) >= 0)) {
	LOG_DAN(0, << "reconsider move is active!!!");
	current_move->terminate(); // terminate current move and decide from new
	mdpInfo::clear_my_intention();
	LOG_DAN(BASELEVEL+0, << "clearing my intention because of goal_shot!!!");
	return playon();
      }	
   }

    
//if (use_trapez && (test_goalshot_trapez() != 0) || (test_catch() != 0)) {
//    LOG_DAN(0, << "reconsider move is active!!!");
//    current_move->terminate(); // terminate current move and decide from new
//    mdpInfo::clear_my_intention();
//    LOG_DAN(BASELEVEL+0, << "clearing my intention!!!");
//    return playon();
    }
    
   }
   return current_move;
}
*/

double Goalie03::dash_power_needed_for_distance(double dist) {
  Vector v = Vector(1.0,0.0);
  v.rotate(WSinfo::me->ang.get_value());

  double vel = skalarprodukt(WSinfo::me->vel, v);
  double power = (dist - vel*2.0/3.0) * 100.0;
  if (power > 100.0) return 100.0;
  else if (power < -100.0) return -100.0;
  else return power;
}

/* moves the goalie to Vector.y
 * 
 * stay on the line
 *
 */
bool Goalie03::go_to_y(Cmd &cmd, Vector pos, int priority ) {
  double power;
  Vector pos2 = pos;

  if (pos2.getY() > left_corner.getY()) {
    pos2.setY( left_corner.getY() );
  } else if (pos2.getY() < right_corner.getY()) {
    pos2.setY( right_corner.getY() );
  }

  if (ball_pos.getX() > -25.0) {
    pos2 = Vector(pos2.getX(), 0.0);
  }

  double y_dist_to_pos = pos2.getY() - my_pos.getY();
  double abs_y_dist_to_pos = fabs(y_dist_to_pos);
  LOG_DAN(0, "distance to position is " << abs_y_dist_to_pos);

  LOG_DAN(0, <<_2D << VC2D(pos, 0.5, "yellow"));
  LOG_DAN(0, <<_2D << VC2D(WSinfo::me->pos, 0.5, "pink"));

  int ball_steps_to_left_corner = ball_steps_to_with_max_speed(left_corner);
  int goalie_steps_to_left_corner = goalie_steps_to(left_corner);
  int ball_steps_to_right_corner = ball_steps_to_with_max_speed(right_corner);
  int goalie_steps_to_right_corner = goalie_steps_to(right_corner);

  LOG_DAN(0, "ball_steps to left corner " << ball_steps_to_left_corner << " goalie_steps to left corner " << goalie_steps_to_left_corner);
  LOG_DAN(0, "ball_steps to right corner " << ball_steps_to_right_corner << " goalie_steps to right corner " << goalie_steps_to_right_corner);

  //if ( ball_dist_to_goal > 20.0) {
  if ( //(ball_dist_to_goal > 22.0) ||
       (((my_pos.getY() > 0.0) && (pos.getY() > my_pos.getY()))||
	((my_pos.getY() < 0.0) && (pos.getY() < my_pos.getY())) ||
	((WSinfo::me->stamina < 0.45 * ServerOptions::stamina_max) && //TG09: 1800
	(fabs(my_pos.getY()) < ServerOptions::goal_width-1.0))) &&
       (ball_steps_to_left_corner > goalie_steps_to_left_corner + 1) && 
       (ball_steps_to_right_corner > goalie_steps_to_right_corner + 1) && 
       (ball_steps_to_left_corner > 2) &&
       (ball_steps_to_right_corner > 2) ) {
    LOG_DAN(0, "ball is far away, don't dash too much!!!");
    if (abs_y_dist_to_pos > 5.0) {//1.5) {
      // Dash noch "vorne"
      power = dash_power_needed_for_distance(abs_y_dist_to_pos);//100
      LOG_DAN(0, "old dash power 100, new : " << power);
      if (((pos2.getY() >= my_pos.getY()) && (direction == BOTTOM)) ||
	  ((pos2.getY() < my_pos.getY()) && (direction == TOP))) {
	// Dash nach "hinten"
	power = dash_power_needed_for_distance(-abs_y_dist_to_pos);//-100;
	LOG_DAN(0, "old dash power -100, new : " << power);
      }
      basic_cmd->set_dash(power, priority);
      basic_cmd->get_cmd(cmd);
      return true;
    } 
    basic_cmd->set_turn(0.0);
    basic_cmd->get_cmd(cmd);
    return true;
  }


  // Bei zu grosser Abweichung wird ein maximaler Dash in die richtige Richtung durchgef�hrt
  if (abs_y_dist_to_pos > 0.8) {
    // Dash noch "vorne"
    power = dash_power_needed_for_distance(abs_y_dist_to_pos);//100;
    LOG_DAN(0, "old dash power 100, new : " << power);
    if (((pos2.getY() >= my_pos.getY()) && (direction == BOTTOM)) ||
	((pos2.getY() < my_pos.getY()) && (direction == TOP))) {
      // Dash nach "hinten"
      power = dash_power_needed_for_distance(-abs_y_dist_to_pos);//-100;
      LOG_DAN(0, "old dash power -100, new : " << power);
    }
    basic_cmd->set_dash(power, priority);
    basic_cmd->get_cmd(cmd);
    return true;
  } else if (abs_y_dist_to_pos > 0.4)  {
    // Dash noch "vorne"
    power = dash_power_needed_for_distance(abs_y_dist_to_pos);//50;
    LOG_DAN(0, "old dash power 50, new : " << power);
    if (((pos2.getY() >= my_pos.getY()) && (direction == BOTTOM)) ||
	((pos2.getY() < my_pos.getY()) && (direction == TOP))) {
      // Dash nach "hinten"
      power = dash_power_needed_for_distance(-abs_y_dist_to_pos);//-50;
      LOG_DAN(0, "old dash power -50, new : " << power);
    }
    basic_cmd->set_dash(power, priority);
    basic_cmd->get_cmd(cmd);
    return true;
  } else if ( (abs_y_dist_to_pos > 0.2) &&
	      (ball_pos - WSinfo::me->pos).sqr_norm() < SQUARE(16.0) ){
    // Falls eine kleinere Korrektur n�tig ist, wird nur ein Dash mit 20 durchgef�hrt
    // Dash nach "vorne"
    power = dash_power_needed_for_distance(abs_y_dist_to_pos);//20;
    LOG_DAN(0, "old dash power 20, new : " << power);
    if (((pos2.getY() >= my_pos.getY()) && (direction == BOTTOM)) ||
	((pos2.getY() < my_pos.getY()) && (direction == TOP))) {
      // Dash nach "hinten"
      power = dash_power_needed_for_distance(-abs_y_dist_to_pos);//-20;
      LOG_DAN(0, "old dash power -20, new : " << power);
    }
    basic_cmd->set_dash(power);
    basic_cmd->get_cmd(cmd);
    return true;
  }
  /* ...sonst, falls die richtige Position erreicht ist, wird mit der
   *  momentanen eigenen Geschwindigkeit abgebremst.
   */
  basic_cmd->set_turn(0.0);
  basic_cmd->get_cmd(cmd);
  return true;
} /* go_to_y */



bool Goalie03::go_to_pos(Cmd &cmd, Vector pos) {
  LOG_DEB(0, << " GO_TO_POS USED");
  if ((Tools::my_angle_to(pos).get_value() > 24 * DEG2RAD) && 
      (Tools::my_angle_to(pos).get_value() < 336 * DEG2RAD)) {
    LOG_DAN(BASELEVEL+0, << "Drehung in go_to_pos");
    basic_cmd->set_turn_inertia(Tools::my_angle_to(pos).get_value());
    basic_cmd->get_cmd(cmd);
    return true;
  }
  double dist = (my_pos - pos).norm();
  double power;
  // Bei zu grosser Abweichung wird ein maximaler Dash in die richtige Richtung durchgef�hrt
  if (dist > 1.0) {
    // Dash noch "vorne"
    power = dash_power_needed_for_distance(dist);
    basic_cmd->set_dash(power);
    basic_cmd->get_cmd(cmd);
    return true;
  } else if (dist > 0.3) { //0.3
    // Falls eine kleinere Korrektur n�tig ist, wird nur ein Dash mit 20 durchgef�hrt
    // Dash nach "vorne"
    power = dash_power_needed_for_distance(dist);
    basic_cmd->set_dash(power);
    basic_cmd->get_cmd(cmd);
    return true;
  }
  /* ...sonst, falls die richtige Position erreicht ist, wird mit der
   *  momentanen eigenen Geschwindigkeit abgebremst.
   */
  basic_cmd->set_turn(0.0);
  basic_cmd->get_cmd(cmd);
  return true;
} /* go_to_pos */


int Goalie03::ball_steps_to_with_max_speed(Vector pos) {
  double ball_vel_norm = 3.0;

  double distance = (pos - ball_pos).norm();
  return (int) (log(1 - (1-ServerOptions::ball_decay)*distance/ball_vel_norm) / log(ServerOptions::ball_decay));
} /* ball_steps_to */



/*  Berechnet die Anzahl der Zeitschritte, die der Ball (bei maximaler Geschwindigkeit) 
    bis zum Punkt pos braucht */
int Goalie03::ball_steps_to(Vector pos) {
  /*
    double x_vel;
    if ((ball_pos - ServerOptions::own_goal_pos).norm() > 22) {
    x_vel = 1.5 * ball_vel.x / ball_vel.norm(); 
    } else {
    x_vel = 2.7 * ball_vel.x / ball_vel.norm(); 
    }
  */
  //double distance = -ball_pos.x + pos.x;
  //return (int) (log(1 - 0.06*(distance)/ball_vel.x) / log(0.94));
  if (ball_vel.norm() == 0.0) return 100;
  double distance = (pos - ball_pos).norm();
  return (int) (   log (   1 
                         -   (1-ServerOptions::ball_decay)
                           * distance/ball_vel.norm()
                       ) 
                 / log(ServerOptions::ball_decay));
} /* ball_steps_to */


double Goalie03::skalarprodukt(Vector v1, Vector v2) {
  return v1.getX()*v2.getX() + v1.getY()*v2.getY();
} /* skalarprodukt */



/*  Berechnet die Anzahl der Zeitschritte, die der Goalie bis zum Punkt pos braucht.
    Das ganze geht iterativ, nicht direkt!
*/
int Goalie03::goalie_steps_to(Vector pos) {
  double v0 = fabs(my_vel.getY());
  if ((pos.getY() - my_pos.getY())* my_vel.getY() < 0) {
    v0 = - v0;
  }

  double distance 
    =   (my_pos - pos).norm() 
      - ServerOptions::catchable_area_l; //TG08: -2.0 durch Parameter ersetzt
  double schwelle = distance + 1/0.6 - v0/0.6 - 1;
  double temp = 2/3 - v0/0.6;
  for (int k = 1; k<10; k++) {
    if (k + pow(0.4,k)*temp > schwelle) {
      return k;
    }
  }
  return 10;
} /* goalie_steps_to */


/* Positioniert den Goalie (hoffentlich) ideal zum Ball .
   Gewichtung ist ein Faktor, der angibt, wie nahe der Goalie am kurzen Pfosten steht.
   Bei 1.6 z.B steht er naeher am kurzen Pfosten als normal, bei Werten kleiner 1 steht er weiter weg.
*/
Vector Goalie03::positioniere(double gewichtung, int anticipate_ball_pos) {
  Vector id_Pos;
  double steigung;
  Vector ball = ball_pos;

  PlayerSet pset_tmp = WSinfo::valid_teammates;
  PPlayer p_tmp = pset_tmp.closest_player_to_point(WSinfo::ball->pos);

  if ((anticipate_ball_pos) && 
      !steps2go.ball_kickable_for_opponent && 
      !(p_tmp == NULL) &&
      //!(mdpInfo::teammate_closest_to_ball()==-1) &&
      !steps2go.ball_kickable_for_teammate ) {
    ball = WSinfo::ball->pos + WSinfo::ball->vel;//mdpInfo::ball_next_time_pos_abs();
  }

  if ((anticipate_ball_pos) &&
      (go2ball_steps[0].steps <= 24) &&
      (go2ball_steps[0].steps > 1) &&
      (go2ball_steps[0].number !=WSinfo::me->number)) 
  {
    if (go2ball_steps[0].steps == 2) 
    {
      ball = ball_pos + 1.94 * ball_vel;
    } 
    else if (go2ball_steps[0].steps >= 5) 
    {
      ball = ball_pos + 5.5 * ball_vel;
    } 
    else 
    {
      ball = ball_pos + 2.82 * ball_vel;
    }
  }

  if (!FIELD_AREA.inside(ball)) {
    ball = ball_pos;
  }


  //TG09: used ball position for corners
  Vector usedLeftCorner = left_corner, usedRightCorner = right_corner;
  if ( ball_pos.getX() < -FIELD_BORDER_X + 2.0 )
  {
    usedRightCorner.subFromX( 1.0);
    usedLeftCorner.subFromX( 1.0);
  }

  Vector upper_intersection = point_on_line(usedLeftCorner  - ball_pos, ball_pos, xA);
  Vector lower_intersection = point_on_line(usedRightCorner - ball_pos, ball_pos, xA);
  
  LOG_DAN(0, <<_2D << VC2D(upper_intersection, 0.5, "blue"));
  LOG_DAN(0, <<_2D << VC2D(lower_intersection, 0.5, "blue"));

  double total_dist= (upper_intersection - lower_intersection).norm();
  
  double upper_dist = (upper_intersection - ball_pos).norm();
  double lower_dist = (lower_intersection - ball_pos).norm();
 
  if ((upper_dist > 1.0) && (lower_dist > 1.0)) {
    upper_dist = upper_dist - 1.0;
    lower_dist = lower_dist - 1.0;
  }

  if (lower_dist == 0.0) {
    id_Pos = lower_intersection;
  } else {
    id_Pos = upper_intersection - ((upper_dist/lower_dist)*total_dist/(1.0+upper_dist/lower_dist)) * Vector(0.0, 1.0);
  }

  PPlayer opp = WSinfo::get_opponent_by_number(steps2go.opponent_number);
  if ((opp !=NULL) &&
      WSinfo::is_ball_kickable_for(opp)) {
    if (ball_vel.getY() > 0.2)  {
      LOG_DAN(0, << "Position wg. Geschw. um 0.3 nach oben verschoben!");
      id_Pos.addToY( 0.3 );
    } else if (ball_vel.getY() < -0.2) {
      LOG_DAN(0, << "Position wg. Geschw. um 0.3 nach unten verschoben!");
      id_Pos.subFromY( 0.3 );
    }
  }


  LOG_DAN(0, << " x ist " << ((upper_dist/lower_dist)*total_dist/(1.0+upper_dist/lower_dist)));
  
  LOG_DAN(0, "positioniere 1. liefert " << xA << "," << id_Pos.getY());
  LOG_DAN(0, <<_2D << L2D(WSinfo::me->pos.getX(),WSinfo::me->pos.getY(),xA, id_Pos.getY(), "red"));


  if ((my_pos.getX() > xA) && (fabs(my_pos.getX() - xA) < 1.2)) //TG: TODO: 2.0 vs. Catch-Radius
  {
    //nicht durch 0 teilen
    if (ball.getX() == xA) {
    } else {
      steigung = (id_Pos.getY() - ball.getY()) /
	(ball.getX() - xA);
      id_Pos.setY( (ball.getX() - my_pos.getX()) * steigung + ball.getY() );
    }
  }

  LOG_DAN(0, "positioniere 2. liefert " << xA << "," << id_Pos.getY());
  LOG_DAN(0, <<_2D << C2D(my_pos.getX(), id_Pos.getY(), 0.2, "red"));

  if ((ball_pos.getX() > -ServerOptions::pitch_length/2.0 + 15.0) &&
      ((ball_pos-Vector(-ServerOptions::pitch_length/2.0, 0.0)).norm() > 18.0)) {
    if (id_Pos.getY() > 4.5) {
      return Vector(xA, 4.5);
      //return left_corner;
    }
    if (id_Pos.getY() < -4.5) {
      return Vector(xA, -4.5);
      //return right_corner;
    }
  }

  if (id_Pos.getY() > left_corner.getY()) {
    return Vector(xA, left_corner.getY());
    //return left_corner;
  }
  if (id_Pos.getY() < right_corner.getY()) {
    return Vector(xA, right_corner.getY());
    //return right_corner;
  }
  
  return Vector(xA, id_Pos.getY());

} /* positioniere */


int Goalie03::am_I_at_home(double faktor) {
  if (xA < -50.0) {
    if (my_pos.getX() < xA) {
      if (fabs(my_pos.getX() - xA) >= goalside_tolerance*faktor) {
	return false;
      } else {
	return true;
      }
    } else {
      if (fabs(my_pos.getX() - xA) >= normal_tolerance*faktor) {
	return false;
      } else {
	return true;
      }
    }
  } else {
    if (fabs(my_pos.getX() - xA) >= faktor) {
      return false;
    } else {
      return true;
    }
  }
} /* am_I_at_home */


bool Goalie03::test_2step_go2pos(Cmd &cmd) {
  LOG_DEB(0, << " (test) last_go2pos = " << last_go2pos << " less_than_two_steps = " << less_than_two_steps);
  if (less_than_two_steps) {
    LOG_DAN(0, << " repositioning special case");
    //ERROR_OUT << "time " << WSinfo::ws->time << " USING SPECIAL CASE REPOSITIONING";
    less_than_two_steps = false;
    LOG_DAN(0,<< "Neuro_Go2Pos in test_2step_go2pos");
    return get_Neuro_Go2Pos_back_forw(cmd, saved_go2pos.getX(), saved_go2pos.getY(), 2);
  }
  return false;
}


/* Laesst den Torwart auf einer Linie vor dem Tor laufen */
bool Goalie03::get_Move_Line(Cmd &cmd, int ball_near_goal) {

  if ((ball_pos.getX() < -ServerOptions::pitch_length/2.0 + 5.0) &&
      (fabs(ball_pos.getY()) < 17.0) &&
      (!am_I_at_home(0.5)) &&
      (fabs(positioniere(1.1, 1).getY() - my_pos.getY()) < 1.5)) {
      //      (xA < normal_xA - 0.1)) {
    LOG_DAN(0, << " repositioning ");
    //ERROR_OUT << "time " << WSinfo::ws->time << " repositioning";
    /*
    if (last_go2pos <= WSinfo::ws->time-2) {
      less_than_two_steps = true;
      last_go2pos = WSinfo::ws->time;
    }
    if (last_go2pos == WSinfo::ws->time-1) {
      less_than_two_steps = false;
      last_go2pos = WSinfo::ws->time;
      }*/

    LOG_DAN(0,<< "Neuro_Go2Pos in get_Move_Line");
    return get_Neuro_Go2Pos_back_forw(cmd, xA, (positioniere(1.1, 1)).getY(),2);
  }


  //2.Zeile ist, wenn der Goalie nicht an der Richtigen Pos. steht
  // Neuro_Go2Pos wird nur verwendet, wenn kein Gegner den Ball hat */
  if ( ((!am_I_at_home(1.0)) && !steps2go.ball_kickable_for_opponent) ||
       (!am_I_at_home(1.0) && !am_I_looking_straight(3)) ||
       (!am_I_at_home(1.5))) {
    LOG_DEB(0, << "last_go2pos = " << last_go2pos << " less_than_two_steps = " << less_than_two_steps);
    /*
    if (last_go2pos <= WSinfo::ws->time-2) {
      less_than_two_steps = true;
      last_go2pos = WSinfo::ws->time;
    }
    if (last_go2pos == WSinfo::ws->time-1) {
      less_than_two_steps = false;
      last_go2pos = WSinfo::ws->time;
      }*/

    LOG_DAN(0,<< "Neuro_Go2Pos in get_Move_Line");
    //ERROR_OUT << "time " << WSinfo::ws->time << " repositioning";
    return get_Neuro_Go2Pos_back_forw(cmd, xA, (positioniere(1.1, 1)).getY(),2);
  }
    
  return go_to_y(cmd, positioniere(1.1, 1));
} /* get_Move_Line */


bool Goalie03::get_Neuro_Go2Pos_back_forw(Cmd &cmd, double x, double y, int steps) {


  if (last_go2pos <= WSinfo::ws->time-2) {
    less_than_two_steps = true;
    last_go2pos = WSinfo::ws->time;
    saved_go2pos = Vector(x, y);
  }
  if (last_go2pos == WSinfo::ws->time-1) {
    less_than_two_steps = false;
    last_go2pos = WSinfo::ws->time;
    saved_go2pos = Vector(x, y);
  }

  LOG_DEB(0, << "last_go2pos = " << last_go2pos << " less_than_two_steps = " << less_than_two_steps);  

  mdpInfo::set_my_intention(DECISION_TYPE_RUNTO);

  if (!LEFT_PENALTY_AREA.inside(my_pos)) {
    go2pos->set_target(Vector(x, y), 0.5);
    go2pos->get_cmd(cmd);
    return true;
  }
  /*
  if ((Tools::get_abs_angle(mdpInfo::my_angle_to(Vector(x,y))) > 1.2 * M_PI/2.0) &&
      !((x > -50.0) && (my_pos.x < x))) {
  */
  if (skalarprodukt(Vector(x,y) - my_pos, ball_pos - my_pos) < 0) {
    if (am_I_at_home(2.0) &&
	(fabs(my_pos.getY()) < ServerOptions::goal_width/2.0 + 0.5)) {
      //go2pos->set_target(Vector(x, y), 0.5);
      //go2pos->get_cmd(cmd);
      LOG_DAN(0, << "using Go2PosBackwards!");
      go2pos_backwards->set_params(x, y, 0.5, 12.0*DEG2RAD);
      go2pos_backwards->get_cmd(cmd);
      return true;
    }
    //go2pos->set_target(Vector(x, y), 0.5);
    //go2pos->get_cmd(cmd);
    LOG_DAN(0, << "using Go2PosBackwards!");
    go2pos_backwards->set_params(x, y, 0.5, 15.0*DEG2RAD);
    go2pos_backwards->get_cmd(cmd);
    return true;
  } else {
    LOG_DAN(0, << "using normal Go2Pos!");
    go2pos->set_target(Vector(x, y), 0.5);
    go2pos->get_cmd(cmd);
    return true;
  }
} /* get_Neuro_Go2Pos_back_forw */


/* Berechnet den Schnittpunkt zweier Geraden */
Vector Goalie03::intersection_point(Vector p1, Vector steigung1, Vector p2, Vector steigung2) {
  
  double x, y, m1, m2;
  if ((steigung1.getX() == 0) || (steigung2.getX() == 0)) {
    return Vector(-51.5, 0);
  }
  m1 = steigung1.getY()/steigung1.getX();
  m2 = steigung2.getY()/steigung2.getX();
  if (m1 == m2) return Vector(-51.5, 0);
  x = (p2.getY() - p1.getY() + p1.getX()*m1 - p2.getX()*m2) / (m1-m2);
  y = (x-p1.getX())*m1 + p1.getY();
  return Vector (x, y);

}



/* Berechnet die y-Koordinate Punktes auf der Linie, der die x-Koordinate x hat
 */
Vector Goalie03::point_on_line(Vector steigung, Vector line_point, double x) {
  //steigung.normalize();
  if (steigung.getX() != 0.0) {
    steigung = (1.0/steigung.getX()) * steigung;
  }
  if (steigung.getX() > 0) {
    return (x - line_point.getX()) * steigung + line_point;
  }
  if (steigung.getX() < 0) {
    return (line_point.getX() - x) * steigung + line_point;
  }
  // Zur Sicherheit, duerfte aber nie eintreten
  return line_point;
} /* point_on_line */



int Goalie03::is_ball_heading_for_goal() {
  double steigung;
  if (ball_vel.getX() == 0.0) return 0;
  steigung = -(ball_vel.getY() / ball_vel.getX());
  id_Punkt_auf_Tor = ServerOptions::own_goal_pos +
    Vector(0, ball_pos.getY() + steigung * (ball_pos.getX() - ServerOptions::own_goal_pos.getX()));

  if ((skalarprodukt(ball_vel, Vector(-1,0)) > 0) 
      && (fabs(id_Punkt_auf_Tor.getY()) <= ServerOptions::goal_width/2.0 + 5)) {
    return 1;
  }
  return 0;
} /* is_ball_heading_for_goal */


/* Wird von test_goalshot aufgerufen, wenn der Goalie nicht ganz an der richtigen
   Position steht oder nicht ganz in die richtige Richtung schaut.
   Die ganzen Abfragen dienen dazu, dass so selten wie moeglich ein Intercept bei
   einem Torschuss aufgerufen werden muss.
   Wenn der Goalie z.B. etwas zu weit vor dem Tor steht und gleichzeitig sein Koerper
   zu weit in Richtung Tor steht, kann er trotzdem einfach geradeaus laufen.
*/
bool Goalie03::goalshot_avoid_intercept(Cmd &cmd, Vector pos) {
  LOG_DAN(BASELEVEL+0,<< "goalshot_avoid_intercept in Aktion");
  if ((fabs(my_pos.getY()) > ServerOptions::goal_width/2.0 + 1) ||
      was_last_move_intercept()) {
    last_intercept = WSinfo::ws->time;
    LOG_DAN(BASELEVEL+0, << "intercept() in goalshot_avoid_intercept");
    return intercept_goalie(cmd, 1);
  }

  double ang;

  if (direction == TOP)
    ang = Tools::my_angle_to(Vector(WSinfo::me->pos.getX(), WSinfo::me->pos.getY()+20)).get_value(); //JTS10: change to me->pos.y+20
  else
    ang = Tools::my_angle_to(Vector(WSinfo::me->pos.getX(), WSinfo::me->pos.getY()-20)).get_value(); //JTS10: change to me->pos.y-20

  double add = 0;
 
  if ( ((pos.getY() >  my_pos.getY()) && (direction == BOTTOM)) ||
       ((pos.getY() <= my_pos.getY()) && (direction == TOP)) )
    add = M_PI;
  else
    add = 0.0;
  
 
  if ((my_pos.getX() <= xA) && (my_pos.getX() >= -52.0)) {

    if ( (ang <= 12*DEG2RAD) || (ang >= 348 * DEG2RAD) ) {
      return go_to_y(cmd, pos, 1);
    } else {
      basic_cmd->set_turn_inertia(Tools::get_angle_between_null_2PI(Tools::my_angle_to(pos).get_value() + add));
      basic_cmd->get_cmd(cmd);
      return true;
    }
    
  } else if ((my_pos.getX() <= xA + 2.5) && (my_pos.getX() > xA))  {
    // -50.5 < my_pos.x <= -48.0

    if (pos.getY() > my_pos.getY())
    {
      if (    ( (ang >= 348*DEG2RAD) || (ang <= 12 * DEG2RAD) )
           && (my_pos.getX() <= -49.0)
	     )
	  {
	    return go_to_y(cmd, pos, 1);
      }
      else
      {
	    basic_cmd->set_turn_inertia(Tools::get_angle_between_null_2PI(Tools::my_angle_to(pos).get_value() + add));
	    basic_cmd->get_cmd(cmd);
	    return true;
      }
    } else {
      //pos.y <= my_pos.y
      if ( ((ang >= 348*DEG2RAD) && (my_pos.getX() <= -49.0))  ||
	   (ang <= 12*DEG2RAD) ) {
	return go_to_y(cmd, pos, 1);
      } else {
	basic_cmd->set_turn_inertia(Tools::get_angle_between_null_2PI(Tools::my_angle_to(pos).get_value() + add));
	basic_cmd->get_cmd(cmd);
	return true;
      }
    }
  }
  LOG_DAN(BASELEVEL+0, << "Selecting Move intercept in test_goalshot");
  last_intercept = WSinfo::ws->time;
  return intercept_goalie(cmd, 1);
} /* goalshot_avoid_intercept */



bool Goalie03::test_corner_goalshot(Cmd &cmd) {
  double steigung;
  Vector x;
  if ((ball_vel.getX() == 0.0) || (!am_I_at_home(1.5))) {
    return false;
  }
  steigung = -(ball_vel.getY() / ball_vel.getX());

  //(ball_pos.x - x) * steigung + ball_pos.y = right_corner.y;
  // => x = ball_pos.x + (ball_pos.y - right_corner.y) / steigung
  if ((ball_pos.getY() < right_corner.getY()) &&
      (ball_vel.getY() > 0) &&
      (ball_vel.getX() < 0) &&
      (fabs(ball_vel.getY()) > fabs(ball_vel.getX()))) {
    x.setX( ball_pos.getX() + (ball_pos.getY() - right_corner.getY()) / steigung );
    x.setY( right_corner.getY() );
  } else if ((ball_pos.getY() > left_corner.getY()) &&
	     (ball_vel.getY() < 0) &&
	     (ball_vel.getX() < 0) &&
	     (fabs(ball_vel.getY()) > fabs(ball_vel.getX()))) {
    x.setX( ball_pos.getX() + (ball_pos.getY() - left_corner.getY() ) / steigung );
    x.setY( left_corner.getY() );
  } else {
    return false;
  }
  LOG_DAN(BASELEVEL+1, << " X ist " << x);
  LOG_DAN(BASELEVEL+1, << " Zeit fuer den Ball bis X: " << ball_steps_to(x));

  if (//(x.x+1 < my_pos.x) && 
      (x.getX() < xA + 0.5) &&
      (x.getX() > -53.5) &&
      (ball_steps_to(x) <= 3) &&
      (ball_steps_to(x) >= 0)) {
    LOG_DAN(BASELEVEL+0, << "go_to_pos in test_corner_goalshot");
    return go_to_pos(cmd, Vector(-51.5, x.getY()));
  }
  return false;
} /* test_corner_goalshot() */



bool Goalie03::test_goal_shot(Cmd &cmd) {
  double steigung = -(ball_vel.getY() / ball_vel.getX());
  
  id_pos_when_goalshot = Vector(my_pos.getX(), (ball_pos.getX() - my_pos.getX()) * steigung + ball_pos.getY());

  LOG_DAN(BASELEVEL+1,<< "Zeit " << ball_steps_to(id_pos_when_goalshot));
  LOG_DAN(BASELEVEL+1,<< "Zeit Goalie " << goalie_steps_to(id_pos_when_goalshot));

  /* Abfrage, ob der Ball aufs Tor geschossen wurde und ob er jetzt abgefangen werden muss */
  if (is_ball_heading_for_goal()
      && (ball_steps_to(id_pos_when_goalshot) - 2 <= goalie_steps_to(id_pos_when_goalshot)) 
      && (ball_steps_to(id_pos_when_goalshot) >= 0)) 
  {

#if LOGGING && BASIC_LOGGING
    Vector d = intersection_point(my_pos, Vector(-ball_vel.getY(), ball_vel.getX()), ball_pos, ball_vel);
#endif
    LOG_DAN(BASELEVEL+1, << _2D 
	    << VC2D(intersection_point(my_pos, Vector(-ball_vel.getY(), ball_vel.getX()), ball_pos, ball_vel), 2.0, "blue"));
    LOG_DAN(BASELEVEL+1,<< "Zeit senkrecht" << ball_steps_to(d));
    LOG_DAN(BASELEVEL+1,<< "Zeit Goalie senkrecht" << goalie_steps_to(d));

    //Wenn der Goalie nicht auf seiner Linie steht oder nicht in die richtige Richtung schaut
    if ((!am_I_at_home(1.0)) || 
        (!am_I_looking_straight(3.0))) 
    {
      //Vom alten Goalie uebernommen
      if (Goalie_Bs03::catch_ban)
      {// the catchban is set !!!
	
        LOG_DAN(BASELEVEL+0, << "Selecting Move Intercept Ball, but I can't catch");
        last_intercept = WSinfo::ws->time;
        intercept->get_cmd(cmd);
        return true;
      }
      else
      {
        return goalshot_avoid_intercept(cmd, id_pos_when_goalshot);
	//LOG_DAN(0, << "Selecting Move Goalie Intercept Ball in test_goalshot");
      }
      
    }

  } 
  else 
  {
    LOG_DAN(BASELEVEL+1,<<"DEBUG: no icpt, because is_ball_heading_for_goal()=="
      <<is_ball_heading_for_goal()<<", ball_steps_to(id_pos_when_goalshot)=="
      <<ball_steps_to(id_pos_when_goalshot)<<", goalie_steps_to(id_pos_when_goalshot)=="
      <<goalie_steps_to(id_pos_when_goalshot)<<", id_pos_when_goalshot=="
      <<id_pos_when_goalshot);
    return false;
  }

  return go_to_y(cmd, id_pos_when_goalshot, 1);

} /* test_goal_shot */



/* Prueft, ob der Ball gefangen werden kann und gibt gegebenenfalls ein Catch zurueck
 */
bool Goalie03::test_catch(Cmd &cmd) {
  LOG_DAN(0, << "catchable?????????");
  if (!LEFT_PENALTY_AREA.inside(my_pos)){
    LOG_DAN(0, << "not inside my penalty area");
    return 0;
  }
  if ( is_ball_catchable ){ // If the ball is catchable
    LOG_DAN(0, << "The ball is catchable");
    Goalie_Bs03::i_have_ball = true;
    Goalie_Bs03::catch_ban = ServerOptions::catch_ban_cycle;
    last_catch_time = WSinfo::ws->time;
    basic_cmd->set_catch(Tools::my_angle_to(WSinfo::ball->pos));
    basic_cmd->get_cmd(cmd);
    return true;
  }
  return false;
} /* test_catch */



bool Goalie03::test_kick(Cmd &cmd) {
  Angle kick_dir = 0;
  double kick_vel = 0;
  const double deg5 = M_PI/36;
  double vel;
  if ( WSinfo::is_ball_kickable() ) 
  { // if the ball is not catchable, but kickable
    LOG_DAN(BASELEVEL+1, << "The ball is kickable");
    LOG_DAN(BASELEVEL+0, << "Selecting Move Hold Ball (so I will be able to catch)") ;
    // Warning : We do not yet check wether we are close to our own goal ! 
    //If we are on our goal_line it might happen that the goalie tries to hold 
    //the ball away from an opponent and thereby kicks it into our own goal !!!
    if (   !(LEFT_PENALTY_AREA.inside(WSinfo::me->pos)) 
        || (WSinfo::me->pos.getX() < -51.0))
    { // if we cannot wait for the ball to be catchable with a hold ball move
      PlayerSet playersNearMe = WSinfo::alive_opponents;
      playersNearMe.join( WSinfo::valid_teammates_without_me );
      playersNearMe.keep_players_in_circle( WSinfo::me->pos, 7.0 );
      if (    playersNearMe.num > 0
           && !(LEFT_PENALTY_AREA.inside(WSinfo::me->pos)) )
      {
        kick_vel = ServerOptions::ball_speed_max * 0.975;
        if ( WSinfo::me->pos.getY() > 0 )
          kick_dir = ( Vector(-FIELD_BORDER_X,FIELD_BORDER_Y) - WSinfo::me->pos).arg();
        else
          kick_dir = ( Vector(-FIELD_BORDER_X,-FIELD_BORDER_Y) - WSinfo::me->pos).arg();
        LOG_DAN(BASELEVEL+0,<<"TG@Goalie: I am not in PEN area, kick hard! v="
          <<kick_vel<<" dir="<<kick_dir);
      }
      else
      if (WSinfo::me->pos.getY() > 0)
      {
        PlayerSet nearPlayers = WSinfo::valid_opponents;
        nearPlayers.keep_players_in_circle( WSinfo::me->pos, 30.0 );
        double evalDir = 0.0, bestEval = -1000.0;
        double dirBonus = 0.0;

        for (double a = M_PI_2 + M_PI_4 - deg5 ; a >= M_PI_4 ; a-= deg5) //TG17: +M_PI_4
        {
          ANGLE ang(a);
          Vector panicDir( ang );
          double danger = 0.0, dangerSum = 0.0;
          for ( int np = 0; np < nearPlayers.num; np++ )
          {
            double dist2Intercept = Tools::get_dist2_line( WSinfo::me->pos, WSinfo::me->pos + panicDir, nearPlayers[np]->pos );
            Vector lf = Tools::get_Lotfuss( WSinfo::me->pos, WSinfo::me->pos + panicDir, nearPlayers[np]->pos );
            double dist2Pass = lf.distance(WSinfo::me->pos);
            if (lf.getY() < WSinfo::me->pos.getY()) dist2Pass = 0.0;
            if (fabs(dist2Intercept) < 0.00001) dist2Intercept = 0.1;
            danger = dist2Pass / dist2Intercept;
            dangerSum += danger;
            LOG_DAN(BASELEVEL+0,<<"TG@Goalie: dir="<<(RAD2DEG(a))<<" opp="
               <<nearPlayers[np]->number<<" danger="<<danger<<" dangerSum="<<dangerSum);
          }
          oneortwo->kick_in_dir_with_max_vel( ANGLE(a) );
          oneortwo->get_vel(vel);
          evalDir = vel - dangerSum + dirBonus;
          LOG_DAN(BASELEVEL+0,<<"TG@Goalie: EVAL FOR DIR "<<(RAD2DEG(a))<<" IS "
               <<"VEL "<<vel<<" minus DANGERSUM "<<dangerSum<<" plus DIRBONUS "<<dirBonus<<" => EVAL "<<evalDir);
          //if (vel > kick_vel)
          if (evalDir >= bestEval)
          {
            bestEval = evalDir;
            kick_vel = vel;
            kick_dir = a;
          }
	  /*
	    if (Move_1Step_Kick::get_max_vel_in_dir(Tools::get_angle_between_null_2PI(a)) > kick_vel){
	    kick_vel = Move_1Step_Kick::get_max_vel_in_dir(Tools::get_angle_between_null_2PI(a));
	    kick_dir = a;
	    }*/
	     dirBonus += 0.03;
       }
        LOG_DAN(BASELEVEL+0,<<"TG@Goalie: ### BEST EVAL FOR DIR "<<(RAD2DEG(kick_dir))<<" IS "
             <<bestEval<<" WITH V="<<kick_vel);
     } 
     else 
     {
        PlayerSet nearPlayers = WSinfo::valid_opponents;
        nearPlayers.keep_players_in_circle( WSinfo::me->pos, 30.0 );
        double evalDir = 0.0, bestEval = -1000.0;
        double dirBonus = 0.0;

       for (double a = -M_PI_2 -M_PI_4 + deg5 ; a <= -M_PI_4 ; a+= deg5)
       {
          ANGLE ang(a);
          Vector panicDir( ang );
          double danger = 0.0, dangerSum = 0.0;
          for ( int np = 0; np < nearPlayers.num; np++ )
          {
            double dist2Intercept = Tools::get_dist2_line( WSinfo::me->pos, WSinfo::me->pos + panicDir, nearPlayers[np]->pos );
            Vector lf = Tools::get_Lotfuss( WSinfo::me->pos, WSinfo::me->pos + panicDir, nearPlayers[np]->pos );
            double dist2Pass = lf.distance(WSinfo::me->pos);
            if (lf.getY() > WSinfo::me->pos.getY()) dist2Pass = 0.0;
            if (fabs(dist2Intercept) < 0.00001) dist2Intercept = 0.1;
            danger = dist2Pass / dist2Intercept;
            dangerSum += danger;
            LOG_DAN(BASELEVEL+0,<<"TG@Goalie: dir="<<(RAD2DEG(a))<<" opp="
               <<nearPlayers[np]->number<<" danger="<<danger<<" dangerSum="<<dangerSum);
          }
         oneortwo->kick_in_dir_with_max_vel( ANGLE(a));
         oneortwo->get_vel(vel);
          evalDir = vel - dangerSum + dirBonus;
          LOG_DAN(BASELEVEL+0,<<"TG@Goalie: EVAL FOR DIR "<<(RAD2DEG(a))<<" IS "
               <<"VEL "<<vel<<" minus DANGERSUM "<<dangerSum<<" plus DIRBONUS "<<dirBonus<<" => EVAL "<<evalDir);
         //if (vel > kick_vel)
          if (evalDir >= bestEval)
         {
            bestEval = evalDir;
           kick_vel = vel;
           kick_dir = a;
         }
	  /*
	    if (Move_1Step_Kick::get_max_vel_in_dir(Tools::get_angle_between_null_2PI(a)) > kick_vel){
	    kick_vel = Move_1Step_Kick::get_max_vel_in_dir(Tools::get_angle_between_null_2PI(a));
	    kick_dir = a;
	    }*/
	     dirBonus += 0.03;
       }
        LOG_DAN(BASELEVEL+0,<<"TG@Goalie: ### BEST EVAL FOR DIR "<<(RAD2DEG(kick_dir))<<" IS "
             <<bestEval<<" WITH V="<<kick_vel);
     }
     if (kick_vel > 0.5)
     { //we have found a good direction
       neuro_kick->kick_in_dir_with_initial_vel(kick_vel, ANGLE(kick_dir));
       neuro_kick->get_cmd(cmd);
       return true;
     } 
     else 
     {
       hold_turn->get_cmd(cmd, ANGLE(0.0));
       return true;
     }
   } 
   else 
   {
     LOG_DAN(0,<<"INFO: me->pos"<<WSinfo::me->pos);
     hold_turn->get_cmd(cmd, ANGLE(0.0));
     return true;
   } 
  }
  return false;
} /* test_kick */

/*
bool Goalie03::test_teammate_control(){
  Main_Move * move2;
  if (mdpInfo::is_ball_kickable_for_teammate(mdpInfo::teammate_closest_to_ball())) { 
    // if it is not catchable, not kickable, but a teammate can kick
    LOG_DAN(1, << "A Teammate controls the ball");
    LOG_DAN(0, << "Selecting Move Line " );
    if ((move2 = test_turn_around()));
    else if ((move2 = test_turn_up()));
    else if ((move2 = get_Move_Line(1)));
    return move2;
  }
  return NULL;
  }*/ /* test_teammate_control */

 
bool Goalie03::test_turn_up(Cmd &cmd) {
  double tolerance;
  if (fabs(my_pos.getY() - positioniere(1.1, 1).getY()) <= 0.3) {
    tolerance = 2.0;
  } else {
    tolerance = 3.0;
  }
  double ang;
  if (direction== TOP) 
    ang = Tools::my_angle_to(Vector(WSinfo::me->pos.getX(), WSinfo::me->pos.getY()+20)).get_value(); //JTS10: change to me->pos.y+20
  else
    ang = Tools::my_angle_to(Vector(WSinfo::me->pos.getX(), WSinfo::me->pos.getY()-20)).get_value(); //JTS10: change to me->pos.y-20
  
  if (am_I_at_home(1.0)) {
    if ( (ang> tolerance*DEG2RAD) && 
	 (ang < (360.0-tolerance)*DEG2RAD) ) {
      if (direction == TOP) {
	LOG_DAN(BASELEVEL+0, << "Drehung Richtung TOP test_turn_up");
      } else {
	LOG_DAN(BASELEVEL+0, << "Drehung Richtung BOTTOM test_turn_up");
      }
      basic_cmd->set_turn_inertia(ang);
      basic_cmd->get_cmd(cmd);
      return true;
    }
  }
  return false;
} /* test_turn_up */


int Goalie03::opponent_steps_to_ball() {
  return static_cast<int>(steps2go.opponent);
} /* opponent_steps_to_ball */


int Goalie03::teammate_steps_to_ball() {
  return static_cast<int>(steps2go.teammate);
} /* teammate_steps_to_ball */

void Goalie03::compute_steps2go(){
  /* calculate number of steps to intercept ball, for every player on the pitch, return sorted list */
  Policy_Tools::go2ball_steps_update();
  go2ball_steps = Policy_Tools::go2ball_steps_list();

  steps2go.me = 1000;
  steps2go.opponent = 1000;
  steps2go.opponent_number = 0;
  steps2go.teammate = 1000;
  steps2go.teammate_number = 0;
  steps2go.my_goalie = 1000;

  
  /* get predicted steps to intercept ball for me, my goalie , fastest teammate and fastest opponent */
  for(int i=0;i<22;i++){
    if(go2ball_steps[i].side == Go2Ball_Steps::MY_TEAM){
      if(WSinfo::me->number == go2ball_steps[i].number){
	steps2go.me = go2ball_steps[i].steps;
      }
      else if(WSinfo::ws->my_goalie_number == go2ball_steps[i].number){
	steps2go.my_goalie = go2ball_steps[i].steps;
	
      }
      else if(steps2go.teammate == 1000){
	steps2go.teammate = go2ball_steps[i].steps;
	steps2go.teammate_number = go2ball_steps[i].number;
      }
    }
    if(go2ball_steps[i].side == Go2Ball_Steps::THEIR_TEAM){
      if(steps2go.opponent == 1000){
	steps2go.opponent = go2ball_steps[i].steps;
	steps2go.opponent_number = go2ball_steps[i].number;
      }
    } 
  }
  /*  char buffer[200];
      sprintf(buffer,"Go2Ball Analysis: Steps to intercept (Me %.2f) (Teammate %d %.2f) (Opponent %d %.2f)", 
      steps2go.me, steps2go.teammate_number, steps2go.teammate,  steps2go.opponent_number,  
      steps2go.opponent);  
      LOG_DAN(2, << buffer);
  */

  PPlayer p_tmp = WSinfo::get_opponent_by_number(steps2go.opponent_number);
  if (p_tmp != NULL) {
    steps2go.opponent_pos = p_tmp->pos;
    steps2go.ball_kickable_for_opponent = WSinfo::is_ball_kickable_for(p_tmp);
  } else {
    steps2go.opponent_pos = Vector(0.0,0.0);
    steps2go.ball_kickable_for_opponent = false;
  }

  PPlayer p_tmp2 = WSinfo::get_teammate_by_number(steps2go.teammate_number);
  if (p_tmp2 != NULL) {
    steps2go.teammate_pos = p_tmp2->pos;
    steps2go.ball_kickable_for_teammate = WSinfo::is_ball_kickable_for(p_tmp2);
  } else {
    steps2go.teammate_pos = Vector(0.0,0.0);
    steps2go.ball_kickable_for_teammate = false;
  }

  steps2go.me = intercept->get_steps2intercept();

}


void Goalie03::log_steps2go(){
  //Policy_Tools::go2ball_steps_update();
  //go2ball_steps = Policy_Tools::go2ball_steps_list();

#if LOGGING && BASIC_LOGGING
  float me = 1000;
#endif
  float opponent = 1000;
#if LOGGING && BASIC_LOGGING
  int opponent_number = 0;
#endif
  float teammate = 1000;
#if LOGGING && BASIC_LOGGING
  int teammate_number = 0;
#endif

  
  for(int i=0;i<22;i++){
    if(go2ball_steps[i].side == Go2Ball_Steps::MY_TEAM){
      if(WSinfo::me->number == go2ball_steps[i].number){
#if LOGGING && BASIC_LOGGING
	me = go2ball_steps[i].steps;
#endif
      }
      else if(teammate == 1000){
	teammate = go2ball_steps[i].steps;
#if LOGGING && BASIC_LOGGING
	teammate_number = go2ball_steps[i].number;
#endif
      }
    }
    if(go2ball_steps[i].side == Go2Ball_Steps::THEIR_TEAM){
      if(opponent == 1000){
	opponent = go2ball_steps[i].steps;
#if LOGGING && BASIC_LOGGING
	opponent_number = go2ball_steps[i].number;
#endif
      }
    } 
    
  }
  LOG_DAN(BASELEVEL+1, << "My steps to ball:" << me);
  LOG_DAN(BASELEVEL+1, << "Opp. steps to ball:" << opponent << " " << opponent_number);
  LOG_DAN(BASELEVEL+1, << "Teamm. steps to ball:" << teammate << " " << teammate_number);
} /* log_steps2go */

 
int Goalie03::was_last_move_intercept() {
  return (last_intercept + 1 == WSinfo::ws->time);
} /* was_last_move_intercept */


int Goalie03::goalie_early_intercept(int *resPtrA, Vector *target_pos) {
  int res_time;
  Vector res_pos;
  Policy_Tools::intercept_min_time_and_pos_hetero(res_time, res_pos, ball_pos, ball_vel, my_pos, 
					   WSinfo::me->number, true, 0.9, -1000.0);
  int second_fastest_steps_to_ball;
  if (teammate_steps_to_ball() < opponent_steps_to_ball()) {
    second_fastest_steps_to_ball = teammate_steps_to_ball();
  } else {
    second_fastest_steps_to_ball = opponent_steps_to_ball();
  }

  Vector bpos = ball_pos;
  Vector bvel = ball_vel;

  if (mdpInfo::is_object_in_my_penalty_area(res_pos, 2.5) || 
      (was_last_move_intercept() && mdpInfo::is_object_in_my_penalty_area(res_pos, 1.5))) {
    *resPtrA = 1;
    LOG_DAN(BASELEVEL+0, << "Goalie early intercept, starting intercept ball");
    return 1;
  }

  for (int i=0; i<second_fastest_steps_to_ball; i++) {
    bpos += bvel;
    if (mdpInfo::is_object_in_my_penalty_area(bpos, 2.5) ||
	(was_last_move_intercept() && mdpInfo::is_object_in_my_penalty_area(bpos, 1.5))) {
      *resPtrA = 2;
      *target_pos = bpos;
      LOG_DAN(BASELEVEL+0, << "Goalie early intercept, starting go2pos");
      return 2;
    }
    bvel /= ServerOptions::ball_decay;
  }
  
  *resPtrA = 0;
  return 0;
}


/* Diese Methode darf nur nach test_goalshot aufgerufen werden, ansonsten fehlen id_pos_when_goalshot
   und id_Punkt_auf_Tor
*/
bool Goalie03::test_fastest_to_ball(Cmd &cmd) {
  int goalie_e_i = 0;
  Vector target_pos;
  log_steps2go();
  
  int intercept_steps = intercept->get_steps2intercept();

  //noch abfragen, ob der Goalie schon vorher ein Intercept gestartet hat, dann weiter laufen,
  if ((go2ball_steps[0].side == 1) 
      && (go2ball_steps[0].steps <= 40) //24
      && (go2ball_steps[0].number == WSinfo::me->number) //am I the fastest player

      && ( (go2ball_steps[1].steps >= intercept_steps + 2) //am I really faster OR
	   || (was_last_move_intercept() && (go2ball_steps[1].steps >= intercept_steps - 1) //did I already start intercepting and I am probably the fastest and I've left my homepos
	       && (!am_I_at_home(1.0) || (fabs(my_pos.getY()) > ServerOptions::goal_width/2.0 + 0.5))) )
      && (mdpInfo::is_object_in_my_penalty_area(WSinfo::ball->pos)
	  || goalie_early_intercept(&goalie_e_i, &target_pos)) 
      && ( ((WSinfo::ball->age == 0) && (WSinfo::ball->age_vel == 0)) ||
	   was_last_move_intercept()) ) {
    
    //test if not a goalshot
    if (!is_ball_heading_for_goal()
	  || (ball_steps_to(id_pos_when_goalshot) - 4 > 
	       goalie_steps_to(id_pos_when_goalshot))
	  || was_last_move_intercept()
	  || !(ball_steps_to(id_pos_when_goalshot) >= 0)) {

      LOG_DAN(BASELEVEL+1, << "I am the fastest player to the ball");
      
      if  (Goalie_Bs03::catch_ban){// the catchban is set !!!
	LOG_DAN(BASELEVEL+0, << "Selecting Move Intercept Ball, but I can't catch");
	last_intercept = WSinfo::ws->time;
	if (goalie_e_i < 2) {
	  intercept->get_cmd(cmd);
	} else {
	  go2pos->set_target(target_pos, 0.5);
	  go2pos->get_cmd(cmd);
	  return true;
	}
      } else {
	LOG_DAN(BASELEVEL+0, << "Selecting Move Goalie Intercept Ball(test_fastest_to_ball)");
	last_intercept = WSinfo::ws->time;
	if (goalie_e_i < 2) {
	  return intercept_goalie(cmd);
	} else {
	  go2pos->set_target(target_pos, 0.5);
	  go2pos->get_cmd(cmd);
	  return true;
	}
      }
    }
  } else if (was_last_move_intercept()) {
    //last move was intercept, but now I think I am much slower than the fastest opponent => continue
    //intercepting for one cycle, DON'T update last_intercept
    if  (Goalie_Bs03::catch_ban){// the catchban is set !!!
      LOG_DAN(BASELEVEL+0, << "Selecting Move Intercept Ball, SECOND but I can't catch");
      if (goalie_e_i < 2) {
	intercept->get_cmd(cmd);
	return true;
      } else {
	go2pos->set_target(target_pos);
	go2pos->get_cmd(cmd);
	return true;
      }
    } else {
      LOG_DAN(BASELEVEL+0, << "Selecting Move Goalie Intercept Ball(test_fastest_to_ball) SECOND");
      if (goalie_e_i < 2) {
	return intercept_goalie(cmd);
      } else {
	go2pos->set_target(target_pos);
	go2pos->get_cmd(cmd);
	return true;
      }
    }
  } 
  return false;
} /* test_fastest_to_ball */


int Goalie03::is_turning_allowed() {
  double steigung2;
  Vector id_Punkt_auf_Tor2;
  steigung2 = -(ball_vel.getY() / ball_vel.getX());
  id_Punkt_auf_Tor2 = ServerOptions::own_goal_pos + 
    Vector(0, ball_pos.getY() + steigung2 * (ball_pos.getX() - ServerOptions::own_goal_pos.getX()));

  if (is_ball_heading_for_goal()
      && (ball_steps_to(id_Punkt_auf_Tor2) < goalie_steps_to(id_Punkt_auf_Tor2) + 4)
      && (ball_steps_to(id_Punkt_auf_Tor2) >= 0)) {
    return 0;
  } else {
    return 1;
  }
} /* is_turning_allowed */


bool Goalie03::test_turn_around(Cmd &cmd) {
  //double offset;
  double angle;
  double turn;

  if (!am_I_at_home(1.0)) {
    return false;
  }
  
  if (direction == TOP) {
    angle = (Tools::my_angle_to(Vector(WSinfo::me->pos.getX(), WSinfo::me->pos.getY()+20)).get_value() - //JTS10: change to me->pos.y+20
	     Tools::my_angle_to(ball_pos).get_value());
  } else {
    angle = (Tools::my_angle_to(Vector(WSinfo::me->pos.getX(), WSinfo::me->pos.getY()-20)).get_value() - //JTS10: change to me->pos.y-20
	     Tools::my_angle_to(ball_pos).get_value());
  }
  angle = Tools::get_angle_between_null_2PI(angle) * RAD2DEG;

  LOG_DAN(BASELEVEL+1, << "my_abs_angle = " << angle);
  LOG_DAN(BASELEVEL+1, << "My_Pos = " << my_pos);
  

  if (((angle >= 95.0) &&
       (angle <= 265.0) &&
       !steps2go.ball_kickable_for_opponent &&
       (opponent_steps_to_ball() >= 2) &&
       (fabs(positioniere(1.1, 0).getY() - my_pos.getY()) < 1.0) &&
       is_turning_allowed()) ||
      ((angle >= 110.0) &&
       (angle <= 250.0) &&
       is_turning_allowed())) {

    if (direction == TOP) {
      direction = BOTTOM ;
      //offset = DEG2RAD;
      if ( ((turn = Tools::my_angle_to(Vector(WSinfo::me->pos.getX(), WSinfo::me->pos.getY()-20)).get_value()) < M_PI) && //JTS10: change to me->pos.y-20
	   (Tools::my_angle_to(ball_pos).get_value() > 90*DEG2RAD) ) {
	turn = M_PI + DEG2RAD;
      }
      LOG_DAN(BASELEVEL+0, << "Drehung(test_turn_around)");
      basic_cmd->set_turn_inertia(turn);
      basic_cmd->get_cmd(cmd);
      return true;
    } else {
      direction = TOP;
      //offset = -DEG2RAD;
      if (((turn = Tools::my_angle_to(Vector(WSinfo::me->pos.getX(), WSinfo::me->pos.getY()+20)).get_value()) > M_PI) && //JTS10: change to me->pos.y+20
	  Tools::my_angle_to(ball_pos).get_value() < 270*DEG2RAD) {
	turn = M_PI - DEG2RAD;
      }
      LOG_DAN(BASELEVEL+0, << "Drehung(test_turn_around)");
      basic_cmd->set_turn_inertia(turn);
      basic_cmd->get_cmd(cmd);
      return true;
    }
    //LOG_DAN(0, << "Drehung");
  }

  return false;
} /* test_turn_around */


#//Wieviele Schritte braucht der Ball, bis der Goalie sich drehen kann und nach der Drehung immer noch den Ball sieht?
int Goalie03::ball_steps_to_turn() {
  Vector sum = ball_pos;
  Vector vel = 0.94 * ball_vel;
  if ((direction == BOTTOM) && (ball_vel.getY() <= 0)) {
    return 100;
  }
  if ((direction == TOP) && (ball_vel.getY() >= 0)){
    return 100;
  }

  for (int i=0; i < 15;i++) {
    sum += vel;
    vel *= 0.94;
    if ((direction == BOTTOM) && (sum.getY() >= my_pos.getY())) {
      return i + 1;
    }
    if ((direction == TOP) && (sum.getY() <= my_pos.getY())){
      return i + 1;
    }
  }
  return 100;
}

Vector Goalie03::expected_ball_pos(int steps) {
  Vector sum = Vector(0.0, 0.0);
  Vector vel = 0.94 * ball_vel;
  for (int i=0; i < steps;i++) {
    sum += vel;
    vel *= 0.94;
  }
  return ball_pos + sum;

} /* expected_ball_pos */


int Goalie03::am_I_looking_straight(double tolerance) {
  double ang;
  if (direction == TOP)
    ang = Tools::my_angle_to(Vector(WSinfo::me->pos.getX(), WSinfo::me->pos.getY()+20)).get_value(); //JTS10: change to me->pos.y+20
  else
    ang = Tools::my_angle_to(Vector(WSinfo::me->pos.getX(), WSinfo::me->pos.getY()-20)).get_value(); //JTS10: change to me->pos.y-20

  if ( (ang > tolerance*M_PI/180) && 
       (ang < (360-tolerance)*M_PI/180) ) {
    return false;
  } else {
    return true;
  }
} /* am_I_looking_straight */

Vector Goalie03::positioniere_trapez() {
  Vector b_pos;
  double move_pos_to_top = 0.0;

  if ((last_player_line > START_MOVE_TRAPEZ) && (ball_pos.getX() > START_MOVE_TRAPEZ)) {
    return Vector(TOP_RIGHT.getX(), 0);
  }

  if (steps2go.ball_kickable_for_opponent) {
    b_pos = steps2go.opponent_pos;
    //b_pos = ball_pos;
  } else {
    b_pos = ball_pos;
    //gefaehrlich, wenn die next_intercept_pos hinter dem Trapez liegt
    //b_pos = Policy_Tools::next_intercept_pos();
  }

  Vector opp_to_goal_vert = steps2go.opponent_pos - Vector(-ServerOptions::pitch_length/2.0,0);
  opp_to_goal_vert.rotate(M_PI/2.0);
  opp_to_goal_vert.normalize();

  /*
  if (steps2go.ball_kickable_for_opponent) {
    move_pos_to_top = 1.2*skalarprodukt(opp_to_goal_vert, ball_pos - steps2go.opponent_pos);
    LOG_DAN(BASELEVEL+0, << "pos um " << move_pos_to_top << " verschoben");
  */
    /*
      if (skalarprodukt(opp_to_goal_vert, ball_pos - mdpInfo::opponent_pos_abs(steps2go.opponent_number)) >= 0.0) {
      move_pos_to_top = 1; 
      LOG_DAN(0, << "pos nach oben verschoben");
      } else if (skalarprodukt(opp_to_goal_vert, ball_pos - mdpInfo::opponent_pos_abs(steps2go.opponent_number)) < 0.0) {
      move_pos_to_top = 0; 
      LOG_DAN(0, << "pos nach unten verschoben");
      }*/
  //} 

  Vector steigung = Vector(-ServerOptions::pitch_length/2.0, 0.0) - b_pos;
  Vector line_pos = point_on_line(steigung, b_pos, TOP_RIGHT.getX());
  if (fabs(line_pos.getY()) <= 5.0) {
    if (fabs(line_pos.getY() + move_pos_to_top) <= 5.0) return Vector(line_pos.getX(), line_pos.getY() + move_pos_to_top);
    //if ((line_pos.y >= -4.0) && (move_pos_to_top == 0)) return Vector(line_pos.x, line_pos.y - 1.0);
    //if ((line_pos.y <= 4.0) && (move_pos_to_top == 1)) return Vector(line_pos.x, line_pos.y + 1.0);
    return line_pos;
  }
  if (b_pos.getY() > 0.0) {
    if (move_pos_to_top < 0.0) {
      line_pos = intersection_point(b_pos,Vector(-ServerOptions::pitch_length/2.0, 2.0*move_pos_to_top)-b_pos,
				    TOP_LEFT,TOP_LEFT-TOP_RIGHT);
    } else if (move_pos_to_top >= 0.0) {
      line_pos = intersection_point(b_pos,Vector(-ServerOptions::pitch_length/2.0, 3.0*move_pos_to_top)-b_pos,
				    TOP_LEFT,TOP_LEFT-TOP_RIGHT);//1.0
    }
    if (line_pos.getX() < -48.0) {
      use_trapez = false;
    }
    return line_pos;
  } else {
    if (move_pos_to_top < 0.0) {
      line_pos = intersection_point(b_pos,Vector(-ServerOptions::pitch_length/2.0, 3.0*move_pos_to_top)-b_pos,
				    BOTTOM_LEFT,BOTTOM_LEFT-BOTTOM_RIGHT);//1.0
    } if (move_pos_to_top >= 0.0) {
      line_pos = intersection_point(b_pos,Vector(-ServerOptions::pitch_length/2.0, 2.0*move_pos_to_top)-b_pos,
				    BOTTOM_LEFT,BOTTOM_LEFT-BOTTOM_RIGHT);
    }
    if (line_pos.getX() < -48.0) {
      use_trapez = false;
    }
    return line_pos;
  }
} /* positioniere_trapez() */


int Goalie03::was_last_move_intercept_p() {
  return (last_intercept_outside_p + 1 == WSinfo::ws->time);
} /* was_last_move_intercept */



bool Goalie03::test_intercept_trapez(Cmd &cmd) {



  int steps;
  Vector next_i_pos;

  Policy_Tools::intercept_min_time_and_pos_hetero(steps, next_i_pos, WSinfo::ball->pos, 
				    WSinfo::ball->vel, WSinfo::me->pos, 
				    WSinfo::me->number, true, -1.0, -1000.0); 
  //LOG_DAN(0, << _2D << C2D(next_i_pos.x, next_i_pos.y, 1, "#000000"));

  if (!steps2go.ball_kickable_for_opponent &&
      (mdpInfo::is_object_in_my_penalty_area(next_i_pos, 2.0) ||
      (mdpInfo::is_object_in_my_penalty_area(next_i_pos, 0.2) && (ball_vel.norm() <= 0.05))) &&
      (go2ball_steps[0].number == WSinfo::me->number) &&
      (steps2go.me + 1 <= steps2go.opponent)) {// ||
    /*
      mdpInfo::is_object_in_my_penalty_area(steps2go.opponent_pos+Vector(ServerOptions::kickable_area+0.5, 0.0)) &&
      steps2go.ball_kickable_for_opponent &&
      ((ball_pos - my_pos).sqr_norm() < 9.0)) { //3.0
    */
    LOG_DAN(BASELEVEL+0, << "test_intercept_trapez() aktiv");
    if (steps2go.ball_kickable_for_opponent) {
      Vector op_pos = steps2go.opponent_pos;
      return get_Neuro_Go2Pos_back_forw(cmd, op_pos.getX(), op_pos.getY(), 1);
    }
    return intercept_goalie(cmd);
  }


  if (intercept_out_of_penalty_area) { //intercept_out_of_penalty_area) {
    if ((go2ball_steps[0].number == WSinfo::me->number) &&
	(steps2go.me + 6 <= steps2go.opponent) &&
	(steps2go.me + 6 <= steps2go.teammate) &&
	//(steps2go.teammate + 4 >= steps2go.opponent) &&
	(fabs(next_i_pos.getY()) <= ServerOptions::penalty_area_width/3.0) &&
	(next_i_pos.getX() >= -ServerOptions::pitch_length/2.0 + ServerOptions::penalty_area_length) &&
	(next_i_pos.getX() <= -22.0) &&
	(mdpInfo::stamina4meters_if_dashing(100) > 1.24 * (next_i_pos-my_pos).norm())) {
      last_intercept_outside_p = WSinfo::ws->time;
      LOG_DAN(BASELEVEL+0, << "neuro intercept ball in test_intercept_trapez");
      intercept->get_cmd(cmd);
      return true;
    }
    if (was_last_move_intercept_p()) {
      if ((go2ball_steps[0].number == WSinfo::me->number) &&
	  (steps2go.me + 4 <= steps2go.opponent) &&
	  (steps2go.me + 4 <= steps2go.teammate) &&
	  //(steps2go.teammate + 4 >= steps2go.opponent) &&
	  (fabs(next_i_pos.getY()) <= ServerOptions::penalty_area_width/3.0 + 2.0) //TG: TODO: Bedeutung dieser 2.0
    && (next_i_pos.getX() >= -ServerOptions::pitch_length/2.0 + ServerOptions::penalty_area_length - 1.0) &&
	  (next_i_pos.getX() <= -20.0) &&
	  (mdpInfo::stamina4meters_if_dashing(100) > 1.2 * (next_i_pos-my_pos).norm())) {
	last_intercept_outside_p = WSinfo::ws->time;
      }
      LOG_DAN(BASELEVEL+0, << "neuro intercept ball in test_intercept_trapez");
      intercept->get_cmd(cmd);
      return true;
    }
  }
  return false;
} /* test_intercept_trapez() */


bool Goalie03::test_goalshot_trapez(Cmd &cmd) {
  Vector b_pos;

  double steigung2 = -(ball_vel.getY() / ball_vel.getX());
  
  Vector i_pos_w_shot = Vector(my_pos.getX(), (ball_pos.getX() - my_pos.getX()) * steigung2 + ball_pos.getY());
  
  LOG_DAN(BASELEVEL+1, << "i_pos_w_shot = " << i_pos_w_shot);
  
  if (!is_ball_heading_for_goal() ||
      (ball_steps_to(i_pos_w_shot) - 2 > goalie_steps_to(i_pos_w_shot)) ||
      (ball_steps_to(i_pos_w_shot) < 0)) {
    return 0;
  }
  /*
    if (!is_ball_heading_for_goal() ||
    (ball_vel.norm() < 0.8) || 
    mdpInfo::is_ball_kickable_for_opponent(steps2go.opponent_number) || 
    (steps2go.opponent < steps2go.me)) {
    return 0;
    }*/

  b_pos = ball_pos;

  LOG_DAN(BASELEVEL+0, << "goal_shot_trapez");

  if (test_turn_up_trapez(cmd, 15.0) != 0) {
    LOG_DAN(BASELEVEL+0, << "intercept in test_goalshot_trapez()");
    return intercept_goalie(cmd, 1);
  }

  Vector steigung = ball_vel;
  Vector line_pos = point_on_line(steigung, b_pos, TOP_RIGHT.getX());
  if (fabs(line_pos.getY()) <= 5.0) return go_to_y_trapez(cmd, line_pos, 1);
  if (b_pos.getY() > 0.0) {
    line_pos = intersection_point(b_pos, steigung, TOP_LEFT, TOP_LEFT - TOP_RIGHT);
    if (line_pos.getX() < -47.0) {
      //use_trapez = false;
    }
  } else {
    line_pos = intersection_point(b_pos, steigung, BOTTOM_LEFT, BOTTOM_LEFT - BOTTOM_RIGHT);
    if (line_pos.getX() < -47.0) {
      //use_trapez = false;
    }
  }
  return go_to_y_trapez(cmd, line_pos, 1);
} /* test_goal_shot_trapez() */


//(xA,yA) gibt an, ab welchem Punkt das Trapez abknickt */
int Goalie03::am_i_at_home_trapez(double tolerance) {
  //if (t && (fabs(my_pos.y) > 4.0)) return 1;
  if (fabs(pos_trap.getY()) <= TOP_RIGHT.getY()) {
    if (fabs(my_pos.getX() - TOP_RIGHT.getX()) < tolerance) {
      return 1;
    }
    else return 0;
  } else {
    if ((fabs(my_pos.getX() - pos_trap.getX()) < tolerance) &&
	(fabs(my_pos.getY() - pos_trap.getY()) < tolerance)) {
      return 1;
    }
    else return 0;
  }
} /* am_i_at_home_trapez */


bool Goalie03::test_turn_around_trapez(Cmd &cmd) {
  //double offset;
  double angle;
  double turn;

  /* Hier nicht noetig
     if (!am_I_at_home(1.0, xA)) {
     return NULL;
     }*/


  //LOG_DAN(0, << " pos_trap = " << pos_trap.x << " " << pos_trap.y);
  //LOG_DAN(0, << " direction_t 1. = " << direction_t);

  if ((direction_t == NORTH) && (pos_trap.getY() >= 5.5)) {
    direction_t = NORTH_WEST;
  }
  else if ((direction_t == SOUTH) && (pos_trap.getY() >= 5.5)) {
    direction_t = SOUTH_EAST;
  }
  else if ((direction_t == NORTH_WEST) && (pos_trap.getY() < 4.5)) {
    direction_t = NORTH;
  }
  else if ((direction_t == SOUTH_EAST) && (pos_trap.getY() < 4.5)) {
    direction_t = SOUTH;
  }

  else if ((direction_t == NORTH_EAST) && (pos_trap.getY() > -4.5)) {
    direction_t = NORTH;
  }
  else if ((direction_t == SOUTH_WEST) && (pos_trap.getY() > -4.5)) {
    direction_t = SOUTH;
  }
  else if ((direction_t == NORTH) && (pos_trap.getY() <= -5.5)) {
    direction_t = NORTH_EAST;
  }
  else if ((direction_t == SOUTH) && (pos_trap.getY() <= -5.5)) {
    direction_t = SOUTH_WEST;
  }

  //LOG_DAN(0, << " direction_t 2. = " << direction_t);

  if ((direction_t == NORTH_WEST) && (my_pos.getY() > 4.5)) {
    angle = Tools::get_angle_between_null_2PI(TOP_TRAPEZ_ANGLE - (ball_pos - my_pos).arg());
    //angle = Tools::get_angle_between_null_2PI(TOP_TRAPEZ_ANGLE - mdpInfo::mdp->me->ang.v - mdpInfo::my_angle_to(ball_pos));
  }
  else if ((direction_t == SOUTH_EAST) && (my_pos.getY() > 4.5)) {
    angle = Tools::get_angle_between_null_2PI(TOP_TRAPEZ_ANGLE - (ball_pos - my_pos).arg() + M_PI);
    //angle = Tools::get_angle_between_null_2PI(TOP_TRAPEZ_ANGLE - mdpInfo::mdp->me->ang.v - mdpInfo::my_angle_to(ball_pos)+ M_PI);
  }
  else if ((direction_t == NORTH) && (fabs(my_pos.getY()) <= 5.5)) {
    angle = Tools::get_angle_between_null_2PI(M_PI/2.0 - (ball_pos - my_pos).arg());
    //angle = Tools::get_angle_between_null_2PI(mdpInfo::mdp->me->ang.v - M_PI/2.0 - mdpInfo::my_angle_to(ball_pos));
  }
  else if ((direction_t == SOUTH) && (fabs(my_pos.getY()) <= 5.5)) {
    angle = Tools::get_angle_between_null_2PI((ball_pos - my_pos).arg() - 1.5*M_PI);
    //angle = Tools::get_angle_between_null_2PI(1.5*M_PI - mdpInfo::mdp->me->ang.v - mdpInfo::my_angle_to(ball_pos));
  }
  else if ((direction_t == NORTH_EAST) && (my_pos.getY() < -4.5)) {
    angle = Tools::get_angle_between_null_2PI((ball_pos - my_pos).arg() - BOTTOM_TRAPEZ_ANGLE + M_PI);
    //angle = Tools::get_angle_between_null_2PI(BOTTOM_TRAPEZ_ANGLE - mdpInfo::mdp->me->ang.v + M_PI - mdpInfo::my_angle_to(ball_pos));
  }
  else if ((direction_t == SOUTH_WEST) && (my_pos.getY() < -4.5)) {
    angle = Tools::get_angle_between_null_2PI((ball_pos - my_pos).arg() - BOTTOM_TRAPEZ_ANGLE);
    //angle = Tools::get_angle_between_null_2PI(BOTTOM_TRAPEZ_ANGLE - mdpInfo::mdp->me->ang.v - mdpInfo::my_angle_to(ball_pos));
  } else return 0;

  angle = angle * RAD2DEG;
  
  double steigung2 = -(ball_vel.getY() / ball_vel.getX());
  
  Vector i_pos_w_shot = Vector(my_pos.getX(), (ball_pos.getX() - my_pos.getX()) * steigung2 + ball_pos.getY());
  
  LOG_DAN(BASELEVEL+1, << "i_pos_w_shot = " << i_pos_w_shot);
  
  if (is_ball_heading_for_goal() &&
      (ball_steps_to(i_pos_w_shot) - 2 < goalie_steps_to(i_pos_w_shot)) &&
      (ball_steps_to(i_pos_w_shot) > 0)) {
    LOG_DAN(0, << "goalshot detected in test_turn_around_trapez, don't turn!");
    return 0;
  }


  if (((angle >= 95.0) && (angle <= 265.0) &&
      !steps2go.ball_kickable_for_opponent &&
      (opponent_steps_to_ball() >= 2) && am_i_at_home_trapez(1.2)) ||

      ((angle >= 110.0) && (angle <= 250.0))
      //is_turning_allowed()
      ) {
    if (direction_t < 3) {
      if (direction_t == NORTH_WEST) {
	angle = Tools::get_angle_between_null_2PI(TOP_TRAPEZ_ANGLE + M_PI - WSinfo::me->ang.get_value());
	direction_t = SOUTH_EAST;
      }
      else if (direction_t == NORTH) {
	angle = Tools::get_angle_between_null_2PI(1.5 * M_PI - WSinfo::me->ang.get_value());
	direction_t = SOUTH;
      }
      else if (direction_t == NORTH_EAST) {
	angle = Tools::get_angle_between_null_2PI(BOTTOM_TRAPEZ_ANGLE - WSinfo::me->ang.get_value());
	direction_t = SOUTH_WEST;
      }
      //offset = DEG2RAD;
      if (((turn = angle) < M_PI) &&
	  Tools::my_angle_to(ball_pos).get_value() > 90*DEG2RAD) {
	turn = M_PI + DEG2RAD;
      }
      LOG_DAN(BASELEVEL+0, << "Drehung(test_turn_around)" << turn << " degrees");
      basic_cmd->set_turn_inertia(turn);
      basic_cmd->get_cmd(cmd);
      return true;
    } else {
      if (direction_t == SOUTH_EAST) {
	angle = Tools::get_angle_between_null_2PI(TOP_TRAPEZ_ANGLE - WSinfo::me->ang.get_value());
	direction_t = NORTH_WEST;
      }
      else if (direction_t == SOUTH) {
	angle = Tools::get_angle_between_null_2PI(0.5 * M_PI - WSinfo::me->ang.get_value());
	direction_t = NORTH;
      }
      else if (direction_t == SOUTH_WEST) {
	angle = Tools::get_angle_between_null_2PI(BOTTOM_TRAPEZ_ANGLE - WSinfo::me->ang.get_value() + M_PI);
	direction_t = NORTH_EAST;
      }
      //offset = -DEG2RAD;
      if (((turn = angle) > M_PI) &&
	  Tools::my_angle_to(ball_pos).get_value() < 270*DEG2RAD) {
	turn = M_PI - DEG2RAD;
      }
      LOG_DAN(BASELEVEL+0, << "Drehung(test_turn_around) " << turn << " degrees");
      basic_cmd->set_turn_inertia(turn);
      basic_cmd->get_cmd(cmd);
      return true;
    }
    //LOG_DAN(0, << "Drehung");
  }

  return false;
} /* test_turn_around_trapez() */


bool Goalie03::test_turn_up_trapez(Cmd &cmd, double tol) {
  double tolerance = tol;

  double angle;

  /*  if ((fabs(my_pos.y) < 4.0) && (fabs(my_pos.x + 40.0) > 1.4)) {
    return 0;
    }*/

  Vector pos_to_goal = pos_trap - Vector(-ServerOptions::pitch_length/2.0, 0);
  pos_to_goal.normalize();
  pos_to_goal.rotate(M_PI/2.0);
  double sp = skalarprodukt(pos_to_goal, my_pos - pos_trap);

  if (fabs(sp) > 1.2) {
    tolerance = 1.5 * tol;
  }

  if (direction_t == NORTH_WEST) {
  //if (((direction_t == NORTH) || (direction_t == NORTH_WEST)) && (pos_trap.y >= 5.0)) {
    angle = Tools::get_angle_between_null_2PI(WSinfo::me->ang.get_value() - TOP_TRAPEZ_ANGLE);
  }
  else if (direction_t == SOUTH_EAST) {
  //else if (((direction_t == SOUTH) || (direction_t == SOUTH_EAST)) && (pos_trap.y >= 5.0)) {
    angle = Tools::get_angle_between_null_2PI(WSinfo::me->ang.get_value() - TOP_TRAPEZ_ANGLE + M_PI);
  }
  else if (direction_t == NORTH) {
  //else if ((direction_t < 3) && (pos_trap.y < 5.0)) {
    angle = Tools::get_angle_between_null_2PI(WSinfo::me->ang.get_value() - M_PI/2.0);
  }
  else if (direction_t == SOUTH) {
  //else if ((direction_t >= 3) && (pos_trap.y < 5.0)) {
    angle = Tools::get_angle_between_null_2PI(WSinfo::me->ang.get_value() + M_PI/2.0);
  }
  else if (direction_t == NORTH_EAST) {
  //else if (((direction_t == NORTH) || (direction_t == NORTH_EAST)) && (pos_trap.y <= -5.0)) {
    angle = Tools::get_angle_between_null_2PI(WSinfo::me->ang.get_value() - BOTTOM_TRAPEZ_ANGLE + M_PI);
  }
  else if (direction_t == SOUTH_WEST) {
  //else if (((direction_t == SOUTH) || (direction_t == SOUTH_WEST)) && (pos_trap.y <= -5.0)) {
    angle = Tools::get_angle_between_null_2PI(WSinfo::me->ang.get_value() - BOTTOM_TRAPEZ_ANGLE);
  }

  LOG_DAN(BASELEVEL+0, << "angle = " << angle*RAD2DEG);
  
  //if (am_I_at_home(1.0, xA)) { Sollte immer der Fall sein, sonst wird test_turn_up_trapez nicht aufgerufen!
  if ((angle > tolerance*DEG2RAD) && 
      (angle < (360.0-tolerance)*DEG2RAD)) {
    LOG_DAN(BASELEVEL+0, << "Drehung um " << angle << " (test_turn_up_trapez)");
    basic_cmd->set_turn_inertia(2*M_PI - angle);
    basic_cmd->get_cmd(cmd);
    return true;
  }
  return false;
} /* test_turn_up_trapez() */



/* Laesst den Torwart auf einer Linie vor dem Tor laufen */
bool Goalie03::get_Move_Trapez(Cmd &cmd) {
  LOG_DAN(BASELEVEL+0, << "get_move_trapez aktiv");
  return go_to_y_trapez(cmd, pos_trap);
} /* get_Move_Trapez */



bool Goalie03::go_to_y_trapez(Cmd &cmd, Vector pos, int priority ) {
  double power;
  Vector pos2 = pos;
  /*
    if (pos2.y > left_corner.y) {
    pos2.y = left_corner.y;
    } else if (pos2.y < right_corner.y) {
    pos2.y = right_corner.y;
    }*/

  Vector pos_to_goal = pos2 - Vector(-ServerOptions::pitch_length/2.0, 0);
  pos_to_goal.normalize();
  pos_to_goal.rotate(M_PI/2.0);
  double sp = skalarprodukt(pos_to_goal, my_pos - pos2);
  
  // Bei zu grosser Abweichung wird ein maximaler Dash in die richtige Richtung durchgef�hrt
  //if ((fabs(my_pos.y - pos2.y) > 1.0) || 
  //  (fabs(pos_trap.y) >= TOP_RIGHT.y) && (fabs(my_pos.x - pos2.x) > 0.5)) {
  if (fabs(sp) > 0.8) {
    // Dash noch "vorne"
    power = dash_power_needed_for_distance(fabs(sp));//100;
    //if (((pos2.y >= my_pos.y) && (direction_t >= 3)) || 
    //((pos2.y < my_pos.y) && (direction_t < 3))) {
    if (((sp <= 0) && (direction_t >= 3)) ||
	((sp > 0) && (direction_t < 3))) {
      // Dash nach "hinten"
      power = dash_power_needed_for_distance(-fabs(sp));//-100;
    }
    basic_cmd->set_dash(power, priority);
    basic_cmd->get_cmd(cmd);
    return true;
    //} else if ((fabs(my_pos.y - pos2.y) > 0.3) ||
    //(fabs(pos_trap.y) >= TOP_RIGHT.y) && (fabs(my_pos.x - pos2.x) > 0.15)) {
  } else if (fabs(sp) > 0.4) {
    // Falls eine kleinere Korrektur n�tig ist, wird nur ein Dash mit 20 durchgef�hrt
    // Dash nach "vorne"
    power = dash_power_needed_for_distance(fabs(sp));//40;
    //if (((pos2.y >= my_pos.y) && (direction_t >= 3)) || 
    //((pos2.y < my_pos.y) && (direction_t < 3))) {
    if (((sp <= 0) && (direction_t >= 3)) ||
	((sp > 0) && (direction_t < 3))) {
      // Dash nach "hinten"
      power = dash_power_needed_for_distance(-fabs(sp));//-30;
    }
    basic_cmd->set_dash(power, priority);
    basic_cmd->get_cmd(cmd);
    return true;
  }
  /* ...sonst, falls die richtige Position erreicht ist, wird mit der
   *  momentanen eigenen Geschwindigkeit abgebremst.
   */
  basic_cmd->set_turn(0.0);
  basic_cmd->get_cmd(cmd);
  return true;
} /* go_to_y_trapez() */


void Goalie03::select_defaultgoalkick_from_mypos( Vector &bestpos, float &vel, float &dir, 
						    int &playernumber){
  // returns true if a goalkick better than goalie_min_gain is found
  // returns false if default kick is selected
  Vector my_position = WSinfo::me->pos;
  Vector best_kickoffpos = my_position; // default
  double bestdir = 30/180. *PI;
  Vector bestipos = Vector(-100,0);
  bool pass2myteam = false;

  double testdir[37];
  testdir[0] = 0;
  for (int i = 1; i <= 18; i++) {
    testdir[i] = DEG2RAD(5*i);
    testdir[i+18] = DEG2RAD(-5*i);
  }
  /*
  testdir[0] = 0;
  testdir[1] = DEG2RAD(5);
  testdir[2] = DEG2RAD(10);
  testdir[3] = DEG2RAD(15);
  testdir[4] = DEG2RAD(20);
  testdir[5] = DEG2RAD(-5);
  testdir[6] = DEG2RAD(-10);
  testdir[7] = DEG2RAD(-15);
  testdir[8] = DEG2RAD(-20);
  */
  const int testdirs = 37;
  const double defaultspeed = 2.5;
  int bestplayer = 0;


  for(int j=0;j<testdirs;j++){
    Vector ipos;
    bool myteam;
    int number;
    bool inpitch=Policy_Tools::earliest_intercept_pos(my_position,defaultspeed,testdir[j],
						      ipos, myteam, number);
#if 1
    LOG_DAN(BASELEVEL+0, <<"Testing Default kick, pos "<<my_position<<" dir "<<testdir[j]
	    <<" intercept pos "<<ipos<<" myteam intercepts "<<myteam<<" player number "
	    <<number);
#endif
    if(inpitch){
      if(pass2myteam == false){ // I haven't found a possibility to pass to my team member yet
	if((ipos.getX() > bestipos.getX()) || (myteam == true)){
	  bestipos = ipos;
	  best_kickoffpos = my_position;
	  bestdir = testdir[j];
	  bestplayer = number;
	  pass2myteam = myteam;
	}
      }
      else{// the best position already goes to my player, so try to improve if possible
	if((ipos.getX() > bestipos.getX()) && (myteam == true)){
	  bestipos = ipos;
	  best_kickoffpos = my_position;
	  bestdir = testdir[j];
	  bestplayer = number;
	  pass2myteam = myteam;
	}
      }
    } // ball is not in pitch, so continue;
  } // for directions j
  vel = defaultspeed;
  dir = bestdir;
  bestpos = best_kickoffpos;
  playernumber = bestplayer;
}


bool Goalie03::test_kick_trapez(Cmd &cmd) {
  if ( WSinfo::is_ball_kickable() ) { // if the ball is not catchable, but kickable
    LOG_DAN(BASELEVEL+1, << "The ball is kickable");
    LOG_DAN(BASELEVEL+0, << "Selecting Move Hold Ball (so I will be able to catch) / trapez") ;
    if (!LEFT_PENALTY_AREA.inside(WSinfo::me->pos))
    {
      PlayerSet playersNearMe = WSinfo::alive_opponents;
      playersNearMe.join( WSinfo::valid_teammates_without_me );
      playersNearMe.keep_players_in_circle( WSinfo::me->pos, 7.0 );
      if (    playersNearMe.num > 0
           && !(LEFT_PENALTY_AREA.inside(WSinfo::me->pos)) )
      {
        double kick_vel, kick_dir;
        kick_vel = ServerOptions::ball_speed_max * 0.975;
        if ( WSinfo::me->pos.getY() > 0 )
          kick_dir = ( Vector(-FIELD_BORDER_X,FIELD_BORDER_Y) - WSinfo::me->pos).arg();
        else
          kick_dir = ( Vector(-FIELD_BORDER_X,-FIELD_BORDER_Y) - WSinfo::me->pos).arg();
        LOG_DAN(BASELEVEL+0,<<"TG@Goalie: I am not in PEN area, kick hard! v="
          <<kick_vel<<" dir="<<kick_dir);
       neuro_kick->kick_in_dir_with_initial_vel(kick_vel, ANGLE(kick_dir));
       neuro_kick->get_cmd(cmd);
       return true;
      }
      if (ball_vel.sqr_norm() > 0.16) 
      {
        LOG_DAN(BASELEVEL,<<"Ball vel larger than 0.16, make hold turn.");
        hold_turn->get_cmd(cmd, ANGLE(0.0));
        return true;
      }
      Vector bestpos;
      float vel, dir;
      int playernumber;
      select_defaultgoalkick_from_mypos(bestpos, vel, dir, playernumber);
      mdpInfo::set_my_intention(DECISION_TYPE_GOALIECLEARANCE,
			    vel,dir,playernumber);

      neuro_kick->kick_in_dir_with_initial_vel(vel, ANGLE(dir));//TG08
      neuro_kick->get_cmd(cmd);
      return true;
      /*
	if(vel >2.0) {
	move = Move_Factory::get_Neuro_Kick2(4,vel,dir );
	else
	move = Move_Factory::get_Neuro_Kick(4,vel,dir ); 
      return move;
      */
    }
  }
  return false;
}

    /*
bool Goalie03::test_kick_trapez() {
  Angle kick_dir = 0;
  double kick_vel = 0;
  const double deg5 = M_PI/36;

  if ( mdpInfo::is_ball_kickable() ) { // if the ball is not catchable, but kickable
    LOG_DAN(1, << "The ball is kickable");
    LOG_DAN(0, << "Selecting Move Hold Ball (so I will be able to catch)") ;
    // Warning : We do not yet check wether we are close to our own goal ! If we are on our goal_line it might happen that the goalie tries to hold the ball away from an opponent and thereby kicks it into our own goal !!!
    if (!(mdpInfo::is_object_in_my_penalty_area(mdpInfo::my_pos_abs()))){ // if we cannot wait for the ball to be catchable with a hold ball move
      if (mdpInfo::my_pos_abs().y > 0){
	for (double a = M_PI/2.0 - deg5 ; a >= M_PI/4.0 ; a-= deg5){
	  if (Move_1Step_Kick::get_max_vel_in_dir(Tools::get_angle_between_null_2PI(a)) > kick_vel){
	    kick_vel = Move_1Step_Kick::get_max_vel_in_dir(Tools::get_angle_between_null_2PI(a));
	    kick_dir = a;
	  }
	}
      } else {
	for (double a = -M_PI/2.0 + deg5 ; a <= -M_PI/4.0 ; a+= deg5){
	  if (Move_1Step_Kick::get_max_vel_in_dir(Tools::get_angle_between_null_2PI(a)) > kick_vel){
	    kick_vel = Move_1Step_Kick::get_max_vel_in_dir(Tools::get_angle_between_null_2PI(a));
	    kick_dir = a;
	  }
	}
      }
      LOG_DAN(0, << "kick_vel is " << kick_vel);
      if (kick_vel > 0.5){ //we have found a good direction
	return Move_Factory::get_1Step_Kick(kick_vel, kick_dir);
      } else {
	return Move_Factory::get_Neuro_Hold_Ball(1);
      }
    } else {
    return Move_Factory::get_Neuro_Hold_Ball(1);//Goalie03::catch_ban +
    } 
  }
  return NULL;
}
*/

void Goalie03::set_vars() {
  my_pos = WSinfo::me->pos;
  my_vel = WSinfo::me->vel;
  ball_vel = WSinfo::ball->vel;
  ball_pos = WSinfo::ball->pos;
  ball_angle = Tools::my_angle_to(ball_pos).get_value();
  dist_to_r_corner = WSinfo::ball->pos.distance(right_corner);
  dist_to_l_corner = WSinfo::ball->pos.distance(left_corner);

  is_ball_catchable = ((mdpInfo::is_ball_catchable() || 
			//mdpInfo::is_ball_catchable_exact() && !mdpInfo::is_ball_catchable_next_time() ||
			(mdpInfo::is_ball_catchable_exact() &&
			((WSinfo::ball->pos + WSinfo::ball->vel).getX() < -ServerOptions::pitch_length/2.0))) &&
		       !Goalie_Bs03::catch_ban);

  compute_steps2go();
  pos_trap = positioniere_trapez();
  last_player_line = mdpInfo::last_player_line();
}


int Goalie03::is_defender_in_rectangle(double x, double y) {

  PlayerSet pset_tmp = WSinfo::valid_teammates_without_me;
  pset_tmp.keep_players_in_rectangle(Vector(x - 2.0, y + 4.0), Vector(-60.0, y - 4.0));
  for (int i = 0; i < pset_tmp.num; i++) {
    if (DeltaPositioning::get_role(pset_tmp[i]->number) == PT_DEFENDER) return 1;
  }
  return 0;

  /*
  Vector pos;
  for(int i=0; i<11; i++) {
    if(!mdpInfo::mdp->my_team[i].alive) continue;
    if(!mdpInfo::is_teammate_pos_valid(mdpInfo::mdp->my_team[i].number)) continue;
    if(mdpInfo::mdp->my_team[i].number == 1) continue;

    pos = mdpInfo::teammate_pos_abs(mdpInfo::mdp->my_team[i].number);
    if ((DeltaPositioning::get_role(mdpInfo::mdp->my_team[i].number) == 0) &&
	(fabs(pos.y-y) <= 4.0) && 
	(pos.x + 2.0 < x)) {
      return 1;
    }
  }
  return 0;
  */
}



/* returns deviation_threshold for determining the next action
 */
Angle Goalie03::calculate_deviation_threshold(double distance) {
  // deviation_threshold determines when the player should turn instead of dash
  Angle deviation_threshold = 0.3;
  if (distance<10) {
    // when near destination only turn if player will miss the destination by more than 1 meter
    Angle deviation_max = asin(1.5 / distance);
    if (deviation_max>deviation_threshold) {
      deviation_threshold=deviation_max;
    }
  }
  return deviation_threshold;
} 

/* returns if destination is reachable (in kick range) within the given number of turns
 */
bool Goalie03::is_destination_reachable(const Vector& destination, Vector my_pos, 
						       Vector my_vel, Angle my_ang, int turns) {
  while (turns-- > 0) {
    Vector me_to_destination = destination-my_pos;
    Angle my_angle_to_destination = me_to_destination.angle()-my_ang;
    if (my_angle_to_destination<-PI)
      my_angle_to_destination+=2*PI;
    if (my_angle_to_destination>PI)
      my_angle_to_destination-=2*PI;
    Angle deviation_threshold = calculate_deviation_threshold(me_to_destination.norm());

    if (fabs(my_angle_to_destination)>deviation_threshold) {
      double inertia_factor = my_vel.norm()*ServerOptions::inertia_moment+1.0;
      Angle turn_angle = my_angle_to_destination*inertia_factor;
      if (turn_angle>3.14159) turn_angle=3.14159;
      if (turn_angle<-3.14159) turn_angle=-3.14159;

      my_ang+=turn_angle/inertia_factor;
      my_ang=Tools::get_angle_between_mPI_pPI(my_ang);
    }
    else {
      my_vel+=100.0*WSinfo::me->dash_power_rate*WSinfo::me->effort
	* Vector(cos(my_ang),sin(my_ang));
      double my_vel_norm = my_vel.norm();
      if (my_vel_norm>ServerOptions::player_speed_max)
	my_vel*=ServerOptions::player_speed_max/my_vel_norm;
    }

    my_pos+=my_vel;
    my_vel*=ServerOptions::player_decay;
    if ((destination-my_pos).norm() <= ServerOptions::catchable_area_l) {
      return true;
    }
  }
  return false;
}


bool Goalie03::intercept_goalie(Cmd &cmd, int priority ) {
  // get current state information
  Vector my_pos_L = WSinfo::me->pos;
  Vector my_vel_L = WSinfo::me->vel;
  Angle my_ang_L = WSinfo::me->ang.get_value();
  Vector ball_pos_L = WSinfo::ball->pos;
  Vector ball_vel_L = WSinfo::ball->vel;

  if (!WSinfo::is_ball_kickable()) {
    // if ball isn't kickable calculate when it is
    int i = 0;
    do {
      ball_pos_L += ball_vel_L;
      ball_vel_L *= ServerOptions::ball_decay;
    } while (!is_destination_reachable(ball_pos_L, my_pos_L, my_vel_L, my_ang_L, ++i));
  }

  Vector destination = ball_pos_L;
  LOG_DAN(0, << "handcoded intercept started!");
  LOG_DAN(0,<< _2D << VC2D(destination,1,"#00ff00"));
  
  if (!mdpInfo::is_position_in_pitch(ball_pos_L, 1.5)) {
    double steigung;
    if (ball_vel.getX() == 0.0) {
      LOG_DAN(0, << "hwgoalie intercept () return 0!");
      return  0;
    }
    steigung = -(ball_vel.getY() / ball_vel.getX());
    Vector id_P_auf_Tor = ServerOptions::own_goal_pos +
      Vector(0, ball_pos.getY() + steigung * (ball_pos.getX() - ServerOptions::own_goal_pos.getX()));
    
    if ((skalarprodukt(ball_vel, Vector(-1.0, 0.0)) > 0) 
	&& (fabs(id_P_auf_Tor.getY()) <= ServerOptions::goal_width/2.0 + 5)) {
      Vector bv = ball_vel;
      bv.normalize(1.5);
      destination = id_P_auf_Tor - bv;
    }
  }

  LOG_DAN(0,<< _2D << VC2D(destination,1,"#00ffff"));

  // check body angle to ball destination
  Angle my_angle_to_destination = Tools::get_angle_between_mPI_pPI(mdpInfo::my_angle_to(destination));
  Angle deviation_threshold = calculate_deviation_threshold((destination - my_pos_L).norm());

  if (fabs(my_angle_to_destination)>deviation_threshold) {
    basic_cmd->set_turn_inertia(my_angle_to_destination);
    basic_cmd->get_cmd(cmd);
    return true;
  } else {
    LOG_DAN(0, << "Return get dash 100, priority "<<priority);
    basic_cmd->set_dash(100, priority);
    basic_cmd->get_cmd(cmd);
    return true;
  }
}


bool Goalie03::get_cmd(Cmd &cmd) 
{
    
if (1)
{
  if ( ivpGoalieTackle08->isGoalieTackleSituation() )
  {
    LOG_DAN(0,<<"Goalie03: GoalieTackle08 indicates that we DO "
      <<"have a GT situation!");
    bool actionFound = ivpGoalieTackle08->get_cmd( cmd );
    if (actionFound)
      return true;
  }
  else
  //checks, for when we are in penalty mode
  if ( WSinfo::ws->play_mode == PM_his_PenaltyKick ) //TG09
  {
    if (WSinfo::is_ball_kickable())
    {
      basic_cmd->set_kick(1.0,ANGLE(0.0));
      basic_cmd->get_cmd(cmd);
      LOG_DAN(0,<<"Goalie03: We have the ball kickable in "
      <<"penalty mode. -> KICK!");
      return true;
    }
  }
  if ( WSinfo::ws->play_mode == PM_his_PenaltyKick )
  {
    LOG_DAN(0,<<"Goalie03: We PROBABLY have a GT situation: "
      <<"Penalty mode -> always use GT.");
    bool actionFound = ivpGoalieTackle08->get_cmd( cmd );
    if (actionFound)
      return true;
  }
  if (   WSinfo::ws->play_mode == PM_his_PenaltyKick  //TG09
      && (   WSinfo::ball->age == 0
          || (   WSinfo::ball->age <= 2
              &&    WSinfo::ws->time - WSinfo::ws->time_of_last_update
                 == WSinfo::ball->age ) 
         ) )
  { 
    Vector ballRealPos = WSinfo::ball->pos, 
           ballNullVel = Vector(0.0,0.0);
    PlayerSet executingOpps = WSinfo::valid_opponents;
    executingOpps.keep_and_sort_closest_players_to_point(1, WSinfo::ball->pos);
    if (   executingOpps.num > 0 )
    {
      LOG_DAN(BASELEVEL+0, << "TG@Goalie: Note: My current opponent has max speed "
        <<executingOpps[0]->speed_max<<", my max speed is "<<WSinfo::me->speed_max<<"!");
      if ( WSinfo::is_ball_kickable_for( executingOpps[0] ) )
      {
        ballRealPos = executingOpps[0]->pos;
//        ballRealPos = WSinfo::ball->pos;
        ballRealPos += executingOpps[0]->vel + executingOpps[0]->vel;
        Vector delta = ballRealPos - MY_GOAL_CENTER;
        delta.normalize(2.0);
        ballRealPos = ballRealPos - delta;
        ballNullVel = executingOpps[0]->vel;
      }
      else
      {
        ballNullVel = WSinfo::ball->vel;
        //compare me and him
        PlayerSet meAndHim;
        meAndHim.join( executingOpps[0] );
        meAndHim.append( WSinfo::get_teammate_by_number(WSinfo::me->number) );
        if (meAndHim.num == 2)
        {
          InterceptResult interceptRes[2];
          meAndHim.keep_and_sort_best_interceptors_with_intercept_behavior_to_WSinfoBallPos
             (2, interceptRes);
          LOG_DAN(BASELEVEL+0,"TG@Goalie: res[0]="<<interceptRes[0].time
            <<"("<<meAndHim[0]->number<<") res[1]="<<interceptRes[1].time<<"("
            <<meAndHim[1]->number<<")");
          if (    interceptRes[0].time < interceptRes[1].time
               && meAndHim[0] == executingOpps[0]
               && meAndHim[1] == WSinfo::me )
          {
            LOG_DAN(BASELEVEL+0,<<"TG@Goalie: Da muss ich mir wohl was einfallen lassen ...");
          }
        }
      }
    }    
    interceptball->get_cmd(cmd, WSinfo::me->pos, WSinfo::me->vel, WSinfo::me->ang, 
                           ballRealPos, ballNullVel);
    LOG_DAN(BASELEVEL+0, << "TG@Goalie: Penalty's intercept ball!");
    return true;
  }
} 
  
  PPlayer bestOppInterceptor = NULL, bestTmmInterceptor = NULL;
  int bestOppIcptSteps = 1000, bestTmmIcptSteps = 1000;
  for (int p=0; p<WSmemory::cvCurrentInterceptPeople.num; p++)
  {
    if (    bestOppInterceptor == NULL
         && WSmemory::cvCurrentInterceptPeople[p]->team == HIS_TEAM )
    {
      bestOppInterceptor = WSmemory::cvCurrentInterceptPeople[p];
      bestOppIcptSteps   = WSmemory::cvCurrentInterceptResult[p].time;
    }
    if (    bestTmmInterceptor == NULL
         && WSmemory::cvCurrentInterceptPeople[p]->team == MY_TEAM
         && WSmemory::cvCurrentInterceptPeople[p]->number != WSinfo::me->number )
    {
      bestTmmInterceptor = WSmemory::cvCurrentInterceptPeople[p];
      bestTmmIcptSteps   = WSmemory::cvCurrentInterceptResult[p].time;
    }
  }
  if ( bestOppInterceptor == NULL )
  {
    PlayerSet allOpps = WSinfo::valid_opponents;
    allOpps.keep_and_sort_closest_players_to_point(1,WSinfo::ball->pos);
    if ( allOpps.num > 0 )
    {
      bestOppInterceptor = allOpps[0];
      bestOppIcptSteps =   allOpps[0]->pos.distance( WSinfo::ball->pos )
                         / allOpps[0]->speed_max;
    }
  }
  LOG_DAN(BASELEVEL+0, << "TG@Goalie: bestOppSteps: "<<bestOppIcptSteps);
  LOG_DAN(BASELEVEL+0, << "TG@Goalie: bestTmmSteps: "<<bestTmmIcptSteps);
  LOG_DAN(BASELEVEL+0, << "TG@Goalie: i am best: "
    <<(WSmemory::cvCurrentInterceptPeople[0] == WSinfo::me)
    <<", i need "<<WSmemory::cvCurrentInterceptResult[0].time);
  for (int i=0; i<WSmemory::cvCurrentInterceptPeople.num; i++)
  {
    LOG_DAN(BASELEVEL+0, << "TG@Goalie: 1. team_"
      <<WSmemory::cvCurrentInterceptPeople[i]->team
      <<", #"<<WSmemory::cvCurrentInterceptPeople[i]->number);
  }
  if (    WSmemory::cvCurrentInterceptPeople.num > 0
       && WSmemory::cvCurrentInterceptPeople[0] == WSinfo::me )
  {
  LOG_DAN(BASELEVEL+0, << "TG@Goalie: requStamina: "
    <<(double)WSinfo::me->stamina_demand_per_meter * 1.5
           * WSinfo::me->pos.distance( WSmemory::cvCurrentInterceptResult[0].pos));
  }
  double fastestIcptStep = (double)(WSmemory::cvCurrentInterceptResult[0].time);
  
  bool basicPredicates =    WSinfo::is_ball_kickable() == false
                         && WSinfo::ws->play_mode == PM_PlayOn
                         && WSinfo::ball->age == 0
                         && WSmemory::cvCurrentInterceptPeople.num > 0
                         && WSmemory::cvCurrentInterceptPeople[0] == WSinfo::me;
  bool basicPredicates4Penalty =    WSinfo::is_ball_kickable() == false
                         && WSinfo::ball->age == 0
                         && WSmemory::cvCurrentInterceptPeople.num > 0
                         && WSmemory::cvCurrentInterceptPeople[0] == WSinfo::me;
                         
  XYRectangle2d extended_LEFT_PENALTY_AREA
                ( Vector(-FIELD_BORDER_X, (-PENALTY_AREA_WIDTH * 0.5) - 2.0 ),
                  Vector(-FIELD_BORDER_X + PENALTY_AREA_LENGTH + 2.0, (PENALTY_AREA_WIDTH*0.5) + 2.0 ) );
  bool icptPosPredicates 
    =    (   !LEFT_PENALTY_AREA.inside(WSmemory::cvCurrentInterceptResult[0].pos)
          && WSmemory::cvCurrentInterceptResult[0].pos.getX() > -FIELD_BORDER_X - 3.0
         )
      || (    ivLastInterceptDecisionBeyondPenaltyArea == WSinfo::ws->time - 1
           && extended_LEFT_PENALTY_AREA.inside(WSmemory::cvCurrentInterceptResult[0].pos) )

      || (    extended_LEFT_PENALTY_AREA.inside(WSmemory::cvCurrentInterceptResult[0].pos) //TG16
           && bestOppIcptSteps - WSmemory::cvCurrentInterceptResult[0].time > 2
           && bestTmmIcptSteps - WSmemory::cvCurrentInterceptResult[0].time > 2);
      
  double staminaBufferFactor = 1.5;
  if ( ivLastInterceptDecisionBeyondPenaltyArea == WSinfo::ws->time - 1 )
    staminaBufferFactor = 1.25;     
  bool staminaPredicates
    = WSinfo::me->stamina - (ServerOptions::effort_dec_thr * ServerOptions::stamina_max + 100.0)
      >   (double)WSinfo::me->stamina_demand_per_meter * staminaBufferFactor
        * WSinfo::me->pos.distance( WSmemory::cvCurrentInterceptResult[0].pos );
  
  bool oppIcptPredicates
    =    bestOppIcptSteps - WSmemory::cvCurrentInterceptResult[0].time > 5
      || (    ivLastInterceptDecisionBeyondPenaltyArea == WSinfo::ws->time - 1
           && bestOppIcptSteps - WSmemory::cvCurrentInterceptResult[0].time > 2 )
      || (    ivLastInterceptDecisionBeyondPenaltyArea == WSinfo::ws->time - 1
           && bestOppIcptSteps - WSmemory::cvCurrentInterceptResult[0].time >= 0
           && bestOppIcptSteps > 8 );//TG09
  bool oppIcptPredicates4Penalty
    =    bestOppIcptSteps - WSmemory::cvCurrentInterceptResult[0].time > 2
      || (    ivLastInterceptDecisionBeyondPenaltyArea == WSinfo::ws->time - 1
           && bestOppIcptSteps - WSmemory::cvCurrentInterceptResult[0].time > 0 );

  int requiredGoalieHeadStartOverTeammate = 3;
  if ( ivLastInterceptDecisionBeyondPenaltyArea == WSinfo::ws->time - 1 )
    requiredGoalieHeadStartOverTeammate = 1;     
  bool tmmIcptPredicates
    = (    bestOppIcptSteps - bestTmmIcptSteps < requiredGoalieHeadStartOverTeammate
           || ((double)bestOppIcptSteps) / ((double)bestTmmIcptSteps) < 1.2
           || (    ((double)bestOppIcptSteps) / fastestIcptStep > 3.0
                && ((double)bestTmmIcptSteps) / fastestIcptStep > 6.0 )
         );
  
  if (  (   basicPredicates
         && icptPosPredicates
         && staminaPredicates
         && oppIcptPredicates
         && tmmIcptPredicates
        )
     ||
        (   WSinfo::ws->play_mode == PM_his_PenaltyKick
         && basicPredicates4Penalty
         && oppIcptPredicates4Penalty
        )
     )
  {
    
    intercept->get_cmd(cmd);
    ivLastInterceptDecisionBeyondPenaltyArea = WSinfo::ws->time;
    LOG_DAN(BASELEVEL+0, << "TG@Goalie: Intercept ball!");
    return true;

  }
  
  
  set_vars();
  //Policy_Tools::go2ball_steps_update();
  //go2ball_steps = Policy_Tools::go2ball_steps_list();

  LOG_DAN(BASELEVEL+1, << "BallPos seen at:" << WSinfo::ball->age);
  LOG_DAN(BASELEVEL+1, << "BallVel seen at:" << WSinfo::ball->age_vel);
  LOG_DAN(BASELEVEL+1, << "mdp_ang" << mdpInfo::my_angle_to(Vector(WSinfo::me->pos.getX(), WSinfo::me->pos.getY()+20)) * RAD2DEG); //JTS10: change to me->pos.y+20
  LOG_DAN(BASELEVEL+1, << "my_pos=" << my_pos);

  LOG_DAN(BASELEVEL+0, << _2D << VC2D(pos_trap, 1, "#0000FF"));

  
  if ((ball_pos.getX() < -ServerOptions::pitch_length/2.0 + 6.0) &&
      (fabs(ball_pos.getY()) < 16.0) &&
      (fabs(ball_pos.getY()) > ServerOptions::goal_width/2.0) &&
      (steps2go.opponent_pos.getX() < -ServerOptions::pitch_length/2.0 + 7.0) &&
      (fabs(steps2go.opponent_pos.getY()) < 17.0) &&
      (fabs(steps2go.opponent_pos.getY()) > ServerOptions::goal_width/2.0)) {
    xA = -50.5;
  } else if ((fabs(my_pos.getY() - positioniere(1.1, 0).getY()) <= 0.5) &&
	     (ball_pos.getX() > -ServerOptions::pitch_length/2.0 + 15.0)) {
    xA = normal_xA;
  }

  if ((use_trapez == 1) && 
      (fabs(steps2go.opponent_pos.getY()) < ServerOptions::penalty_area_width/2.0 - 5.0) &&
      ((steps2go.opponent_pos.getX() + 1.0 <= last_player_line) ||
       ((steps2go.opponent_pos.getX() <= steps2go.teammate_pos.getX()) &&
       (DeltaPositioning::get_role(steps2go.teammate_number) == 0) &&
       (steps2go.opponent_pos.getX() <= last_player_line + 6.0) &&
       !is_defender_in_rectangle(steps2go.opponent_pos.getX(), steps2go.opponent_pos.getY())))) {
    //opponent broke through
    use_trapez = 2;
    break_through_nr = steps2go.opponent_number;
  }

  LOG_DAN(BASELEVEL+0, << "use_trapez = " << use_trapez);
  if (use_trapez == 2) {
    if ((steps2go.opponent_number != break_through_nr) || 
	((steps2go.opponent_pos.getX() - 0.9 < steps2go.teammate_pos.getX()) && //-0.4 >
	(steps2go.opponent_pos.getX() > steps2go.teammate_pos.getX()) &&
	(fabs(steps2go.opponent_pos.getY() - steps2go.teammate_pos.getY()) < 0.5)) ||
	(last_player_line + 5.0 < steps2go.opponent_pos.getX())) {
      use_trapez = 1;
    }
  }
  LOG_DAN(BASELEVEL+0, << "use_trapez = " << use_trapez);
  if (use_trapez != 2) {
    if ( (    (WSinfo::me->stamina > 0.95*ServerOptions::stamina_max) //TG09: 3800 
           || use_trapez) &&
	 (last_player_line > -25.0) &&
	 (ball_pos.getX() > -30.0)) {
	  //||((ball_pos.x > -25.0) || use_trapez))) {
      use_trapez = 1;
    } else {
      LOG_DAN(0, << "last_player_line = " << last_player_line);
      use_trapez = 0; 
    }
  }     

  if (   WSinfo::me->stamina < 0.35*ServerOptions::stamina_max //TG09: 1400 
      || WSinfo::ws->play_mode == PM_his_FreeKick) {
    LOG_DAN(0, << "stamina too low");
    use_trapez = 0;
  }
  
  LOG_DAN(BASELEVEL+0, << "use_trapez = " << use_trapez);

  // ball far from goal ?
  //if (ball_pos.x > -24.0) {
  if (use_trapez) {
    if ((Goalie_Bs03::time_flag == WSinfo::ws->time)&&(Goalie_Bs03::catch_ban >= ServerOptions::catch_ban_cycle-1)){
      //    We reset the catchban... 
      Goalie_Bs03::catch_ban = 0;
      
    }
    LOG_DAN(0, << "HAUKE: BIN DA!!!");
    if (!WSinfo::is_ball_pos_valid()) {
      /*if ((last_time_ball_pos_invalid < WSinfo::ws->time - 1) && (WSinfo::ws->play_mode == PM_PlayOn)) {
	last_time_ball_pos_invalid = WSinfo::ws->time;
	LOG_DAN(0, << "dash backwards because ball is invalid!");
	basic_cmd->set_dash(-100);
	basic_cmd->get_cmd(cmd);
	} else {*/
	face_ball->turn_to_ball();
	face_ball->get_cmd(cmd);
	//}
    } else if (test_catch(cmd));
      
    else if (test_kick_trapez(cmd));
    else if (test_goalshot_trapez(cmd));
    else if (test_intercept_trapez(cmd));
    else if (!am_i_at_home_trapez(1.4) &&
	     //!steps2go.ball_kickable_for_opponent ||
	     ((steps2go.opponent >= 3) ||
	     (WSinfo::ball->pos.distance(WSinfo::me->pos) > 7.8))) {//((fabs(my_pos.x + 40.0) >= 1.0)) { // vorher 0.5
      //NO
      LOG_DAN(BASELEVEL+0, << "Selecting Move NeuroGo2Pos " << pos_trap.getX() << " " << pos_trap.getY());
      get_Neuro_Go2Pos_back_forw(cmd, pos_trap.getX(), pos_trap.getY(), 2);
    } else {
      // YES (I'm at home position)
      if ((last_player_line > START_MOVE_TRAPEZ) && (ball_pos.getX() > START_MOVE_TRAPEZ)) {
	double angle = WSinfo::me->ang.get_value();
	if (WSinfo::me->pos.distance(pos_trap) > 0.8) {
	  go2pos->set_target(pos_trap);
	  go2pos->get_cmd(cmd);
	} else if ((angle > 4.0*DEG2RAD) && 
		   (angle < (360.0-4.0)*DEG2RAD)) {
	  LOG_DAN(BASELEVEL+0, << "Drehung um " << angle << " (playon)");
	  basic_cmd->set_turn_inertia(2*M_PI - angle);
	  basic_cmd->get_cmd(cmd);
	} else {
	  basic_cmd->set_turn(0.0);
	  basic_cmd->get_cmd(cmd);
	}
      } else
	if (test_turn_around_trapez(cmd)){}
	else if (test_turn_up_trapez(cmd)){}
	else if (get_Move_Trapez(cmd)){}
    }
    /* 
       } else if (ball_pos.x > -34.0) {
       //if (mdpInfo::ball_distance_to(my_pos) >= 24) {
       // YES
       // at home position ?
       if ((fabs(my_pos.x + 50.5) >= 1.0)) { // vorher 0.5
       //NO
       LOG_DAN(0, << "Selecting Move NeuroGo2Pos -50.5,my_pos.y ( I want to go home)" );
       return get_Neuro_Go2Pos_back_forw(-50.5,(positioniere(1.2, 0)).y,3);
       } else {
       // YES (I'm at home position)
       if ((move = test_turn_around()));
       else if ((move = test_turn_up()));
       else if ((move = get_Move_Line(0, -50.5)));
       return move;
       } */
  } else {
    /*-------------------------*/
    /* Start important stuff ! */
    /*-------------------------*/
    
    /* So what's the plan ?
       Here's how the goalie should behave :
       -If the ball is catchable, catch it !
       -If the ball is kickable, control it until it can be caught !
       -If one of the opponents controls the ball, it gets tricky !
       -Wait until he shoots at the goal, then try to reach the ball ,
          don't intercept if the opponent is too close !
       -If neither controls the ball, I stay on the line, except if I am the fastest !
    */
    
    // Some tricky stuff because the procedure might be called !twice! a turn...
    if ((Goalie_Bs03::time_flag == WSinfo::ws->time) && (Goalie_Bs03::catch_ban >= ServerOptions::catch_ban_cycle-1)){
      //    We reset the catchban...
      Goalie_Bs03::catch_ban = 0;
      
    }
    
   
    
    
    LOG_DAN(BASELEVEL+1, << "Ball an Pos:" << ball_pos);

    if (WSinfo::ws->play_mode != PM_my_GoalKick && WSinfo::ws->play_mode != PM_his_FreeKick) {

      if (!WSinfo::is_ball_pos_valid()) {
	face_ball->turn_to_ball();
	face_ball->get_cmd(cmd);
      } else
      if (test_catch(cmd));
      else if (test_kick(cmd));
      //else if ((move = test_teammate_control()));
      //else if ((move = test_opponent_control()));
      else { // if it cannot be caught etc... (The ball is running free)
	if (test_corner_goalshot(cmd));
	else if (test_goal_shot(cmd));
	else if (test_fastest_to_ball(cmd));
	else if (test_2step_go2pos(cmd));
	else if (test_turn_around(cmd));
	else if (test_turn_up(cmd));
	else {
	  LOG_DAN(BASELEVEL+0, << "I am not the fastest player to reach the ball ") ;
	  LOG_DAN(BASELEVEL+0, << "Selecting Move Line ") ;
	  get_Move_Line(cmd, 1);
	} 
      }

    } else {

      if (test_turn_around(cmd));
      else if (test_turn_up(cmd));
      else {
	get_Move_Line(cmd, 1);
      } 

    }
  }
  /*-------------------------*/
    /* End important stuff ! */
    /*-------------------------*/
    
  //} else { // ball position is not valid (i.e. the timer is 0)
  //  LOG_DAN(0, << "Selecting Move Face Ball (the ball position is not valid) ") ;
  //}

  if (Goalie_Bs03::catch_ban && 
      (last_catch_time + Goalie_Bs03::catch_ban 
       <= WSinfo::ws->time)) 
    Goalie_Bs03::catch_ban = 0;




  Goalie_Bs03::time_flag = WSinfo::ws->time;
  return true; 
  
} /* play_on */
