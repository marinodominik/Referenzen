#include "go2pos_backwards_bms.h"
#include "tools.h"
#include "ws_info.h"

bool Go2PosBackwards::initialized = false;

Go2PosBackwards::Go2PosBackwards() {
  basic_cmd = new BasicCmd;
}

Go2PosBackwards::~Go2PosBackwards() {
  delete basic_cmd;
}

void Go2PosBackwards::set_params(double p_x, double p_y, double p_accuracy, double p_angle_accuracy){
  target.setXY( p_x, p_y );
  accuracy = p_accuracy;
  angle_accuracy = p_angle_accuracy;
}

double Go2PosBackwards::dash_power_needed_for_distance(double dist) {
  Vector v = Vector(1.0,0.0);
  v.rotate(WSinfo::me->ang.get_value());

  double vel = WSinfo::me->vel.getX() * v.getX() + WSinfo::me->vel.getY() * v.getY();
  double power = (dist - vel*2.0/3.0) * 100.0;
  if (power > 100.0) return 100.0;
  else if (power < -100.0) return -100.0;
  else return power;
}

bool Go2PosBackwards::get_cmd(Cmd & cmd){
  double dist = target.distance(WSinfo::me->pos);
  if (dist < accuracy) {
    basic_cmd->set_turn(0.0);
    basic_cmd->get_cmd(cmd);
    return true;
  }

  if (fabs(Tools::get_angle_between_mPI_pPI(Tools::my_angle_to(target).get_value() - M_PI)) <= 
      angle_accuracy){
    basic_cmd->set_dash(-dash_power_needed_for_distance(dist));
    basic_cmd->get_cmd(cmd);
    return true;
    //cmd.cmd_main.set_dash(-fabs(dash_power));
  } else {
    basic_cmd->set_turn_inertia(Tools::get_angle_between_null_2PI(Tools::my_angle_to(target).get_value() - M_PI));
    basic_cmd->get_cmd(cmd);
    return true;
  }
}









