#include "mdpstate.h"


std::ostream& operator<< (std::ostream& o, const FVal& fv) {
  //  return o << "(" << fv.v << ", " << fv.p << ", " << fv.t << ")";  
  return o << fv.v;
}

std::ostream& operator<< (std::ostream& o, const FBall& fb) {
   o << "\nBall "
     << ", Pos= ("<< fb.pos_x <<"," << fb.pos_y << ")"
     << ", Vel= (" << fb.vel_x << ","<< fb.vel_y << ")";
   return o;
}

std::ostream& operator<< (std::ostream& o, const FPlayer& fp) {
   o << "\nPlayer= " << fp.number;
   if (!fp.alive) {
      o << " NOT alive";
      return o;
   }
   o  << ", Pos= ("<< fp.pos_x <<"," << fp.pos_y << ")"
      << ", angle= " << fp.ang 
      << ", neck_angle= " << fp.neck_angle
      << ", vel= ("<< fp.vel_x << "," << fp.vel_y  << ")"
      << ", stamina= " << fp.stamina;
   return o;
}


std::ostream& operator<< (std::ostream& o, const MDPstate& mdp) {
   int i;
   o << "\nMDPstate     --------------------------------"
     << "\ntime           = " << mdp.time_current
     << "\nlast_update    = " << mdp.time_of_last_update
     << "\nplay_mode      = " << play_mode_str(mdp.play_mode)
     << "\n--"
     << "\nmy_team_score  = " << mdp.my_team_score
     << "\nhis_team_score = " << mdp.his_team_score
     << "\n--"
     << "\nview_angle     = ";
   switch (mdp.view_angle) {
   case WIDE: o<< "WIDE"; break;
   case NARROW: o<< "NARROW"; break;
   case NORMAL: o<< "NORMAL"; break;
   default: o << "invalid value";
   }
   o << "\nview_quality   = ";
   switch (mdp.view_quality) {
   case HIGH: o<< "HIGH"; break;
   case LOW: o<< "LOW"; break;
   default: o << "invalid value";
   }
   o << "\n--"
     << "\nball           = " << mdp.ball;
   if (mdp.me)
     o << "\nme             =   " << *mdp.me;
   o << "\n--"
     << "\nmy_team        = ";
   for (i=0;i < NUM_PLAYERS; i++) 
     o << mdp.my_team[i];
   o << "\n--"
     << "\nhis_team       = ";
   for (i=0;i < NUM_PLAYERS; i++) 
     o << mdp.his_team[i];
   o << "\nMDPstate end --------------------------------";
   return o;
}


FVal::FVal() {
   set_zero_probability();
}

FVal::FVal(double value,double pp) {
   v= value;
   p= pp;
}

FVal::FVal(double value,double pp,double time) {
   v= value;
   p= pp;
   t= time;
}


void FVal::set_zero_probability() {
   p= 0.0;
}


void FVal::operator=(const FVal & fval) {
   v= fval.v;
   p= fval.p;
   t= fval.t;
}

FBall::FBall() {
   pos_x.set_zero_probability();
   pos_y.set_zero_probability();
   vel_x.set_zero_probability();
   vel_y.set_zero_probability();
}

void FBall::init(const FVal &_pos_x,const FVal &_pos_y) {
   pos_x= _pos_x;
   pos_y= _pos_y;
   vel_x.set_zero_probability();
   vel_y.set_zero_probability();
}


// FBall::init(const FVal &pos_x,const FVal &pos_y,const FVal &angle) {
//    x= pos_x;
//    y= pos_y;
//    a= angle;
//    v.set_zero_probability();
// }

void FBall::init(const FVal &_pos_x,const FVal &_pos_y,const FVal &_vel_x,const FVal &_vel_y) {
   pos_x= _pos_x;
   pos_y= _pos_y;
   vel_x= _vel_x;
   vel_y= _vel_y;
}


FPlayer::FPlayer() {
   alive = 0;
   number= 0;
   pos_x.set_zero_probability();
   pos_y.set_zero_probability();
   vel_x.set_zero_probability();
   vel_y.set_zero_probability();
   ang.set_zero_probability();
   stamina.set_zero_probability();
   effort.set_zero_probability();
   recovery.set_zero_probability();
   neck_angle.set_zero_probability();
}

void FPlayer::init(int n,const FVal &_pos_x,const FVal &_pos_y) {
   alive= 1;
   number= n;
   pos_x= _pos_x;
   pos_y= _pos_y;
   stamina.set_zero_probability();
}


void FPlayer::init(int n,const FVal &_pos_x,const FVal &_pos_y,const FVal &_vel_x,const FVal &_vel_y) {
   alive= 1;
   number= n;
   pos_x= _pos_x;
   pos_y= _pos_y;
   vel_x= _vel_x;
   vel_y= _vel_y;
   stamina.set_zero_probability();
}

// FPlayer::init(int n,const FVal &pos_x,const FVal &pos_y,const FVal &angle,const FVal &vel) {
//    alive= 1;
//    number= n;
//    x= pos_x;
//    y= pos_y;
//    a= angle;
//    v= vel;
//    stamina.set_zero_probability();
// }

void FPlayer::init(int n,const FVal &_pos_x,const FVal &_pos_y,const FVal &_vel_x,const FVal &_vel_y,
	      const FVal &_ang, const FVal &_stamina, const FVal &_effort, const FVal &_recovery, const FVal &_neck_angle) {
   alive= 1;
   number= n;
   pos_x= _pos_x;
   pos_y= _pos_y;
   vel_x= _vel_x;
   vel_y= _vel_y;
   ang = _ang;
   stamina= _stamina;
   effort = _effort;
   recovery = _recovery;
   neck_angle = _neck_angle;
}

double FPlayer::neck_angle_rel() const {
  double rel_neck = neck_angle.v-ang.v;
  if (rel_neck<0)
    rel_neck += 2* PI;
  return(rel_neck);
}



MDPstate::MDPstate() {
  me= 0;
  last_message.heart_at = -1;
  last_message.from = 0;
}

void MDPstate::reset()
{
    me = 0;
    last_message.heart_at = -1;
    last_message.from = 0;
}
