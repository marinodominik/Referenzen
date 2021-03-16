#include <stdio.h>
#include <iomanip>
#include <math.h>
#include <sstream>

#include "mdpstate.h"
#include "mdp_info.h"
#include "ws_info.h"
#include "options.h"
#include "tools.h"
#include "policy_tools.h"
#include "log_macros.h"
#include "positioning.h"
//#include "log_intention.h"
#include "intention.h"
#include "blackboard.h"

#define FLOAT_EPS .001
#define VALID .5

//const  MDPstate* mdpInfo::mdp = NULL; 
MDPstate* mdpInfo::mdp = NULL; //const kommt wieder, dazu muss aber das Konzept in update_game_state verbessert werden!
MDPstate* mdpInfo::server_state = NULL;
MDPmemory*  mdpInfo::memory = NULL;
Cmd* mdpInfo::my_current_cmd;
IntentionType mdpInfo::intention[11];
const int mdpInfo::stamina4infinite_steps = 100000;
//int mdpInfo::next_view_angle;

StatData mdpInfo::stats;

void mdpInfo::init() {
  clear_intentions();
  //clear_my_neck_intention();
  clear_mdp_statistics();
}

void mdpInfo::clear_intentions(){
  for(int i=0;i<11;i++){
    intention[i].type = DECISION_TYPE_NONE;
    intention[i].p1 = 0.0;
    intention[i].p2 = 0.0;
    intention[i].p3 = 0.0;
    intention[i].p4 = 0.0;
    intention[i].p5 = 0.0;
    intention[i].time = 0;  
  }
}

void mdpInfo::clear_my_intention(){
  set_my_intention(DECISION_TYPE_NONE);
}
void mdpInfo::clear_my_neck_intention(){
  Blackboard::set_neck_request(NECK_REQ_NONE);
}

#define ARRAY_IDX(x)(x-1)
void mdpInfo::set_intention(int player_no, int type, int time, float p1, float p2, float p3, 
			    float p4, float p5){
  if(player_no >11 || player_no <1)
    return;
  intention[ARRAY_IDX(player_no)].type = type;
  intention[ARRAY_IDX(player_no)].time = time;  
  intention[ARRAY_IDX(player_no)].p1 = p1;
  intention[ARRAY_IDX(player_no)].p2 = p2;
  intention[ARRAY_IDX(player_no)].p3 = p3;
  intention[ARRAY_IDX(player_no)].p4 = p4;
  intention[ARRAY_IDX(player_no)].p5 = p5;
}

void mdpInfo::set_my_intention(int type, float p1, float p2, float p3, 
			       float p4, float p5 ){
  set_intention(mdp->me->number, type, mdp->time_current,p1,p2,p3,p4,p5);
}

void mdpInfo::set_my_neck_intention(int type, float p1){
  int req = NECK_REQ_NONE;
  switch(type) {
  case NECK_INTENTION_NONE: req = NECK_REQ_NONE; break;
  case NECK_INTENTION_LOOKINDIRECTION: req = NECK_REQ_LOOKINDIRECTION; break;
  case NECK_INTENTION_PASSINDIRECTION: req = NECK_REQ_PASSINDIRECTION; break;
  case NECK_INTENTION_BLOCKBALLHOLDER: req = NECK_REQ_BLOCKBALLHOLDER; break;
  case NECK_INTENTION_SCANFORBALL: req = NECK_REQ_SCANFORBALL; break;
  case NECK_INTENTION_FACEBALL: req = NECK_REQ_FACEBALL; break;
  default: ERROR_OUT << "\nWARNING: Unsupported neck intention! ("<<type<<")"; break;
  }  
  Blackboard::set_neck_request(req,p1);
}

void mdpInfo::set_intention_by_heart_info(char* heartinfo){
   int type,time,sender;
   float p1,p2,p3,p4,p5;

   sscanf(heartinfo,"%d %d %d %f %f %f %f %f",&sender,&type,&time,&p1,&p2,&p3,&p4,&p5);
   mdpInfo::set_intention(sender,type,time,p1,p2,p3,p4,p5);
   if(type==2 && ((int)p5==mdpInfo::mdp->me->number)){
     //LOG_INFO("Mdp-Info parse heart: pass intention of "<<sender<<" to "<<(int) p5);
   }
   /*
     cout<<"Intention of "<<sender<<" : type: "<<type<<" "<<time<<" "<<" "<<p1<<" "<<p2<<" "<<p3
	 <<" "<<p4<<" "<<p5<<endl;
   */
}


int mdpInfo::get_intention(int player_no, IntentionType &player_intention){
/** returns intention type of teammate player_no and also returns its intention 
    if player == me, then my current decision is returned, otherwise heart information from teammates
    is given */
  if(player_no >11 || player_no <1)
    return DECISION_TYPE_NONE;
  player_intention.type =intention[ARRAY_IDX(player_no)].type;
  player_intention.time=intention[ARRAY_IDX(player_no)].time;  
  player_intention.p1=intention[ARRAY_IDX(player_no)].p1;
  player_intention.p2=intention[ARRAY_IDX(player_no)].p2;
  player_intention.p3=intention[ARRAY_IDX(player_no)].p3;
  player_intention.p4=intention[ARRAY_IDX(player_no)].p4;
  player_intention.p5=intention[ARRAY_IDX(player_no)].p5;
  return intention[ARRAY_IDX(player_no)].type;
}

int mdpInfo::get_intention(int player_no){
  IntentionType tmp;
  return get_intention(player_no,tmp);
}

int mdpInfo::get_my_intention(IntentionType &player_intention){
  return(get_intention(mdp->me->number,player_intention));
}

int mdpInfo::get_my_intention(){
  return(get_intention(mdp->me->number));
}

int mdpInfo::get_my_neck_intention() {
  double dum;
  int req=Blackboard::get_neck_request(dum);
  int type = NECK_INTENTION_NONE;
  switch(req) {
  case NECK_REQ_NONE: type = NECK_INTENTION_NONE; break;
  case NECK_REQ_DIRECTOPPONENTDEFENSE: type = NECK_INTENTION_DIRECTOPPONENTDEFENSE; break;
  case NECK_REQ_LOOKINDIRECTION: type = NECK_INTENTION_LOOKINDIRECTION; break;
  case NECK_REQ_PASSINDIRECTION: type = NECK_INTENTION_PASSINDIRECTION; break;
  case NECK_REQ_BLOCKBALLHOLDER: type = NECK_INTENTION_BLOCKBALLHOLDER; break;
  case NECK_REQ_SCANFORBALL: type = NECK_INTENTION_SCANFORBALL; break;
  case NECK_REQ_FACEBALL: type = NECK_INTENTION_FACEBALL; break;
  default: ERROR_OUT << "\nWARNING: Neck request not supported in mdp_info!";
  }
  return type;
}

int mdpInfo::get_my_neck_intention(IntentionType &intention) {
  int type = get_my_neck_intention();
  intention.type = type;
  if(type != NECK_INTENTION_NONE) {
    double dum;
    Blackboard::get_neck_request(dum);
    intention.p1 = dum;
  }
  intention.time = WSinfo::ws->time;
  return type;
}

void mdpInfo::update( MDPstate &m) {
  mdp = &m;
}

void mdpInfo::update_memory( MDPmemory &m) {
  memory = &m;
}

Vector mdpInfo::my_expected_pos_abs_with_dash(double dash_power){
  Vector a = Vector(cos(mdp->me->ang.v), sin(mdp->me->ang.v));
  a *= dash_power * WSinfo::me->dash_power_rate * WSinfo::me->effort;
  Vector expected_pos = WSinfo::me->pos + WSinfo::me->vel + a;
  // cout << "mdpInfo::my_expected_pos_abs_with_dash" << my_pos_abs() << expected_pos << endl;
  return expected_pos;
}

Vector mdpInfo::my_expected_vel_with_dash(double dash_power){
  Vector a = Vector(cos(mdp->me->ang.v), sin(mdp->me->ang.v));
  a *= dash_power * WSinfo::me->dash_power_rate * WSinfo::me->effort;
  Vector expected_vel = WSinfo::me->decay * WSinfo::me->vel + a;
  // cout << "mdpInfo::my_expected_vel_with_dash" << my_vel_abs() << expected_vel << endl;
  return expected_vel;
}

/** NOTE: If next_view_angle is set (e.g. by BS02_View_Policy), this returns the view angle
    of the NEXT cycle!
*/
/** NOTE #2: These functions are obsolete. You should use the methods found in Blackboard
    instead.
*/
Angle mdpInfo::view_angle_width_rad() {
  return next_view_angle_width().get_value();
}

ANGLE mdpInfo::next_view_angle_width() {
  int vang = Blackboard::get_next_view_angle();
  if(!(WSinfo::ws->synch_mode_ok || Blackboard::get_guess_synched()))
  {
	  Angle normal = ServerOptions::visible_angle * PI/180.0;
	  if ( WIDE == vang )
		return ANGLE(normal*2.0);
	  if ( NARROW == vang )
		return ANGLE(normal*0.5); 
	  return ANGLE(normal);
  }
  else
  {
  	   Angle narrow = ServerOptions::visible_angle * ((ServerOptions::simulator_step * PI)  / (ServerOptions::send_step * 180.0));
       if ( NORMAL == vang )
         return ANGLE(narrow*2.0);
       if ( WIDE == vang )
       	 return ANGLE(narrow*3.0);
       // otherwise return narrow angle
       return ANGLE(narrow); 
  }
}

ANGLE mdpInfo::cur_view_angle_width() {
  int vang = WSinfo::ws->view_angle;
  if(!(WSinfo::ws->synch_mode_ok || Blackboard::get_guess_synched()))
  {
	  Angle normal = ServerOptions::visible_angle * PI/180.0;
	  if ( WIDE == vang )
		return ANGLE(normal*2.0);
	  if ( NARROW == vang )
		return ANGLE(normal*0.5); 
	  return ANGLE(normal);
  }
  else
  {
  	   Angle narrow = ServerOptions::visible_angle * ((ServerOptions::simulator_step * PI)  / (ServerOptions::send_step * 180.0));
       if ( NORMAL == vang )
         return ANGLE(narrow*2.0);
       if ( WIDE == vang )
       	 return ANGLE(narrow*3.0);
       // otherwise return narrow angle
       return ANGLE(narrow); 
  }
}
Angle mdpInfo::my_angle_to(Vector target){
  Angle result;
  target -= WSinfo::me->pos;
  result = target.angle() - WSinfo::me->ang.get_value();
  if(result<0)
    result += 2*PI;
  if(result>2*PI)
    result -= 2*PI;
  return (result);
}

Angle mdpInfo::my_abs_angle_to(Vector target) {
  return Tools::get_angle_between_null_2PI(my_angle_to(target)+WSinfo::me->ang.get_value());
}

Angle mdpInfo::my_neck_angle_to(Vector target){
  Angle result;
  target -= WSinfo::me->pos;
  result = target.angle() - WSinfo::me->neck_ang.get_value();
  if(result<0)
    result += 2*PI;
  if(result>2*PI)
    result -= 2*PI;
  return (result);
}

/** returns expected relative body turn angle */
Angle mdpInfo::my_expected_turn_angle() {
  if (my_current_cmd->cmd_body.get_type() != my_current_cmd->cmd_body.TYPE_TURN) return 0;
  double turnangle=0;
  my_current_cmd->cmd_body.get_turn(turnangle);
  //cout << "turnangle: " << turnangle;
  if(turnangle>PI) turnangle-=2*PI;
  double newangle=turnangle/(1.0+WSinfo::me->inertia_moment*mdp->me->vel().norm());
#if 0  
  if(mdp->me->number==1) { 
    cout << mdp->time_current
	 << ": current angle: " << server_state->me->ang.v << ", expected: "
	 << Tools::get_angle_between_null_2PI(newangle+mdp->me->ang.v)
	 << ", turnangle: " << turnangle
	 << ", current speed: " << server_state->me->vel().norm() << std::endl << std::flush;
    my_last_angle=newangle;
  }
#endif  
  return Tools::get_angle_between_null_2PI(newangle);  
}

double mdpInfo::moment_to_turn_neck_to_abs(Angle abs_dir) {

  Angle minang=my_minimum_abs_angle_to_see()+.5*mdpInfo::next_view_angle_width().get_value();
  Angle maxang=my_maximum_abs_angle_to_see()-.5*mdpInfo::next_view_angle_width().get_value();
  if(minang<maxang) {
    if(abs_dir<minang) abs_dir=minang;
    if(abs_dir>maxang) abs_dir=maxang;
  } else {
    if(abs_dir>maxang && abs_dir<(minang-maxang)/2.+maxang) abs_dir=maxang;
    else if(abs_dir<minang && abs_dir>maxang) abs_dir=minang;
  }      
 
  return Tools::get_angle_between_null_2PI(abs_dir-mdp->me->neck_angle.v-my_expected_turn_angle());
}

double mdpInfo::moment_to_turn_neck_to_rel(Angle rel_dir) {
  return Tools::get_angle_between_null_2PI(rel_dir-mdp->me->neck_angle_rel());
}

/** gets the maximum angle I could see (abs, right) */
Angle mdpInfo::my_maximum_abs_angle_to_see() {
  Angle max_angle=mdp->me->ang.v+my_expected_turn_angle();
  max_angle+=ServerOptions::maxneckang.get_value()+.5*mdpInfo::next_view_angle_width().get_value();
  return Tools::get_angle_between_null_2PI(max_angle);
}

/** gets the minimum angle I could see (abs, left) */
Angle mdpInfo::my_minimum_abs_angle_to_see() {
  Angle min_angle=mdp->me->ang.v+my_expected_turn_angle();
  min_angle+=ServerOptions::minneckang.get_value()-.5*mdpInfo::next_view_angle_width().get_value();
  return Tools::get_angle_between_null_2PI(min_angle);
}

/** returns true if abs direction can be in view_angle with appropriate neck turn */
bool mdpInfo::could_see_in_direction(Angle target) {
  Angle minang=my_minimum_abs_angle_to_see();
  Angle maxang=my_maximum_abs_angle_to_see();
  target=Tools::get_angle_between_null_2PI(target);
  if(minang>maxang && (target<minang && target>maxang)) return false;
  if(minang<maxang && (target<minang || target>maxang)) return false;
  return true;
}
	
  
bool mdpInfo::is_ball_catchable(){
  return WSinfo::is_ball_pos_valid() && 
    fabs((WSinfo::ball->pos - WSinfo::me->pos).norm()) <= ServerOptions::catchable_area_l -0.07 &&
    LEFT_PENALTY_AREA.inside(WSinfo::ball->pos) && (WSinfo::ws->play_mode == PM_PlayOn || WSinfo::ws->play_mode == PM_his_PenaltyKick);

}

bool mdpInfo::is_ball_catchable_exact() {
  return  WSinfo::is_ball_pos_valid() && 
    fabs((WSinfo::ball->pos - WSinfo::me->pos).norm()) <= ServerOptions::catchable_area_l&&
    mdpInfo::is_object_in_my_penalty_area(WSinfo::ball->pos) && (WSinfo::ws->play_mode == PM_PlayOn);

}

bool mdpInfo::is_ball_catchable_next_time() {
  if ((mdp->me->pos_x.p < VALID) || (mdp->me->pos_y.p < VALID)) return false;
  if ((mdp->ball.pos_x.p < VALID) || (mdp->ball.pos_y.p < VALID)) return false;
  if ((mdp->ball.vel_x.p < VALID) || (mdp->ball.vel_y.p < VALID)) return false;
  
  //  printf("I know enough to decide!\n");

  double dx = (WSinfo::ball->pos + WSinfo::ball->vel).getX() - mdp->me->pos_x.v ;
  double dy = (WSinfo::ball->pos + WSinfo::ball->vel).getY() - mdp->me->pos_y.v ;

  double sqr_dist= ServerOptions::catchable_area_l;
  sqr_dist *= sqr_dist;
  if ( dx*dx + dy*dy <= sqr_dist ) return true;
  return false;

}

bool mdpInfo::is_ball_kickable_next_time() {
  if ((mdp->me->pos_x.p < VALID) || (mdp->me->pos_y.p < VALID)) return false;
  if ((mdp->ball.pos_x.p < VALID) || (mdp->ball.pos_y.p < VALID)) return false;
  if ((mdp->ball.vel_x.p < VALID) || (mdp->ball.vel_y.p < VALID)) return false;
  
  //  printf("I know enough to decide!\n");

  double dx = (WSinfo::ball->pos + WSinfo::ball->vel).getX() - mdp->me->pos_x.v ;
  double dy = (WSinfo::ball->pos + WSinfo::ball->vel).getY() - mdp->me->pos_y.v ;

  // 0.9 => 10% tolerance to be sure if true is right
  double sqr_dist= 0.9 * ServerOptions::kickable_area;
  sqr_dist *= sqr_dist;
  if ( dx*dx + dy*dy < sqr_dist ) 
    return true;
  return false;
}

bool mdpInfo::is_ball_infeelrange() {
  if ((mdp->me->pos_x.p < VALID) || (mdp->me->pos_y.p < VALID)) return false;
  if ((mdp->ball.pos_x.p < VALID) || (mdp->ball.pos_y.p < VALID)) return false;

  double dx = mdp->ball.pos_x.v - mdp->me->pos_x.v ;
  double dy = mdp->ball.pos_y.v - mdp->me->pos_y.v ;

  double sqr_dist= ServerOptions::visible_distance;
  sqr_dist *= sqr_dist;
  if ( dx*dx + dy*dy < sqr_dist ) 
    return true;
  return false;
}

bool mdpInfo::is_ball_infeelrange_next_time() {
  if ((mdp->me->pos_x.p < VALID) || (mdp->me->pos_y.p < VALID)) return false;
  if ((mdp->ball.pos_x.p < VALID) || (mdp->ball.pos_y.p < VALID)) return false;
  if ((mdp->ball.vel_x.p < VALID) || (mdp->ball.vel_y.p < VALID)) return false;
  
  //  printf("I know enough to decide!\n");

  double dx = (WSinfo::ball->pos + WSinfo::ball->vel).getX() - mdp->me->pos_x.v ;
  double dy = (WSinfo::ball->pos + WSinfo::ball->vel).getY() - mdp->me->pos_y.v ;

  // 0.9 => 10% tolerance to be sure if true is right
  double sqr_dist= ServerOptions::visible_distance;
  sqr_dist *= sqr_dist;
  if ( dx*dx + dy*dy < sqr_dist ) 
    return true;
  return false;
}

bool mdpInfo::is_ball_outfeelrange_next_time() {
  if ((mdp->me->pos_x.p < VALID) || (mdp->me->pos_y.p < VALID)) return false;
  if ((mdp->ball.pos_x.p < VALID) || (mdp->ball.pos_y.p < VALID)) return false;
  if ((mdp->ball.vel_x.p < VALID) || (mdp->ball.vel_y.p < VALID)) return false;
  
  //  printf("I know enough to decide!\n");

  double dx = (WSinfo::ball->pos + WSinfo::ball->vel).getX() - mdp->me->pos_x.v ;
  double dy = (WSinfo::ball->pos + WSinfo::ball->vel).getY() - mdp->me->pos_y.v ;

  // 0.9 => 10% tolerance to be sure if true is right
  double sqr_dist= ServerOptions::visible_distance;
  sqr_dist *= sqr_dist;
  if ( dx*dx + dy*dy >= sqr_dist )
    return true;
  return false;
}
  
bool mdpInfo::is_position_in_pitch(Vector position, float safety_margin ) {
  if ( (position.getX()-safety_margin < -ServerOptions::pitch_length / 2.) || (position.getX()+safety_margin > ServerOptions::pitch_length / 2.) ) return false;
  if ( (position.getY()-safety_margin < -ServerOptions::pitch_width  / 2.) || (position.getY()+safety_margin > ServerOptions::pitch_width  / 2.) ) return false;
  return true;
}


int mdpInfo::teammate_array_index(int number) {
  int index = -1;
  int index2 = -1;
  for (int i=0;i<11;i++) {
    if(mdp->my_team[i].alive) {
      if (mdp->my_team[i].number == number) {
	if(index != -1){
	  //printf("teammate %d occurred twice!\n", number);
	  index2 = i;
	} else
	  index = i;
      }
    }
  }

  //if (index == -1) cerr << "Can't get the teammate number " << number << " array index\n";
  if(index2 != -1){
    //found two indices!
    if(mdp->my_team[index2].pos_x.p  > mdp->my_team[index].pos_x.p){
      //select index with better confidence!
      index = index2;
    }
  }
  if(index<0 || index>10) index=-1;
  return index;
}


bool mdpInfo::is_teammate_pos_valid( const FPlayer & player ) {
  if ( player.pos_x.p > VALID && player.pos_y.p > VALID) 
    return true;
  return false;
}

bool mdpInfo::is_teammate_pos_valid(int number) {
  if ((number <= 0) || (number > 11)) {
    //    cerr << "mdpInfo::is_teammate_pos_valid(int number): Invalid number !!! \n";
    return false;
  }
  // @ralf: Bad hack, but all player are alive in MDPstate.
  //if( !is_position_in_pitch(teammate_pos_abs(number) ) ) return false;

  int index = teammate_array_index(number);
  if ((index < 0) || (index > 10)) {
//     cerr << "mdpInfo::is_teammate_pos_valid(int number): Invalid array_index !!! \n";
    return false;
  }

  return is_teammate_pos_valid(mdp->my_team[index]);
}

bool mdpInfo::is_teammate_vel_valid(int number) {
  if(!is_teammate_pos_valid(number)) return false;
  
  int index = teammate_array_index(number);
  if ((index < 0) || (index > 10)) {
//     cerr << "mdpInfo::is_teammate_pos_valid(int number): Invalid array_index !!! \n";
    return false;
  }

  if ((mdp->my_team[index].vel_x.p > VALID) && ( mdp->my_team[index].vel_y.p > VALID)) return true;
  return false;
}

double mdpInfo::last_player_line() {
  double min_x = 1000.0; // Value must be bigger then the pitch length!
  for(int i=0; i<11; i++) {
    if(!mdp->my_team[i].alive) continue;
    if(!is_teammate_pos_valid(mdp->my_team[i])) continue;
    if(mdp->my_team[i].number == 1) continue;
    if(mdp->my_team[i].pos_x.v > min_x) continue;

    min_x = mdp->my_team[i].pos_x.v;
  }
  
  return min_x;
}

//void bla() {}
      
Vector mdpInfo::teammate_pos_abs(int number) {
  if ((number <= 0) || (number > 11)) {
    LOG_ERR(0, << "mdpInfo::teammate_pos_abs(" <<  number << "): Invalid number !!! ");
    //bla();
    return Vector(0.0,10000.0);// Error: No valid number -> no valid position
  }

  int index = teammate_array_index(number);
  if ((index < 0) || (index > 10)) {
    //    cerr << "mdpInfo::teammate_pos_abs(int number): Invalid array_index !!! \n";
    return Vector(0.0,10000.0); // Error: No valid index -> no valid position
  }
  
  return Vector((mdp->my_team[index].pos_x.v) , (mdp->my_team[index].pos_y.v));
} 

Vector mdpInfo::teammate_vel_abs(int number) {
  if ((number <= 0) || (number > 11)) {
    LOG_ERR(0, << "mdpInfo::teammate_vel_abs(" << number << "): Invalid number !!! ");
    return Vector(0.0,0.0);// Error: No valid number -> no valid position
  }
  
  int index = teammate_array_index(number);
  if ((index < 0) || (index > 10)) {
    //    cerr << "mdpInfo::teammate_pos_abs(int number): Invalid array_index !!! \n";
    return Vector(0.0,0.0); // Error: No valid index -> no valid position
  }
  
  return Vector((mdp->my_team[index].vel_x.v) , (mdp->my_team[index].vel_y.v));
} 


 
int mdpInfo::opponent_array_index(int number) {
  int index = -1;
  int index2 = -1;
  for (int i=0;i<11;i++) {
    if (mdp->his_team[i].alive) {
      if (mdp->his_team[i].number == number) {
	if(index != -1){
	  //printf("opponent %d occurred twice!\n", number);
	  index2 = i;
	}
	else
	  index = i;
      }
    }
  }
  //if (index == -1) cerr << "Can't get the opponent array index\n";
  if(index2 != -1){
    //found two indices!
    if(mdp->his_team[index2].pos_x.p  > mdp->his_team[index].pos_x.p){
      //select index with better confidence!
      index = index2;
    }
  }
  if(index<0 || index>10) index=-1;
  return index;
}

bool mdpInfo::is_opponent_pos_valid( const FPlayer & player ) {
  if ( player.pos_x.p > VALID && player.pos_y.p > VALID) 
    return true;
  return false;
}

bool mdpInfo::is_opponent_pos_valid(int number) {

  if(number == 0)
    return false; // ridi: an opponent with such a number does not exist
  if ((number <= 0) || (number > 11)) {
    LOG_ERR(0, << "mdpInfo::opponent_pos_valid(" << number << "): Invalid number !!! ");
    return false;
  }
  
  // @ralf: Bad hack, but all player are alive in MDPstate.
  //if( !is_position_in_pitch(opponent_pos_abs(number) ) ) return false;

  int index = opponent_array_index(number);
  if ((index < 0) || (index > 10)) {
//     cerr << "mdpInfo::opponent_pos_valid(int number): Invalid array_index !!! \n";
    return false;
  }
  
  return is_opponent_pos_valid( mdp->his_team[index] );

}


Vector mdpInfo::opponent_pos_abs(int number) {
  if ((number <= 0) || (number > 11)) {
    LOG_ERR(0, << "mdpInfo::opponent_pos_abs(" << number << "): Invalid number!!! " << number );
    return Vector(0.0,0.0);
  }

  int index = opponent_array_index(number);
  if ((index < 0) || (index > 10)) {
    //    cerr << "mdpInfo::opponent_pos_abs(int number): Invalid array_index !!! \n";
    return Vector(0.0,0.0);
  }
  
  return Vector((mdp->his_team[index].pos_x.v) , (mdp->his_team[index].pos_y.v));
}

Vector mdpInfo::opponent_vel_abs(int number) {
  if ((number <= 0) || (number > 11)) {
    LOG_ERR(0, << "mdpInfo::oppoennt_vel_abs(" <<  number << "): Invalid number !!!");
    return Vector(0.0,0.0);// Error: No valid number -> no valid position
  }
  
  int index = opponent_array_index(number);
  if ((index < 0) || (index > 10)) {
    //    cerr << "mdpInfo::teammate_pos_abs(int number): Invalid array_index !!! \n";
    return Vector(0.0,0.0); // Error: No valid index -> no valid position
  }
  
  return Vector((mdp->his_team[index].vel_x.v) , (mdp->his_team[index].vel_y.v));
} 

int mdpInfo::age_opponent_ang(int number) {
  if ((number <= 0) || (number > 11)) {
    LOG_ERR(0, << "mdpInfo::age_playerang(" <<  number << "): Invalid number !!!");
    return -1;// Error: No valid number -> no valid position
  }
  
  int index = opponent_array_index(number);
  if ((index < 0) || (index > 10)) {
    //    cerr << "mdpInfo::teammate_pos_abs(int number): Invalid array_index !!! \n";
    return -1; // Error: No valid index -> no valid position
  }
  return (mdp->time_current - (int)mdp->his_team[index].ang.t);  
}

Angle mdpInfo::opponent_ang_abs(int number) {
  if ((number <= 0) || (number > 11)) {
    LOG_ERR(0, << "mdpInfo::oppoennt_ang_abs(" <<  number << "): Invalid number !!!");
    return 0.0;// Error: No valid number -> no valid position
  }
  
  int index = opponent_array_index(number);
  if ((index < 0) || (index > 10)) {
    //    cerr << "mdpInfo::teammate_pos_abs(int number): Invalid array_index !!! \n";
    return 0.0; // Error: No valid index -> no valid position
  }
  
  return mdp->his_team[index].ang.v;
} 


double mdpInfo::opponent_distance_to(int number, Vector target) {
  return (opponent_pos_abs(number) - target).norm();
}

double mdpInfo::opponent_distance_to_ball(int number) {
  return opponent_distance_to(number, WSinfo::ball->pos);
}

double mdpInfo::opponent_distance_to_me(int number) {
  return opponent_distance_to(number, WSinfo::me->pos);
}

int mdpInfo::opponent_closest_to(Vector target) {
  int number = -1;
  double dist = 1000.0; // Value must be bigger then the pitch length!
  for (int i = 0; i<11; i++) {
    if (mdp->his_team[i].alive && (mdp->his_team[i].pos_x.p >0.0)) {
      if ( (mdp->his_team[i].pos() - target).norm() < dist) {
	dist = (mdp->his_team[i].pos() - target).norm();
	number = mdp->his_team[i].number;
      }
    }
  }
  return number;
}

int mdpInfo::opponent_closest_to(Vector target, double &dist, double &dir) {
  Vector pos;
  return opponent_closest_to(target,dist,dir,pos);
}

int mdpInfo::opponent_closest_to(Vector target, double &dist, double &dir, Vector &pos) {
  int number = -1;
  dist = 1000.0; // Value must be bigger then the pitch length! 
  dir = 0;
  pos = Vector(200,200);
  for (int i = 0; i<11; i++) {
    if (mdp->his_team[i].alive && (mdp->his_team[i].pos_x.p >0.0)) {
      if ( (mdp->his_team[i].pos() - target).norm() < dist) {
	dist = (mdp->his_team[i].pos() - target).norm();
	dir = (mdp->his_team[i].pos() - target).arg();
	pos = mdp->his_team[i].pos();
	number = mdp->his_team[i].number;
      }
    }
  }
  // use additional information
  if(mdp->closest_attacker.seen_at >= mdp->time_of_last_update){ // I have recent sensor information!!
    Vector relpos;
    relpos.init_polar(mdpInfo::mdp->closest_attacker.dist,mdpInfo::mdp->closest_attacker.dir);
    Vector closestpos = mdp->me->pos() + relpos;
    if(((closestpos-target).norm() + 0.01 < dist) &&
       (relpos.norm() <= ServerOptions::visible_distance)){
      // consider a felt player only in the 'feel' range of a player
      // the felt player seems to be the closest to the target
      /*
      dist = mdp->closest_attacker.dist;
      dir = mdp->closest_attacker.dir;
      */
      dist = (closestpos - target).norm();
      dir = (closestpos - target).arg();

      //number = mdp->closest_attacker.number;
      number = -2;
      pos = closestpos;
    } 
  }
  return number;
}

int mdpInfo::opponent_closest_to_opponent_goal() {
  Vector goalpos = Vector(ServerOptions::pitch_length/2.,0);
  return opponent_closest_to(goalpos);
}

int mdpInfo::opponent_goalie_age(){
  
  if(mdp->his_goalie_number <= 0)
    return -1; // don't know current goalie
  for (int i=0; i<11; i++) {
    if (mdp->his_team[i].alive){
      if(mdp->his_team[i].number == mdp->his_goalie_number)
	return(age_playerpos(&mdp->his_team[i]));
    }
  }  
  return -2;
}

int mdpInfo::teammate_age(int number){
  
  for (int i=0; i<11; i++) {
    if (mdp->my_team[i].alive){
      if(mdp->my_team[i].number == number)
	return(age_playerpos(&mdp->my_team[i]));
    }
  }  
  return -2;
}

int mdpInfo::opponent_closest_to_ball(double &dist, double &dir, Vector &pos) {
  if (WSinfo::is_ball_pos_valid()){ //if the ball position is known
    return opponent_closest_to(WSinfo::ball->pos,dist,dir,pos);
  } else { // the ball position is unknown
    dist = 1000.;
    dir = 0;
    pos = Vector(0);
    return -1;
  }
}
 
bool mdpInfo::is_ball_kickable_for_opponent(int number){
  int index = opponent_array_index(number);

  if ( ! is_opponent_pos_valid(mdp->his_team[index]) )
    return false;

  if ((mdp->ball.pos_x.p < VALID) || (mdp->ball.pos_y.p < VALID)) return false;
  
  double dx = mdp->ball.pos_x.v - mdp->his_team[index].pos_x.v ;
  double dy = mdp->ball.pos_y.v - mdp->his_team[index].pos_y.v ;

  // no 10 % tolerance here
  double sqr_dist= 1.0*ServerOptions::kickable_area;
  sqr_dist *= sqr_dist;
  if ( dx*dx + dy*dy < sqr_dist ) return true;
  return false;
}

Vector mdpInfo::get_pos_abs(FPlayer player) {
  return Vector(player.pos_x.v , player.pos_y.v);
}

int mdpInfo::age_playerpos(const FPlayer *player) {
  return (mdp->time_current - player->pos_seen_at());
}

int mdpInfo::age_playervel(const FPlayer *player) {
  return (mdp->time_current - player->vel_seen_at());
}

int mdpInfo::fastest_team_to_ball(){
  int mintime = 1000;
  int time;
  Vector intercept_point;
  int fastest_team = UNKNOWN_TEAM;

  Vector ball_pos = WSinfo::ball->pos;
  Vector ball_vel = WSinfo::ball->vel;
  //  double ball_vel_decay = ServerOptions::ball_decay;
  //  double player_radius = ServerOptions::player_size;
  //  double player_vel_max = ServerOptions::player_speed_max;
  Vector player_pos;
  for(int i=0; i<11;i++){
    if((mdp->my_team[i].alive) && (is_teammate_pos_valid(mdp->my_team[i]))){
      player_pos = Vector(mdp->my_team[i].pos_x.v, mdp->my_team[i].pos_y.v);
      Policy_Tools::intercept_min_time_and_pos_hetero( time, intercept_point, ball_pos, ball_vel, player_pos, 
						mdp->my_team[i].number, true, -1.0, -1000.0);
      if(time < mintime){
        mintime = time;
        fastest_team = MY_TEAM;
      }
    }
  }
  for(int i=0; i<11;i++){
    if((mdp->his_team[i].alive) && (is_opponent_pos_valid(mdp->his_team[i]))){
      player_pos = Vector(mdp->his_team[i].pos_x.v, mdp->his_team[i].pos_y.v);
      Policy_Tools::intercept_min_time_and_pos_hetero( time, intercept_point, ball_pos, ball_vel, player_pos, 
						mdp->his_team[i].number, false, -1.0, -1000.0);
      if(time < mintime){
        mintime = time;
        fastest_team = HIS_TEAM;
      }
    }
  }
  return fastest_team;
}

int mdpInfo::fastest_team_to_ball_momentum(){
  return memory->team_in_attack;
}

bool mdpInfo::is_my_team_attacking(){
  if (fastest_team_to_ball_momentum() == MY_TEAM)
    return true;
  return false;
}

bool mdpInfo::is_object_in_my_penalty_area(Vector obj, double safety_margin ) {
  double y1 = ServerOptions::own_goal_pos.getY() - (ServerOptions::penalty_area_width / 2) + safety_margin;
  double y2 = ServerOptions::own_goal_pos.getY() + (ServerOptions::penalty_area_width / 2) - safety_margin;
  double x  = ServerOptions::own_goal_pos.getX() +  ServerOptions::penalty_area_length     - safety_margin;

  return (obj.getY() >= y1) && (obj.getY() <= y2) && (obj.getX() >= ServerOptions::own_goal_pos.getX()) && (obj.getX() <= x) ? true : false;
}

  /** get self.
      @return pointer to the player himself
  */
FPlayer* mdpInfo::me(){
  return mdp->me;
}

  /** get the current playmode */
int mdpInfo::play_mode(){
  return mdp->play_mode;
}

  /** get a player from our team.
     @param player_nbr (range = 0-10)
     @return a pointer to a player
  */
const FPlayer* mdpInfo::my_team(int player_nbr){
  return &(mdp->my_team[player_nbr]);
}
  /** get a player from their team.
     @param player_nbr (range = 0-10)
     @return a pointer to a player
  */
const FPlayer* mdpInfo::his_team(int player_nbr){
  return &(mdp->his_team[player_nbr]);
}


double mdpInfo::stamina4meters_if_dashing(double power){
  /** computes the number of meters that are possible if player dashes with power 
   */
  return(stamina4steps_if_dashing(power) * WSinfo::me->speed_max);
}

int mdpInfo::stamina4steps_if_dashing(double power){
  /** computes the number of steps that are possible if player dashes with power 
      avoiding that stamina goes below recover_dec_thr * stamina_max
      if no restrictions apply then mdpInfo::stamina4infinite_steps is returned
      @return int number of steps that are possible
  */
  
  //int cycles_per_second = 1000/ServerOptions::simulator_step;
  //int cycles2go = ServerOptions::half_time * cycles_per_second;
  double stamina = mdp->me->stamina.v;
  //double effort = 1.0;
  //double recovery = 1.0;
  //int i = 0; //counter variable

  double tmp_power = power;
  if (power < 0.0) {
    tmp_power = -power * 2.0;
  }

  if ((power <= WSinfo::me->stamina_inc_max) && 
      (power > -WSinfo::me->stamina_inc_max/2.0)) {
    if (stamina <= ServerOptions::recover_dec_thr * ServerOptions::stamina_max + 12.0 + tmp_power) {
      return 0;
    } else {
      return stamina4infinite_steps;
    }
  }

  double tmp_stamina = stamina - ServerOptions::recover_dec_thr*ServerOptions::stamina_max - 12.0;
  
  if (tmp_power - WSinfo::me->stamina_inc_max < 1.0) return stamina4infinite_steps;
  
  int tmp_steps = static_cast<int>(tmp_stamina/(tmp_power-WSinfo::me->stamina_inc_max));

  if (tmp_steps > stamina4infinite_steps) tmp_steps = stamina4infinite_steps;
  if (tmp_steps < 0) tmp_steps = 0;

  return tmp_steps;

  //Was hier drunter steht, ist FALSCH, bevor das wieder verwendet wird, lieber 2-mal 
  //auf Fehler durchsuchen!!! DANIEL

  /*
  if (ServerOptions::half_time <= 0) //infinite game
    cycles2go = 5000;

  for (i = time_current();i < time_current() + 60; i++){
    if (power < 0.0) {
      stamina = stamina + 2.0 * power * effort + recovery * ServerOptions::stamina_inc_max;
    } else {
      stamina = stamina - power * effort + recovery * ServerOptions::stamina_inc_max;
    }

    //Stamina darf Max-Wert nicht ueberschreiten
    if (stamina>ServerOptions::stamina_max) 
      stamina = ServerOptions::stamina_max;

    //berechne effort
    if (stamina <= ServerOptions::effort_dec_thr*ServerOptions::stamina_max)
      effort = effort - ServerOptions::effort_dec;
    if (stamina >= ServerOptions::effort_inc_thr*ServerOptions::stamina_max)
      effort = effort + ServerOptions::effort_inc;
    if (effort < ServerOptions::effort_min) 
      effort = ServerOptions::effort_min;

    //berechne recovery
    if (stamina <= ServerOptions::recover_dec_thr*ServerOptions::stamina_max)
      recovery = recovery - ServerOptions::recover_dec;
    if (recovery < ServerOptions::recover_min) 
      recovery = ServerOptions::recover_min;

    //Stamina darf nicht unter recover_dec_thr*stamina_max fallen!!!
    if (stamina <= ServerOptions::recover_dec_thr*ServerOptions::stamina_max + 12.0) {
      break;
    }
  } 
  //korrigiere Counter
  if (i<cycles2go) {
    i--;
  }

  //LOG_DEB(0, << "stamina = " << stamina);  
  //LOG_DEB(0, << "power = " << power);
  //LOG_DEB(0, << "new method : " << tmp_steps);
  //LOG_DEB(0, << "return value is " << i-time_current());
  return i-time_current();
  */
}

double mdpInfo::stamina_left(){
  double player_stamina = mdp->me->stamina.v;
  double left_stamina = player_stamina - ServerOptions::recover_dec_thr*ServerOptions::stamina_max - 12.0;
  //(-1) wichtig (wegen recovery-Pruefung auf <=!)
  if (left_stamina < 0) left_stamina = 0;
  return left_stamina;
}

int mdpInfo::get_stamina_state(){
  return(Stamina::get_state());
}


#define PLAYER_RADIUS 1.8
#define BALL_RADIUS 1.2

/* log data to visualize server state 
   Currently no need to implement     */
void mdpInfo::visualize_server_state() {
  
}

bool mdpInfo::do_I_play_with_fullstate() {
  if (mdp == server_state) 
    return true;
  else 
    return false;
}

 
void mdpInfo::clear_mdp_statistics() {
  stats.ball_near_pos_avg=stats.ball_dist_pos_avg=stats.ball_near_age_avg=stats.ball_dist_age_avg=0;
  stats.ball_near_vel_avg=stats.ball_dist_vel_avg=stats.ball_kr_vel_avg=stats.ball_kr_pos_avg=0;
  stats.ball_near_cyc=stats.ball_dist_cyc=stats.ball_kr_cyc=0;
  stats.ball_near_unk=stats.ball_dist_unk=stats.ball_kr_unk=0;
  stats.me_pos_avg=stats.me_vel_avg=stats.me_age_avg=stats.ball_kr_age_avg=0;
  stats.me_cyc=0;
  stats.team_near_pos_avg=stats.team_dist_pos_avg=stats.team_unk_avg=0;
  stats.team_near_age_avg=stats.team_dist_age_avg=0;
  stats.opp_near_pos_avg=stats.opp_dist_pos_avg=stats.opp_unk_avg=0;
  stats.opp_near_age_avg=stats.opp_dist_age_avg=0;
  stats.all_cycles=stats.called_cyc=0;
  stats.team_near_cyc=stats.opp_near_cyc=stats.team_dist_cyc=stats.opp_dist_cyc=0;
  stats.goalie_near_pos_avg=stats.goalie_dist_pos_avg=stats.goalie_near_age_avg=0;
  stats.goalie_dist_age_avg=stats.goalie_near_cyc=stats.goalie_dist_cyc=stats.goalie_unk=0;
  stats.my_offside_l_avg=stats.my_offside_r_avg=stats.my_offside_l_cyc=stats.my_offside_r_cyc=0;
  stats.his_offside_l_avg=stats.his_offside_r_avg=stats.his_offside_l_cyc=stats.his_offside_r_cyc=0;
  stats.ball_in_kr_err=stats.ball_out_kr_err=0;
  stats.ang_with_ball_avg=stats.ang_without_ball_avg=0;
  stats.ang_with_ball_cyc=stats.ang_without_ball_cyc=0;
}

#if 0
void mdpInfo::collect_mdp_statistics() {
  if(server_state==NULL || server_state->play_mode!=MDPstate::PM_PlayOn) return;
  stats.all_cycles++;
  
  
}
#endif

void mdpInfo::print_mdp_statistics() {
  char str[2048];
  std::stringstream stream;

  	  // << "                           MDP Statistics for player " << ClientOptions::player_no << "\n"
	  // << "                           ===========================\n\n"
  sprintf(str,
	  "\n=========================================================================\n"
	  
	  "MDP Statistics for player %d (%s)\n\n"
	  "Counted %ld play_on cycles (no fullstate: %ld cycles)\n\n" 
	  "             | Avg pos | Avg vel | Avg age | # cyc | # ukn | Avg ukn pl |\n"
	  "-------------+---------+---------+---------+-------+-------+------------|\n"
	  "Me           | %6.2f  | %6.2f  | %6.2f  |       | %4ld  |            |\n"
	  "Ball  ( <kr) | %6.2f  | %6.2f  | %6.2f  | %4ld  | %4ld  |            |\n"
	  "Ball  [kr,5] | %6.2f  | %6.2f  | %6.2f  | %4ld  | %4ld  |            |\n"
	  "Ball  (>=5 ) | %6.2f  | %6.2f  | %6.2f  | %4ld  | %4ld  |            |\n"
	  "Goalie( <25) | %6.2f  |         | %6.2f  | %4ld  | %4ld  |            |\n"
	  "Goalie(>=25) | %6.2f  |         | %6.2f  | %4ld  |       |            |\n"	 
	  "Team  ( <20) | %6.2f  |         | %6.2f  | %4ld  |       |   %5.2f    |\n"
	  "Team  (>=20) | %6.2f  |         | %6.2f  | %4ld  |       |            |\n"
	  "Opp   ( <20) | %6.2f  |         | %6.2f  | %4ld  |       |   %5.2f    |\n"
	  "Opp   (>=20) | %6.2f  |         | %6.2f  | %4ld  |       |            |\n"
	  "-------------------------------------------------------------------------\n"
	  "Body angle wrong: with ball %6.2f (%4ld cyc), without %6.2f (%4ld cyc)|\n" 
	  "Ball wrongly seen in kick_area: %4ld cyc, outside kick_area: %4ld cyc)  |\n"
	  "Ball position wrong by more than 0.3 in kick_area: %4ld times           |\n"
	  "Ball velocity wrong by more than 0.3 in kick_area: %4ld times           |\n"
	  "-------------------------------------------------------------------------\n" 
	  "My  offside line: l %6.2f (%4ld cyc), r %6.2f (%4ld cyc)              |\n"
	  "His offside line: l %6.2f (%4ld cyc), r %6.2f (%4ld cyc)              |\n"
	  "=========================================================================\n\n",
	  
	  ClientOptions::player_no,ClientOptions::teamname,
	  stats.called_cyc,stats.called_cyc-stats.all_cycles,
	  stats.me_pos_avg,stats.me_vel_avg,stats.me_age_avg,stats.all_cycles-stats.me_cyc,
	  stats.ball_kr_pos_avg,stats.ball_kr_vel_avg,stats.ball_kr_age_avg,
	  stats.ball_kr_cyc,stats.ball_kr_unk,
	  stats.ball_near_pos_avg,stats.ball_near_vel_avg,stats.ball_near_age_avg,
	  stats.ball_near_cyc,stats.ball_near_unk,stats.ball_dist_pos_avg,
	  stats.ball_dist_vel_avg,stats.ball_dist_age_avg,
	  stats.ball_dist_cyc,stats.ball_dist_unk,
	  stats.goalie_near_pos_avg,stats.goalie_near_age_avg,stats.goalie_near_cyc,
	  stats.goalie_unk,
	  stats.goalie_dist_pos_avg,stats.goalie_dist_age_avg,stats.goalie_dist_cyc,
	  stats.team_near_pos_avg,stats.team_near_age_avg,stats.team_near_cyc,
	  stats.team_unk_avg,stats.team_dist_pos_avg,stats.team_dist_age_avg,
	  stats.team_dist_cyc,stats.opp_near_pos_avg,stats.opp_near_age_avg,
	  stats.opp_near_cyc,stats.opp_unk_avg,stats.opp_dist_pos_avg,stats.opp_dist_age_avg,
	  stats.opp_dist_cyc,stats.ang_with_ball_avg,stats.ang_with_ball_cyc,
	  stats.ang_without_ball_avg,stats.ang_without_ball_cyc,
	  stats.ball_in_kr_err,stats.ball_out_kr_err,stats.ball_kr_pos_wrong,
	  stats.ball_kr_vel_wrong,
	  stats.my_offside_l_avg,stats.my_offside_l_cyc,
	  stats.my_offside_r_avg,stats.my_offside_r_cyc,stats.his_offside_l_avg,
	  stats.his_offside_l_cyc,stats.his_offside_r_avg,stats.his_offside_r_cyc);
  
  stream << str << std::ends;
  LOG_MDP(0,<< stream.str());
  if(LogOptions::log_stat) std::cout << stream.str() << std::flush;
}

/*

Offside lines:

my offside line:  left avg 12.8 (123/200 cyc), right 10.5 (1234/32 cyc)    
his offside line: left avg 13.5 (12/23 cyc)


MDP statistics for player 1 (_pvq)

Missing fullstate information: 12 cycles







*/
