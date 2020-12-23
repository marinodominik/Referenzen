#include "pos_tools.h"

#include "server_options.h"
#include "tools.h"
#include "log_macros.h"
#include "macro_msg.h"

#define DEBUG(XXX) LOG_DEB(0,XXX)

void PosSet::evaluate_position( PosValue & pos, PlayerSet & pset ) {
  pos.valid= false; // will be set to true if this position should to be considered in further calculations!

  if ( pset.num < 1 ) 
    return;
  
  double sqr_time_to_pos[pset.num];
  double sqr_time_max= -1.0;

  for ( int i=0; i< pset.num; i++) {
    sqr_time_to_pos[i]= pos.pos.sqr_distance( pset.p[i]->pos );
    
    double sqr_speed_max= pset.p[i]->speed_max;
    sqr_speed_max *= sqr_speed_max;  

    sqr_time_to_pos[i] /= sqr_speed_max;
    if (sqr_time_to_pos[i] > sqr_time_max )
      sqr_time_max= sqr_time_to_pos[i];
  }
  
  sqr_time_max += 10.0;

  pos.first_time_to_pos= sqr_time_max;
  int idx= -1;
  
  for (int i=0; i<pset.num; i++) {
    if ( sqr_time_to_pos[i] < pos.first_time_to_pos ) {
      pos.first_time_to_pos= sqr_time_to_pos[i];
      idx= i;
    }
  }
  pos.first= pset.p[idx];

  if ( require_my_team_to_be_first && pos.first->team != MY_TEAM ) 
    return;

  if ( require_me_to_be_first && pos.first != WSinfo::me )
    return;
    
  pos.first_time_to_pos= sqrt( pos.first_time_to_pos );
  sqr_time_to_pos[idx]= sqr_time_max;  //eliminate from array

  //second fastest player
  pos.second_time_to_pos= sqr_time_max;
  if ( pset.num < 2 ) 
    pos.second= 0;
  else {
    for (int i=0; i<pset.num; i++) {
      if ( sqr_time_to_pos[i] < pos.second_time_to_pos ) {
	pos.second_time_to_pos= sqr_time_to_pos[i];
	idx= i;
      }
    }
    pos.second= pset.p[idx];
    pos.second_time_to_pos= sqrt( pos.second_time_to_pos );
    sqr_time_to_pos[idx]= sqr_time_max;
  }

  //several check to let the pos invalid
  if ( require_my_team_to_be_at_least_second && 
       pos.first->team != MY_TEAM && pos.second->team != MY_TEAM )
    return;
  if ( require_me_to_be_at_least_second &&
       pos.first != WSinfo::me && pos.second != WSinfo::me )
    return;

  //fastest opponent
  if ( pos.first->team != MY_TEAM ) {
    pos.first_opponent= pos.first;
    pos.first_opponent_time_to_pos= pos.first_time_to_pos;
  }
  else if ( pos.second && pos.second->team != MY_TEAM ) {
    pos.first_opponent= pos.second;
    pos.first_opponent_time_to_pos= pos.second_time_to_pos;
  }
  else {
    pos.first_opponent= 0;
    pos.first_opponent_time_to_pos= sqr_time_max;
    int idx= -1;
    for (int i=0; i<pset.num; i++) {
      if ( pset.p[i]->team != MY_TEAM && sqr_time_to_pos[i] < pos.first_opponent_time_to_pos ) {
	pos.first_opponent_time_to_pos= sqr_time_to_pos[i];
	idx= i;
      }
    }
    if (idx >= 0) {
      pos.first_opponent= pset.p[idx];
      pos.first_opponent_time_to_pos= sqrt( pos.first_opponent_time_to_pos); 
      sqr_time_to_pos[idx]= sqr_time_max;  //eliminate from array (in case you ever wanted to look for second_opponent)
    }
  }

  if ( pos.first_opponent_time_to_pos < pos.first_time_to_pos)
    ERROR_OUT << "something went wrong";

  //computer initial ball velocities
  double distance= WSinfo::ball->pos.distance( pos.pos );
  {
    int power= int( floor( pos.first_time_to_pos ) ) + 1;
    pos.initial_ball_vel_to_get_to_pos_in_time_of_first=  
      (1.0 - ServerOptions::ball_decay) * distance / (1.0 - Tools::get_ball_decay_to_the_power( power ) );
  }

  if ( pos.second ) {
    int power= int( floor( pos.second_time_to_pos ) ) + 1;
    pos.initial_ball_vel_to_get_to_pos_in_time_of_second=  
      (1.0 - ServerOptions::ball_decay) * distance / (1.0 - Tools::get_ball_decay_to_the_power( power ) );
  }

  if ( pos.first_opponent ) {
    int power= int( floor( pos.first_opponent_time_to_pos ) ) + 1;
    pos.initial_ball_vel_to_get_to_pos_in_time_of_first_opponent=  
      (1.0 - ServerOptions::ball_decay) * distance / (1.0 - Tools::get_ball_decay_to_the_power( power ) );
  }
    

  pos.valid= true;
}

void PosSet::evaluate_positions( PlayerSet & pset ) {
  for (int i=0; i<num; i++) {
    evaluate_position( position[i], pset);
  }
}

bool PosSet::add_grid(Vector pos, int res1, Vector & dir1, int res2, Vector & dir2) {
  const double  d1= 1.0/double(res1);
  const double  d2= 1.0/double(res2);

  for (int i=0; i < res2; i++) {
    Vector point= pos + i * d2 * dir2;
    for (int j=0; j < res1; j++) {
      point += d1 * dir1;
      
      if ( ! Tools::point_in_field(point, -2.0) ) //don't set points which are not in field!
	continue;

      if (num < max_num) {
	position[num].pos= point;
	num++;
      }
      else {
	ERROR_OUT << " to much positions";
      }
    }
  }
  return true;
}

void PosSet::add_his_goal_area() {
  Vector start= HIS_GOAL_LEFT_CORNER;
  Vector end  = HIS_GOAL_RIGHT_CORNER;
  start.subFromY( 0.3 );
  end.addToY( 0.3 );
  int num_steps= 2; // must be >= 2
  Vector step = 1.0/double(num_steps-1) * ( end - start);
  
  for (int i=0; i < num_steps; i++) {
    if (num < max_num) {
      position[num].pos= start;
      start+= step;
      num++;
    }
    else {
      ERROR_OUT << " to much positions";
    }
  }
}

void PosSet::draw_positions() const{
  for (int i=0; i<num; i++) {
    const PosValue & pos = position[i];
    if ( pos.valid ) {
      LOG_DEB(0, _2D << VP2D(pos.pos,"ff0000"));
      LOG_DEB(1, _2D << VSTRING2D(pos.pos, pos.initial_ball_vel_to_get_to_pos_in_time_of_first /*) << "," << pos.initial_ball_vel_to_get_to_pos_in_time_of_second*/, "000000"));
      //LOG_DEB(1, _2D << STRING2D(pos.pos.x, pos.pos.y, int(pos.first_time_to_pos) << "," << int(pos.second_time_to_pos), "000000"));
    }
    else {
      LOG_DEB(0, _2D << VP2D(pos.pos,"0000ff"));
    }
  }
}
