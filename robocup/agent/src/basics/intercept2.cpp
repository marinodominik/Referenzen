#include "intercept2.h"
#include "globaldef.h"
#include "server_options.h"
#include "log_macros.h"

/******************************************************************************
   class Intercept (for questions ask ART!)
*******************************************************************************/

//#define ART_DEB(LLL,DDD) LOG_DEB(LLL,DDD)
#define ART_DEB(LLL,DDD)

void Intercept2::InterceptPositions::show2d() {
  ART_DEB(0,"Intercept Result num_pos= " << num_pos );
  for (int i=0; i<num_pos; i++) {
    ART_DEB(0, _2D << C2D(pos[i].ball_pos.x,pos[i].ball_pos.y,0.2,"0000ff") << STRING2D(pos[i].ball_pos.x+0.4,pos[i].ball_pos.y, pos[i].ball_time << "," << pos[i].my_time, "0000ff"));
  }
}

void Intercept2::minimal_time_player_reaches_ball_at_pos( int & res_time, Vector & res_pos ) {

  double sqr_ball_vel_considered_slow= SQUARE(0.1);

  Vector ball_pos= input.ball_pos;
  Vector ball_vel= input.ball_vel;

  int ball_start_time= input.ball_start_time;
  while ( ball_start_time < 0 ) {
    ball_pos += ball_vel;
    ball_vel *= ServerOptions::ball_decay;
    ball_start_time++;
  }

  struct Player {
    PPlayer ref;
    Vector pos;
    int min_time;
    int first_intercept_time_beg;
    int first_intercept_time_end;
    int second_intercept_time_beg;
  };

  const int player_num = input.pset.num;
  Player *player = new Player [player_num];
  
  for (int i=0; i<player_num; i++) {
    Player & p= player[i];
    p.ref= input.pset[i];
    p.pos= p.ref->pos;
    p.min_time= -2;
    p.first_intercept_time_beg= -1;
    p.first_intercept_time_end= -1;
    p.second_intercept_time_beg= -1;

    //p.first_intercept_ball_pos;
  }
  int time= 0;
  Player * his_first_interceptor= 0;

  while ( true ) {
    if (  ! FIELD_AREA.inside(ball_pos) ) {
      time--; //reduce the most possible time (don't want to intercept the ball outside of the field
      break;
    }

    if ( ball_vel.sqr_norm() > sqr_ball_vel_considered_slow  ) 
      break;

    for ( int i=0; i< player_num; i++) {
      Player & p= player[i];

      //this is an optimization, don't compute the intercept position if the player in ANY case needs longer the the current time
      if ( p.pos.sqr_distance( ball_pos ) > SQUARE( (p.ref->age+time) * p.ref->speed_max + p.ref->radius ) ) {
	p.min_time= time+1;
	continue;
      }

      int need_time; 
      time_to_target(ball_pos, p.ref->kick_radius, p.ref, need_time);
      if ( need_time <= time ) {
	if ( need_time < p.min_time ) 
	  p.min_time= need_time;

	if ( p.first_intercept_time_beg < 0 ) {
	  if ( his_first_interceptor == 0 && p.ref->team == HIS_TEAM )
	    his_first_interceptor= &p;
	    
	  p.first_intercept_time_beg= need_time;
	  //p.first_intercept_ball_pos= ball_pos;
	}
	else if ( p.first_intercept_time_end >= 0 && p.second_intercept_time_beg < 0 ) 
	  p.second_intercept_time_beg= time;
      }
      else {
	if ( p.first_intercept_time_beg >= 0 )
	  p.first_intercept_time_end= time-1;
      }
    }
    
    if ( his_first_interceptor )
      break;
  }

#if 0
  if ( LogOptions::is_on() ) {
    for ( int i=0; i< player_num; i++) {
      Player & p= player[i];
      
      for (int i=0; i<player_num; i++) {
	Player & p= player[i];
	p.ref= input.pset[i];
	p.pos= p.ref->pos;
	p.min_time= -2;
	p.first_intercept_time_beg= -1;
	p.first_intercept_time_end= -1;
	p.second_intercept_time_beg= -1;

	//p.first_intercept_ball_pos;
      }
      PlayerSet const& pset= input.pset;
      int time= 0;
      int my_team_intercept_time= -1;
      Player * his_first_interceptor= 0;
      
      while ( true ) {
	if (  fabs(ball_pos.x) > FIELD_BORDER_X || fabs(ball_pos.y) > FIELD_BORDER_Y ) {
	  time--; //reduce the most possible time (don't want to intercept the ball outside of the field
	  break;
	}
	
	if ( ball_vel.sqr_norm() > sqr_ball_vel_considered_slow  ) 
	  break;
      }
    }
  }
#endif
    delete [] player;
}

void Intercept2::time_to_target( Vector const & target, double tolerance, PPlayer player, int & time, Cmd_Body & cmd  ) const {
  Object obj(player);
  Vector tmp_target= target;
  
  int turn_time= 0;
  int dash_time= 0;
  while (true) {
    ART_DEB(1, << " === turn_time= " << turn_time << " target= " << tmp_target);
    NoneDashTurn ndt;
    turn_to_pos(tmp_target,obj, dash_time, tolerance, ndt);
#if 1 //set the first cmd
    if ( turn_time == 0 ) {
      if ( ndt.type == ndt.turn ) {
	//double tmp= ndt.val * ( 1.0 + obj.vel.norm() * obj.param->inertia_moment );
	cmd.set_turn( ndt.val * ( 1.0 + obj.vel.norm() * obj.param->inertia_moment ));
      }
      else if ( ndt.type == ndt.dash )
	cmd.set_dash( ndt.val / (obj.param->dash_power_rate*obj.param->effort));
    }
#endif
    if ( dash_time >= 0 ) 
      break;
    obj.apply(ndt);
    turn_time++;
  }

#if 0
  if ( need_long_turn ) {
    ART_DEB(1,<< " need long turn, so increasing the time ");
    turn_time++;
  }
#endif

  time= turn_time + dash_time;
}


#define BLA(XXX,CCC,OFFSET) C2D( XXX.pos.x, XXX.pos.y, 1.0,CCC) << STRING2D(XXX.pos.x+1.5, XXX.pos.y+OFFSET, XXX.time << ',' << XXX.time_left,CCC)
#define BLA2(XXX)  "pos= " << XXX.pos << " time= " << XXX.time << " time_left= " << XXX.time_left << " done= " << XXX.done

#if 0
void Intercept2::intercept_old( Vector const & ball_pos, Vector const & ball_vel, PPlayer player, int & time, Cmd_Body & cmd ) const {
  double sqr_ball_vel_considered_slow= SQUARE( player->speed_max );

  Vector bpos= ball_pos;
  Vector bvel= ball_vel;

  int ball_start_time= 0;
  while ( ball_start_time < 0 ) {
    bpos += bvel;
    bvel *= ServerOptions::ball_decay;
    ball_start_time++;
  }

  struct Res {
    Res() { time = -1; done= false; }
    Vector pos;
    int time;
    int time_left;
    bool done;
  };

  Res shortest_time, shortest_way, maximum_x;

  time= 0;
  while ( !shortest_time.done || !shortest_way.done || !maximum_x.done ) {
    if ( time > 100 )
      break;

    bpos += bvel;
    bvel *= ServerOptions::ball_decay;
    time++;

    ART_DEB(0,"time= " << time << " bpos= " << bpos << " bvel= " << bvel << " playerpos= " << player->pos);
    if ( ! FIELD_AREA.inside(bpos) ) //|| bvel.sqr_norm() < sqr_ball_vel_considered_slow )
      break;

    //this is an optimization, don't compute the intercept position if the player in ANY case needs longer the the current time
    if ( player->pos.sqr_distance( bpos ) > SQUARE( (player->age+time) * player->speed_max + player->kick_radius ) ) {
      //ART_DEB(0, "player->pos.sqr_distance( bpos )= " << player->pos.sqr_distance( bpos ) << " > " << SQUARE( (player->age+time) * player->speed_max + player->kick_radius ) << " age= " << player->age << " time= " << time  << " speed= " << player->speed_max << " r=" << player->kick_radius << " 2*2=" <<SQUARE(2));
      continue;
    }
    int need_time;
    time_to_target(bpos, player, need_time);
    ART_DEB(0, << "need_time= " << need_time );
    if ( need_time > time ) 
      continue;

    if ( shortest_time.time < 0 || time < shortest_time.time ) {
      shortest_time.done= true;
      shortest_time.pos= bpos;
      shortest_time.time= time;//need_time;
      shortest_time.time_left= time-need_time;
    }
    if ( shortest_way.time < 0 || player->pos.sqr_distance( bpos) < player->pos.sqr_distance( shortest_way.pos ) ) {
      shortest_way.pos= bpos;
      shortest_way.time= need_time;
      shortest_way.time_left= time-need_time;
    }
    else 
      shortest_way.done= true; 


    if ( maximum_x.time < 0 || maximum_x.pos.x < bpos.x ) {
      maximum_x.pos= bpos;
      maximum_x.time= need_time;
      maximum_x.time_left= time-need_time;
    }

#if 1
    { 
      Vector tmp= bpos - player->pos;
      if ( bvel.x * tmp.x + bvel.y * tmp.y < 0 ) //I'm running towards the ball
	sqr_ball_vel_considered_slow= SQUARE( player->speed_max * 0.5 );
    }
#endif

    if ( bvel.sqr_norm() < sqr_ball_vel_considered_slow  
	 && maximum_x.time >= 0 && shortest_way.time >= 0 && shortest_time.time >= 0 ) {
      shortest_way.done= true;
      maximum_x.done= true;
    }
#if 0
    if ( maximum_x.time >= 0 ) {
      if ( bpos.x - maximum_x.pos.x < 0.2 ) //doesn't improve any more
	  maximum_x.done= true;
    }
#endif
  }
  ART_DEB(0,_2D << BLA(shortest_time,"ff0000",0) << BLA(shortest_way,"00ff00",1.5) << BLA(maximum_x,"0000ff",-1.5));
  ART_DEB(0, << "\nshortest_time= " << BLA2(shortest_time) << "\nshortest_way= " << BLA2(shortest_way) << "\nmaximum_x= " << BLA2(maximum_x));

  //time_to_target( shortest_way.pos, player, time, cmd);
  time_to_target( shortest_time.pos, player, time, cmd);
  //time_to_target( maximum_x.pos, player, time, cmd);
}
#endif

Vector old_pos;
int old_time;

void Intercept2::intercept( Vector const & ball_pos, Vector const & ball_vel, PPlayer player, int & time, Cmd_Body & cmd ) {
  ART_DEB(0,"New intercept");
  compute_possible_intercept_positions(ball_pos, ball_vel, player );  
  intercept_positions.show2d(); //debug
  if ( intercept_positions.is_empty() ) {
    time= -1;
    return;
  }

  int my_best_time= intercept_positions.pos[0].my_time;
  int fastest_opponent_time;
  compute_fastest_opponent(ball_pos, ball_vel, fastest_opponent_time);

  for ( int i=1; i< intercept_positions.num_pos; i++) {  //start at 1, so we don't erase all possibilities
    if ( intercept_positions.pos[i].my_time > fastest_opponent_time )
      intercept_positions.num_pos= i;
  }

  int best_idx= 0;

  Vector new_pos= intercept_positions.pos[0].ball_pos;
  bool dont_change_new_pos= false;

  ART_DEB(0,"old_time= " << old_time << " cur_time= " << WSinfo::ws->time );

  Vector pos;
  if ( can_reach_ball_with_dashing(ball_pos, ball_vel, player->radius + 0.2 * player->kick_radius, time, pos) ) {
    ART_DEB(0, << "can_reach_ball_with_dashing, time= " << time << " pos= " << pos);
    ART_DEB(0, << _2D << C2D(pos.x,pos.y,3,"00ffff"));
    new_pos= pos;
    if ( time <= fastest_opponent_time + 2 && time <= my_best_time + 3)
      dont_change_new_pos= true;

    //if ( time <= fastest_opponent_time + 3 )
    //  dont_change_new_pos= true;

    if ( time <=  my_best_time + 1 )
      dont_change_new_pos= true;

    ART_DEB(0, " dont_change_new_pos= " << dont_change_new_pos << " dash_time= " << time << "  my_best_time= " << my_best_time << " fastest_opponent_time= " << fastest_opponent_time );
  }
  else {
    ART_DEB(0, << "CANNOT  reach_ball_with_dashing, time= " << time << " pos= " << pos);
  }

  if ( ! dont_change_new_pos ) {
    int best_advantage= intercept_positions.pos[0].ball_time - intercept_positions.pos[0].ball_time - 1;
    for (int i=0; i < intercept_positions.num_pos; i++ ) {
      InterceptPositions::ResPos const& p= intercept_positions.pos[i];
      int advantage=  p.ball_time - p.my_time;
      if ( advantage > best_advantage && best_advantage <= 2 && advantage <= 5) {
	best_idx= i;
	best_advantage= advantage;
      }
      else 
	break;
    }
    new_pos= intercept_positions.pos[best_idx].ball_pos;
  }

  old_time= WSinfo::ws->time;
  
  ART_DEB(0,_2D << C2D( new_pos.x, new_pos.y, 1.2,"ff0000") << C2D( old_pos.x, old_pos.y, 1.0,"00ff00") 
	  << STRING2D(new_pos.x+1.5, new_pos.y,
		      intercept_positions.pos[best_idx].ball_time << ',' <<
		      intercept_positions.pos[best_idx].my_time, "ff0000"));
  old_pos= new_pos;

  double min_tolerance= player->radius + ( player->kick_radius - player->radius ) * 0.5;
  double tolerance= player->kick_radius;
  if ( player->pos.sqr_distance( new_pos ) < SQUARE(tolerance) )
    tolerance *= 0.5;
  if ( tolerance < min_tolerance )
    tolerance = min_tolerance;

  time_to_target( new_pos, tolerance , player, time, cmd);
}

void Intercept2::compute_possible_intercept_positions( Vector const & ball_pos, Vector const & ball_vel, PPlayer player) {
  double ball_vel_considered_slow= 0.5;

  Vector bpos= ball_pos;
  Vector bvel= ball_vel;
  double  bspeed= ball_vel.norm();

#if 0
  int ball_start_time= 0;
  while ( ball_start_time < 0 ) {
    bpos += bvel;
    bvel *= ServerOptions::ball_decay;
    bspeed *= ServerOptions::ball_decay;
    ball_start_time++;
  }
#endif
  intercept_positions.reset();
  const double tolerance= player->kick_radius * 0.8;

  int time= 0;
  while ( true ) {

    bpos += bvel;

    if ( ! FIELD_AREA.inside(bpos) ) {
      bpos -= bvel;
      if ( intercept_positions.is_empty() ) { // take the last position in the field as intercept position
	InterceptPositions::ResPos res;
	res.ball_time= time;
	time_to_target(bpos, tolerance, player, res.my_time); 
	res.ball_pos= bpos;
	res.ball_vel= bvel;
	res.ball_speed= bspeed;
	intercept_positions.add(res);
      }
      break;
    }

    bvel *= ServerOptions::ball_decay;
    bspeed *= ServerOptions::ball_decay;
    time++;

    if ( bspeed < ball_vel_considered_slow && !intercept_positions.is_empty() )
      break;

#if 0
    ART_DEB(0,"time= " << time << " bpos= " << bpos << " bvel= " << bvel << " playerpos= " << player->pos);
    if ( ! FIELD_AREA.inside(bpos) ) //|| bvel.sqr_norm() < sqr_ball_vel_considered_slow )
      break;
#endif

    //this is an optimization, don't compute the intercept position if the player in ANY case needs longer the the current time
    if ( player->pos.sqr_distance( bpos ) > SQUARE( (player->age+time) * player->speed_max + player->kick_radius ) ) {
      //ART_DEB(0,"skipping time= " << time << " bpos= " << bvel << " ppos= " << player->pos << " sqr_distance( bpos )= " << player->pos.sqr_distance( bpos ) << " > " << SQUARE( (player->age+time) * player->speed_max + player->kick_radius ) << " age= " << player->age << " time= " << time  << " speed= " << player->speed_max << " r=" << player->kick_radius);
      continue;
    }
    int need_time;
    time_to_target(bpos, tolerance, player, need_time);

#if 0
    ART_DEB(0, << "need_time= " << need_time );
#endif

    if ( need_time > time ) 
      continue;

    InterceptPositions::ResPos res;
    res.ball_time= time;
    res.my_time= need_time;
    res.ball_pos= bpos;
    res.ball_vel= bvel;
    res.ball_speed= bspeed;

    if ( ! intercept_positions.add(res) )
      break;
  }
}

void Intercept2::compute_fastest_opponent(Vector const & ball_pos, Vector const & ball_vel, int & time) {
  PlayerSet pset= WSinfo::valid_opponents;

  Vector bpos= ball_pos;
  Vector bvel= ball_vel;
  //double  bspeed= ball_vel.norm();

  if ( pset.num <= 0 ) {
    time= 100;
    return;
  }
  time= 0;
  while ( true ) {
    for (int i=0; i<pset.num; i++) {
      PPlayer player= pset[i];
      if ( player->pos.sqr_distance( bpos ) < SQUARE(  time * player->speed_max + player->kick_radius ) ) 
	return;
    }
    bpos += bvel;
    bvel *= ServerOptions::ball_decay;

    //bspeed *= ServerOptions::ball_decay;
    time++;
  }
}


bool Intercept2::can_reach_ball_with_dashing(Vector const & ball_pos, Vector const & ball_vel, double tolerance, int & time, Vector & pos) {
  Object obj(WSinfo::me);

  Vector intersection_pos;
  Vector projection_pos;
  double s1, s2;
  Vector acc(obj.ang);
  bool intersection_res= Geometry2d::intersect_lines(intersection_pos, s1, s2, 
					Line2d(obj.pos, acc), 
					Line2d(ball_pos, ball_vel));

  if ( intersection_res ) {
    if ( ! FIELD_AREA.inside( intersection_pos ) )
      intersection_res= false;

    if ( s1 < 0 || s2 < 0 )  //in future negative s1 near zero will be handled
      intersection_res= false;
  }


  bool projection_res;
  {
    //check if ball will go towards the player
    Geometry2d::projection_to_line( projection_pos, obj.pos, Line2d(ball_pos, ball_vel) );

    if ( projection_pos.sqr_distance( obj.pos ) < SQUARE(obj.param->kick_radius * 0.8) &&
	 ball_vel.dot_product( projection_pos - ball_pos ) > 0 ) {
      projection_res= true;
    }
    else 
      projection_res= false;
  }

  if ( projection_res && intersection_res ) {  //eliminate one of the possibilities!
    if ( projection_pos.sqr_distance(ball_pos) + 1.0  < intersection_pos.sqr_distance(ball_pos) )
      intersection_res= false;
    else
      projection_res= false;
  }

  double players_distance;
  double players_speed;

  if ( projection_res ) {
    pos= projection_pos;
    players_distance= pos.distance(obj.pos);
    players_speed= 0;
  }
  else if ( intersection_res ) {
    pos= intersection_pos;
    players_distance= s1;
    players_speed= obj.vel.norm();
  }
  else
    return false;
  

  double balls_distance= ball_pos.distance(pos);
  double balls_speed= ball_vel.norm();

  ART_DEB(0, << _2D << C2D( pos.x, pos.y, 2.0, "00ff00") << L2D(ball_pos.x, ball_pos.y, ball_pos.x + 10*ball_vel.x, ball_pos.y + 10*ball_vel.y, "00ff00"));
  ART_DEB(0, << " projection_res= " << projection_res << " intersection_res= " << intersection_res << " ball_vel= " << ball_vel << " players_distance= " << players_distance << " balls_distance= " << balls_distance);

  double players_way= 0;
  double balls_way= 0;

  time= 0;

  while (true) {  //check if the player can reach intersection_pos earlier then the ball!
    if ( players_way >= players_distance - tolerance  ) {
      while ( true ) {
	if ( balls_way >= balls_distance )
	  return true;

	if ( balls_speed <= 0.3 )
	  return false;

	balls_way += balls_speed;
	balls_speed *= ServerOptions::ball_decay;
	time++;
      }
    }

    if ( balls_way >= balls_distance ) //+ tolerance ) 
      return false;
      
    players_speed += 100 * (obj.param->dash_power_rate*obj.param->effort);
    if ( players_speed > obj.param->speed_max )
      players_speed= obj.param->speed_max;
    players_way+= players_speed;
    players_speed *= obj.param->decay;

    balls_way += balls_speed;
    balls_speed *= ServerOptions::ball_decay;

    time++;
  }
  return false;
}

void Intercept2::conv_to_canonical_form( Vector & target, Vector & pos, Vector  & vel, ANGLE  & ang) const {
  //this method uses 2 costly functions: norm() and atan2 (inside of ANGLE(x,y)
  //this amounts to the calculation of the polar coordinates of an angle
  Vector tmp= target-pos;
  double norm= tmp.norm();
  if ( norm < 0.001 ) 
    tmp= Vector(1,0);
  else
    tmp *= 1.0/norm;
  
  if ( LogOptions::is_on() ) {
    Vector v(ang);
    Vector v2= vel;
    v2.normalize();
    ART_DEB(1,<< "my pos= " << pos << " my_vel= " << vel << "(" << v2 << ") my_ang= " << ang << " acc= " << v);
    v*= 3;
    v+= pos;
    ART_DEB(1,<< _2D << C2D(target.x,target.y,1,"0000ff")
	    << C2D(pos.x,pos.y,1,"ff0000")
	    << L2D(pos.x,pos.y,v.x,v.y,"ff0000")
	    << L2D(pos.x,pos.y,pos.x+vel.x,pos.y+vel.y,"00ff00")
	    );
  }
  target= Vector(0,0);
  pos= Vector(-norm,0); 
  /* Rotation Matrix (counterclockwise)
   tmp.x= cos, tmp.y= sin
  [ cos     sin]
  [ -sin     cos]
  */
  double old_x= vel.getX();
  vel.setX(  tmp.getX() * old_x + tmp.getY() * vel.getY() );
  vel.setY(- tmp.getY() * old_x + tmp.getX() * vel.getY() );

  ang -= ANGLE(tmp.getX(),tmp.getY());

  if ( LogOptions::is_on() ) {
    Vector v(ang);
    Vector v2= vel;
    v2.normalize();
    ART_DEB(1,<< "my pos= " << pos << " my_vel= " << vel << "(" << v2 << ") my_ang= " << ang << " acc= " << v);
    v*= 3;
    v+= pos;
    ART_DEB(1,<< _2D << C2D(target.x,target.y,1,"0000ff")
	    << C2D(pos.x,pos.y,1,"ff0000")
	    << L2D(pos.x,pos.y,v.x,v.y,"ff0000")
	    << L2D(pos.x,pos.y,pos.x+vel.x,pos.y+vel.y,"00ff00")
	    );
  }
}

void Intercept2::dash_to_minimize_vel( Object const & obj, double & dash, Vector & new_vel ) const {
  Vector acc(obj.ang);
  
  /* 
     norm of acc is 1 !

     minimize f(dash)= (vel + dash * acc)^2 

     use derivation

     f'(dash) == 0    <---->   vel.x * acc.x + vel.y * acc.y + dash * acc.sqr_norm() == 0

  */

  dash =  - ( obj.vel.getX() * acc.getX() + obj.vel.getY() * acc.getY() ) / (obj.param->dash_power_rate*obj.param->effort);
  if ( dash >= 100 )
    dash= 100;
  else if ( dash <= -100 )
    dash= -100;

  new_vel= obj.vel + dash * (obj.param->dash_power_rate*obj.param->effort) * acc;
  ART_DEB(1, << "\n result of dash_to_min_vel, vel= " << obj.vel << " to " << new_vel << " by dash = " << dash << " acc= " << acc << " ang= " << obj.ang);
}

void Intercept2::turn_to_pos( Vector & target, Object & obj, int & time, double tolerance, NoneDashTurn & ndt ) const {
  conv_to_canonical_form(target, obj.pos, obj.vel, obj.ang);

  time= -1;

  double distance= - obj.pos.getX();

  if ( distance < tolerance ) {
    double dash;
    Vector new_vel;
    if ( tolerance < 0.8 ) { //just try reduce speed, if the tolerance is small!
      dash_to_minimize_vel(obj, dash, new_vel ); //maybe later set dash to minimize dist to pos
      ndt.set_dash( dash * (obj.param->dash_power_rate*obj.param->effort) );
    }
    else
      ndt.set_none();
    time= 0;
    return;
  }
    
  const double velocity= obj.vel.norm();
#if 0
  const Vector pos_in_two_cycles= obj.pos + obj.vel + obj.param->decay*obj.vel;
  if ( pos_in_two_cycles.sqr_norm() < tolerance*tolerance ) {
    time= 1;
    ndt.set_none();
  }
#endif
  
  double ang = obj.ang.get_value_mPI_pPI();

  const double slope= tan(fabs(ang));  //CHECK THIS, can be infinite

  ART_DEB(1,<< "ang_val= " << ang_val << " slope*dist= " << slope*distance);
  //if ( slope * distance < obj.param->kick_radius   //original
  if ( slope * distance < tolerance + distance * 0.1   //test
       && fabs(ang) < PI * 0.5 ) //this asserts, that the angle is heading toward the destination
  { 
    ndt.set_dash( 100 * (obj.param->dash_power_rate*obj.param->effort) );
    double my_way= 0.0;
    time= 0;
    double old_inc= velocity;
    double cur_velocity= velocity;
    while ( my_way < distance - obj.param->kick_radius * 0.9 ) {
      double inc= cur_velocity + 100 * (obj.param->dash_power_rate*obj.param->effort);
      if ( inc >= obj.param->speed_max )
	inc= obj.param->speed_max;
#if 1
      if ( inc - old_inc < 0.001 ) { //now I'm keeping a constant velocity, don't need to iterate to the end
	time += int ( ceil( ( distance - my_way ) / inc ) );
	//ART_DEB(1, << " keeping vel of = " << inc << " speed_max= " << obj.param->speed_max  << " orig. velocity= " << velocity);
	break;
      }
      old_inc= inc;
#endif
      cur_velocity = inc * obj.param->decay;
      my_way+= inc;
      time++;
    }
    //time= 1000; //to be improved
    return;
  }
  
  //now we are going to turn
  ART_DEB( 1, " velocity= " << velocity );
  if ( velocity > 0.2 ) { 
    //simulate next position
    Vector tmp_pos = obj.pos + obj.vel;
    Vector tmp_vel = obj.param->decay * obj.vel;
    ANGLE  tmp_ang =  obj.ang;
    conv_to_canonical_form(target,tmp_pos,tmp_vel, tmp_ang);
    ang = tmp_ang.get_value_mPI_pPI();
  }
  else
  {
    ang = obj.ang.get_value_mPI_pPI();
  }

  double moment= fabs(ang) * ( 1.0 + velocity * obj.param->inertia_moment );
  ART_DEB(1, << " moment= " << moment << " velocity= " << velocity);

  bool moment_deficit= false;
  if ( moment > PI_MINUS_EPS ) {
    moment_deficit= true; //moment- PI;
    moment = PI_MINUS_EPS;
  }
  moment *= -(ang < 0 ? -1 : 1);

  if ( ! moment_deficit || velocity < 0.2 ) {
    ndt.set_turn( moment / ( 1.0 + velocity * obj.param->inertia_moment ) );
    return;
  }

  /*  this code was commented out in padua2003 to save stamina
  { //try to reduce the own velocity significantly
    double dash;
    Vector new_vel;
    dash_to_minimize_vel(obj, dash, new_vel );
    
    if ( new_vel.sqr_norm() < SQUARE(0.1) ) {
      ART_DEB(1, << "\n reducing velocity from " << obj.vel << " to " << new_vel << " by dash = " << dash );
      ndt.set_dash( dash * (obj.param->dash_power_rate*obj.param->effort) );
      return;
    }
    else
      ART_DEB(1, << "\n NOT reducing velocity from " << obj.vel << "(" << obj.vel.norm() << ") to " << new_vel << "(" << new_vel.norm() << ") by dash = " << dash );
  }
  */
  //try the turn at last
  ndt.set_turn( moment / ( 1.0 + velocity * obj.param->inertia_moment ) );

  return;

}

/******************************************************************************
  end class Intercept
*******************************************************************************/
