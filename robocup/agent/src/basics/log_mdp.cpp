#include "log_mdp.h"
#include <stdio.h>
#include <sstream>
#include "log_macros.h"
#include "mdp_info.h"

void log_mdp_me( const MDPstate *mdp, std::string tag )
{
    char buffer[ 200 ];
    Vector my_pos = mdp->me->pos();
    Vector my_vel = mdp->me->vel();

    sprintf( buffer, "Me: Abs (%.1f %.1f age: %d (%.2f)) V (%d %.2f /%.2f %.2f age: %d (%.2f)) NeckAngle %d BodyAngle %d Stamina: %d TheirGoalie: %d OurGoalie: %d", my_pos.getX(), my_pos.getY(), mdpInfo::age_playerpos( mdp->me ), mdp->me->pos_x.p, ( int ) RAD2DEG( my_vel.arg() ), my_vel.norm(), my_vel.getX(), my_vel.getY(), mdpInfo::age_playervel( mdp->me ), mdp->me->vel_x.p, ( int ) RAD2DEG( mdp->me->neck_angle.v ), ( int ) RAD2DEG( mdp->me->ang.v ), ( int ) mdp->me->stamina.v, ( int ) mdp->his_goalie_number, mdp->my_goalie_number );

    // show neck angle
    Vector target;
    target.init_polar( 1., mdpInfo::mdp->me->neck_angle.v );
    if( mdp->time_current > 0 )
    {
        //LOG_MDP(1,<< tag <<" "<< buffer);
        LOG_MDP( 1, << _2D << VL2D( my_pos, target, "red" ) );
    }
}

void log_mdp_ball(const MDPstate *mdp,std::string tag){
  char buffer[200];
  Vector my_pos =  mdp->me->pos();
  Vector ball_pos =  mdp->ball.pos();
  Vector rel_ball_pos = ball_pos-my_pos;

  sprintf(buffer, 
     "Ball: Abs (%.1f %.1f age: %d (%.2f)) Rel (%.1f %.1f;%.2f) V (%d %.2f /%.1f %.1f age %d)",
      ball_pos.getX(),ball_pos.getY(),
	  WSinfo::ball->age,
	  mdp->ball.pos_x.p,
	  rel_ball_pos.getX(), rel_ball_pos.getY(),
	  rel_ball_pos.norm(),
	  ( int ) RAD2DEG(WSinfo::ball->vel.arg()),WSinfo::ball->vel.norm(),
	  WSinfo::ball->vel.getX(),
	  WSinfo::ball->vel.getY(),
	  WSinfo::ball->age);

  // show ball, ball vel
#if LOGGING && BASIC_LOGGING
  float x1,x2,x3,x4;
#endif
  Vector ball =WSinfo::ball->pos;

#if LOGGING && BASIC_LOGGING
  x1= ball.getX();
  x2= ball.getY();
#endif
  ball = ball + mdp->ball.vel();
#if LOGGING && BASIC_LOGGING
  x3= ball.getX();
  x4= ball.getY();
#endif
  LOG_MDP(1,<< _2D <<C2D( x1,x2,0.085,"blue")<<L2D( x1,x2,x3,x4,"blue"));
  if(mdpInfo::server_state!=NULL){
    if(mdpInfo::server_state->time_current >= mdpInfo::mdp->time_current){
      ball = mdpInfo::server_state->ball.pos();
      ball = ball - mdpInfo::server_state->me->pos();
      ball = ball + mdp->me->pos();
#if LOGGING && BASIC_LOGGING
      x1= ball.getX();
      x2= ball.getY();
#endif
      ball = ball + mdpInfo::server_state->ball.vel();
#if LOGGING && BASIC_LOGGING
      x3= ball.getX();
      x4= ball.getY();
#endif
      LOG_MDP(1,<< _2D << C2D( x1,x2,0.085,"black")<<L2D( x1,x2,x3,x4,"black"));
    }
  }
}

void log_mdp_all(const MDPstate *mdp){
  log_mdp_me(mdp,"W");
  if(mdpInfo::server_state!=NULL){
    if(mdpInfo::server_state->time_current >= mdpInfo::mdp->time_current)
      log_mdp_me(mdpInfo::server_state,"SW");
  }
  log_mdp_ball(mdp,"W");
  if(mdpInfo::server_state!=NULL){
    if(mdpInfo::server_state->time_current >= mdpInfo::mdp->time_current)
      log_mdp_ball(mdpInfo::server_state,"SW");
  }
  log_mdp_all(mdp,"W");
  if(mdpInfo::server_state!=NULL){
    if(mdpInfo::server_state->time_current >= mdpInfo::mdp->time_current)
      log_mdp_all(mdpInfo::server_state,"SW");
  }
}

void log_mdp_all(const MDPstate *mdp,std::string tag){
  char buffer[2000];
  std::stringstream stream_2d;
  int i;
  int seen_before;

  sprintf(buffer,"%s","");
  for(i=0;i<11;i++){  // I am always my_team[0]
    if(mdp->my_team[i].pos_x.p){  // >0: position valid
      seen_before = mdpInfo::age_playerpos(&(mdp->my_team[i]));
      sprintf(buffer,"%s (F%d (%.0f,%.0f);(%.2f,%.2f) age: %d (%.2f)) ",
	      buffer,
	      mdp->my_team[i].number,
	      mdp->my_team[i].pos_x.v,mdp->my_team[i].pos_y.v,
	      mdp->my_team[i].vel_x.v,mdp->my_team[i].vel_y.v,
	      seen_before,
	      mdp->my_team[i].pos_x.p);

      //@andi: um den Radius fuer Feinpositionierung zu zeigen
      DashPosition t = DeltaPositioning::get_position(mdp->my_team[i].number);
      //LOG_DRAWCIRCLE_D<<(int)t.x<<" "<<(int)t.y<<" "<<(int)t.radius<<" purple1";
      stream_2d <<VC2D(t, t.radius, "purple1");

      if(strcmp(tag.c_str(),"W") == 0){
	if(seen_before < 4){
	  //LOG_DRAWCIRCLE_D<<(int)mdp->my_team[i].pos_x.v<<" "<<(int)mdp->my_team[i].pos_y.v<<" 2 blue";
	  stream_2d<<C2D(mdp->my_team[i].pos_x.v,mdp->my_team[i].pos_y.v ,.3,"blue");	  
	}
	else{
	  //LOG_DRAWCIRCLE_D<<(int)mdp->my_team[i].pos_x.v<<" "<<(int)mdp->my_team[i].pos_y.v<<" 2 pink";
	  stream_2d<<C2D(mdp->my_team[i].pos_x.v,mdp->my_team[i].pos_y.v ,.3,"pink");	  
	}
      }
    }
  }
  //LOG_MDP(1,<<tag<<"      "<<buffer);
  sprintf(buffer,"%s","");

  // Information about Opponents:
  for(i=0;i<11;i++){  
    if(mdp->his_team[i].pos_x.p){  // >0: position valid
      seen_before = mdpInfo::age_playerpos(&(mdp->his_team[i]));
      sprintf(buffer,"%s (O%d (%.0f,%.0f);(%.2f,%.2f) age: %d (%.2f)) ",
	      buffer,
	      mdp->his_team[i].number,
	      mdp->his_team[i].pos_x.v,mdp->his_team[i].pos_y.v,
	      mdp->his_team[i].vel_x.v,mdp->his_team[i].vel_y.v,
	      seen_before,
	      mdp->his_team[i].pos_x.p);
      if(strcmp(tag.c_str(),"W") == 0){
	if(seen_before < 4){
	  stream_2d<<C2D(mdp->his_team[i].pos_x.v,mdp->his_team[i].pos_y.v ,.3,"red");	  
	}
	else{
	  stream_2d<<C2D(mdp->his_team[i].pos_x.v,mdp->his_team[i].pos_y.v ,.3,"magenta");
	}
      }
    }
  }
  
  stream_2d << '\0';
  LOG_MDP(1,<< _2D << stream_2d.str());
}

