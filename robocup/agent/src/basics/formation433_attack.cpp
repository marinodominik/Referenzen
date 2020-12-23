#include "formation433_attack.h"
#include "macro_msg.h"
#include "ws_info.h"
#include "valueparser.h"
#include "log_macros.h"
#include "ws_memory.h"
#include <stdio.h>

#define DBLOG_POL(LLL,XXX) LOG_POL(LLL,<<"Formation433Attack: "<<XXX)
//#define DBLOG_MOV(LLL,XXX) 
#define DBLOG_DRAW(LLL,XXX) LOG_POL(LLL,<<_2D<<XXX)
//#define DBLOG_DRAW(LLL,XXX) 


bool Formation433Attack::init(char const * conf_file, int argc, char const* const* argv) {
  if (NUM_PLAYERS < 11) {
    ERROR_OUT << "\nwrong number of players: " << NUM_PLAYERS; 
    return false;
  }
  //the following entries are (usually) in [0,1]x[-1,1]
  //x values smaller then 0 violate our defence line
  //x values greater then 1 violate our offence line
  //y values > 1 or < -1 violate the top or bottom field borders
  //goalie doesn't retrieve his pos from a Formation, but just to be consistent: he would run out of the field!

  for(int i=0; i<NUM_FORMATIONS; i++){
    // init all
    for(int j=1; j<= 11; j++){
      home[i][j].role = 0; 
      home[i][j].pos = Vector(0); 
      home[i][j].delta_x = 0; 
      home[i][j].delta_y = 0; 
    }
  }

  home[STANDARD][1].role = PT_GOALIE;
  home[STANDARD][2].role = PT_DEFENDER;
  home[STANDARD][3].role = PT_DEFENDER;
  home[STANDARD][4].role = PT_DEFENDER;
  home[STANDARD][5].role = PT_DEFENDER;
  home[STANDARD][6].role = PT_MIDFIELD;
  home[STANDARD][7].role = PT_MIDFIELD;
  home[STANDARD][8].role = PT_MIDFIELD;
  home[STANDARD][9].role = PT_ATTACKER;
  home[STANDARD][10].role= PT_ATTACKER;
  home[STANDARD][11].role= PT_ATTACKER;

  // defenders
  home[STANDARD][1].pos = Vector(0, 0); 
  home[STANDARD][2].pos = Vector(0, 0.6);
  home[STANDARD][3].pos = Vector(0, 0.2);
  home[STANDARD][4].pos = Vector(0,-0.2);
  home[STANDARD][5].pos = Vector(0,-0.6);

  // midfielders and attackers: 
  home[STANDARD][6].pos = Vector(0.5, 0.5);
  home[STANDARD][7].pos = Vector(0.5, 0.2);
  home[STANDARD][8].pos = Vector(0.5,-0.4);
  home[STANDARD][9].pos = Vector(1.0, 0.8);
  home[STANDARD][10].pos= Vector(1.0, 0.0);
  home[STANDARD][11].pos= Vector(1.0,-0.6);

  // special left wing formation
  home[LEFT_WING][6].delta_y = .1;
  home[LEFT_WING][7].delta_y = .1;
  home[LEFT_WING][8].delta_y = .2;
  home[LEFT_WING][9].delta_y = 0;
  home[LEFT_WING][10].delta_y = .1;
  home[LEFT_WING][11].delta_y = .5;

  // special right wing formation
  home[RIGHT_WING][6].delta_y = -.4;
  home[RIGHT_WING][7].delta_y = -.5;
  home[RIGHT_WING][8].delta_y = -.1;
  home[RIGHT_WING][9].delta_y = -.4;
  home[RIGHT_WING][10].delta_y = -.1;
  home[RIGHT_WING][11].delta_y = -.1;

  // special close2 goal formation - unsymmetric
  home[CLOSE2GOAL][6].delta_y = .0;
  home[CLOSE2GOAL][7].delta_y = 0;
  home[CLOSE2GOAL][8].delta_y = .2;
  home[CLOSE2GOAL][9].delta_y = -0.4;
  home[CLOSE2GOAL][10].delta_y = 0;
  home[CLOSE2GOAL][11].delta_y = .3;

  for(int i=1; i<NUM_FORMATIONS; i++){
    for(int j=1; j<= 11; j++){
      home[i][j].role = home[STANDARD][j].role; // init with default values
      home[i][j].pos = home[STANDARD][j].pos; // init with default values
      home[i][j].pos += Vector(home[i][j].delta_x,home[i][j].delta_y);
    }
  }

  formation_state_updated_at = 0;
  previous_formation_state = STANDARD;
  current_formation_state = STANDARD;

  return true;
}

int Formation433Attack::get_role(int number) {
  return home[get_formation_state()][number].role;
}

int Formation433Attack::get_formation_state(){
  
  if (formation_state_updated_at == WSinfo::ws->time){
    //DBLOG_POL(1,"No Need to update! Current Formation state "<<current_formation_state<<" updated at "<<formation_state_updated_at);
    return current_formation_state;
  }

  // do state transition
  // allowed transitions: s <-> l, s <-> r, s <-> c
  // d.h. wenn ich vorne links bin, muss der Ball in die Mitte, um dann von standard nach close2goal zu wechseln.
  Vector ballpos = WSinfo::ball->pos;
  const float s2l_thresh = 25;  // standard to left threshold
  const float l2s_thresh = 20;   // left to standard  threshold
  const float s2r_thresh = -15;
  const float r2s_thresh = -10;
  const float s2c_thresh = 35;
  const float c2s_thresh = 25;

  //DBLOG_POL(1,"Check Transition Previous Formation state: "<<previous_formation_state<<" Ballpos "<<ballpos);
  
  // from standard (central node)
  // ordering determines priority: first  transition has least priority; others can overwrite !
#define STATEMACHINE_1 // this is type 1: wing has priority over close2goal

#ifdef STATEMACHINE_1
  if(previous_formation_state == STANDARD && ballpos.getX() > s2c_thresh)
    current_formation_state = CLOSE2GOAL;
#endif
  //  if(previous_formation_state == STANDARD && ballpos.y > s2l_thresh)
  if(ballpos.getY() > s2l_thresh) // allways switch if ball is left
    current_formation_state = LEFT_WING;
  //  if(previous_formation_state == STANDARD && ballpos.y < s2r_thresh)
  if(ballpos.getY() < s2r_thresh) // allways switch if ball is right
    current_formation_state = RIGHT_WING;

#ifndef STATEMACHINE_1
  if(ballpos.x > s2c_thresh)  
    current_formation_state = CLOSE2GOAL;
#endif


  // from left
  if(previous_formation_state == LEFT_WING && ballpos.getY() < l2s_thresh)
    current_formation_state = STANDARD;

  // from right
  if(previous_formation_state == RIGHT_WING && ballpos.getY() > r2s_thresh)
    current_formation_state = STANDARD;

  // from close2goal
  if(previous_formation_state == CLOSE2GOAL && ballpos.getX() < c2s_thresh)
    current_formation_state = STANDARD;

  formation_state_updated_at = WSinfo::ws->time;
  previous_formation_state = current_formation_state;
  DBLOG_POL(1,"Current Formation state "<<current_formation_state<<" updated at "<<formation_state_updated_at
	    <<"Previous Formation state: "<<previous_formation_state<<" Ballpos "<<ballpos);

  return current_formation_state;
}

Vector Formation433Attack::get_grid_pos(int number) {
  const float midfielder_maxdist2offence_line = 20;

  int formation_state = get_formation_state();
  DBLOG_POL(1,"Formation state: "<<formation_state);

  Vector res;
  Home & h = home[formation_state][number]; 

  get_boundary(defence_line, offence_line); 

  double x_stretch= offence_line-defence_line;
  double y_stretch= FIELD_BORDER_Y;

  res.setX(defence_line + h.pos.getX() * x_stretch);
  res.setY(h.pos.getY() * y_stretch);


  // special rules
  if(get_role(number) == 1 && WSmemory::team_last_at_ball() == 0){// midfielder, we attack
    if(res.getX() < offence_line - midfielder_maxdist2offence_line){
      res.setX(offence_line - midfielder_maxdist2offence_line);
    }
  }

  if(number == WSinfo::me->number && // I ask my target position
     WSinfo::me->pos.distance(res) < 5.){ // I'm close to target
    // check whether teammate is close to my target position, too
    PlayerSet pset= WSinfo::valid_teammates_without_me;
    pset.keep_players_in_circle(res,3.);
    if (pset.num >0){
      DBLOG_POL(0,"Compute Grid pos: There's a player close to my home -> stay ");
      res = WSinfo::me->pos;
    }

  }

  DBLOG_DRAW(1,VC2D(res,1.5,"aaaa00") << STRING2D(res.getX()+0.4,res.getY(), number,"aaaa00"));
  DBLOG_POL(0,"Get Grid pos: "<<res);
  return res;
}

bool Formation433Attack::need_fine_positioning(int number) {
  Vector grid_pos= get_grid_pos(number);

  if ( WSinfo::me->pos.sqr_distance( grid_pos ) > SQUARE(3.0) )
    return true;
  return false;
}

Vector Formation433Attack::get_fine_pos(int number) {
  Vector res= get_grid_pos(number);
  return res;
}

void Formation433Attack::get_boundary(double & defence, double & offence) {
  if ( boundary_update_cycle == WSinfo::ws->time ) {
    defence= defence_line;
    offence= offence_line;
  }

  boundary_update_cycle= WSinfo::ws->time;

  Vector ball_pos= WSinfo::ball->pos;
  Vector ball_vel= WSinfo::ball->vel;
  if ( ! WSinfo::is_ball_pos_valid() ) //don't use the velocity, it the ball is too old!
    ball_vel= Vector(0,0);

  PlayerSet pset= WSinfo::valid_teammates;
  pset+= WSinfo::valid_opponents;
  InterceptResult ires[1];
  
  pset.keep_and_sort_best_interceptors(1, ball_pos, ball_vel, ires);

  defence_line= ires[0].pos.getX() ;

  //don't go into the opponent's half
  if ( defence_line > -4.0 )
    defence_line= -4.0;

  //don't go out of the field
  if ( defence_line < -FIELD_BORDER_X + 10 )
    defence_line= -FIELD_BORDER_X + 10;

  offence_line= WSinfo::his_team_pos_of_offside_line();
  if ( offence_line < 0.0 )
    offence_line= 0.0;

  if ( offence_line > FIELD_BORDER_X - 5.0 )
    offence_line= FIELD_BORDER_X- 5.0;
  

  if ( offence_line - defence_line > 50.0 )
    offence_line= defence_line+ 50.0;
  
  defence= defence_line;
  offence= offence_line;
}





