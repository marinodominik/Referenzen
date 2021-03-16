
#include "mdp_memory.h"
#include "mdp_info.h"
#include "options.h"
#include "mdpstate.h"
#include "log_macros.h"

#define TEAM_SIZE 11

MDPmemory::MDPmemory() {
  opponent_last_at_ball_number = 0;
  opponent_last_at_ball_time = 0;
  teammate_last_at_ball_number = 0;
  teammate_last_at_ball_time = 0;
  counter = 0;
}

void MDPmemory::update(){
  Vector ball, player;
  if (WSinfo::is_ball_pos_valid()) {
    //berechnet mit Gedaechtnis wer im Angriff ist (=schnellster Spieler zum Ball)
    int summe = 0;
    momentum[counter] = mdpInfo::fastest_team_to_ball();
    counter = (counter + 1 ) % 5;
    for(int i=0; i<MAX_MOMENTUM;i++){
      summe += momentum[i];
    }
    team_in_attack = summe / MAX_MOMENTUM;

    ball = WSinfo::ball->pos;
    for (int i = 1 ; i <= TEAM_SIZE ; i++) {
      if (mdpInfo::is_teammate_pos_valid(i)){
	player =  mdpInfo::teammate_pos_abs(i);
	if (fabs((player - ball).norm()) < ServerOptions::kickable_area) {
	  teammate_last_at_ball_number = i;
	  teammate_last_at_ball_time = mdpInfo::mdp->time_current;
	}
      }
      if (mdpInfo::is_opponent_pos_valid(i)) {
	player =  mdpInfo::opponent_pos_abs(i);
	if (fabs((player - ball).norm()) < ServerOptions::kickable_area) {
	  opponent_last_at_ball_number = i;
	  opponent_last_at_ball_time = mdpInfo::mdp->time_current;
	}
      }
    }
    if(teammate_last_at_ball_time >= opponent_last_at_ball_time)
      team_last_at_ball = 0;
    else
      team_last_at_ball = 1;
  }   
  mdpInfo::update_memory(*this);
}

  
