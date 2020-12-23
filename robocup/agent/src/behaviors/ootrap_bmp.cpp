#include "ootrap_bmp.h"
//#include "random.h"
#include "ws_memory.h"
#include "../policy/planning.h"
#include "ws.h"
#include "../policy/positioning.h"
#include "../policy/policy_tools.h"
#include "mdp_info.h"
#include "blackboard.h"

//#include <cstdlib>
//#include <sstream>


#define P1_HOLD 1
#define P1_PASS 3
#define P1_WAIT_OR_RUN 5
#define P1_RUN 7
#define P1_INTERCEPT 9

#define P2_GO2POS2 20
#define P2_WAIT 22
#define P2_INTERCEPT 24
#define P2_PASS 26
#define P2_SAY 28

#define TIME_TO_REACT 1
#define MAX_TIME 20

//#define P1_MIN_OL_DIST 8
#define P1_MIN_RADIUS 8
#define P2_MIN_OL_DIST 16
#define P2_MIN_RADIUS 16

#define POS_2_X 4
#define POS_2_Y 6
#define MSG_FACTOR 1000

#define ERROR_MSG -50

#define MAX_DIST_PASS_POS 13


#define DBLOG_POL(LLL,XXX) LOG_POL(LLL,<<"OOTRAP_NO_BALL: " XXX)
#define DBLOG_DRAW(LLL,XXX) LOG_POL(LLL,<<_2D <<XXX)

OvercomeOffsideTrap::OvercomeOffsideTrap() {
  go2pos= new NeuroGo2Pos;
  neuroIntercept = new NeuroIntercept;
  neuroKick = new NeuroKick05;
  faceBall = new FaceBall;
  onetwoholdturn = new OneTwoHoldTurn;

  mode=0;
  
  
}



/*void OvercomeOffsideTrap::reset_intention(){
  DBLOG_POL(0, <<"KLAPPT!!!");
  DBLOG_DRAW(0, Circle2d(taropp->pos,33));
}*/


bool OvercomeOffsideTrap::init(char const * conf_file, int argc, char const* const* argv){
  return ( NeuroGo2Pos::init(conf_file,argc,argv)
          && NeuroIntercept::init(conf_file,argc,argv)
          && NeuroKick05::init(conf_file,argc,argv)
          && FaceBall::init(conf_file,argc,argv)
	  && OneTwoHoldTurn::init(conf_file,argc,argv));
}





bool OvercomeOffsideTrap::test_nb(Cmd & cmd){
   
    DBLOG_POL(0, << "MODE"<<mode<<" " <<WSinfo::me->number);
    if (mode==3){
      run_time=WSinfo::ws->time;
      got_already=false;
      mode=P1_RUN;
    }
    if(mode==26) mode=28;
    if (mode==28){
      if (WSinfo::ws->time==say_time){
        cmd.cmd_say.set_msg(player->number,(WSinfo::ball->pos.getX()+WSinfo::ball->vel.getX()+WSinfo::ball->vel.getX())*MSG_FACTOR,
	                    (WSinfo::ball->pos.getY()+WSinfo::ball->vel.getY()+WSinfo::ball->vel.getY())*MSG_FACTOR);
        DBLOG_POL(0, <<"ball_pos"<<(WSinfo::ball->pos.getX()+WSinfo::ball->vel.getX()+WSinfo::ball->vel.getX())*MSG_FACTOR<<","<<
	                    (WSinfo::ball->pos.getY()+WSinfo::ball->vel.getY()+WSinfo::ball->vel.getY())*MSG_FACTOR);
      }
      else if (WSinfo::ws->time==say_time+1){
        cmd.cmd_say.set_msg(player->number,WSinfo::ball->vel.getX()*MSG_FACTOR,WSinfo::ball->vel.getY()*MSG_FACTOR);
	DBLOG_POL(0, <<"ball_vel"<<WSinfo::ball->vel.getX()*MSG_FACTOR<<","<<WSinfo::ball->vel.getY()*MSG_FACTOR);
	mode=0;
      }
      return false;
    }
    if (mode>=5&&mode<26){
      return true;
    }
    else mode=0;
    if (WSinfo::me->number==6) player=WSinfo::get_teammate_by_number(9);
    else if (WSinfo::me->number==7) player=WSinfo::get_teammate_by_number(10);
    else if (WSinfo::me->number==8) player=WSinfo::get_teammate_by_number(11);
   /* else if (WSinfo::me->number==9) player=WSinfo::get_teammate_by_number(6);
    else if (WSinfo::me->number==10) player=WSinfo::get_teammate_by_number(7);
    else if (WSinfo::me->number==11) player=WSinfo::get_teammate_by_number(8);*/
    else return false;
    
    if (player==0){
      DBLOG_POL(0, <<"null-pointer!");
      return false;
      
    }
    
    if (!WSinfo::is_teammate_pos_valid(player)){
      DBLOG_POL(0, <<"teammate_pos_ not valid!");
      return false; 
    }
    
    /*if (player->pos.x<WSinfo::me->pos.x){
      DBLOG_POL(0, <<"teammate_pos_ behind me valid!");
      return false;
    }*/
    if (!WSinfo::is_ball_pos_valid()){
      DBLOG_POL(0, <<"ball_pos_not valid?"); 
      return false;
    }
    
    if (player->pos.distance(WSinfo::ball->pos)>2.5){
      DBLOG_DRAW(0,VL2D( player->pos,
                        WSinfo::ball->pos,"green"));
      DBLOG_POL(0, <<"player is not ballowner");
      return false;
    }
    
    if (player->pos.getX()>23){
      DBLOG_POL(0, <<"player not in area");
      return false;
    }
    
    cmd.cmd_att.set_attentionto(player->number);
    dist_ol = WSinfo::his_team_pos_of_offside_line()-player->pos.getX();
    DBLOG_DRAW(0,L2D( WSinfo::his_team_pos_of_offside_line() ,-30, 
                        WSinfo::his_team_pos_of_offside_line() ,30,"green"));
    if (dist_ol>P2_MIN_OL_DIST){
      DBLOG_POL(0, << "O-line!"<<dist_ol);
      return false;
    }
    PlayerSet oppset = WSinfo::valid_opponents;
    Vector p1=Vector(1,1);
    p1.normalize(P2_MIN_RADIUS);
    Vector p2=Vector(1,-1);
    p2.normalize(P2_MIN_RADIUS);
    oppset.keep_players_in_quadrangle(player->pos,player->pos,player->pos+p1,player->pos+p2);
    DBLOG_DRAW(0, Quadrangle2d(player->pos,player->pos,player->pos+p1,player->pos+p2));
    
    if (oppset.num==1){
      //if (oppset.num==0) taropp = player; //???
     
      taropp=oppset.closest_player_to_point(player->pos);
      DBLOG_DRAW(0, Circle2d(taropp->pos,1));  
      pos2 = player->pos + Vector(-POS_2_X,POS_2_Y);
    
      if (pos2.getY()>FIELD_BORDER_Y-2){
         DBLOG_POL(0, << "Aus-links!");
      }
      else{
        oppset = WSinfo::valid_opponents;
        oppset.keep_players_in_quadrangle(Vector(player->pos.getX(),taropp->pos.getY()+0.5),Vector(player->pos.getX(),taropp->pos.getY()+8),
                                          Vector(player->pos.getX()+8,taropp->pos.getY()+8),Vector(player->pos.getX()+8,taropp->pos.getY()+0.5));
        DBLOG_DRAW(0, Quadrangle2d(Vector(player->pos.getX(),taropp->pos.getY()+0.5),Vector(player->pos.getX(),taropp->pos.getY()+8),
                                   Vector(player->pos.getX()+8,taropp->pos.getY()+8),Vector(player->pos.getX()+8,taropp->pos.getY()+0.5)));
        oppset.remove(taropp);
        go_left=true;
        if (oppset.num!=0){
          DBLOG_POL(0, << "links nicht frei!");
        }
        else{
          double dist=WSinfo::me->pos.distance(pos2);
	  if (dist<MAX_DIST_PASS_POS){ 
	    mode = P2_GO2POS2;
	    return true;
	  }
	  else{
	    DBLOG_POL(0, << "links too far away: "<<dist);
	  
	  }
        }
      }
      pos2 = player->pos + Vector(-POS_2_X,-POS_2_Y);
      if (pos2.getY()<-FIELD_BORDER_Y+2){
        DBLOG_POL(0, << "rechts aus !");
        return false;
      }
      oppset = WSinfo::valid_opponents;
      oppset.keep_players_in_quadrangle(Vector(player->pos.getX(),taropp->pos.getY()-0.5),Vector(player->pos.getX(),taropp->pos.getY()-8),
                                          Vector(player->pos.getX()+8,taropp->pos.getY()-8),Vector(player->pos.getX()+8,taropp->pos.getY()-0.5));
      DBLOG_DRAW(0, Quadrangle2d(Vector(player->pos.getX(),taropp->pos.getY()-0.5),Vector(player->pos.getX(),taropp->pos.getY()-8),
                                 Vector(player->pos.getX()+8,taropp->pos.getY()-8),Vector(player->pos.getX()+8,taropp->pos.getY()-0.5)));
      oppset.remove(taropp);
      go_left=false;
      if (oppset.num!=0){
        DBLOG_POL(0, << "rechts nicht frei !");
        return false;
      }
//      p2_active=false; 
      double dist=WSinfo::me->pos.distance(pos2);
      if (dist<MAX_DIST_PASS_POS){ 
        mode = P2_GO2POS2;
	return true;
      }
      DBLOG_POL(0, << "rechts too far away: "<<dist);
      return false;
    }
  return false;  
}



bool OvercomeOffsideTrap::p2_go2pos2(Cmd & cmd){
   /*if ( WSinfo::ws->time == WSinfo::ws->msg.time && WSinfo::ws->msg.type==WSinfo::me->number){ 
   if (WSinfo::ws->msg.param1 == 0){
     p2_active = true;
     DBLOG_POL(0, << "AKTIVIERT"<<taropp->pos<<taropp->number);
   }
   if (WSinfo::ws->msg.param1 == -1){
     mode=0;
     DBLOG_POL(0, << "DE-AKTIVIERT"<<taropp->pos<<taropp->number);
     return false;
   }
  }
  Vector target, ballpos, ballvel;
  if(Policy_Tools::check_go4pass(target,ballpos,ballvel)){
     DBLOG_POL(0, << "lese pas_info");
    mode= P2_INTERCEPT;
    mdpInfo::set_my_intention(DECISION_TYPE_INTERCEPTBALL); // tell neck to turn2 ball
    neuroIntercept->set_virtual_state(ballpos,ballvel);
    start_intercept_time=WSinfo::ws->time;
    return p2_intercept(cmd);
  }*/
  
  if ( WSinfo::ws->time == WSinfo::ws->msg.time && WSinfo::ws->msg.type==WSinfo::me->number){ 
    if (WSinfo::ws->msg.param1 == ERROR_MSG){
      mode=0;
      DBLOG_POL(0, << "DE-AKTIVIERT"<<taropp->pos<<taropp->number);
      return false;
    }
    else {
  
      DBLOG_POL(0, << "lese pas_info");
      mode= P2_INTERCEPT;
      mdpInfo::set_my_intention(DECISION_TYPE_INTERCEPTBALL); // tell neck to turn2 ball
      Vector ballpos= Vector((float)WSinfo::ws->msg.param1/MSG_FACTOR,(float)WSinfo::ws->msg.param2/MSG_FACTOR);
      Vector ballvel= WSinfo::me->pos-ballpos;
      ballvel.normalize(2.);
      neuroIntercept->set_virtual_state(ballpos,ballvel);
      start_intercept_time=WSinfo::ws->time;
      return p2_intercept(cmd);
    }
  }
  if (WSinfo::is_teammate_pos_valid(player)){
  
    if(WSinfo::is_ball_pos_valid() && player->pos.distance(WSinfo::ball->pos)>2.5){
      
      DBLOG_POL(0, << "ER HAT DEN BALL VERLOREN");
      mode =0;
      cmd.cmd_say.set_msg(player->number,ERROR_MSG,ERROR_MSG);
      return false;
    }
    //if (pos2.distance(player->pos + Vector(-POS_2_X,POS_2_Y))>1.5) pos2 = player->pos + Vector(-POS_2_X,POS_2_Y);
    if (go_left) { if (pos2.getY()>FIELD_BORDER_Y-1) { pos2.setY(FIELD_BORDER_Y-1); }
    else if (pos2.getY()<-FIELD_BORDER_Y+1) { pos2.setY(-FIELD_BORDER_Y+1); } }
  }
  
  double myposx = (WSinfo::me->pos.getX()+WSinfo::me->vel.getX())*MSG_FACTOR;
  double myposy = (WSinfo::me->pos.getY()+WSinfo::me->vel.getY())*MSG_FACTOR;
  DBLOG_DRAW(0, Circle2d(Vector(myposx,myposy),1));
  cmd.cmd_say.set_msg(player->number,myposx,myposy);
  cmd.cmd_neck.set_turn(mdpInfo::moment_to_turn_neck_to_abs((WSinfo::ball->pos-WSinfo::me->pos).arg()));
  if (WSinfo::me->pos.distance(pos2)<1){
    DBLOG_POL(0, << "reached POS2!");
    mode=P2_WAIT; 
  
    return p2_wait(cmd);
  }
  go2pos->set_target(pos2);
  return go2pos->get_cmd(cmd);
}



bool OvercomeOffsideTrap::p2_wait(Cmd & cmd){
  /*if ( WSinfo::ws->time == WSinfo::ws->msg.time && WSinfo::ws->msg.type==WSinfo::me->number){ 
  if (WSinfo::ws->msg.param1 == 0){
     p2_active = true;
     DBLOG_POL(0, << "AKTIVIERT"<<taropp->pos<<taropp->number);
   }
   if (WSinfo::ws->msg.param1 == -1){
     mode=0;
     DBLOG_POL(0, << "DE-AKTIVIERT"<<taropp->pos<<taropp->number);
     return false;
   }
  }
   Vector target, ballpos, ballvel;
   if(Policy_Tools::check_go4pass(target,ballpos,ballvel)){
   DBLOG_POL(0, << "lese pas_info");
    mode= P2_INTERCEPT;
    mdpInfo::set_my_intention(DECISION_TYPE_INTERCEPTBALL); // tell neck to turn2 ball
    neuroIntercept->set_virtual_state(ballpos,ballvel);
    start_intercept_time=WSinfo::ws->time;
    return p2_intercept(cmd);
  }  */
  if ( WSinfo::ws->time == WSinfo::ws->msg.time && WSinfo::ws->msg.type==WSinfo::me->number){ 
    if (WSinfo::ws->msg.param1 == ERROR_MSG){
      mode=0;
      DBLOG_POL(0, << "DE-AKTIVIERT"<<taropp->pos<<taropp->number);
      return false;
    }
    else {
      DBLOG_POL(0, << "lese pas_info");
      mode= P2_INTERCEPT;
      mdpInfo::set_my_intention(DECISION_TYPE_INTERCEPTBALL); // tell neck to turn2 ball
      Vector ballpos= Vector((float)WSinfo::ws->msg.param1/MSG_FACTOR,(float)WSinfo::ws->msg.param2/MSG_FACTOR);
      Vector ballvel= WSinfo::me->pos-ballpos;
      ballvel.normalize(2.);
      neuroIntercept->set_virtual_state(ballpos,ballvel);
      start_intercept_time=WSinfo::ws->time;
      return p2_intercept(cmd);
    }
  }
  double myposx = (WSinfo::me->pos.getX()+WSinfo::me->vel.getX())*MSG_FACTOR;
  double myposy = (WSinfo::me->pos.getY()+WSinfo::me->vel.getY())*MSG_FACTOR;
  if (WSinfo::is_ball_pos_valid()){
    cmd.cmd_neck.set_turn(mdpInfo::moment_to_turn_neck_to_abs((WSinfo::ball->pos-WSinfo::me->pos).arg()));
    if (WSinfo::is_teammate_pos_valid(player) &&  player->pos.distance(WSinfo::ball->pos)<2.5){
      
      if (go_left){
        pos2 = player->pos + Vector(-POS_2_X,POS_2_Y);
        if (pos2.getY()>FIELD_BORDER_Y-1) pos2.setY(FIELD_BORDER_Y-1);
      }
      else{
        pos2 = player->pos + Vector(-POS_2_X,-POS_2_Y);
        if (pos2.getY()<-FIELD_BORDER_Y+1) pos2.setY(-FIELD_BORDER_Y+1);
      }
      if (WSinfo::me->pos.distance(pos2)>1.5){
        mode=P2_GO2POS2;
	DBLOG_POL(0, << "Korrigiere POS!");
	return  p2_go2pos2(cmd);
      } 
      cmd.cmd_say.set_msg(player->number,myposx,myposy);
      DBLOG_POL(0, << "WARTE!");
      //faceBall->turn_to_point(player->pos);
      return true;//faceBall->get_cmd(cmd);
    }
    DBLOG_POL(0, << "ER HAT DEN BALL VERLOREN");
    mode =0;
    cmd.cmd_say.set_msg(player->number,ERROR_MSG,ERROR_MSG);
    return false;
  }
  else{
    DBLOG_POL(0, << "SUCHE BALL");
    cmd.cmd_say.set_msg(player->number,myposx,myposy);
    return faceBall->get_cmd(cmd);
  }
}
   

bool OvercomeOffsideTrap::p2_intercept(Cmd & cmd){
  if( WSmemory::team_last_at_ball()==1 || WSinfo::ws->time-start_intercept_time>8 
     || WSinfo::me->pos.distance(WSinfo::ball->pos)>15){
    DBLOG_POL(0, << "intercept ball: time out oder nicht aktiviert");
    cmd.cmd_say.set_msg(player->number,ERROR_MSG,ERROR_MSG);
    mode=0;
    return false;
  }
    
  return neuroIntercept->get_cmd(cmd);
}


  
bool OvercomeOffsideTrap::p2_pass(Cmd & cmd){
  Quadrangle2d check_area = Quadrangle2d(WSinfo::me->pos, pos3, 4 * ServerOptions::kickable_area);
  PlayerSet oppset = WSinfo::valid_opponents;
  oppset.keep_players_in(check_area);
  DBLOG_DRAW(0,check_area);
  if (oppset.num!=0){
    mode=0;
    cmd.cmd_say.set_msg(player->number,ERROR_MSG,ERROR_MSG);
    DBLOG_POL(0, << "passway is not empty");
    return false;
  } 

  say_time=WSinfo::ws->time+5;
  neuroKick->kick_to_pos_with_initial_vel(2.0,pos3);
  
  if (neuroKick->get_cmd(cmd)) return true;
  DBLOG_POL(0, << "kick not possible: intercept ball!");
  return neuroIntercept->get_cmd(cmd);
}





bool OvercomeOffsideTrap::test_wb(Cmd & cmd){
    DBLOG_POL(0, << "MODE"<<mode);
    if (mode==P2_INTERCEPT) {
      if (go_left) pos3=WSinfo::me->pos+Vector(30,-3);
      else pos3=WSinfo::me->pos+Vector(30,3);
      mode=P2_PASS;
      Intention lp;
      DBLOG_DRAW(0, Circle2d(pos3,0.5));
      lp.set_pass(pos3,2.5,WSinfo::ws->time,player->number);
      Blackboard::pass_intention=lp; 
      start_intercept_time=WSinfo::ws->time;
      return true;
    }
    if (mode==P2_PASS) return true;
    if (mode>0 && mode<5){
      return true;  
    }
    else mode=0; 
   /* if (WSinfo::me->number==6) player=WSinfo::get_teammate_by_number(9);
    else if (WSinfo::me->number==7) player=WSinfo::get_teammate_by_number(10);
    else if (WSinfo::me->number==8) player=WSinfo::get_teammate_by_number(11);*/
    if (WSinfo::me->number==9) player=WSinfo::get_teammate_by_number(6);
    else if (WSinfo::me->number==10) player=WSinfo::get_teammate_by_number(7);
    else if (WSinfo::me->number==11) player=WSinfo::get_teammate_by_number(8);
    else return false;
    if (player==0) return false;
    cmd.cmd_att.set_attentionto(player->number);
    if ( WSinfo::ws->time == WSinfo::ws->msg.time && WSinfo::ws->msg.type==WSinfo::me->number ){       
      playerpos=Vector(WSinfo::ws->msg.param1,WSinfo::ws->msg.param2);
    }
    else return false;
    dist_ol = WSinfo::his_team_pos_of_offside_line()-WSinfo::me->pos.getX();
    DBLOG_POL(0, << "test"<<dist_ol);
    DBLOG_DRAW(0,L2D( WSinfo::his_team_pos_of_offside_line() ,-30, 
                        WSinfo::his_team_pos_of_offside_line() ,30,"green"));
    //if (dist_ol>P1_MIN_OL_DIST+5) return false;
    dist_ol = WSinfo::his_team_pos_of_offside_line()-WSinfo::me->pos.getX();
    PlayerSet oppset = WSinfo::valid_opponents;
    PlayerSet oppset2 = WSinfo::valid_opponents;
    Vector p1=Vector(1,1);
    p1.normalize(P1_MIN_RADIUS);
    Vector p2=Vector(1,-1);
    p2.normalize(P1_MIN_RADIUS);
    oppset.keep_players_in_quadrangle(WSinfo::me->pos,WSinfo::me->pos,WSinfo::me->pos+p1,WSinfo::me->pos+p2);
    DBLOG_DRAW(0, Quadrangle2d(WSinfo::me->pos,WSinfo::me->pos,WSinfo::me->pos+p1,WSinfo::me->pos+p2));
    p1.normalize(P1_MIN_RADIUS+dist_ol);
    p2.normalize(P1_MIN_RADIUS+dist_ol);
    oppset2.keep_players_in_quadrangle(WSinfo::me->pos,WSinfo::me->pos,WSinfo::me->pos+p1,WSinfo::me->pos+p2);
    DBLOG_DRAW(0, Quadrangle2d(WSinfo::me->pos,WSinfo::me->pos,WSinfo::me->pos+p1,WSinfo::me->pos+p2));
    //PPlayer tmpopp;
    if (oppset.num==1 && oppset2.num==1){
      taropp=oppset.closest_player_to_point(WSinfo::me->pos);
      DBLOG_DRAW(0, Circle2d(taropp->pos,1));
     
        DBLOG_POL(0, << "AKTIVIERT"<<taropp->pos<<taropp->number);
	//cmd.cmd_say.set_msg(player->number,0,0);
	//player1=tmpopp; //misteri�er Absturz!!!
	//target=tmpopp->pos;
	mode = 1;
	
	return true;
      
    }
    return false;
    
}



bool OvercomeOffsideTrap::p1_hold(Cmd & cmd){
 
  dist_ol = WSinfo::his_team_pos_of_offside_line()-WSinfo::me->pos.getX();
            DBLOG_DRAW(0,L2D( WSinfo::his_team_pos_of_offside_line() ,-30, 
                        WSinfo::his_team_pos_of_offside_line() ,30,"green"));
           
	    if ( WSinfo::ws->time == WSinfo::ws->msg.time && WSinfo::ws->msg.type==WSinfo::me->number ){
	      playerpos=Vector((float)WSinfo::ws->msg.param1/MSG_FACTOR,(float)WSinfo::ws->msg.param2/MSG_FACTOR);
	    }
	    else{ 
	      mode=0;
	      cmd.cmd_say.set_msg(player->number,ERROR_MSG,ERROR_MSG);
	      DBLOG_POL(0, << "anderer Spieler nicht gesendet");
	      return false;
	    }
	    /*if (WSinfo::is_teammate_pos_valid(player)){
	      playerpos = player->pos;
	    }*/
	     pos2= WSinfo::me->pos+Vector(-POS_2_X,POS_2_Y);
  if (pos2.getY()>FIELD_BORDER_Y-1){
    DBLOG_POL(0, << "links aus");
  }
  else{
  DBLOG_DRAW(0,Circle2d(pos2,0.5));
	    DBLOG_DRAW(0,VL2D( playerpos,
                        pos2,"green"));
            DBLOG_POL(0,<<"dist: player-POS2LINKS"<<playerpos.distance(pos2));			
	    if (playerpos.distance(pos2)<3){
	       Quadrangle2d check_area = Quadrangle2d(WSinfo::me->pos, pos2, 4 * ServerOptions::kickable_area);
	      
	      PlayerSet oppset = WSinfo::valid_opponents;
	      oppset.keep_players_in(check_area);
	      DBLOG_DRAW(0,check_area);
	      if (oppset.num!=0){
	        mode=0;
                cmd.cmd_say.set_msg(player->number,ERROR_MSG,ERROR_MSG);
                DBLOG_POL(0, << "passway is not empty");
                return false;
	      } 
	      speed=Planning::compute_pass_speed(WSinfo::ball->pos,playerpos,playerpos,2.5);
	      Intention lp;
	      DBLOG_DRAW(0, Circle2d(pos2,0.5));
                /*lp.set_pass(pos2,speed,WSinfo::ws->time,player->number);
              Blackboard::pass_intention=lp; */
	      cmd.cmd_say.set_msg(player->number,(WSinfo::ball->pos.getX()+WSinfo::ball->vel.getX())*MSG_FACTOR,
	                    (WSinfo::ball->pos.getY()+WSinfo::ball->vel.getY())*MSG_FACTOR);
	      mode=P1_PASS;
	      go_left=true;
	      DBLOG_POL(0, << "schreibe pass_info");
	      return p1_pass(cmd);
	    }
	    }
    pos2= WSinfo::me->pos+Vector(-POS_2_X,-POS_2_Y);
  if (pos2.getY()<-FIELD_BORDER_Y+1){
    mode=0;
    cmd.cmd_say.set_msg(player->number,ERROR_MSG,ERROR_MSG);
    DBLOG_POL(0, << "rechts aus");
    return false;
  }
  DBLOG_DRAW(0,Circle2d(pos2,0.5));
	    DBLOG_DRAW(0,VL2D( playerpos,
                        pos2,"green"));
            DBLOG_POL(0,<<"dist: player-POS2RECHTS"<<playerpos.distance(pos2));			
	    if (playerpos.distance(pos2)<3){
	      
	      Quadrangle2d check_area = Quadrangle2d(WSinfo::me->pos, pos2, 4 * ServerOptions::kickable_area);
	      
	      PlayerSet oppset = WSinfo::valid_opponents;
	      oppset.keep_players_in(check_area);
	      DBLOG_DRAW(0,check_area);
	      if (oppset.num!=0){
	        mode=0;
                cmd.cmd_say.set_msg(player->number,ERROR_MSG,ERROR_MSG);
                DBLOG_POL(0, << "passway is not empty");
                return false;
	      } 
	      speed=Planning::compute_pass_speed(WSinfo::ball->pos,playerpos,playerpos,2.5);
	      Intention lp;
	      DBLOG_DRAW(0, Circle2d(pos2,0.5));
              /*lp.set_pass(pos2,speed,WSinfo::ws->time,player->number);
              Blackboard::pass_intention=lp; */
	      cmd.cmd_say.set_msg(player->number,(WSinfo::ball->pos.getX()+WSinfo::ball->vel.getX())*MSG_FACTOR,
	                    (WSinfo::ball->pos.getY()+WSinfo::ball->vel.getY())*MSG_FACTOR);
	      mode=P1_PASS;
	      go_left=false;
	      DBLOG_POL(0, << "schreibe pass_info");
	      return p1_pass(cmd);
	    }
	    
	    
	    
	    
	    
	    return onetwoholdturn->get_cmd(cmd, (playerpos-WSinfo::me->pos).ARG());
}	   


	    
bool OvercomeOffsideTrap::p1_pass(Cmd & cmd){
  //ball_pos=Vector(ERROR_MSG,ERROR_MSG);
  
  if (go_left) pos3 = playerpos+Vector(30,-2);
  else pos3 = playerpos+Vector(30,2);
  neuroKick->kick_to_pos_with_initial_vel(speed,pos2);
  return neuroKick->get_cmd(cmd);
}    



bool OvercomeOffsideTrap::p1_run(Cmd & cmd){
  LOG_POL(0,<<"msg.time "<<WSinfo::ws->msg.type<<" ws->msg.param1 " << 
           WSinfo::ws->msg.param1);
  if (WSinfo::ws->time-run_time>15){
    mode=0;
    DBLOG_POL(0, << "time out");
    return false;
  }
  if (WSinfo::ws->time == WSinfo::ws->msg.time && WSinfo::ws->msg.type==WSinfo::me->number){
    if (WSinfo::ws->msg.param1 == ERROR_MSG){
      mode=0;
      DBLOG_POL(0, << "anderer Spieler sendet abbruch");
      return false;
    }
    else {
      if (WSinfo::ws->time-run_time>3){ 
        DBLOG_POL(0, <<"ball_pos"<<ball_pos);
        if (!got_already){
          ball_pos= Vector((float)WSinfo::ws->msg.param1/MSG_FACTOR,(float)WSinfo::ws->msg.param2/MSG_FACTOR);
          DBLOG_POL(0, <<"bekommen: ball_pos"<<ball_pos.getX()<<" " <<ball_pos.getY());
          DBLOG_DRAW(0, Circle2d(ball_pos,3));
          got_already=true;
        }
        else{  
          mode= P1_INTERCEPT;
          mdpInfo::set_my_intention(DECISION_TYPE_INTERCEPTBALL); // tell neck to turn2 ball
          Vector ball_vel= Vector((float)WSinfo::ws->msg.param1/MSG_FACTOR,(float)WSinfo::ws->msg.param2/MSG_FACTOR);
	  DBLOG_POL(0, <<"ball_pos"<<ball_pos);
	  DBLOG_POL(0, <<"ball_vel"<<Vector(WSinfo::ws->msg.param1,WSinfo::ws->msg.param2));
          neuroIntercept->set_virtual_state(ball_pos,ball_vel);
          start_intercept_time=WSinfo::ws->time;
        
          start_intercept_time=WSinfo::ws->time;
          return p1_intercept(cmd);
        } 
      }
    }
  }
  
  DBLOG_DRAW(0, Circle2d(pos3,0.5));
 /* if(Policy_Tools::check_go4pass(target,ballpos,ballvel)){
    neuroIntercept->set_virtual_state(ballpos,ballvel);
    DBLOG_DRAW(0, Circle2d(ballpos,0.5));
    mode=P1_INTERCEPT;
    return p1_intercept(cmd);
  }*/
 /* if (WSinfo::is_ball_pos_valid()&&WSinfo::ball->pos.x+0.5>WSinfo::me->pos.x){
    mode=P1_INTERCEPT;
    return p1_intercept(cmd);
  }*/
  //if (WSinfo::me->pos.distance(pos3)<10 && WSinfo::is_ball_pos_valid()) return inter;
  dist_ol = WSinfo::his_team_pos_of_offside_line()-WSinfo::me->pos.getX();
  if (dist_ol<1.5) return false; //Notbremse!!
  
  go2pos->set_target(pos3);
  return go2pos->get_cmd(cmd);
}



bool OvercomeOffsideTrap::p1_intercept(Cmd & cmd){
  if(WSmemory::team_last_at_ball()==1 || WSinfo::ws->time-start_intercept_time>15){
    DBLOG_POL(0, << "intercept ball: time out");
    cmd.cmd_say.set_msg(player->number,-1,-1);
    mode=0;
    return false;
  }
  return neuroIntercept->get_cmd(cmd);
}



	    
bool OvercomeOffsideTrap::get_cmd(Cmd & cmd){
  //oldtime=WSinfo::ws->time;
  
  switch(mode){
    case P1_HOLD:      cmd.cmd_att.set_attentionto(player->number);
                       return p1_hold(cmd);
    case P1_PASS:      cmd.cmd_att.set_attentionto(player->number);
                       return p1_pass(cmd);  
    case P1_RUN:       cmd.cmd_att.set_attentionto(player->number);
                       return p1_run(cmd);
    case P1_INTERCEPT: return p1_intercept(cmd);
    case P2_GO2POS2:   cmd.cmd_att.set_attentionto(player->number);
                       return p2_go2pos2(cmd);
    case P2_WAIT:      cmd.cmd_att.set_attentionto(player->number);
                       return p2_wait(cmd);
    case P2_INTERCEPT: cmd.cmd_att.set_attentionto(player->number);
                       return p2_intercept(cmd);
    case P2_PASS:      cmd.cmd_att.set_attentionto(player->number);
                       return p2_pass(cmd); 
    default: return false;
  }
}








