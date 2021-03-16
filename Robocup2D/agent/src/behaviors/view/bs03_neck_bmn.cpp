#include "bs03_neck_bmn.h"
#include "mdp_info.h"
#include <stdlib.h>
#include <stdio.h>

#include "../../basics/Cmd.h"
#include "tools.h"
#include "valueparser.h"
#include "options.h"
#include "log_macros.h"
#include "../policy/positioning.h"
#include "angle.h"
#include "ws_info.h"
#include "../policy/policy_tools.h"
#include "geometry2d.h"

//#define LOG_AGE
#define BASELEVEL 3

#if 0
#define DBLOG_POL(LLL,XXX) LOG_POL(LLL,<<"BS02Neck: "<<XXX)
#define DBLOG_DRAW(LLL,XXX) LOG_POL(LLL,<<_2D<<XXX)
#else
#define DBLOG_POL(LLL,XXX) 
#define DBLOG_DRAW(LLL,XXX)
#endif 

#if 0
#define MYLOG_POL(LLL,XXX) LOG_POL(LLL,<<"NECK: "<<XXX)
#define MYLOG_DRAW(LLL,XXX) LOG_POL(LLL,<<_2D<<XXX)
#else
#define MYLOG_POL(LLL,XXX) 
#define MYLOG_DRAW(LLL,XXX)
#endif 



/** Parameters */

#define NUM_ANGLES 17                    /* Number of possible view angles (Caution: time!) */
#define MAX_VISIBLE_DISTANCE 50.0        /* Ignore objects/points further away */
#define MAX_BALL_AGE 3                   /* Look to ball if older! */
#define MAX_FASTEST_OPP_AGE 2            /* ...if DECISION_TYPE_BLOCKBALLHOLDER is set! */
#define MAX_GOALIE_AGE 2                 /* only if we are carrying the ball... */

#define SQR_MAX_VISIBLE_DISTANCE SQUARE(MAX_VISIBLE_DISTANCE)

#define OFFSIDE_TOLERANCE 6
#define IGNORE_OFFSIDE_GOAL_DISTANCE 18 
#define MAX_OFFSIDE_LINE_AGE 3

#define SECT_X BS02_NECK_SECT_X
#define SECT_Y BS02_NECK_SECT_Y

//#define VALUE_BASED

bool BS03Neck::initialized = false;

BS03Neck::Neck_Info::Neck_Info()
{
    for( int x = 0; x < BS02_NECK_SECT_X; x++ )
        for( int y = 0; y < BS02_NECK_SECT_Y; y++ )
            sector[ x ][ y ] = 0;

    for( int i = 0; i < NUM_PLAYERS; i++ )
    {
        opp_player[ i ] = -1;
        own_player[ i ] = -1;
    }

    sector_avg = penalty_avg = -1;
    ball = opp_goalie = -1;
    opp_goal = opp_goal_left = opp_goal_right = own_goal = opp_offside = own_offside = 0;
    opponents = teammates = 0;

}

BS03Neck::BS03Neck()
{
  xunit=ServerOptions::pitch_length/SECT_X;
  yunit=ServerOptions::pitch_width/SECT_Y;
  norm_view_ang=ServerOptions::visible_angle * PI/180.0;
  ball_already_searched=false;
  need_lookto_ball=false;
  last_looked_to_goalie=-2;
  
  ValueParser vp(CommandLineOptions::policy_conf,"BS03Neck");
  vp.get("intercept_look_mode",intercept_look_mode,1);
  vp.get("opp_has_ball_look_mode",opp_has_ball_look_mode,1);
  vp.get("own_has_ball_look_mode",own_has_ball_look_mode,1);
  vp.get("ball_holder_look_mode",ball_holder_look_mode,1);
  // ridi 04: Warum?  vp.get("ignore_neck_intentions",ignore_neck_intentions,true);
  vp.get("ignore_neck_intentions",ignore_neck_intentions,false);
  vp.get("use_1v1_mode",use_1v1_mode,false);

  neck_cmd = new NeckCmd();
  goalie_neck = new GoalieNeck();

  center_target = false;
  single_sector_divisor = 0;
}

BS03Neck::~BS03Neck() {
  delete neck_cmd;
  delete goalie_neck;
}


bool BS03Neck::get_cmd(Cmd &cmd) {
  //long starttime=Tools::get_current_ms_time();


  if(!WSinfo::me) {
    cerr << "\nERROR [BS02_Neck]: WSinfo::me not set!\n";
    return true;
  }

  if (   WSinfo::ws->play_mode == PM_my_BeforeKickOff
      || WSinfo::ws->play_mode == PM_his_BeforeKickOff
      || WSinfo::ws->play_mode == PM_my_KickOff
      || WSinfo::ws->play_mode == PM_his_KickOff)
  {
    neck_cmd->set_turn_neck_abs( Tools::range_random( -PI/2.0, PI/2.0));
    return neck_cmd->get_cmd(cmd);
  }
  
  if (ClientOptions::consider_goalie) {
    if(WSinfo::ws->play_mode == PM_PlayOn
       || WSinfo::ws->play_mode == PM_my_BeforePenaltyKick
       || WSinfo::ws->play_mode == PM_his_BeforePenaltyKick
       || WSinfo::ws->play_mode == PM_my_PenaltyKick
       || WSinfo::ws->play_mode == PM_his_PenaltyKick) {  // Sput03: goalie should turn neck during standards...
      return goalie_neck->get_cmd(cmd);
      //::get_Dynamic_Neck_Goalie_Policy();
    }
  }

  if(WSinfo::ws->ms_time_of_see<WSinfo::ws->ms_time_of_sb) {
    //LOG_ERR(1,<<"WARNING: BS02_Neck: Did not get see update!");
    MYLOG_POL(0,"TURN NECK- WARNING: Did not get see update!");
    //cerr << "\n### Player "<<WSinfo::me->number<<", Cycle "<<WSinfo::ws->time<<": NO SEE UPDATE!";
    got_update=false;
  } else got_update=true;

  init_cycle(cmd);

  MYLOG_POL(0,"NECK: regular looking!");
  
  /* This is necessary for Face_Ball move! Do not remove! */
  if(neck_intention.type==NECK_INTENTION_SCANFORBALL) {
    MYLOG_POL(0,"NECK: schanforball!");
    //cerr << "\nP"<<WSinfo::me->number<<"cyc #"<<WSinfo::ws->time<<": Neck Lock!";
    return neck_lock(cmd);
  } 

  Angle target=-1;
  center_target=false;
  
  /* Daisy Chain */
  
  if(target!=-1){}
    else if(-1!=(target=check_neck_1v1())){}  // for penalty situations; use_1v1_mode must be 1
    else if(-1!=(target=check_direct_opponent_defense())){}
    else if(-1!=(target=check_search_ball())){}
    else if(-1!=(target=check_neck_intention())){}
    else if(-1!=(target=check_players_near_ball())){}
  //else if(-1!=(target=check_block_ball_holder())){}   // obsolete - covered by neck intention!
    else if(-1!=(target=check_intercept())){}
    else if(-1!=(target=check_offside())){}
#ifndef VALUE_BASED
    else if(-1!=(target=check_goalie())){}
#endif
    else if((-1!=(target=check_relevant_teammates()))){}
  /* Wertfunktion / Defaultmove */
  if(target==-1 || !center_target) {

    
#ifdef VALUE_BASED
    Angle vb_target=neck_value_based(target);
#else
    Angle vb_target=this->neck_default(target);  /* new (pre-Padova) default move */
    //Angle vb_target=-1;
#endif
    
    if(target!=-1) {
      if(vb_target!=-1 &&
	 fabs(target-vb_target)<Tools::next_view_angle_width().get_value()*.5-DEG2RAD(4)) {
	MYLOG_POL(0,"Modifying selected target by "<<fabs(target-vb_target)<<".");
	target=vb_target;
      }
    } else target=vb_target;
  }

  /* Move ausfuehren */
  
  //long endtime=Tools::get_current_ms_time();
  //DBLOG_POL(0,"Needed "<<endtime-starttime<<"ms.");
  
  if(target>-1) {  
#if 1
    Vector pol;
    pol.init_polar(40,target);pol+=WSinfo::me->pos;
    DBLOG_DRAW(1,"LINE("<<WSinfo::me->pos.x<<","<<WSinfo::me->pos.y<<","<<pol.x<<","<<pol.y<<");");
#endif
    DBLOG_POL(0, "Turning to " <<target<<" ("<<RAD2DEG(target)<<")...");
    neck_cmd->set_turn_neck_abs(target);
    return neck_cmd->get_cmd(cmd);
  }
  DBLOG_POL(0,"WARNING: No neck turn executed!");
  ERROR_OUT << "WARNING: No neck turn executed!";
  return true;
}

bool BS03Neck::init(char const * conf_file, int argc, char const* const* argv) {
  if(initialized) return initialized;
  initialized = ( NeckCmd::init(conf_file, argc, argv) && GoalieNeck::init(conf_file, argc, argv) );
  if(!initialized) { std::cout << "\nBS03Neck behavior NOT initialized!!!"; }
  return initialized;
}

/** prepare all for current cycle, init variables and so on */
void BS03Neck::init_cycle(Cmd &cmd) {

  mdpInfo::get_my_neck_intention(neck_intention);

  minang=Tools::my_minimum_abs_angle_to_see().get_value();
  maxang=Tools::my_maximum_abs_angle_to_see().get_value();
  //bool center_target=false;  // look exactly to target?
  
  /* Model next cycle */
  Tools::model_cmd_main(WSinfo::me->pos,WSinfo::me->vel,WSinfo::me->ang.get_value(),WSinfo::ball->pos,
			WSinfo::ball->vel,cmd.cmd_body,
			my_new_pos,my_new_vel,my_new_angle,new_ball_pos,new_ball_vel);
  
  own_players_near_ball=WSinfo::valid_teammates_without_me;
  opp_players_near_ball=WSinfo::valid_opponents;
  own_players_near_ball.keep_players_in_circle(WSinfo::ball->pos,1.8);
  opp_players_near_ball.keep_players_in_circle(WSinfo::ball->pos,1.8);
  players_near_ball=own_players_near_ball;
  players_near_ball+=opp_players_near_ball;

  distance_to_ball=(my_new_pos-new_ball_pos).norm();
  ballinfeelrange=distance_to_ball<0.9*ServerOptions::visible_distance; //changed from 0.95 to 0.9 /*TG_OSAKA*/
  potentialcollision = distance_to_ball < 2.5 * WSinfo::me->radius;
  dir_to_ball=(new_ball_pos-my_new_pos).ARG();

  /* Update age information */


#ifdef VALUE_BASED
  get_single_sector_weights();
#endif
  
  Vector dum;
  dum.init_polar(1.0,WSinfo::me->neck_ang);
  Cone2d viewcone(WSinfo::me->pos,dum,ANGLE(Tools::cur_view_angle_width().get_value()-DEG2RAD(7)));
    
  Vector pos;
  if(WSinfo::ws->time>0) {
    cur_info.sector_avg=0;
    pos.setY( -(SECT_Y-1)/2*yunit );
    for(int y=0; y<SECT_Y;y++,pos.addToY( yunit )) {
      pos.setX( -(SECT_X-1)/2*xunit );
      for(int x=0;x<SECT_X;x++,pos.addToX( xunit )) {
	if(got_update && (WSinfo::me->pos-pos).sqr_norm()<MAX_VISIBLE_DISTANCE*MAX_VISIBLE_DISTANCE && 
	   viewcone.inside(pos))
	  cur_info.sector[x][y]=0;
	else 
	  cur_info.sector[x][y]++;
#ifdef VALUE_BASED	
	cur_info.sector_avg+=cur_info.sector[x][y]*single_sector_weight[x][y];
#endif
#ifdef LOG_AGE
	if(x%2) {
	  DBLOG_DRAW(0,"STRING("<<pos.x<<","<<pos.y<<",\""<<cur_info.sector[x][y]<<"\");");
	}
#endif
      }
    }
#ifdef VALUE_BASED
    cur_info.sector_avg/=single_sector_divisor;
    DBLOG_POL(1,"cur_info.sector_avg="<<cur_info.sector_avg);
#endif
  }

  cur_info.ball=WSinfo::ball->age;
  if(WSinfo::ball->age<MAX_BALL_AGE) ball_already_searched=false;

#ifdef VALUE_BASED  
  cur_info.opponents=0;
  for(int p=0;p<WSinfo::valid_opponents.num;p++) {
    cur_info.opponents+=WSinfo::valid_opponents[p]->age;
  }
  if(WSinfo::valid_opponents.num>0) cur_info.opponents/=(double)WSinfo::valid_opponents.num;
#endif  
#ifdef LOG_AGE
  DBLOG_DRAW(1,"STRING col=555555 ("<<WSinfo::ball->pos.x<<","<<WSinfo::ball->pos.y
	  <<",\"[-"<<cur_info.ball<<"-]\");");
#endif

  //pos.x=mdpInfo::opponent_goalpos().x;
  //pos.y=mdpInfo::opponent_goalpos().y;
  //DBLOG_POL(1,"col=000000 CIRCLE("<<pos.x<<","<<pos.y<<",3);");

  if(WSinfo::his_goalie) {
    cur_info.opp_goalie=WSinfo::his_goalie->age;
  } else {
    cur_info.opp_goalie=-1;
  }
  
  if(got_update && viewcone.inside(HIS_GOAL_RIGHT_CORNER)) {
    cur_info.opp_goal_right=0;
  } else {
    cur_info.opp_goal_right++;
  }
  if(got_update && viewcone.inside(HIS_GOAL_LEFT_CORNER)) {
    cur_info.opp_goal_left=0;
  } else {
    cur_info.opp_goal_left++;
  }
  if(got_update && viewcone.inside(HIS_GOAL_CENTER)) {
    cur_info.opp_goal=0;
  } else {
    cur_info.opp_goal++;
  }
  
#ifdef LOG_AGE
  //DBLOG_POL(1,"STRING col=ffffff ("<<mdpInfo::opponent_goalpos().x<<","
  //	  <<mdpInfo::opponent_goalpos().y<<",\"[-"<<cur_info.opp_goal_left<<"-]\");");	  
#endif

}

bool BS03Neck::can_see_object(Vector mypos,ANGLE neckdir,Vector objpos) {
  if((objpos-mypos).sqr_norm()>SQR_MAX_VISIBLE_DISTANCE) return false;
  Angle target=Tools::get_angle_between_null_2PI((objpos-mypos).angle());
  Angle minang=(neckdir-ANGLE(.5*mdpInfo::view_angle_width_rad())).get_value();
  Angle maxang=(neckdir+ANGLE(.5*mdpInfo::view_angle_width_rad())).get_value();
  if(minang>maxang && (target<minang && target>maxang)) return false;
  if(minang<maxang && (target<minang || target>maxang)) return false;
  return true;
}


#ifdef VALUE_BASED
/*****************************************************************************************
 * Value based neck policy
 *****************************************************************************************/

/** This function creates a new neck_info based on given parameters.
 *  Attention: This does not check if ball is in feel range!
 *  Invalid objects/players will be ignored and need to be already set to -1
 *  in the given neckinfo
 */
Neck_Info BS03Neck::get_neck_info(Vector mypos,ANGLE neckdir,Vector ballpos,
					  Neck_Info neckinfo) {

  Vector dum;
  dum.init_polar(1.0,neckdir);
  Cone2d viewcone(mypos,dum,Tools::next_view_angle_width());
  
  Vector pos;
  neckinfo.sector_avg=0;
  pos.y=-(SECT_Y-1)/2*yunit;
  //int sectors_counted=0;
  for(int y=0; y<SECT_Y;y++, pos.y+= yunit) {
    pos.x=-(SECT_X-1)/2*xunit;
    for(int x=0;x<SECT_X;x++,pos.x+= xunit) {
      if((mypos-pos).sqr_norm()<SQR_MAX_VISIBLE_DISTANCE && viewcone.inside(pos))
	neckinfo.sector[x][y]=0;
      else 
	neckinfo.sector[x][y]++;
      
      neckinfo.sector_avg+=neckinfo.sector[x][y]*single_sector_weight[x][y];
      //sectors_counted++;
    }
  }
  neckinfo.sector_avg/=single_sector_divisor;

  if(WSinfo::is_ball_pos_valid() && 
     //(mypos-WSinfo::ball->pos).sqr_norm()<MAX_VISIBLE_DISTANCE*MAX_VISIBLE_DISTANCE &&
     viewcone.inside(ballpos))
    neckinfo.ball=0;
  else neckinfo.ball++;
  
#if 0  
  for(int p=0;p<WSinfo::valid_opponents.num;p++) {
    if(can_see_object(mypos,neckdir,WSinfo::valid_opponents[p]->pos))
      neckinfo.opp_player[WSinfo::valid_opponents[p]->number]=0;
    else neckinfo.opp_player[WSinfo::valid_opponents[p]->number]++;
  }
#endif
#if 0
  for(int p=0;p<WSinfo::valid_teammates_without_me.num;p++) {
    if(can_see_object(mypos,neckdir,WSinfo::valid_teammates_without_me[p]->pos))
      neckinfo.own_player[WSinfo::valid_teammates_without_me[p]->number]=0;
    else neckinfo.own_player[WSinfo::valid_teammates_without_me[p]->number]++;
  }
#endif

  neckinfo.opponents=0;
  for(int p=0;p<WSinfo::valid_opponents.num;p++) {
    if(!viewcone.inside(WSinfo::valid_opponents[p]->pos)) {
      neckinfo.opponents+=WSinfo::valid_opponents[p]->age;
    }
  }
  if(WSinfo::valid_opponents.num>0) neckinfo.opponents/=(double)WSinfo::valid_opponents.num;
  
  if(WSinfo::his_goalie) {
    if(viewcone.inside(WSinfo::his_goalie->pos)) {
      neckinfo.opp_goalie=0;
    } else {
      if(neckinfo.opp_goalie>=0) neckinfo.opp_goalie++;
    }
  } else {
    neckinfo.opp_goalie=-1;
  }
  
  if(viewcone.inside(HIS_GOAL_RIGHT_CORNER)) {
    neckinfo.opp_goal_right=0;
  } else {
    neckinfo.opp_goal_right++;
  }
  if(viewcone.inside(HIS_GOAL_LEFT_CORNER)) {
    neckinfo.opp_goal_left=0;
  } else {
    neckinfo.opp_goal_left++;
  }
  if(viewcone.inside(HIS_GOAL_CENTER)) {
    neckinfo.opp_goal=0;
  } else {
    neckinfo.opp_goal++;
  }
  
  return neckinfo;    
}

void BS03Neck::get_neckinfo_weights() {
  ball_weight=0,goalie_weight=0;

  //if(WSinfo::is_ball_pos_valid()) {
  if(WSinfo::ball->age>MAX_BALL_AGE) ball_weight=20;
  else {
    ball_weight=(1-distance_to_ball/100);
    ball_weight*=(0.05+WSinfo::ball->vel.norm()/2);
    if((mdpInfo::get_my_intention() == DECISION_TYPE_INTERCEPTBALL) ||
       (mdpInfo::get_my_intention() == DECISION_TYPE_INTERCEPTSLOWBALL) ||
       (mdpInfo::get_my_intention() == DECISION_TYPE_EXPECT_PASS) ) {
      ball_weight*=2;
    }    
    if(players_near_ball.num>0) {
      if(WSinfo::ball->age==0) {
	need_lookto_ball=true;
      } else {
	ball_weight*=2.5;  // Ball nicht aus den Augen verlieren!
	//DBLOG_POL(1,"Ball near other player, so look at it!");
      }
    }
  }
  //}

  if(mdpInfo::my_distance_to(mdpInfo::opponent_goalpos())<30) {
    if(ballinfeelrange) {
      goalie_weight=0.8;
      if(mdpInfo::my_distance_to(mdpInfo::opponent_goalpos())<16) {
	goalie_weight=0.95;
      }
    }
    else goalie_weight=0.3;
  } else goalie_weight=0;
  
  sector_weight=1.2;
  //get_single_sector_weights();

  opp_weight=1.5;
  team_weight=1.0;
  
  DBLOG_POL(3,"Calculated value weights are");
  DBLOG_POL(3,"ball_weight="<<ball_weight<<", goalie_weight="<<goalie_weight
	  <<", sector_weight="<<sector_weight);
}

/** calculate the individual weight for every sector on field */
void BS03Neck::get_single_sector_weights() {
  Set2d *high_importance=0;
  Set2d *med_importance=0;
  
  switch(DeltaPositioning::get_role(WSinfo::me->number)) {
  case PT_DEFENDER:
    /* rectangle around the enemy offside line, including me */
    high_importance =
      new XYRectangle2d(Vector(WSinfo::my_team_pos_of_offside_line()-10,
			       FIELD_BORDER_Y),
			Vector(Tools::max(WSinfo::my_team_pos_of_offside_line()+20,
				   WSinfo::me->pos.x+5),-FIELD_BORDER_Y));
    break;
  case PT_MIDFIELD:
    high_importance =
      new XYRectangle2d(Vector(WSinfo::me->pos.x-10,FIELD_BORDER_Y),
			Vector(FIELD_BORDER_X,-FIELD_BORDER_Y));
    break;
  case PT_ATTACKER:
    high_importance =
      new Quadrangle2d(WSinfo::me->pos,HIS_GOAL_CENTER,PENALTY_AREA_WIDTH,PENALTY_AREA_WIDTH);
    med_importance =
      new XYRectangle2d(Vector(WSinfo::me->pos.x-10,FIELD_BORDER_Y),
			Vector(FIELD_BORDER_X,-FIELD_BORDER_Y));
    break;
  }

  if(high_importance) {
    DBLOG_DRAW(0,*high_importance);
    //DBLOG_POL(0,"high importance " << *high_importance);
  }
  if(med_importance) {
    DBLOG_DRAW(0,*med_importance);
    //DBLOG_POL(0,"med importance " << *high_importance);
  }

  /* now value the sectors... */
  single_sector_divisor=0;
  Vector pos;
  pos.y=-(SECT_Y-1)/2*yunit;
  for(int y=0; y<SECT_Y;y++, pos.y+= yunit) {
    pos.x=-(SECT_X-1)/2*xunit;
    for(int x=0;x<SECT_X;x++,pos.x+= xunit) {
      if(high_importance && high_importance->inside(pos)) single_sector_weight[x][y]=1;
      else if(med_importance && med_importance->inside(pos)) single_sector_weight[x][y]=.5;
      else single_sector_weight[x][y]=0;
      single_sector_divisor+=single_sector_weight[x][y];
    }
  }
      
  delete high_importance; delete med_importance;
}

/** Get the value of a given neckinfo */
/*  This one really does the trick... */
double BS03Neck::get_neckinfo_value(Neck_Info neckinfo) {

  //double totalval=0;
  double val_ball=neckinfo.ball;
  double val_sector;
  double val_goalie;
  double val_opp;
  double val_team;
  
  val_sector=(neckinfo.sector_avg-(cur_info.sector_avg+1));

  if(neckinfo.opp_goalie>=0) val_goalie=neckinfo.opp_goalie;
  else val_goalie=(2*neckinfo.opp_goal+neckinfo.opp_goal_left+neckinfo.opp_goal_right)/4.0;
  if(val_goalie==1) val_goalie=0;  // nicht zweimal nacheinander zum Goalie schauen!

  val_opp=(neckinfo.opponents-(cur_info.opponents+1));
  val_team=(neckinfo.teammates-(cur_info.teammates+1));
  
  DBLOG_POL(4,"val_ball="<<val_ball*ball_weight
	    <<", val_sector="<<val_sector*sector_weight
	    <<", val_goalie="<<val_goalie*goalie_weight
	    <<", val_opp="<<val_opp*opp_weight);
  return val_ball*ball_weight+val_sector*sector_weight+val_goalie*goalie_weight
    + val_opp*opp_weight+val_team*team_weight;
}

Angle BS03Neck::get_best_angle(Angle &target) {

  //Vector my_new_pos,my_new_vel,new_ball_pos,new_ball_vel;
  //Angle my_new_angle;
  //LOG_MOV(2,<<"_2D_ CIRCLE col=ff0000 ("<<new_ball_pos.x<<","<<new_ball_pos.y<<",3);");
  
  ANGLE myang=ANGLE(my_new_angle);
  ANGLE startang=myang+ServerOptions::minneckang;
  ANGLE preset;
  if(target!=-1) preset=ANGLE(target);
  
  Angle bestang=-1;
  double bestval=10000000;
  for(int i=0;i<NUM_ANGLES;i++) {

    ANGLE aktang=startang+ ANGLE(
      i*(ServerOptions::maxneckang-ServerOptions::minneckang).get_value()/(NUM_ANGLES-1.0)
      );    
    /* ignore angle if preset target can't be seen */
    if(target!=-1 && fabs((aktang-preset).get_value()) >
       (mdpInfo::next_view_angle_width()-ANGLE(DEG2RAD(8/.5))).get_value()*.5) {
      continue;
    }
    
#if 0
    Vector pol;
    pol.init_polar(20,aktang.get_value());pol+=WSinfo::me->pos;
    //DBLOG_DRAW(2,L2D(WSinfo::me->pos.x,WSinfo::me->pos.y,pol.x,pol.y,"#cc88ff"));
#endif
    //long time1=Tools::get_current_ms_time();
  
    next_info=get_neck_info(my_new_pos,aktang,new_ball_pos,cur_info);
    //long time2=Tools::get_current_ms_time();
    
    if(ballinfeelrange) next_info.ball=0;
    double dum=get_neckinfo_value(next_info);
    DBLOG_POL(4,"aktang: "<<aktang<<", value: "<<dum);
    if(dum<bestval) {
      bestval=dum;bestang=aktang.get_value();
    }
    //long time2=Tools::get_current_ms_time();
    //LOG_MOV(2,<<"BS02_Neck: Check needed "<<time2-time1<<" ms.");
  }

  if(bestang!=-1) target=bestang;

  return target;
  
}

Angle BS03Neck::neck_value_based(Angle target) {
  DBLOG_POL(0,"Calculating value based neck policy...");
  get_neckinfo_weights();
  return get_best_angle(target);
}

#else
/***********************************************************************
 * New (sector based) default neck policy
 ***********************************************************************/

Angle BS03Neck::neck_default(Angle preset) {
  ANGLE minang,maxang;
  minang=Tools::my_minimum_abs_angle_to_see();
  maxang=Tools::my_maximum_abs_angle_to_see();

  ANGLE myang=ANGLE(my_new_angle);
  ANGLE startang=myang+ServerOptions::minneckang;
  Vector mypos=my_new_pos;
  
  // TODO: preset

  int mark[SECT_X][SECT_Y];
  for(int x=0;x<SECT_X;x++)
    for(int y=0;y<SECT_Y;y++)
      mark[x][y]=0;
  
  Cone2d cone(mypos,maxang,minang); // cone is the _invisible_ area!
  Vector dum1,dum2;
  dum1=10*cone.dir1;
  dum2=10*cone.dir2;
  DBLOG_DRAW(0,"LINE("<<mypos.x<<","<<mypos.y<<","<<(mypos+dum1).x<<","<<(mypos+dum1).y<<");");
  DBLOG_DRAW(0,"LINE("<<mypos.x<<","<<mypos.y<<","<<(mypos+dum2).x<<","<<(mypos+dum2).y<<");");

  PlayerSet players=WSinfo::valid_opponents;
  players+=WSinfo::valid_teammates_without_me;
  for(int p=0;p<players.num;p++) 
  {
    int x=(int)((players[p]->pos.getX()+FIELD_BORDER_X)/xunit);
    int y=(int)((players[p]->pos.getY()+FIELD_BORDER_Y)/yunit);
    //This line has been added on April 3rd, 2006. It is hoped that by adding
    //this condition the well-known bug, which results in agent mass mortality
    //when performing a corner kick, is removed 
    //BEGIN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if ( x>=0 && x<SECT_X && y>=0 && y<SECT_Y ) //tg
    //END ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      mark[x][y]=-1;
    if(players[p]->age>2) 
    {
      for(int xi=x-1;xi<=x+1;xi++)
      {
        for(int yi=y-1;yi<=y+1;yi++) 
        {
          if (   xi >= 0 
              && xi<SECT_X 
              && yi>=0 
              && yi<SECT_Y ) 
            mark[xi][yi]=-1;
        }
      }
    }
  }
  
  long maxage=0;
  Vector pos;
  pos.setY( -(SECT_Y-1)/2*yunit );
  for(int y=0; y<SECT_Y;y++, pos.addToY( yunit )) {
    pos.setX( -(SECT_X-1)/2*xunit );
    for(int x=0;x<SECT_X;x++,pos.addToX( xunit )) {
      if(!mark[x][y]) continue; // no objects!
      if((mypos-pos).sqr_norm()<SQR_MAX_VISIBLE_DISTANCE && !cone.inside(pos)) {
	if(cur_info.sector[x][y]>maxage) maxage=cur_info.sector[x][y];
      } else mark[x][y]=0;
    }
  }
  
#if 1
  pos.setY( -(SECT_Y-1)/2*yunit );
  for(int y=0; y<SECT_Y;y++, pos.addToY( yunit )) {
    pos.setX( -(SECT_X-1)/2*xunit );
    for(int x=0;x<SECT_X;x++,pos.addToX( xunit )) {
      if(mark[x][y]<0) {
	DBLOG_DRAW(0,"STRING("<<pos.x<<","<<pos.y<<",\""<<cur_info.sector[x][y]<<"\");");
      }
    }
  }
#endif
  
  Angle bestang=-1;
  int maxagesect=0;
  double bestage=0;
  for(int i=0;i<NUM_ANGLES;i++) {
    ANGLE aktang=startang+ ANGLE(i*(ServerOptions::maxneckang-ServerOptions::minneckang).get_value()
				 /(NUM_ANGLES-1.0));    
    /* ignore angle if preset target can't be seen */
    //if(target!=-1 && fabs((aktang-preset).get_value()) >
    //   (mdpInfo::next_view_angle_width()-ANGLE(DEG2RAD(8/.5))).get_value()*.5) {
    //  continue;
    //}
    Vector dum;
    dum.init_polar(1.0,aktang);
    Cone2d viewcone(mypos,dum,ANGLE(Tools::next_view_angle_width().get_value()-DEG2RAD(7)));

    int mas=0;
    double age=0;
    int sectcnt=0;
    pos.setY( -(SECT_Y-1)/2*yunit );
    for(int y=0; y<SECT_Y;y++, pos.addToY( yunit )) {
      pos.setX( -(SECT_X-1)/2*xunit );
      for(int x=0;x<SECT_X;x++,pos.addToX( xunit )) {
	if(mark[x][y]>=0) continue;
	if(!viewcone.inside(pos)) continue;
	if(cur_info.sector[x][y]>=maxage) mas++;
	age+=cur_info.sector[x][y];
	sectcnt++;
      }
    }
    if(sectcnt==0) continue;
    //age/=sectcnt;
    if(mas>maxagesect || (mas==maxagesect && age>bestage)) {
      maxagesect=mas;
      bestang=aktang.get_value();
      bestage=age;
    }    
  }
  if(bestang<0) {
    DBLOG_POL(0,"WARNING: No default neck move selected!");
    return -1;
  }
  DBLOG_POL(0,"neck_default selected ang "<<bestang<<" ("<<RAD2DEG(bestang)<<"); maxage sect="
	    <<maxagesect<<", avg age="<<bestage);
	
  return bestang;

}




#endif  // VALUE_BASED

/***********************************************************************
 * Simple moves (or special moves)
 ***********************************************************************/

/** Special consideration when defense is based on direct opponents.*/
Angle BS03Neck::check_direct_opponent_defense()
{
  MYLOG_POL(0,"TURN NECK: check direct opponent defense ");
  Angle target=-1;
  if(    ! ignore_neck_intentions 
      && neck_intention.type == NECK_INTENTION_DIRECTOPPONENTDEFENSE) 
  {
    target = intention_direct_opponent_defense();

    if( !Tools::could_see_in_direction(target) ) 
    {
      MYLOG_POL(0,"WARNING: I cannot look in the direction ("<<target<<") forced by NECK_INTENTION!");
      target=-1;center_target=false; // statt dessen Standardmove machen!
    } 
    else
    {
      MYLOG_POL(0,"I am forced to look to target NECK_INTENTION, target = "<<target<<"!");
      Vector pol;
      pol.init_polar(20.,target);
      pol+=WSinfo::me->pos;      
      MYLOG_DRAW(0,L2D(WSinfo::me->pos.x,WSinfo::me->pos.y,pol.x,pol.y,"cyan"));
      center_target=true;
    }
  }
  else {
    MYLOG_POL(0,"TURN NECK: neck_intention.type =="<<neck_intention.type);
  }
  return target;
}


/** Fallback routine if ball not known - looks directly to guessed ball pos! */
Angle BS03Neck::check_search_ball() {
  MYLOG_POL(0,"TURN NECK: check search ball ");
  if(WSinfo::ball->age>=MAX_BALL_AGE /*|| need_lookto_ball*/) {
    //cerr << "\nP"<<WSinfo::me->number<<"cyc #"<<WSinfo::ws->time<<": Search ball!";
    if(got_update) need_lookto_ball=false;
    if(got_update && ball_already_searched) {
      DBLOG_POL(1,"check_search_ball(): Ball not at guessed position, continuing");
      return -1;  // Ball not at guessed position
    }
    if(!Tools::could_see_in_direction(dir_to_ball.get_value())) {
      DBLOG_POL(1,"check_search_ball(): Cannot look to guessed ballpos!");
      return -1; 
    }
    DBLOG_POL(0,"check_search_ball(): Looking to guessed ballpos "<<dir_to_ball<<"!");
    ball_already_searched=true;  
    return dir_to_ball.get_value();
  }
  return -1;
}

Angle BS03Neck::check_relevant_teammates() {
  MYLOG_POL(0,"TURN NECK: Check relevant teammates ");

  Angle target=-1; // default
  center_target=false; // default
  PPlayer teammate;
  
  for(int i=0; i<WSinfo::num_relevant_teammates;i++){
    if(WSinfo::get_teammate(WSinfo::relevant_teammate[i],teammate)){
      MYLOG_POL(0,"TURN NECK: Checking relevant teammate "
	      <<teammate->number<<" age "<<teammate->age);
      if(teammate->age >1){
	target=(teammate->pos - WSinfo::me->pos).arg();

	Vector pol;
	pol.init_polar(25,target);
	pol+=WSinfo::me->pos;
	MYLOG_DRAW(0, L2D(WSinfo::me->pos.x,WSinfo::me->pos.y,pol.x,pol.y,"#000080"));

	if(!Tools::could_see_in_direction(target)) {
	  MYLOG_POL(0,"TURN NECK: Cannot look in direction of teammate "<<teammate->number);
	  target = -1;
	}
	else{
	  MYLOG_POL(0,"TURN NECK: YEP! Let's look to teammate  "<<teammate->number
		  <<" in dir "<<RAD2DEG(target));
	  center_target=true;
	  break;
	}
      } // if teammate age > 1
    } // if teammate is alive
  } // for all relevant teammates
  return target;
}





Angle BS03Neck::check_neck_intention() {
  MYLOG_POL(0,"Entered neck intention type: "<<neck_intention.type<<" ignore_neck_intentions: "<<ignore_neck_intentions);

  Angle target=-1;
  if(!ignore_neck_intentions && neck_intention.type!=NECK_INTENTION_NONE) {
    switch(neck_intention.type) {
    case NECK_INTENTION_LOOKINDIRECTION: target=intention_lookindirection();break;
    case NECK_INTENTION_PASSINDIRECTION: target=intention_passindirection();break;
    case NECK_INTENTION_FACEBALL: target=intention_faceball();break;
    case NECK_INTENTION_BLOCKBALLHOLDER: target=intention_blockballholder();break;
    default: DBLOG_POL(0,"WARNING: NECK_INTENTION type "
		     <<neck_intention.type
		     <<" not (yet) supported, continuing in normal mode!");
    //cerr << "\nBS02_Neck: WARNING: NECK_INTENTION type "
    //	 <<neck_intention.type
    //	 <<" not (yet) supported, continuing in normal mode!";
    break;
    } // end switch   
    if(!Tools::could_see_in_direction(target)) {
      MYLOG_POL(0,"WARNING: I cannot look in the direction ("<<target<<") forced by NECK_INTENTION!");
      target=-1;center_target=false; // statt dessen Standardmove machen!
    } 
    else{
      MYLOG_POL(0,"I am forced to look to target NECK_INTENTION, target is "<<target<<"!");
      Vector pol;
      pol.init_polar(20.,target);
      pol+=WSinfo::me->pos;      
      MYLOG_DRAW(0,L2D(WSinfo::me->pos.x,WSinfo::me->pos.y,pol.x,pol.y,"cyan"));
      center_target=true;
    }
  }
  return target;
}

Angle BS03Neck::check_intercept() {
  MYLOG_POL(0,"TURN NECK: check intercept ");

  LOG_POL(0,"TURN NECK: check intercept ! ballinfeelrange "<<ballinfeelrange<<" potent. coll "<<potentialcollision);

  if(!WSinfo::is_ball_pos_valid()) {
    DBLOG_POL(1,"check_intercept: I don't know where the ball is, ignoring!");
    return -1;
  } 
  if(intercept_look_mode!=0 &&
     ((mdpInfo::get_my_intention() == DECISION_TYPE_INTERCEPTBALL) ||
      (mdpInfo::get_my_intention() == DECISION_TYPE_INTERCEPTSLOWBALL) //||
      /*(mdpInfo::get_my_intention() == DECISION_TYPE_EXPECT_PASS)*/)) {
    if(ballinfeelrange && potentialcollision == false) {
      LOG_POL(0,"Turn Neck: Intercepting,ball will be in feelrange next time. But Not close. Ridi05!");
      return -1;
    } else {
      if(!Tools::could_see_in_direction(dir_to_ball.get_value())) {
	DBLOG_POL(1,"Intercepting, but I can't look to ball!");
	return -1;
      } else {
	DBLOG_POL(0,"Intercepting, trying to look to ball...");
	if(intercept_look_mode==2) center_target=true;
	return dir_to_ball.get_value();
      }
    }
  }
  return -1;
}

Angle BS03Neck::check_players_near_ball() {
  MYLOG_POL(0,"TURN NECK: check players near ball ");

  if(!WSinfo::is_ball_pos_valid()) {
    DBLOG_POL(1,"check_players_near_ball: I don't know where the ball is, ignoring!");
    return -1;
  }
  Angle target=-1;
  
  if(target<0 && opp_has_ball_look_mode!=0 && opp_players_near_ball.num>0) {
    DBLOG_POL(0,"Opponents near ball, trying to look...");
    target=0;
    if(opp_has_ball_look_mode==2) center_target=true;
  }
  if(target<0 && own_has_ball_look_mode!=0 && own_players_near_ball.num>0) {
    DBLOG_POL(0,"Teammates near ball, trying to look...");
    target=0;
    if(own_has_ball_look_mode==2) center_target=true;
  }
  if(!target) {
    if(!Tools::could_see_in_direction(dir_to_ball.get_value())) {
      DBLOG_POL(0,"Intercepting, but I can't look to ball!");
      center_target=false;
      return -1;
    } else {
      return dir_to_ball.get_value();
    }
  }
  return -1;
}

/* obsolete, should be covered by neck intention! */
Angle BS03Neck::check_block_ball_holder() {
  if(ball_holder_look_mode==0) return -1;
  if(mdpInfo::get_my_intention()!=DECISION_TYPE_BLOCKBALLHOLDER) return -1;
  LOG_POL(3,<<"BS02_Neck - check_block_ball_holder: Got BLOCKBALLHOLDER intention!");
  Angle target;
  if((target=get_dir_of_nearest_opponent())==-1) return -1;
  if(!Tools::could_see_in_direction(target)) {
    //MYLOG_POL(3,"BS02_Neck - check_block_ball_holder: Can't look to fastest opp!");
    return -1;
  }
  Angle target_dist=Tools::get_angle_between_mPI_pPI(dir_to_ball.get_value()-target);
  if(fabs(target_dist)
     < (mdpInfo::next_view_angle_width().get_value())-DEG2RAD(10)) {
    //MYLOG_POL(3,"BS02_Neck - check_block_ball_holder: Looking to ball and fastest opponent!");
    target+=target_dist*.5;
    center_target=true;
  } else {
    MYLOG_POL(3,"BS02_Neck - check_block_ball_holder: Looking to fastest opponent!");
    ;
  }
  if(ball_holder_look_mode==2) center_target=true;
  return target;
}

Angle BS03Neck::check_offside() {
  MYLOG_POL(0,"TURN NECK: check offside ");

  if(mdpInfo::is_ball_infeelrange() || ballinfeelrange) {
    DBLOG_POL(1,"check_offside: I am carrying the ball, ignoring!");
    return -1; // I am carrying the ball!
  }
  if(WSinfo::ball->pos.getX()>WSinfo::me->pos.getX()+OFFSIDE_TOLERANCE) {
    DBLOG_POL(1,"check_offside: Ball in front of me, ignoring!");
    return -1; // Ball in front of me
  }
  if(WSinfo::me->pos.getX()>(FIELD_BORDER_X-IGNORE_OFFSIDE_GOAL_DISTANCE)) {
    DBLOG_POL(1,"check_offside: Too close to goal, ignoring!");
    return -1;  // too close to goal!
  }
  if(WSinfo::me->pos.getX() < WSinfo::his_team_pos_of_offside_line()-OFFSIDE_TOLERANCE) {
    DBLOG_POL(1,"check_offside: Too far away from last known offside line, ignoring!");
    return -1;
  }

  PlayerSet opp_in_left_offside_box,opp_in_right_offside_box,opp_in_offside_box;
  opp_in_offside_box=WSinfo::valid_opponents;
  opp_in_offside_box.keep_players_in_rectangle(Vector(WSinfo::me->pos.getX()-OFFSIDE_TOLERANCE,-FIELD_BORDER_Y),
					       Vector(FIELD_BORDER_X,+FIELD_BORDER_Y));
  opp_in_offside_box.keep_and_sort_players_by_x_from_right(11);
  if(WSinfo::his_goalie) opp_in_offside_box.remove(WSinfo::his_goalie);
  opp_in_left_offside_box=opp_in_offside_box;
  opp_in_right_offside_box=opp_in_offside_box;
  opp_in_left_offside_box.keep_players_in_halfplane(WSinfo::me->pos,Vector(0,1));
  opp_in_right_offside_box.keep_players_in_halfplane(WSinfo::me->pos,Vector(0,-1));
  
  //for(int i=0;i<opp_in_left_offside_box.num;i++) {
  //  LOG_MOV(0,<<_2D<<C2D(opp_in_left_offside_box[i]->pos.x,opp_in_left_offside_box[i]->pos.y,2.5,"#111111"));
  //}
  //for(int i=0;i<opp_in_right_offside_box.num;i++) {
  //  LOG_MOV(0,<<_2D<<C2D(opp_in_right_offside_box[i]->pos.x,opp_in_right_offside_box[i]->pos.y,2.5,"#ff0000"));
  //}
  
  MYLOG_POL(1,_2D<<L2D(WSinfo::his_team_pos_of_offside_line(),FIELD_BORDER_Y,
		       WSinfo::his_team_pos_of_offside_line(),-FIELD_BORDER_Y,"#000000"));
  
  long left_age=1000;
  long right_age=1000;

  static long last_left_check=-10;
  static long last_right_check=-10;
  static long last_left_age=0;
  static long last_right_age=0;
  
  if(opp_in_left_offside_box.num>0 && 
     Tools::could_see_in_direction(Tools::my_abs_angle_to(opp_in_left_offside_box[0]->pos))) {
    for(int i=0;i<opp_in_left_offside_box.num;i++) {
      if(opp_in_left_offside_box[i]->age<left_age) left_age=opp_in_left_offside_box[i]->age;
    }
  }
  if(opp_in_right_offside_box.num>0 && 
     Tools::could_see_in_direction(Tools::my_abs_angle_to(opp_in_right_offside_box[0]->pos))) {
    for(int i=0;i<opp_in_right_offside_box.num;i++) {
      if(opp_in_right_offside_box[i]->age<right_age) right_age=opp_in_right_offside_box[i]->age;
    }
  }  
  
  if(left_age<right_age && right_age<999 && right_age>=MAX_OFFSIDE_LINE_AGE) {
    DBLOG_POL(0,"check_offside: checking righthand offside line!");
    if(last_right_check==WSinfo::ws->time-1 && last_right_age==right_age-1) {
      DBLOG_POL(0,"check_offside: Something is wrong, I already checked last cycle!");
      return -1;
    }
    last_right_check=WSinfo::ws->time;
    last_right_age = right_age;
    center_target=true;
    //DBLOG_POL(3,<<_2D<<C2D(opp_in_right_offside_box[0]->pos.x,opp_in_right_offside_box[0]->pos.y,2.5,"#ffffff"));
    return Tools::my_abs_angle_to(opp_in_right_offside_box[0]->pos).get_value();
  } else {
    if(left_age<999 && left_age>=MAX_OFFSIDE_LINE_AGE) {
      DBLOG_POL(0,"check_offside: checking lefthand offside line!");
      //LOG_POL(3,<<_2D<<C2D(opp_in_left_offside_box[0]->pos.x,opp_in_left_offside_box[0]->pos.y,2.5,"#ffffff"));
      if(last_left_check==WSinfo::ws->time-1 && last_left_age==left_age-1) {
	DBLOG_POL(0,"check_offside: Something is wrong, I already checked last cycle!");
	return -1;
      }
      last_left_check=WSinfo::ws->time;
      last_left_age = left_age;
      center_target=true;
      return Tools::my_abs_angle_to(opp_in_left_offside_box[0]->pos).get_value();
    }
  }
  return -1;
}

Angle BS03Neck::check_neck_1v1() {
  MYLOG_POL(0,"TURN NECK: check 1v1 ");

  switch(WSinfo::ws->play_mode) {
  case PM_my_BeforePenaltyKick:
  case PM_his_BeforePenaltyKick:
  case PM_my_PenaltyKick:
  case PM_his_PenaltyKick:
    break;
  default:
    if(!use_1v1_mode) return -1;
    break;
  }
  Angle target;
  //if(mdpInfo::my_distance_to(mdpInfo::opponent_goalpos())>30) return -1;
  if(!WSinfo::is_ball_pos_valid() || !ballinfeelrange) return -1;
  if(!WSinfo::his_goalie) return -1;
  if(WSinfo::is_opponent_pos_valid(WSinfo::his_goalie)) {
    target=Tools::my_abs_angle_to(WSinfo::his_goalie->pos).get_value();
    if(Tools::could_see_in_direction(target)) return target;
  }
  target=Tools::my_abs_angle_to(mdpInfo::opponent_goalpos()).get_value();
  if(Tools::could_see_in_direction(target)) return target;
  return -1;
}

Angle BS03Neck::check_goalie() {
  MYLOG_POL(0,"TURN NECK: check goalie ");

  if(last_looked_to_goalie>=WSinfo::ws->time-1) return -1;
  if(my_new_pos.distance(HIS_GOAL_CENTER)>30) return -1;
  Angle target=-1;
  if(WSinfo::his_goalie) {
    if(WSinfo::his_goalie->age<MAX_GOALIE_AGE) return -1;
    if(!ballinfeelrange) return -1;
    DBLOG_POL(0,"check_goalie: Goalie too old, trying to look");
    if(Tools::could_see_in_direction(Tools::my_abs_angle_to(WSinfo::his_goalie->pos))) {
      target=Tools::my_abs_angle_to(WSinfo::his_goalie->pos).get_value();
      DBLOG_POL(0,"check_goalie: Looking to last goalie pos");
    }
  }
  if(!WSinfo::his_goalie &&
     (cur_info.opp_goal_left+2*cur_info.opp_goal+cur_info.opp_goal_right)/4.0
     < MAX_GOALIE_AGE) return -1;
  if(Tools::could_see_in_direction(Tools::my_abs_angle_to(HIS_GOAL_CENTER))) {
    center_target=true;
    DBLOG_POL(0,"check_goalie: Looking to opp goal center");
    target=Tools::my_abs_angle_to(HIS_GOAL_CENTER).get_value();
  } else if(Tools::could_see_in_direction(Tools::my_abs_angle_to(HIS_GOAL_LEFT_CORNER))) {
    center_target=true;
    DBLOG_POL(0,"check_goalie: Looking to left goal corner");
    target=Tools::my_abs_angle_to(HIS_GOAL_LEFT_CORNER).get_value();
  } else if(Tools::could_see_in_direction(Tools::my_abs_angle_to(HIS_GOAL_RIGHT_CORNER))) {
    center_target=true;
    DBLOG_POL(0,"check_goalie: Looking to right goal corner");
    target=Tools::my_abs_angle_to(HIS_GOAL_RIGHT_CORNER).get_value();
  }
  if(target!=-1) {
    last_looked_to_goalie=WSinfo::ws->time;
  }
  return target;
}

/***********************************************************************************
 * Reaction to NECK_INTENTION
 **********************************************************************************/

Angle BS03Neck::intention_lookindirection() {
  DBLOG_POL(0,"intention_lookindirection: Should turn to "<<neck_intention.p1<<"!");
  return neck_intention.p1;
}

Angle BS03Neck::intention_direct_opponent_defense() {
  DBLOG_POL(0,"intention_direct_opponent_defense: Should turn to "<<neck_intention.p1<<"!");
  return neck_intention.p1;
}

Angle BS03Neck::intention_passindirection() {
  DBLOG_POL(0,"intention_passindirection: Should turn to "<<neck_intention.p1<<"!");
  return neck_intention.p1;
}

Angle BS03Neck::intention_faceball() {
  Angle target = Tools::my_abs_angle_to(WSinfo::ball->pos+WSinfo::ball->vel).get_value();
  DBLOG_POL(0,"intention_faceball: Should turn to "<<target<<"!");
  return target;
}

Angle BS03Neck::intention_blockballholder() {
  if(ball_holder_look_mode==0) return -1;
  
  DBLOG_POL(0,"intention_blockballholder: Got BLOCKBALLHOLDER neck intention!");
  Angle target;
  if((target=get_dir_of_nearest_opponent())==-1) return -1;
  if(!Tools::could_see_in_direction(target)) {
    DBLOG_POL(0,"intention_blockballholder: Can't look to fastest opp!");
    return -1;
  }
  Angle target_dist=Tools::get_angle_between_mPI_pPI(dir_to_ball.get_value()-target);
  if(fabs(target_dist)
     < (mdpInfo::next_view_angle_width().get_value())-DEG2RAD(10)) {
    DBLOG_POL(0,"intention_blockballholder: Looking to ball and fastest opponent!");
    target+=target_dist*.5;
    center_target=true;
  } else {
    DBLOG_POL(0,"intention_blockballholder: Looking to fastest opponent!");
  }
  if(ball_holder_look_mode==2) center_target=true;
  return target; 
}

/********************************************************************************
 * Helper functions
 ********************************************************************************/


/** locks neck angle to body angle */
bool BS03Neck::neck_lock(Cmd &cmd) {
  DBLOG_POL(0,"neck_lock: Locking neck angle to body angle");
  neck_cmd->set_turn_neck_rel(0.0);
  return neck_cmd->get_cmd(cmd);
}

/** returns -1 also when opponent has an age < MAX_FASTEST_OPP_AGE! */
Angle BS03Neck::get_dir_of_nearest_opponent() {
  static long last_calculated=-1;
  static Angle cache=0;

  int opp_number=-1;
  
  if(last_calculated==WSinfo::ws->time) return cache;
  last_calculated=WSinfo::ws->time;
  Policy_Tools::go2ball_steps_update();

  for(int i=0;i<22;i++) {
    if(Policy_Tools::go2ball_steps_list()[i].side == Go2Ball_Steps::THEIR_TEAM) {
      opp_number=Policy_Tools::go2ball_steps_list()[i].number;
      if(opp_number==WSinfo::ws->his_goalie_number) {
	opp_number=-1;continue;
      }
      break;
    }
  }
  if(opp_number==-1) {
    DBLOG_POL(1,"I don't know which opponent is fastest to ball!");
    return -1;
  }
  PPlayer p=WSinfo::get_opponent_by_number(opp_number);
  if(p->age<MAX_FASTEST_OPP_AGE) {
    DBLOG_POL(1,"Fastest opponent has been seen recently, ignoring...");
    return -1;
  }
  cache=(p->pos-my_new_pos).angle();
  DBLOG_POL(1,"Fastest opponent is #"<<opp_number<<"!");
  return cache;
}






////////////////////////////////////////////////////////////////////////////////////////////////////////

#if 0
/************************************************************************************
 * OBSOLETE
 *
 * This is all old unused stuff, but still kept for reference purposes.
 ************************************************************************************/

/** This routine replaces the buggy one in mdpInfo - dirty workaround! */

bool BS03Neck::is_ball_in_feelrange_next_time() {
  return ballinfeelrange;
#if 0
  //return mdpInfo::is_ball_infeelrange();
  FPlayer *me=mdpInfo::mdp->me;
  Vector my_new_pos,my_new_vel,new_ball_pos,new_ball_vel;
  Angle my_new_angle;
  Tools::model_cmd_main(me->pos(),me->vel(),me->ang.v,mdpInfo::mdp->ball.pos(),mdpInfo::mdp->ball.vel(),
			cmd->cmd_body,
			my_new_pos,my_new_vel,my_new_angle,new_ball_pos,new_ball_vel);
  if((my_new_pos-new_ball_pos).norm()<0.95*ServerOptions::visible_distance) return true;
  
  return false;
#endif
}


/** old routines, will maybe be changed after Seattle */

//#define ATTACKER_DIST 6.

/** lock neck angle to body angle */
Neck_Move* BS03Neck::neck_face_ball() {
  LOG_POL(2,<<"BS02_Neck - neck_face_ball: Locking neck angle to body angle");
  return Move_Factory::get_Neck_Turn_Rel(0);
}

/* Subroutines return -1 if they do not want to turn neck; target vector otherwise */
/* Add true as param if they should be forced to react.                            */
/* They do nothing if target is already a valid angle.                             */

/* Reaction to NECK_INTENTION */
/* They have priority!        */

Neck_Move* BS03Neck::bs02_neck_main() {

  LOG_POL(3,<<"BS02_Neck: Executing main behaviour...");
  //FPlayer *me=mdpInfo::mdp->me;
  Angle minang,maxang;
  
  //Vector pol;
#if 1
  Vector pol;
  FPlayer *me=mdpInfo::mdp->me;
  pol.init_polar(45,minang);pol+=me->pos();
  LOG_MOV(2,<< _2D << L2D(me->pos().x,me->pos().y,pol.x,pol.y,"#000080"));
  pol.init_polar(25,maxang);pol+=me->pos();
  LOG_MOV(2,<< _2D << L2D(me->pos().x,me->pos().y,pol.x,pol.y,"#000080"));
#endif
  Angle target=-1;

  LOG_POL(3,<<"BS02_Neck: Checking for special situations!");
  
  // care for neck intention types

  if(!ignore_neck_intentions && neck_intention.type!=NECK_INTENTION_NONE) {
    LOG_POL(3,<<"BS02_Neck: Got NECK_INTENTION type "<<neck_intention.type);
    //cout << "\nGot NECK_INTENTION!" << endl;
    switch(neck_intention.type) {
      //case NECK_INTENTION_LOCKNECK: intention_lockneck(target);break;
      //case NECK_INTENTION_LOOKTOGOAL: intention_looktogoal(target);break;
      //case NECK_INTENTION_LOOKTOGOALIE: intention_looktogoalie(target);break;
      //case NECK_INTENTION_LOOKTOBALL: intention_looktoball(target);break;
      //case NECK_INTENTION_LOOKTOCLOSESTOPPONENT: intention_looktoclosestopponent(target);break;
      //case NECK_INTENTION_CHECKOFFSIDE: intention_checkoffside(target);break;
    case NECK_INTENTION_LOOKINDIRECTION: intention_lookindirection(target);break;
    case NECK_INTENTION_PASSINDIRECTION: intention_passindirection(target);break;
      //case NECK_INTENTION_SCANFORBALL: intention_scanforball(target);break;
    case NECK_INTENTION_FACEBALL: intention_faceball(target);break;
      //case NECK_INTENTION_EXPECTPASS: intention_expectpass(target);break;
    case NECK_INTENTION_BLOCKBALLHOLDER: intention_blockballholder(target);break;
    default: LOG_MOV(2,<<"BS02_Neck: WARNING: NECK_INTENTION type "
		     <<neck_intention.type
		     <<" not (yet) supported, continuing in normal mode!");
    //cerr << "\nBS02_Neck: WARNING: NECK_INTENTION type "
    // <<neck_intention.type
    // <<" not (yet) supported, continuing in normal mode!";
    break;
    }    
  } else {
    // look always to ball in certain situations!
    if(intercept_look_mode!=0 &&
       ((mdpInfo::get_my_intention() == DECISION_TYPE_INTERCEPTBALL) ||
	(mdpInfo::get_my_intention() == DECISION_TYPE_INTERCEPTSLOWBALL) //||
	/*(mdpInfo::get_my_intention() == DECISION_TYPE_EXPECT_PASS)*/)) {
      if(is_ball_in_feelrange_next_time()) {
	LOG_POL(3,<<"BS02_Neck: Intercepting, but ball will be in feelrange next time!");
      } else {
	LOG_POL(3,<<"BS02_Neck: Intercepting, trying to look to ball...");
	target=0;
	if(intercept_look_mode==2) center_target=true;
      }
    }
    //LOG_POL(2,<<"BS02_Neck: Opp "<<opp_players_near_ball.num<<" Own "<<own_players_near_ball.num);
    if(target<0 && opp_has_ball_look_mode!=0 && opp_players_near_ball.num>0) {
      LOG_POL(3,<<"BS02_Neck: Opponents near ball, trying to look...");
      target=0;
      if(opp_has_ball_look_mode==2) center_target=true;
    }
    if(target<0 && own_has_ball_look_mode!=0 && own_players_near_ball.num>0) {
      LOG_POL(3,<<"BS02_Neck: Teammates near ball, trying to look...");
      target=0;
      if(own_has_ball_look_mode==2) center_target=true;
    }
    if(!target) {
      if(!mdpInfo::is_ball_pos_valid()) {
	LOG_POL(3,<<"BS02_Neck: I don't know where the ball is, ignoring!");
	target=-1;
      } else {
	ANGLE balldir=(new_ball_pos-my_new_pos).angle();
	target=balldir.get_value();
      }
    }
  }
    
  // check, ob g?ltiger Winkel
  if(target!=-1) {
    LOG_POL(3,"BS02_Neck: Should look to "<<target<<" ("<<RAD2DEG(target)<<")!");
    if(!mdpInfo::could_see_in_direction(target)) {
      LOG_POL(3,"BS02_Neck: WARNING: I cannot look into that direction - ignoring!");
      //LOG_ERR(2,"BS02_Neck: WARNING: I cannot look in the direction forced by NECK_INTENTION!");
      //cerr << "\nBS02_Neck: WARNING: I cannot look in the direction forced by NECK_INTENTION!";
      target=-1;center_target=false; // statt dessen Standardmove machen!
    }
  }

  if(target==-1 || !center_target) {

    LOG_POL(0,<<"TURN_NECK: I am free to look where I want!");

    LOG_POL(3,<<"BS02_Neck: Starting value-based calculation of best neck direction!");
    // hier kommt die Wertfunktion ins Spiel.
    //long time1=Tools::get_current_ms_time();
    double oldtarget=-1;
    if(target!=-1) {
      oldtarget=target;
    }
    get_neckinfo_weights();
    get_best_angle(target);
    
    if(oldtarget!=-1) {
      //Neck_Info next_info=get_neck_info(my_new_pos,oldtarget,new_ball_pos,cur_info);
      if(fabs(target-oldtarget)<mdpInfo::view_angle_width_rad()*.5-DEG2RAD(4)) {
	LOG_POL(3,<<"BS02_Neck: Modifying preset target by "<<fabs(target-oldtarget)<<".");
      } //else {
      //target=oldtarget;
      //}
    }
  }
  if(target>-1) {
    
#if 1
    Vector pol;
    FPlayer *me=mdpInfo::mdp->me;
    pol.init_polar(40,target);pol+=me->pos();
    LOG_MOV(3,<< _2D << L2D(me->pos().x,me->pos().y,pol.x,pol.y,"#cccccc"));
#endif
    LOG_POL(3,<< "BS02_Neck: Turning to " <<target<<" ("<<RAD2DEG(target)<<")...");
    return Move_Factory::get_Neck_Turn_Abs(target);
  }
  LOG_POL(1,<<"BS02_Neck: WARNING: No neck turn executed!");
  return Move_Factory::get_Neck_Move();
}

#endif

/****************************************************************************
 * Old Stuff - taken from BS01_Neck and only put here for reference!
 ****************************************************************************/


#if 0
/* This is the old daisy chain of BS01_Neck, only here for reference! */
// highest priority first!

if(mdpInfo::get_my_intention()==DECISION_TYPE_LOOKFORGOALIE)
     look_for_goalie(target);  
     if(is_ball_in_feelrange_next_time())
     ball_in_feelrange(target);  
     if((mdpInfo::get_my_intention() == DECISION_TYPE_INTERCEPTBALL) ||
	(mdpInfo::get_my_intention() == DECISION_TYPE_INTERCEPTSLOWBALL) ||
	(mdpInfo::get_my_intention() == DECISION_TYPE_EXPECT_PASS) )
     intercept_ball(target);
     check_offside(target);
     look_to_ball(target);
     if(mdpInfo::my_distance_to(mdpInfo::opponent_goalpos())<GOAL_NEAR)
     goal_is_near(target);
     if(mdpInfo::my_distance_to_ball()<BALL_NEAR)
     ball_is_near(target);
     
     do_nothing_special(target);  /* nothing to do, so simply scan */
     

/* Normal neck moves */

Angle BS03Neck::look_for_goalie(Angle &target,bool force) {
  if(!force && target>-1) return target;
  Angle toobj=mdpInfo::my_abs_angle_to(mdpInfo::opponent_goalpos());
  int nr=mdpInfo::mdp->his_goalie_number;
  if(nr>=0) {
    int i=mdpInfo::opponent_array_index(nr);
    if(i>=0) {
      if(!force && mdpInfo::age_playerpos(&(mdpInfo::mdp->his_team[i]))<MAX_GOALIE_AGE+1) {
	LOG_MOV(2,<<"BS02_Neck - look_for_goalie: Goalie information newer than "
		<< MAX_GOALIE_AGE+1 << " cycles");
	return -1;
      }
      Angle toobj2=mdpInfo::my_abs_angle_to(mdpInfo::mdp->his_team[i].pos());
      if(mdpInfo::could_see_in_direction(toobj2)) toobj=toobj2;
      if(!mdpInfo::could_see_in_direction(toobj)) {
	LOG_MOV(2,<<"BS02_Neck - look_for_goalie: Cannot look into goalie direction!");
	return -1;
      }
      LOG_MOV(2,<<"BS02_Neck - look_for_goalie: Looking to goalie!");
      target=toobj;return toobj;
    }
  }
  LOG_MOV(2,<<"BS02_Neck - look_for_goalie: Goalie unknown!");
  return -1;
}


Angle BS03Neck::look_to_ball(Angle &target,bool force) {
  if(target>-1) return target;
  if(!mdpInfo::is_ball_pos_valid()) {
    LOG_MOV(2,<<"BS02_Neck - look_to_ball: Ball position not known!");
    return -1;
  }
  if(WSinfo::ball->age<MAX_BALL_AGE+1) {
    LOG_MOV(2,<<"BS02_Neck - look_to_ball: Ball information newer than "
	    << MAX_BALL_AGE+1 << " cycles");
    return -1;
  }  
  Angle toobj=mdpInfo::my_abs_angle_to(mdpInfo::ball_next_time_pos_abs());
  if(!mdpInfo::could_see_in_direction(toobj)) {
    LOG_MOV(2,<<"BS02_Neck - look_to_ball: Cannot turn neck into ball direction!");
    return -1;
  }
  LOG_MOV(2,<<"BS02_Neck - look_to_ball: Looking to ball!");
  target=toobj;return target;  
}

Angle BS03Neck::ball_is_near(Angle &target,bool force) {
  if(target>-1) return target;
  int maxage=MAX_PLAYER_AGE;Angle toobj=-1,akttarget=-1;
  int targetplayer=0;
  for(int i=0;i<11;i++) {
    if(!mdpInfo::mdp->my_team[i].alive) continue;
    if(mdpInfo::mdp->my_team[i].number==mdpInfo::mdp->me->number) continue;
    toobj=mdpInfo::my_abs_angle_to(mdpInfo::mdp->my_team[i].pos());
    if(!mdpInfo::could_see_in_direction(toobj)) continue;
    if(mdpInfo::my_distance_to(mdpInfo::mdp->my_team[i].pos())>MAX_TEAMMATE_DISTANCE) continue;
    if(mdpInfo::age_playerpos(&mdpInfo::mdp->my_team[i])>maxage) {	
      maxage=mdpInfo::age_playerpos(&mdpInfo::mdp->my_team[i]);
      targetplayer=mdpInfo::mdp->my_team[i].number;akttarget=toobj;
    }
  }
  for(int i=0;i<11;i++) {
    if(!mdpInfo::mdp->his_team[i].alive) continue;
    toobj=mdpInfo::my_abs_angle_to(mdpInfo::mdp->his_team[i].pos());
    if(!mdpInfo::could_see_in_direction(toobj)) continue;
     if(mdpInfo::my_distance_to(mdpInfo::mdp->his_team[i].pos())>MAX_OPPONENT_DISTANCE) continue;
    if(mdpInfo::age_playerpos(&mdpInfo::mdp->his_team[i])>maxage) {	
      maxage=mdpInfo::age_playerpos(&mdpInfo::mdp->his_team[i]);
      targetplayer=-mdpInfo::mdp->his_team[i].number;akttarget=toobj;
    }
  }
  if(targetplayer==0) {
    LOG_MOV(2,<<"BS02_Neck - ball_is_near: No players to update available!");
    return -1;
  }
  if(targetplayer>0) { 
    LOG_MOV(2,<<"BS02_Neck - ball_is_near: Looking to teammate #"<<targetplayer<<"!");
  } else {
    LOG_MOV(2,<<"BS02_Neck - ball_is_near: Looking to opponent #"<<-targetplayer<<"!");
  }
  target=akttarget;
  return target;
}

Angle BS03Neck::goal_is_near(Angle &target,bool force) {
  if(target>-1) return target;
  Angle akttarget=-1;
  LOG_MOV(2,<<"BS02_Neck - goal_is_near: Executing look_for_goalie.");
  if(look_for_goalie(akttarget)>-1) {
    target=akttarget;return target;
  }
  return -1;
}

Angle BS03Neck::intercept_ball(Angle &target,bool force) {
  if(target>-1) return target;
  
  if(is_ball_in_feelrange_next_time()) {
    if(mdpInfo::my_distance_to(mdpInfo::opponent_goalpos())<30
       && (!mdpInfo::is_ball_infeelrange()) // always look to goal when getting the ball!
       && (mdpInfo::opponent_goalie_age()!=0)) {
      LOG_MOV(2,<<"BS02_Neck - intercept_ball: Ball is just entering feelrange, looking "
	      <<"to goal!");
      return look_for_goalie(target,true);
    }
       
    LOG_MOV(2,<<"BS02_Neck - intercept_ball: Ball in feelrange, doing nothing");
    return -1;
  }
  if(!mdpInfo::is_ball_pos_valid()) {
    LOG_MOV(2,<<"BS02_Neck - intercept_ball: Ball position not known!");
    return -1;
  }
  int maxage=1;
  double ballvel=WSinfo::ball->vel().norm();
  if(ballvel<1.0) maxage=3;
  else if(ballvel<1.5) maxage=2;
  if(WSinfo::ball->age<maxage) {
    LOG_MOV(2,<<"BS02_Neck - intercept_ball: Ball information newer than "<<maxage<<" cycles!");
    return -1;
  }  
  Angle toobj=mdpInfo::my_abs_angle_to(mdpInfo::ball_next_time_pos_abs());
  if(!mdpInfo::could_see_in_direction(toobj)) {
    LOG_MOV(2,<<"BS02_Neck - intercept_ball: Cannot turn neck into ball direction!");
    return -1;
  }
  LOG_MOV(2,<<"BS02_Neck - intercept_ball: Looking to ball!");
  time_intercept=mdpInfo::mdp->time_current;
  target=toobj;return target;  
}

Angle BS03Neck::ball_in_feelrange(Angle &target,bool force) {
  if(target>-1) return target;

  if(mdpInfo::my_distance_to(mdpInfo::opponent_goalpos())<30
     && (!mdpInfo::is_ball_infeelrange())   // always look to goal when getting the ball!
     && (mdpInfo::opponent_goalie_age!=0)) {
    LOG_MOV(2,<<"BS02_Neck - ball_in_feelrange: Ball is just entering feelrange, looking "
	    <<"to goalie!");
    return look_for_goalie(target,true);
  }

  int attacker=mdpInfo::opponent_closest_to(mdpInfo::my_pos_abs());  
  Vector attackerpos=0;int attackeridx=-1;Angle attackerdir=0;
  if(attacker>0) {
    attackerpos = mdpInfo::opponent_pos_abs(attacker);
    attackeridx = mdpInfo::opponent_array_index(attacker);
    attackerdir = mdpInfo::my_abs_angle_to(attackerpos);
    if(attackeridx<0) {
      LOG_MOV(2,<<"BS02_Neck - ball_in_feelrange: Could not find attacker #"<<attacker
	      <<" in mdpstate!");
      attacker=0;
    }
  }
  else attacker=0;
  
  if(!attacker || mdpInfo::my_distance_to(attackerpos)>ATTACKER_DIST) {
    LOG_MOV(2,<<"BS02_Neck - ball_in_feelrange: No attacker nearby!");
  } else if(mdpInfo::age_playerpos(&mdpInfo::mdp->his_team[attackeridx])<MAX_ATTACKER_AGE+1) {
    LOG_MOV(2,<<"BS02_Neck - ball_in_feelrange: Attacker information newer than "
	     <<MAX_ATTACKER_AGE+1 <<" cycles!");
  } else if(!mdpInfo::could_see_in_direction(attackerdir)) {
    LOG_MOV(2,<<"BS02_Neck - ball_in_feelrange: Cannot turn neck to attacker #"
	    <<attacker<<"!");
  } else {
    LOG_MOV(2,<<"BS02_Neck - ball_in_feelrange: Turning neck to attacker #"<<attacker<<"!");
    target=attackerdir;
    return target;
  }
  if(mdpInfo::my_distance_to(mdpInfo::opponent_goalpos())<GOAL_NEAR) {      
    Angle toobj=mdpInfo::my_abs_angle_to(mdpInfo::opponent_goalpos());
    int nr=mdpInfo::mdp->his_goalie_number;
    if(nr<0) {
      LOG_MOV(2,<<"BS02_Neck - ball_in_feelrange: Goalie unknown!");
      return -1;
    }
    int i=mdpInfo::opponent_array_index(nr);
    if(i<0) {
      LOG_MOV(2,<<"BS02_Neck - ball_in_feelrange: Could not find goalie in mdpstate!");
      return -1;
    }
    if(mdpInfo::age_playerpos(&mdpInfo::mdp->his_team[i])>=MAX_GOALIE_AGE+1) {
      Angle toobj2=mdpInfo::my_abs_angle_to(mdpInfo::mdp->his_team[i].pos());
      if(mdpInfo::could_see_in_direction(toobj2)) toobj=toobj2;  
      if(mdpInfo::could_see_in_direction(toobj)) {
	LOG_MOV(2,<<"BS02_Neck - ball_in_feelrange: Looking to goalie!");
	target=toobj;return target;
      }
    } else {
      LOG_MOV(2,<<"BS02_Neck - ball_in_feelrange: No need to look to goal, so do nothing");
      return -1;
    }
  }
  return -1;
} 

Angle BS03Neck::check_offside(Angle &target,bool force) {
  if(target>-1) return target;

  if(mdpInfo::is_ball_infeelrange() || is_ball_in_feelrange_next_time()) {
    LOG_MOV(2,<<"BS02_Neck - check_offside: I am carrying the ball, no check needed!");
    return -1;
  }
  
  if(mdpInfo::mdp->me->pos_x.v > DeltaPositioning::get_my_offence_line()-OFFENCE_LINE_DIST) {    

    double line=-1000,pline=-1000,posx;
    int ind=-1,pind=-1;
    for(int i=0;i<11;i++) {
      if(!mdpInfo::mdp->his_team[i].alive) continue;
      posx=mdpInfo::mdp->his_team[i].pos_x.v;
      if(posx>pline) {line=pline;ind=pind;pline=posx;pind=i;continue;}
      if(posx>line) {line=posx;ind=i;}
    }
    if(ind<0) return -1;
    //LOG_MOV(2,<<"BS02_Neck - check_offside: Last opponent is #"
    //    <<mdpInfo::mdp->his_team[ind].number<<"!");
    if(mdpInfo::age_playerpos(&mdpInfo::mdp->his_team[ind])<OFFENCE_LINE_AGE+1) {      
      LOG_MOV(2,<<"BS02_Neck - check_offside: Last opponent information newer than "
	      <<OFFENCE_LINE_AGE+1<<" cycles");
      return -1;
    }
    Angle toobj=mdpInfo::my_abs_angle_to(mdpInfo::mdp->his_team[ind].pos());
    if(!mdpInfo::could_see_in_direction(toobj)) {
      LOG_MOV(2,<<"BS02_Neck - check_offside: Cannot turn neck to last opponent!");
      return -1;
    }
    LOG_MOV(2,<<"BS02_Neck - check_offside: Looking to last opponent #"
	    <<mdpInfo::mdp->his_team[ind].number<<"!");
    target=toobj;
    return target;
  }
  return -1;
}
     
Angle BS03Neck::do_nothing_special(Angle &target,bool force) {
  if(target>-1) return target;
  
  if(mdpInfo::my_distance_to(mdpInfo::opponent_goalpos())<30
     && (mdpInfo::opponent_goalie_age()>1)) {
    LOG_MOV(2,<<"BS02_Neck - do_nothing_special: Nothing to do, looking "
	    <<"to goalie!");
    target = look_for_goalie(target,true);
    if(target>-1) return target;
  }
  int oldestval=100000;int oldestrad=-1;int i;
  for(i=0;i<16;i++) {
    Angle ang=i*2.*PI/16;
    if(!mdpInfo::could_see_in_direction(ang)) {
      i++; // don't let the sectors overlap so often [dirty stuff!]
      continue;
    }
    if(radian_last_seen[i]<oldestval) {
      oldestval=radian_last_seen[i];oldestrad=i;
    }
  }
  target=oldestrad*2.*PI/16;
  LOG_MOV(2,<<"BS02_Neck - do_nothing_special: Turning to radian "<<oldestrad
	  << " at "<<RAD2DEG(target)<<", last seen "
	  <<mdpInfo::mdp->time_current-radian_last_seen[oldestrad]<<" cycles ago");
  return target;
}
#endif
