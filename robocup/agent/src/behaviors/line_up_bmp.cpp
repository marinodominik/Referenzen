#include "line_up_bmp.h"
//#include "mdp_info.h"
#include <stdlib.h>
#include <stdio.h>
#include "valueparser.h"
#include "options.h"
#include "../Agent.h"
#include "../basics/tools.h"
#include "../basics/ws_info.h"

bool LineUp::initialized = false;

LineUp::LineUp() {
  //char key_kick_off[50];

  // open config file
  //ValueParser parser( CommandLineOptions::policy_conf, "Line_Up_Policy" );
  //parser.set_verbose(true);

  /*
  // read home positions
  for(int i=1;i<=11;i++){
    sprintf(key_kick_off, "home_kick_off_%d",i);

    parser.get(key_kick_off, home_kick_off[i], 2);
  }
  */

  basic_cmd = new BasicCmd();
  go2pos = new NeuroGo2Pos();
  face_ball = new FaceBall();

  ivSectorAngle = ANGLE(0);

}

LineUp::~LineUp() {
  delete basic_cmd;
  delete go2pos;
  delete face_ball;
}

bool LineUp::get_cmd(Cmd &cmd){
  
  if ( !ClientOptions::consider_goalie ) {
    /* Daniel
    int temp_matching = DeltaPositioning::get_matching();
    DeltaPositioning::select_matching(0);
    DeltaPositioning::select_formation(1);
    DeltaPositioning::update();
    */
    Vector my_form = DeltaPositioning::get_position_base(WSinfo::me->number);
    double delta = 1.0;
    adaptMyFormPosition( my_form, delta );
    //D DeltaPositioning::select_matching(temp_matching);
    if ( (WSinfo::me->pos - my_form).sqr_norm() < delta ) {
LOG_POL(0,"LINEUP: USE FACEBALL");
      face_ball->turn_to_ball();
      return face_ball->get_cmd(cmd);
    } else {
LOG_POL(0,"LINEUP: USE MOVETO -> my_form="<<my_form);
      basic_cmd->set_move(my_form);
      return basic_cmd->get_cmd(cmd);
    }
  }
  //line up goalie 
  Vector home_pos = Vector(-50.5, 0.0);//Vector(home_kick_off[WSinfo::me->number][0], home_kick_off[WSinfo::me->number][1]);
  Vector pos;
  pos.clone( WSinfo::me->pos );

  double sqr_target_accuracy = 1.0;

  if ( !WSinfo::is_my_pos_valid() ) {
    double moment = DEG2RAD(40.0);//mdpInfo::view_angle_width_rad();
    INFO_OUT << "\nPlayer_no= " << ClientOptions::player_no << " my pos is not VALID";
    basic_cmd->set_turn(moment);
    return basic_cmd->get_cmd(cmd);
  }
  
  if( (pos - home_pos).sqr_norm() <= sqr_target_accuracy ) {
    face_ball->turn_to_ball();
    return face_ball->get_cmd(cmd);
  }
  else { 
    switch (WSinfo::ws->play_mode) {
    case PM_my_BeforeKickOff:  
    case PM_his_BeforeKickOff: 
    case PM_my_AfterGoal:	   
    case PM_his_AfterGoal:	   
    case PM_Half_Time:
      basic_cmd->set_move(home_pos);
      return basic_cmd->get_cmd(cmd);
      break;
    default:
      go2pos->set_target(home_pos, 0.7, 0);
      return go2pos->get_cmd(cmd);
    }
  }
}

bool LineUp::get_cmd(Cmd &cmd, bool sectorBased)
{
LOG_POL(0,"LINEUP: INFOS myPos="<<WSinfo::me->pos<<" ballPos="<<WSinfo::ball->pos<<" ballDist="<<WSinfo::me->pos.distance(WSinfo::ball->pos)<<" pm="<<WSinfo::ws->play_mode);
  Vector my_form = DeltaPositioning::get_position_base(WSinfo::me->number);
  double delta = 1.0;
  adaptMyFormPosition( my_form, delta );
  if (    (WSinfo::me->pos - my_form).sqr_norm() > delta
       || WSinfo::ball->age > 2 ) 
    return get_cmd(cmd);
  if (!sectorBased)
    return get_cmd(cmd);
  ANGLE nextWidth = Tools::next_view_angle_width();
  if (WSinfo::ws->time_of_last_update == WSinfo::ws->time)
    ivSectorAngle += nextWidth;
  ANGLE turnAngle = ivSectorAngle - WSinfo::me->ang;
  basic_cmd->set_turn_inertia( turnAngle.get_value_0_p2PI() );
  return basic_cmd->get_cmd(cmd);
}

void LineUp::adaptMyFormPosition( Vector& p, double& delta )
{
  LOG_POL(4,"LINEUP: initial my_form="<<p);
  if (WSinfo::ws->time > 50 && WSinfo::ws->play_mode == PM_his_AfterGoal)
  {
    if (WSinfo::me->number==2) {p.mulYby( 0.5 ); p.subFromX( 0.0 ); }
    if (WSinfo::me->number==3) {p.setY( 0.0 );   p.subFromX( 0.0 ); }
    if (WSinfo::me->number==4) {p.setY( 0.0 );   p.subFromX( 5.0 ); }
    if (WSinfo::me->number==5) {p.mulYby( 0.5 ); p.subFromX( 0.0 ); }
    if (WSinfo::me->number==6) {                 p.subFromX( 0.0 ); }
    if (WSinfo::me->number==7) {                 p.subFromX( 0.0 ); }
    if (WSinfo::me->number==8) {                 p.subFromX( 0.0 ); }
    LOG_POL(4,"LINEUP: adaptation for t>50 (2,3,4,5,6,7,8) @PM_my_AfterGoal -> my_form="<<p);
  }
  if (WSinfo::ws->time > 50 && WSinfo::ws->play_mode == PM_my_AfterGoal)
  {
    if (WSinfo::me->number==2) {p.mulYby( 0.5 ); p.subFromX(  5.0 ); }
    if (WSinfo::me->number==3) {p.setY( 0.0 );   p.subFromX(  5.0 ); }
    if (WSinfo::me->number==4) {p.setY( 0.0 );   p.subFromX( 10.0 ); }
    if (WSinfo::me->number==5) {p.mulYby( 0.5 ); p.subFromX(  5.0 ); }
    if (WSinfo::me->number==6) {                 p.subFromX(  5.0 ); }
    if (WSinfo::me->number==7) {                 p.subFromX(  5.0 ); }
    if (WSinfo::me->number==8) {                 p.subFromX(  5.0 ); }
    LOG_POL(4,"LINEUP: adaptation for t>50 (2,3,4,5,6,7,8) @PM_his_AfterGoal -> my_form="<<p);
  }
  if (   WSinfo::last_cmd->cmd_body.get_type() == WSinfo::last_cmd->cmd_body.TYPE_DASH
      && fabs(WSinfo::me->ang.get_value_mPI_pPI()) < PI*(60.0/180.0) )
  {
    p.subFromX( 2.0 );
    LOG_POL(4,"LINEUP: adaptation due to last DASH forward -> my_form="<<p);
  }
  if (p.getX() > 0.0)
  {
    p.setX( -0.5 ); //added by TG
    LOG_POL(4,"LINEUP: adaptation due to x>0 -> my_form="<<p);
  }
  if ( p.getX() + WSinfo::me->vel.getX() > 0.0)
  {
      p.setX( -1.0 ); //added by TG
      LOG_POL(4,"LINEUP: adaptation due to sliding over x=0 -> my_form="<<p);
  }
  if ( WSinfo::me->number == 10 ) //TG16
  {
    if (WSinfo::ws->play_mode == PM_my_AfterGoal)
      p.setX( -9.5 );
    else
      p.setX( -0.45 );
    p.setY( 0.0 );
    delta = 0.05;
    LOG_POL(4,"LINEUP: adaptation for #10 with delta=0.1 -> my_form="<<p);
  }
}
