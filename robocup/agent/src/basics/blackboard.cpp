#include "blackboard.h"

Intention Blackboard::main_intention;
Intention Blackboard::pass_intention;
Blackboard::ViewInfo Blackboard::viewInfo;
NeckRequest Blackboard::neckReq;
AttentionToRequest Blackboard::cvAttentionToRequest;
PassRequestRequest Blackboard::cvPassRequestRequest;

bool Blackboard::need_goal_kick;
bool Blackboard::force_highq_view;
bool Blackboard::force_wideang_view;

bool Blackboard::get_pass_info(const int current_time, double &speed, Vector & target, int & target_player_number){
  if (pass_intention.valid_at() != current_time)
    return false;
  return pass_intention.get_kick_info(speed,target, target_player_number);
}

void Blackboard::init() {
  viewInfo.ang_set=-1;
  viewInfo.last_lowq_cycle = -1;
  viewInfo.ball_pos_set = -1;
  force_highq_view=false;
  force_wideang_view=false;
}

void Blackboard::set_next_view_angle_and_quality(int ang,int qty) {
  viewInfo.ang_set = WSinfo::ws->time;
  viewInfo.next_view_angle = ang;
  viewInfo.next_view_qty = qty;
}

bool Blackboard::get_guess_synched() {
	return viewInfo.guess_synched;
}

void Blackboard::set_guess_synched() {
	viewInfo.guess_synched = true;
}

int Blackboard::get_next_view_angle() {
  if(viewInfo.ang_set<WSinfo::ws->time) {
    ERROR_OUT << "WARNING: View behavior has not set next_view_angle!";
    return Cmd_View::VIEW_ANGLE_NARROW;
  }
  return viewInfo.next_view_angle;
}

int Blackboard::get_next_view_quality() {
  if(viewInfo.ang_set<WSinfo::ws->time) {
    ERROR_OUT << "WARNING: View behavior has not set next_view_quality!";
    return Cmd_View::VIEW_QUALITY_HIGH;
  }
  return viewInfo.next_view_qty;
}

void Blackboard::set_last_lowq_cycle(long cyc) {
  viewInfo.last_lowq_cycle = cyc;
}

long Blackboard::get_last_lowq_cycle() {
  return viewInfo.last_lowq_cycle;
}

void Blackboard::set_last_ball_pos(Vector pos) {
  viewInfo.ball_pos_set = WSinfo::ws->time;
  viewInfo.last_ball_pos = pos;
}

Vector Blackboard::get_last_ball_pos() {
  if(viewInfo.ball_pos_set == -1) {
    return Vector(0,0);
  }
  return viewInfo.last_ball_pos;
}

/**********************************************************************/

void Blackboard::set_neck_request(int type, double param) {
  neckReq.set_request(type,param);
}

int Blackboard::get_neck_request(double &param) {
  if(!neckReq.is_set()) return NECK_REQ_NONE;
  param = neckReq.get_param();
  return neckReq.get_type();
}

/**********************************************************************/

void Blackboard::set_attention_to_request(int plNr)
{
  cvAttentionToRequest.set_request(plNr);
}

int Blackboard::get_attention_to_request()
{
  if ( ! cvAttentionToRequest.is_set())
    return ATT_REQ_NONE;
  return cvAttentionToRequest.get_param();
}

//TGpr: begin
void Blackboard::set_pass_request_request(int plNr, int passAngle)
{
  cvPassRequestRequest.set_request(plNr, passAngle);
}

bool Blackboard::get_pass_request_request(int &plNr, int &passParam)
{
  return cvPassRequestRequest.get_param(plNr, passParam);
}
//TGpr: end

//TGdoa: begin
/**********************************************************************/
int  Blackboard::cvAcceptedDirectOpponentAssignment        = -1;
long Blackboard::cvTimeOfAcceptingDirectOpponentAssignment = -1;
bool Blackboard::get_direct_opponent_assignment_accepted( long t, int & a )
{
  if ( t != WSinfo::ws->time ) 
    return false;
  a = cvAcceptedDirectOpponentAssignment;
  return true;
}
void Blackboard::set_direct_opponent_assignment_accepted( int a )
{
  cvAcceptedDirectOpponentAssignment = a;
  cvTimeOfAcceptingDirectOpponentAssignment = WSinfo::ws->time;
}
//TGdoa: end



