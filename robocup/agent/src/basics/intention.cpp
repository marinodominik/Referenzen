#include "intention.h"
#include "globaldef.h"

#include "angle.h"

#include "ws_info.h"

void Intention::copyFromIntention( Intention &intn )
{
    subtype           = intn.subtype;
    valid_since_cycle = intn.valid_since_cycle;
    valid_at_cycle    = intn.valid_at_cycle;
    V                 = intn.V;
    risky_pass        = intn.risky_pass;
    immediatePass     = intn.immediatePass;
    type              = intn.type;
    target_player     = intn.target_player;
    priority          = intn.priority;
    resultingpos      = intn.resultingpos;
    target_body_dir.set_value( intn.target_body_dir.get_value_0_p2PI() );
    kick_speed        = intn.kick_speed;
    kick_dir          = intn.kick_dir;
    kick_target       = intn.kick_target;
    player_target     = intn.player_target;
    potential_pos     = intn.potential_pos;
    attacker_num      = intn.attacker_num;
    wait_then_pass    = intn.wait_then_pass;
    advantage         = intn.advantage;
}

bool Intention::is_pass2teammate(){
  if(type == PASS)
    return true;
  if(type == LAUFPASS)
    return true;
  if(type == PANIC_KICK)
    return true;
  if(type == KICKNRUSH)
    return true;
  return false;
}

bool Intention::is_selfpass(){
  if(type == SELFPASS)
    return true;
  if(type == IMMEDIATE_SELFPASS)
    return true;
  return false;
}

bool Intention::is_dribble(){
  if(type == DRIBBLE)
    return true;
  return false;
}

void Intention::confirm_intention(const Intention intention, const int valid_at){
  valid_at_cycle = valid_at; // update
  valid_since_cycle = intention.valid_since_cycle; 
  kick_speed = intention.kick_speed;
  kick_target = intention.kick_target;
  type = intention.type;
  player_target = intention.player_target;
  target_body_dir = intention.target_body_dir;
  target_player = intention.target_player;
  risky_pass = intention.risky_pass;
  potential_pos = intention.potential_pos;
  kick_dir = intention.kick_dir;
  attacker_num = intention.attacker_num;
  resultingpos = intention.resultingpos;
  wait_then_pass = intention.wait_then_pass;
  subtype = intention.subtype;
  advantage = intention.advantage;
}

void Intention::set_valid_since(const long cycle){
  valid_since_cycle = cycle;
}

void Intention::set_valid_at(const long cycle){
  valid_at_cycle = cycle;
}

long Intention::valid_since(){
  if(valid_at_cycle >0) 
    return valid_since_cycle;
  else
    return 999999;  // a large value to indicate that its not valid
}



void Intention::reset(){
  subtype = SUBTYPE_NONE;
  V=0;
  valid_at_cycle = 0; 
  valid_since_cycle = 999999;
  kick_speed = 0;
  kick_target = Vector(0);
  type = 0;
  player_target = Vector(0);
  target_body_dir = ANGLE(0);
  target_player = 0;
  risky_pass = false;
  potential_pos = Vector(0);
  attacker_num = 0;
  resultingpos = Vector(0);
  kick_dir = 0;
  wait_then_pass = false;
  advantage = 0;
}

void Intention::set_goto(const Vector &target, const int valid_at) {
  type = GOTO;
  player_target = target;
  valid_at_cycle = valid_at;
  valid_since_cycle = valid_at;
}

void Intention::set_score(const Vector &target, const double &speed, const int valid_at){
  type = SCORE;
  kick_speed = speed;
  kick_target = target;
  valid_at_cycle = valid_at;
  valid_since_cycle = valid_at;
}

void Intention::set_pass(const Vector &target, const double &speed, const int valid_at,
			 int target_player_number,const double tmp_priority, const Vector ipos,
			 const   Vector potential_position ){
  type = PASS;
  kick_speed = speed;
  kick_target = target;
  valid_at_cycle = valid_at;
  valid_since_cycle = valid_at;
  target_player= target_player_number;
  resultingpos= ipos;
  potential_pos = potential_position;
  priority = tmp_priority;
  wait_then_pass = false;
}


void Intention::set_laufpass(const Vector &target, const double &speed, const int valid_at,
			     int target_player_number,const double tmp_priority,
			     const Vector ipos,
			     const bool is_risky,const   Vector potential_position ){
  type = LAUFPASS;
  kick_speed = speed;
  kick_target = target;
  valid_at_cycle = valid_at;
  valid_since_cycle = valid_at;
  target_player= target_player_number;
  resultingpos = ipos;
  priority = tmp_priority;
  risky_pass = is_risky;
  potential_pos = potential_position;
  wait_then_pass = false;
}

void Intention::set_pass(const Vector &target, const double &speed, const double &kickdir, const int valid_at,
			 int target_player_number,const double tmp_priority, const Vector ipos,
			 const   Vector potential_position ){
  type = PASS;
  kick_speed = speed;
  kick_target = target;
  kick_dir = kickdir;
  valid_at_cycle = valid_at;
  valid_since_cycle = valid_at;
  target_player= target_player_number;
  resultingpos= ipos;
  potential_pos = potential_position;
  priority = tmp_priority;
  wait_then_pass = false;
}


void Intention::set_laufpass(const Vector &target, const double &speed, const double &kickdir, const int valid_at,
			     int target_player_number,const double tmp_priority,
			     const Vector ipos,
			     const bool is_risky,const   Vector potential_position ){
  type = LAUFPASS;
  kick_speed = speed;
  kick_target = target;
  kick_dir = kickdir;
  valid_at_cycle = valid_at;
  valid_since_cycle = valid_at;
  target_player= target_player_number;
  resultingpos = ipos;
  priority = tmp_priority;
  risky_pass = is_risky;
  potential_pos = potential_position;
  wait_then_pass = false;
}


void Intention::set_opening_seq(const Vector &target, const double &speed, const int valid_at, int target_player_number){
  type = OPENING_SEQ;
  kick_speed = speed;
  kick_target = target;
  valid_at_cycle = valid_at;
  valid_since_cycle = valid_at;
  target_player= target_player_number;
}

void Intention::set_selfpass(const ANGLE & targetdir, const Vector &target, 
			     const double &speed, const int valid_at){
  type = SELFPASS;
  target_body_dir = targetdir;
  kick_speed = speed;
  kick_target = target;
  valid_since_cycle = valid_at;
  valid_at_cycle = valid_at;
}


void Intention::set_selfpass(const ANGLE & targetdir, const Vector &target, const double &speed, const int valid_at, const int attackernum){
  set_selfpass(targetdir, target, speed, valid_at);
  attacker_num = attackernum;
}

/*
void Intention::set_selfpass(const ANGLE & target_dir, const Vector &target, 
			     const double &speed, const int &attacker_num, const int &evaluation,
			     const Vector &resulting_pos, const int valid_at){
  type = SELFPASS;
  target_body_dir = targetdir;
  kick_speed = speed;
  kick_target = target;
  valid_since_cycle = valid_at;
  valid_at_cycle = valid_at;

  attacker_num = attackernum;
  V = evaluation;
  resultingpos = resulting_pos;
}
*/


void Intention::set_immediateselfpass(const Vector &target, const double &speed, const int valid_at){
  type = IMMEDIATE_SELFPASS;
  kick_speed = speed;
  kick_target = target;
  valid_since_cycle = valid_at;
  valid_at_cycle = valid_at;
}


void Intention::set_tingletangle(const Vector &target, const int valid_at){
  type = TINGLETANGLE;
  player_target = target;
  valid_since_cycle = valid_at;
  valid_at_cycle = valid_at;
}
void Intention::set_dribble(const Vector &target, const int valid_at){
  type = DRIBBLE;
  player_target = target;
  valid_since_cycle = valid_at;
  valid_at_cycle = valid_at;
}

void Intention::set_dribblequickly(const Vector &target, const int valid_at){
  type = DRIBBLE_QUICKLY;
  player_target = target;
  valid_since_cycle = valid_at;
  valid_at_cycle = valid_at;
}


void Intention::set_kicknrush(const Vector &target, const double &speed, const int valid_at){
  type = KICKNRUSH;
  kick_speed = speed;
  kick_target = target;
  valid_since_cycle = valid_at;
  valid_at_cycle = valid_at;
}

void Intention::set_panic_kick(const Vector &target, const int valid_at){
  type = PANIC_KICK;
  kick_target = target;
  valid_since_cycle = valid_at;
  valid_at_cycle = valid_at;
}

void Intention::set_tackling( const double &speed, const int valid_at )
{
  type = TACKLING;
  kick_speed = speed;
  valid_since_cycle = valid_at;
  valid_at_cycle = valid_at;
}

void Intention::set_backup(const int valid_at){
  type = BACKUP;
  valid_since_cycle = valid_at;
  valid_at_cycle = valid_at;
}

void Intention::set_turnanddash(const int valid_at){
  type = TURN_AND_DASH;
  valid_since_cycle = valid_at;
  valid_at_cycle = valid_at;
}

void Intention::set_waitandsee(const int valid_at){
  type = WAITANDSEE;
  valid_since_cycle = valid_at;
  valid_at_cycle = valid_at;
}

void Intention::set_holdturn(const ANGLE body_dir, const int valid_at){
  type = HOLDTURN;
  target_body_dir = body_dir;
  valid_since_cycle = valid_at;
  valid_at_cycle = valid_at;
}

int Intention::get_type(){
  return type;
}

bool Intention::get_kick_info(double &speed, Vector &target, int & target_player_number){
  if(type <SELFPASS || type > PANIC_KICK){
    speed  = 0.0;
    target = Vector(0);
    return false;
  }
  speed = kick_speed;
  target = kick_target;
  target_player_number= target_player;
  return true;
}

/********************************************************************/

NeckRequest::NeckRequest()
{
    valid_at_cycle = -1;
}

void NeckRequest::set_request(int t,double p) {
  valid_at_cycle = WSinfo::ws->time;
  type = t;
  p0 = p;
}

int NeckRequest::get_type() {
  if(valid_at_cycle != WSinfo::ws->time) return NECK_REQ_NONE;
  return type;
}

double NeckRequest::get_param() {
  if(valid_at_cycle != WSinfo::ws->time) {
return 0;
}
  return p0;
}

bool NeckRequest::is_set() {
  return (valid_at_cycle == WSinfo::ws->time);
}

/*******************************************************************/

AttentionToRequest::AttentionToRequest()
{
    ivValidAtCycle = -1;
}

void AttentionToRequest::set_request(int plNr)
{
  ivValidAtCycle = WSinfo::ws->time;
  ivAttentionToPlayer = plNr;
}

int AttentionToRequest::get_param()
{
  if (ivValidAtCycle != WSinfo::ws->time)
    return -1;
  return ivAttentionToPlayer;
}

bool AttentionToRequest::is_set()
{
  return (ivValidAtCycle == WSinfo::ws->time);
}

/*******************************************************************/

//TGpr: begin

PassRequestRequest::PassRequestRequest()
{
    ivValidAtCycle = -1;
}

void PassRequestRequest::set_request(int plNr, int passAngleParameter)
{
  ivValidAtCycle = WSinfo::ws->time;
  ivRequestPassFromPlayer = plNr;
  ivRequestPassInDirection = passAngleParameter;
}

bool PassRequestRequest::get_param(int &plNr, int &passAngleParameter)
{
  if (ivValidAtCycle != WSinfo::ws->time)
    return false;
  plNr = ivRequestPassFromPlayer;
  passAngleParameter = ivRequestPassInDirection;
  return true;
}

bool PassRequestRequest::is_set()
{
  return (ivValidAtCycle == WSinfo::ws->time);
}
//TGpr: end


