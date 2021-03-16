#include "neuro_kick_wrapper_bms.h"
#define BASELEVEL 0

//============================================================================
//DECLARATION OF STATIC VARIABLES
//============================================================================
bool    NeuroKickWrapper::cvInitialized = false;

//============================================================================
// NeuroKickWrapper::init()
//============================================================================
bool NeuroKickWrapper::init ( char const * confFile, 
                              int          argc, 
                              char const* const* argv) 
{
  if(cvInitialized) 
    return true;
    
  if ( ! NeuroKick05::init(confFile,argc,argv) ) 
  {
    ERROR_OUT 
    << "\nCould not initialize NeuroKick05 behavior - stop loading.";
    exit(1);
  }
  if ( ! NeuroKick08::init(confFile,argc,argv) ) 
  {
    ERROR_OUT 
    << "\nCould not initialize NeuroKick08 behavior - stop loading.";
    exit(1);
  }

  cvInitialized = true;
  return true;
}

//============================================================================
// CONSTRUCTOR
// NeuroKickWrapper::NeuroKickWrapper()
//============================================================================
NeuroKickWrapper::NeuroKickWrapper()
{
  ivpNeuroKick05 = new NeuroKick05();
  ivpNeuroKick08 = new NeuroKick08();
  ivInitInCycle = -1;
}

//============================================================================
// DESTRUCTOR
// NeuroKickWrapper::~NeuroKickWrapper()
//============================================================================
NeuroKickWrapper::~NeuroKickWrapper()
{
  if (ivpNeuroKick05)
    delete ivpNeuroKick05;
  if (ivpNeuroKick08)
    delete ivpNeuroKick08;
}

//============================================================================
// SKILL'S MAIN METHOD
// NeuroKickWrapper::get_cmd()
//============================================================================
bool
NeuroKickWrapper::get_cmd(Cmd & cmd)
{
  if ( !cvInitialized ) 
  {
    ERROR_OUT << "\nNeuroKickWrapper not initialized!";
    return false;
  }
  if ( WSinfo::ws->time != ivInitInCycle) 
  {
    ERROR_OUT << "\nNeuroKickWrapper::get_cmd() called "
      <<"without prior initialization!";
    return false;
  }
  LOG_MOV(0,<<"Starting NeuroKickWrapper behavior (vel: "
    <<ivTargetKickVelocity<<").");

  if ( ivTargetKickVelocity < 2.1 )
    return ivpNeuroKick05->get_cmd( cmd );
  else
    return ivpNeuroKick08->get_cmd( cmd );
}

//============================================================================
// SKILL'S FAKE STATE METHOD
// NeuroKickWrapper::set_state()
//============================================================================
void
NeuroKickWrapper::set_state (const Vector &myPos,
                             const Vector &myVel,
                             const ANGLE  &myAng,
                             const Vector &ballPos,
                             const Vector &ballVel)
{
  ivpNeuroKick05->set_state (myPos, myVel, myAng, ballPos, ballVel);
  ivpNeuroKick08->set_state (myPos, myVel, myAng, ballPos, ballVel);
}

//============================================================================
// SKILL'S INITIALIZATION METHOD
// NeuroKickWrapper::kick_to_pos_with_initial_vel()
//============================================================================
void 
NeuroKickWrapper::kick_to_pos_with_initial_vel  (  double         vel,
                                                   const Vector & pos) 
{
  ivpNeuroKick05->kick_to_pos_with_initial_vel( vel, pos );                  
  ivpNeuroKick08->kick_to_pos_with_initial_vel( vel, pos );     
  ivTargetKickVelocity = vel;             
  ivInitInCycle = WSinfo::ws->time;
}

//============================================================================
// SKILL'S INITIALIZATION METHOD
// NeuroKickWrapper::kick_to_pos_with_final_vel()
//============================================================================
void 
NeuroKickWrapper::kick_to_pos_with_final_vel( double        vel,
                                              const Vector &pos )
{
  ivpNeuroKick05->kick_to_pos_with_final_vel( vel, pos );
  ivpNeuroKick08->kick_to_pos_with_final_vel( vel, pos );
  ivTargetKickVelocity = ivpNeuroKick08->getTargetKickVelocity();
  ivInitInCycle = WSinfo::ws->time;
}

//============================================================================
// SKILL'S INITIALIZATION METHOD
// NeuroKickWrapper::kick_to_pos_with_max_vel()
//============================================================================
void 
NeuroKickWrapper::kick_to_pos_with_max_vel( const Vector &pos ) 
{
  kick_to_pos_with_initial_vel ( ServerOptions::ball_speed_max, pos );
  ivTargetKickVelocity = ServerOptions::ball_speed_max;
}

//============================================================================
// SKILL'S INITIALIZATION METHOD
// NeuroKickWrapper::kick_in_dir_with_max_vel()
//============================================================================
void NeuroKickWrapper::kick_in_dir_with_max_vel(const ANGLE &dir ) 
{
  kick_in_dir_with_initial_vel( ServerOptions::ball_speed_max, dir );
  ivTargetKickVelocity = ServerOptions::ball_speed_max;
}

//============================================================================
// SKILL'S INITIALIZATION METHOD
// NeuroKick08::kick_in_dir_with_initial_vel()
//============================================================================
void NeuroKickWrapper::kick_in_dir_with_initial_vel( double vel,
                                                const ANGLE &dir ) 
{
  ivpNeuroKick05->kick_in_dir_with_initial_vel( vel, dir );
  ivpNeuroKick08->kick_in_dir_with_initial_vel( vel, dir );
  ivTargetKickVelocity = vel;
  ivInitInCycle = WSinfo::ws->time;
}

