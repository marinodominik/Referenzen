#include "test_skills_bmc.h"

#include "ws_info.h"
#include "log_macros.h"

bool TestSkillsPlayer::initialized = false;

TestSkillsPlayer::TestSkillsPlayer()
{
    go2pos = new NeuroGo2Pos;
}

TestSkillsPlayer::~TestSkillsPlayer()
{

}

bool TestSkillsPlayer::init( char const *conf_file, int argc, char const* const* argv )
{
    if( initialized ) return initialized;

    initialized = NeuroGo2Pos::init( conf_file, argc, argv );

    return initialized;
}

bool TestSkillsPlayer::get_cmd( Cmd & cmd )
{
    switch( WSinfo::ws->play_mode )
    {
        case PM_PlayOn :
            //cmd.cmd_main.set_dash(100);
            go2pos->set_target( Vector( 0, 0 ) );
            return go2pos->get_cmd( cmd );
            break;
        default :
            cmd.cmd_body.set_turn( 1.57 );
    }

    LOG_DEB( 0, << "now in cycle : " << WSinfo::ws->time );

    LOG_DEB( 0, << _2D << VC2D(WSinfo::me->pos, 2, "ff0000") );
    if( WSinfo::ws->time % 100 == 1 )
    {
        std::cout << "\nnow in cycle " << WSinfo::ws->time << std::flush;
    }

    return true;
}
