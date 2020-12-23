#ifndef _TEST_SKILLS_BMC_H_
#define _TEST_SKILLS_BMC_H_

#include "base_bm.h"
#include "skills/neuro_go2pos_bms.h"

class TestSkillsPlayer : public BodyBehavior
{

private:
    static bool initialized;
    NeuroGo2Pos *go2pos;

public:
    TestSkillsPlayer();
    virtual ~TestSkillsPlayer();
    static bool init( char const * conf_file, int argc, char const* const * argv );
    bool get_cmd( Cmd & cmd );
};

#endif
