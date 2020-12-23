#ifndef _TRAIN_SKILLS_BMC_H_
#define _TRAIN_SKILLS_BMC_H_

#include "base_bm.h"

#include "skills/neuro_go2pos_bms.h"
#include "skills/neuro_intercept_bms.h"
#include "skills/neuro_kick_bms.h"

class TrainSkillsPlayer : public BodyBehavior
{
    static bool initialized;

    NeuroGo2Pos    *go2pos;
    NeuroIntercept *neurointercept;
    NeuroKick      *neurokick;

    bool playon( Cmd & cmd );

public:
    TrainSkillsPlayer();
    virtual ~TrainSkillsPlayer();

    static bool init( char const *conf_file, int argc, char const* const* argv );

    bool get_cmd( Cmd & cmd );
};

#endif
