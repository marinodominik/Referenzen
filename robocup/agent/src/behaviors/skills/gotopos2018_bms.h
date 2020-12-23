#ifndef _GOTOPOS2018_BMS_H_
#define _GOTOPOS2018_BMS_H_

#include "../base_bm.h"
#include "basic_cmd_bms.h"
#include "Cmd.h"
#include "log_macros.h"

class GoToPos2018 : public BodyBehavior
{
  private:
    static bool cvInitialized;
    BasicCmd *  ivpBasicCmdBehavior;
    Vector      ivTarget;
    long        ivLastTimeTargetHasBeenSet;
    double      ivTolerance;

  public:
    //Kon-/Destruktor
    GoToPos2018();
    virtual ~GoToPos2018();
    //Schnittstellenmethoden
    bool        get_cmd(Cmd &cmd);
    static bool init(char const * conf_file, int argc, char const* const* argv);
    void        set_target( Vector target, double tolerance = 1.0 );
};

#endif
