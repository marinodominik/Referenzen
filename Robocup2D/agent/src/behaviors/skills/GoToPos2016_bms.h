#ifndef _GOTOPOS2016_BMS_H_
#define _GOTOPOS2016_BMS_H_

#include "../../basics/Cmd.h"
#include "../base_bm.h"
#include "basic_cmd_bms.h"
#include "log_macros.h"

#include "../../basics/tools.h"

class GoToPos2016 : public BodyBehavior
{

private:
    // Static-Attributes
    static bool   cvInitialized;

    static int    cvSpeedFindIteartionCount; // = 8;
    static double cvMaxDiffDefaultVal; // = 0.5 ~ 30 Grad

    // Management-Attributes
    double ivMaxDiff; // = 0.5 ~ 30 Grad
    bool ivAllowAlreadyReachedTarget;
    int  ivAllowAlreadyReachedTargetTimeStep;

    BasicCmd *ivpBasicCmdBehavior;

    Vector   ivTarget;
    int      ivTargetSetAt;
    bool     ivTargetFirstUse;
    double   ivTolerance;
    int      ivEstimatedAt;
    int      ivEstimatedSteps;
    bool     ivUseBackDashes;


    // Calculation-Attributes
    bool   ivOptUseful[3];

    Cmd    ivCmdOpt[3], ivCmdOnFirstRun[3];

    Vector ivPlayerPosOpt[3];
    Vector ivPlayerVelOpt[3];
    Angle  ivPlayerAngOpt[3];
    ANGLE  ivAngToTargetOpt[3];

    double ivTurnRads;
    double ivTurnBackRads;
    double ivMomentToTurnReal;

    Vector ivLastOpt0PlayerPos;

    Vector ivSpeedPos;
    Vector ivDummyVec;
    Angle  ivDummyAng;
    Cmd    ivSpeedCmd;
    double ivSpeedDist[2];
    bool   ivSpeedBefore[2];
    Vector ivSpeedFindPos[3];

    double ivMinSpeed;
    double ivMidSpeed;
    double ivMaxSpeed;

    bool   ivFirstRun;
    bool   ivFound;
    int    ivOptFound;

    Cmd_Body ivCmdToUse;

    //-------------------------------------------------------------------------
    //
    // Idee für eine Caching-Tabelle um das bereits berechnete Werte weg zu
    // speichern, damit man viele Werte vergleichen kann ohne diese immer wieder
    // neu berechnen zu müssen.
    //
    //-------------------------------------------------------------------------

//    class CacheTable
//    {
//    private:
//        class PlayerTable
//        {
//        private:
//            class TargetTable
//            {
//            private:
//                class ToleranceTable
//                {
//                private:
//                    class UseBackDashesTable
//                    {
//                    private:
//                        UseBackDashesTable *next;
//                        int steps;
//                    public:
//                        void insertResult(                                                             int steps /* UND Aktion mit nötigen Parametern */ );
//                    } *toleraceTableHeadRow;
//                public:
//                    void insertResult(                                              bool useBackDashes, int steps /* UND Aktion mit nötigen Parametern */ );
//                } *targetTableHeadRow;
//            public:
//                void insertResult(                                double tolerance, bool useBackDashes, int steps /* UND Aktion mit nötigen Parametern */ );
//            } *playerTableHeadRow;
//        public:
//            void insertResult(              const Vector &target, double tolerance, bool useBackDashes, int steps /* UND Aktion mit nötigen Parametern */ );
//        } *cacheTableHeadRow;
//
//    public:
//        void insertResult( PPlayer &player, const Vector &target, double tolerance, bool useBackDashes, int steps /* UND Aktion mit nötigen Parametern */ );
//
//        bool isAlreadyCached();
//        void clearCacheTable();
//    } cacheTable;

public:

    //Kon-/Destruktor
    GoToPos2016();
    virtual ~GoToPos2016();

    //Schnittstellenmethoden
    static bool init( char const * conf_file, int argc, char const* const* argv );

    void    set_target(                                               const Vector &target );
    void    set_target(                                               const Vector &target,                         bool useBackDashes );
    void    set_target(                                               const Vector &target, double tolerance      , bool useBackDashes = false );
    Vector* get_target(              bool useTargetMoreThanOnceInSameCycle = false );
    bool    is_target_valid(         bool useTargetMoreThanOnceInSameCycle = false );

    bool    get_cmd(       Cmd &cmd );
    bool    get_cmd(       Cmd &cmd, bool useTargetMoreThanOnceInSameCycle );
    bool    get_cmd_go_to( Cmd &cmd,                                  const Vector &target, double tolerance = 1.0, bool useBackDashes = false, bool useTargetMoreThanOnceInSameCycle = false );

    int     estimate_duration(       bool useTargetMoreThanOnceInSameCycle = false );
    int     estimate_duration_to_target(                              const Vector &target, double tolerance = 1.0, bool useBackDashes = false, bool useTargetMoreThanOnceInSameCycle = false );
    int     estimate_duration_from_player_to_target( PPlayer &player, const Vector &target, double tolerance = 1.0, bool useBackDashes = false, bool useTargetMoreThanOnceInSameCycle = false );

    //Customitationmethods
    void    setAngleDiff( double newAngleDiff);
    double  getAngleDiff();
    void    resetAngleDiffToDefault();

    void    setPermissionToGoToAlreadyReachedTarget( bool allow = true );
    bool    getPermissionToGoToAlreadyReachedTarget();
    void    resetPermissionToGoToAlreadyReachedTarget();

    void    resetAllCustomizations();
};

#endif
