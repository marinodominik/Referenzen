#include "infprojrss_bmc.h"

#include "ws_info.h"
#include "ws_memory.h"
#include "log_macros.h"
#include "tools.h"
#include "blackboard.h"
#include "skills/Dribble2018.h"

//#############################################################################
// CLASS InfProjRSS
//#############################################################################
//Dieses Verhalten repraesentiert, ein "Komplettverhalten", welches auf
//"oberster Ebene" (d.h. in der Hauptschleife in client.c, die 6000 mal pro
//Spiel durchlaufen wird) aufgerufen wird. Konkret wird einmal pro Zyklus
//die Methode get_cmd (sh.u.) aufgerufen, wo die Aktion fuer den aktuellen
//Zeitschritt ermittelt wird, die dann (ausserhalb dieses Verhaltens) an
//den Soccer Server gesendet wird.

bool InfProjRSS::cvInitialized = false;

//=============================================================================
// KONSTRUKTOR
//=============================================================================
InfProjRSS::InfProjRSS()
{
    //Instanziierung eines anderen Verhaltens, dass im Kontext dieses
    //(Haupt-)Verhaltens verwendet wird. Man koennte natuerlich auch andere
    //Verhalten (z.B. Dribbling) verwenden.
    ivpLineUpBehavior = new LineUp();
    ivpTingleTangleBehavior = TingleTangle::getInstance();
    ivpNeuroDribble2017Behavior = new NeuroDribble2017();
    ivpInterceptBallBehavior = new InterceptBall();
    ivpNeuroKickBehavior = new NeuroKickWrapper();
    ivpOneStepKickBehavior = new OneStepKick();
    ivpSearchBallBehavior = new SearchBall();
    ivpGo2PosBehavior = new GoToPos2016();
    ivpNeuroGo2Pos = new NeuroGo2Pos();
    ivpDribble2018 = new Dribble2018();
    ivState = 0;
}

//=============================================================================
// DESTRUKTOR
//=============================================================================
InfProjRSS::~InfProjRSS()
{
    if (ivpLineUpBehavior) delete ivpLineUpBehavior;
    if (ivpInterceptBallBehavior) delete ivpInterceptBallBehavior;
    if (ivpNeuroDribble2017Behavior) delete ivpNeuroDribble2017Behavior;
    if (ivpNeuroKickBehavior) delete ivpNeuroKickBehavior;
    if (ivpSearchBallBehavior) delete ivpSearchBallBehavior;
}

//=============================================================================
// INITIALISIERUNGSMETHODE
//=============================================================================
bool
InfProjRSS::init(char const * conf_file, int argc, char const* const* argv)
{
    if ( cvInitialized )
        return true;

    //Hier ist erst einmal nicht viel zu tun.
    cvInitialized = true;

    //Alle zu verwendenden Verhalten muessen ebenfalls initialisiert werden.
    bool returnValue =    LineUp::init(conf_file,argc,argv)
                          && TingleTangle::init(conf_file,argc,argv)
                          && InterceptBall::init(conf_file,argc,argv)
                          && SearchBall::init(conf_file,argc,argv)
                          && NeuroKickWrapper::init(conf_file,argc,argv);
    //returnValue =    BasicCmd::init(conf_file,argc,argv)
    //              && OneStepKick::init(conf_file,argc,argv)

    return returnValue;
}


//=============================================================================
// HAUPTMETHODE
//=============================================================================
//
//Die Methode get_cmd ist die zentrale Methode aller im Brainstormers-Framework
//verwendeten Methoden. Ihr wird ein "Kommando-Formular" uebergeben, dass
//diese Methode quasi "ausfuellt"
//
//Das uebergebene Cmd-Objekt enthaelt mehrere Unterobjekte, die zu den
//Controllern korrespondieren (d.h. zum Neck-/View-/AttentionTo-/ und
//natuerlich zum MainController).
//An dieser Stelle ist natuerlich das Main-Unterobjekt von Interesse, weil das
//Hauptkommando (das die Kontrolle ueber den Koerper des Spielers uebernehmen
//soll) gesetzt werden soll (z.B kick, dash, turn ...).
bool
InfProjRSS::get_cmd(Cmd & cmd)
{
    bool cmd_set = false;

    long start_infprojrss = Tools::get_current_ms_time();
    LOG_POL(0,<<"### ENTERED InfProjRSS "
              <<start_infprojrss - WSinfo::ws->ms_time_of_sb
              <<" ms after sense body. Current play mode = "<<WSinfo::ws->play_mode<<" ball "
              <<WSinfo::ball->pos.getX()<<"/"<<WSinfo::ball->pos.getY()<<" ball_v "
              <<WSinfo::ball->vel.getX()<<"/"<<WSinfo::ball->vel.getY());
    LOG_POL(0,<<_2D<<VC2D(WSinfo::ball->pos+WSinfo::ball->vel,0.3,"ffff00"));
    //Im Folgenden wird gemaess des aktuellen Spielmodus (die der Soccer
    //Server bestimmt, z.B. Eckstoß, Einwurf, normales Spiel) verzweigt.
    //Normales Spiel (play_on) ist hierbei natuerlich am interessantesten.

    switch (WSinfo::ws->play_mode)
    {

        case PM_PlayOn: {
            // std::cout<<" test "<<std::endl;


            //ivpInterceptBallBehavior intercept;
            if (   WSinfo::is_ball_kickable()
                   && WSinfo::me->pos.distance(HIS_GOAL_CENTER) < 5.0)
            {
                //Spieler ist mit Ball direkt vor das gegnerische Tor gesetzt worden.
                //Ein Schuss mit voller Staerke nach vorn.
                LOG_POL(0,"InfProjRSS: Coach has beamed me in front of the opponent goal. Get my reward now! :-)");
                cmd.cmd_body.set_kick(100.0, -WSinfo::me->ang);
                cmd_set = true;
                break;
            }

            //Dribble2018 dribble;

            //Vector target =  MY_GOAL_LEFT_CORNER;



           // Vector target = HIS_GOAL_CENTER; //nur als Beispiel
           // target.setX( target.getX() + PENALTY_AREA_LENGTH );
           // target.setY( target.getY() + PENALTY_AREA_WIDTH / 2.0 ); // linke obere Ecke des gegnerischen Strafraums


            //Überprüfe wo der ball liegt, wenn hinter spieler, dann kick ansonsten dash
            Vector ball = WSinfo::ball->pos;
            Vector ballNext = WSinfo::ball->pos+WSinfo::ball->vel*0.8;
            Vector me = WSinfo::me->pos;
            Vector meNext = WSinfo::me->pos+WSinfo::me->vel*0.8;
/*
            ANGLE test;
            test.set_value(DEG2RAD(90));
*/         // dribble.set_value(target);
            Vector target(0, -16);

            if (ivpDribble2018->is_safe()){
                ivpDribble2018->set_target(target);
                //dribble.set_target(target);

                //dribble.get_cmd(cmd);
                ivpDribble2018->get_cmd(cmd);
               // std::cout<<"Command: "<<cmd.cmd_body.check_cmd()<<std::endl;

            }else{

            }



            break;
        }
        case PM_my_KickOff:
        case PM_my_BeforeKickOff:
        case PM_his_BeforeKickOff:
        case PM_my_AfterGoal:
        case PM_his_AfterGoal:
        case PM_Half_Time:
        {
            LOG_POL(1, "InfProjRSS: Standardsituationen und verwandte Spielmodi werden"
                    <<" nicht weiter berücksichtigt: "
                    <<WSinfo::ws->play_mode);
            break;
        }
        case PM_my_FreeKick:
        {
            //nur eine Log-Ausgabe
            LOG_POL(1, "InfProjRSS: Ein Freistoss fuer unser Team.");
            break;
        }
            //Fuer alle folgenden Spielmodi ist in diesem Verhalten nichts
            //ausimplementiert.
        case PM_my_GoalieFreeKick:
        case PM_his_GoalKick:
        case PM_his_GoalieFreeKick:
        case PM_my_GoalKick:
        case PM_my_KickIn:
        case PM_his_KickIn:
        case PM_my_CornerKick:
        case PM_his_CornerKick:
        case PM_his_FreeKick:
        case PM_my_OffSideKick:
        case PM_his_OffSideKick:
        default:
        {
            LOG_POL(1,"InfProjRSS: In der durch die switch-Anweisung realisierten"
                    <<" Fallunterscheidung, sind *nicht* alle Spielmodi unterstuetzt, die"
                    <<" der Soccer Server verwendet.");
            ERROR_OUT << "Time " << WSinfo::ws->time << " player nr. " << WSinfo::me->number
                      << " play_mode is " << WSinfo::ws->play_mode << ". No command was set by behavior!";
            return false;  // behaviour is currently not responsible for that case
        }
    }

    LOG_POL(1,<<"InfProjRSS-Verhalten. Entscheidung benoetigte "
              <<Tools::get_current_ms_time() - start_infprojrss<<" ms"<<std::flush);

    //AB HIER: nur Debugging
    //AUSGABE des Kommandos, fuer das sich entschieden wurde.
    Angle angle;
    double power, x, y;
    switch( cmd.cmd_body.get_type())
    {
        case Cmd_Body::TYPE_KICK:
            cmd.cmd_body.get_kick(power, angle);
            LOG_POL(0, << "infprojrss_bmc.c: cmd KICK, power "<<power<<", angle "<< RAD2DEG(angle) );
            break;
        case Cmd_Body::TYPE_TURN:
            cmd.cmd_body.get_turn(angle);
            LOG_POL(0, << "infprojrss_bmc.c: cmd Turn, angle "<< RAD2DEG(angle) );
            break;
        case Cmd_Body::TYPE_DASH:
            cmd.cmd_body.get_dash(power,angle);
            LOG_POL(0, << "infprojrss_bmc.c: cmd DASH, power "<< power <<", angle "<<RAD2DEG(angle));
            break;
        case Cmd_Body::TYPE_CATCH:
            cmd.cmd_body.get_catch(angle);
            LOG_POL(0, << "infprojrss_bmc.c: cmd Catch, angle "<< RAD2DEG(angle) );
            break;
        case Cmd_Body::TYPE_TACKLE:
            cmd.cmd_body.get_tackle(power, angle);
            LOG_POL(0, << "infprojrss_bmc.c: cmd Tackle, power "<< power << ", angle " << RAD2DEG(angle) );
            break;
        case Cmd_Body::TYPE_MOVETO:
            cmd.cmd_body.get_moveto(x, y);
            LOG_POL(0, << "infprojrss_bmc.c: cmd Moveto, target "<< x << " " << y );
        default:
            LOG_POL(0, << "infprojrss_bmc.c: No CMD was set " << std::flush);
    }
    //Linie von mir zum Ball
    LOG_POL(0,<<_2D<<L2D(WSinfo::me->pos.getX(),WSinfo::me->pos.getY(),WSinfo::ball->pos.getX(),WSinfo::ball->pos.getY(),"5555ff"));
    //Rueckgabe (true oder false), ob ein Kommando erfolgreich
    //gesetzt werden konnte.
    LOG_POL(0,<<"### LEAVING InfProjRSS ");
    return cmd_set;
}
