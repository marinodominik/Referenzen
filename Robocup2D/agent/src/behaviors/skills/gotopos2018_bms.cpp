#include "gotopos2018_bms.h"

//Klassenvariablen
bool GoToPos2018::cvInitialized = false;

//Konstruktor
GoToPos2018::GoToPos2018()
{
  ivpBasicCmdBehavior = new BasicCmd();
  ivLastTimeTargetHasBeenSet = -1;
  ivTolerance = 1.0;
}

//Destruktor
GoToPos2018::~GoToPos2018()
{
  if (ivpBasicCmdBehavior) delete ivpBasicCmdBehavior;
}

//Klasseninitialisierung
bool
GoToPos2018::init(char const * conf_file, int argc, char const* const* argv)
{
   if (cvInitialized)
     return true;

  bool cvInitialized = BasicCmd::init(conf_file,argc,argv);
  std::cout<<"GoToPos2018: init result: "<<cvInitialized<<std::endl<<std::flush;

  return cvInitialized;
}
//Hauptmethode get_cmd
bool
GoToPos2018::get_cmd( Cmd & cmd )
{
  if (WSinfo::ws->time != ivLastTimeTargetHasBeenSet)
  {
    LOG_POL(0,<<"GoToPos2018: Behavior not usable. Target has not been "
              <<"set (use set_target(.) first!).");
    return false;
  }
  if (WSinfo::me->pos.distance(ivTarget) < ivTolerance)
  {
    LOG_POL(0,<<"GoToPos2018: I have already reached the target point ("<<ivTarget
              <<"). Method get_cmd returns false.");
    return false;
  }
  //##############################################################
  //TODO: sinnvolle Implementierung des GoToPos2018-Verhaltens ...
  //      Aufgabenstellung B1 ...
  //##############################################################
  cmd.cmd_body.set_dash(100.0,0.0);
  return true; //Das ist natuerlich eigentlich nicht ok: Hier wird true
               //zurueckgegeben, obwohl kein (sinnvoller) Befehl gesetzt wurde.
}

//Zielpunkt festlegen
//[Optional (Teilaufgabe 2): Toleranz variieren; Standardwert ist 1.0]
void
GoToPos2018::set_target( Vector target, double tolerance )
{
  LOG_POL(1,<<"GoToPos2018: I am setting my 'go-to-pos' target to "<<target);
  LOG_POL(0,<<_2D<<C2D(target.getX(),target.getY(),1.0,"4444ff"));
  ivLastTimeTargetHasBeenSet = WSinfo::ws->time;
  ivTarget = target;
  ivTolerance = tolerance;
}
