#ifndef _CMD_H_
#define _CMD_H_

#include "globaldef.h"
#include "Vector.h"
#include "angle.h"
#include "macro_msg.h"
#include "comm_msg.h"
#include <iostream> 

#include "PlayerSet.h"

/** 
Die Klasse Cmd_Base dient als Basisklasse fuer die Klassen:

Cmd_Body, Cmd_Neck, Cmd_View, Cmd_Say, Cmd_Point

Cmd_Base implementiert einen Standardmechanismus fuer diese
Kommando-Klassen. Dieser besteht darin, dass man ein Kommando sperren und
setzen kann. 

Fuer den Anwender ist die Klasse Cmd von Bedeutung. Sie enthaelt im wesentlichen 
vier Felder, die den entsprechnden Kommandoklassen entsprechen.

Struktur:

                      Cmd_Base

                         |
      __________________/ \____________________________
     |             |           |            |          |

  Cmd_Body    Cmd_Neck    Cmd_View     Cmd_Say      Cmd_Point


und 

                        Cmd
 
              enthaelt je ein Objekt vom Typ:
              
              Cmd_Body
              Cmd_Neck
              Cmd_View
              Cmd_Say
              Cmd_Point

 
Autor: ART!
*/

// |----------------------------------------------------------------|
// |  ,-----.             ,--.      ,-----.                         |
// | '  .--./,--,--,--. ,-|  |      |  |) /_  ,--,--. ,---.  ,---.  |
// | |  |    |        |' .-. |      |  .-.  \' ,-.  |(  .-' | .-. : |
// | '  '--'\|  |  |  |\ `-' |,----.|  '--' /\ '-'  |.-'  `)\   --. |
// |  `-----'`--`--`--' `---' '----'`------'  `--`--'`----'  `----' |
// |                                                                |
// |----------------------------------------------------------------|
class Cmd_Base
{

protected:

    /// Interner Schalter: Sperre (ein/aus)
    bool lock_set;

    /// Interner Schalter: Kommando (gesetzt/nicht gesetzt)
    bool cmd_set;


public:

    ///Initialzustand ist immer : (nicht gesperrt) und (Kommando nicht gesetzt)
    Cmd_Base();

    /// setze Sperre, muss meistens nicht explizit aufgerufen werden, da das
    /// Setzen von Kommandos in den abgeleiteten Klassen automatisch eine Sperre erzeugt
    bool set_lock();
    /// entferne Sperre, wird evtl. nicht gebraucht
    bool unset_lock();
    ///Pruefe ob Objekt gesperrt ist
    bool is_lock_set() const;
    /// Fehlermeldung falls auf ein gesperrtes Objekt geschrieben wird
    bool check_lock() const;

    /// setze Kommando, muss meistens nicht explizit aufgerufen werden, da das
    /// Setzen von Kommandos in den abgeleiteten Klassen automatisch erfolgt
    bool set_cmd();
    /// Falls ein gesetztes Kommando doch nicht beruecksichtigt werden soll, kann 
    /// explizit mit dieser Methode tun
    bool unset_cmd();
    ///Pruefe ob ein Kommando gesetzt ist, nur bei Ergebnis true ist in den
    ///abgeleitete Klassen das Auslesen der Werte sinnvoll.
    bool is_cmd_set() const;
    /// Fehlermeldung falls kein Kommando gesetzt wurde
    bool check_cmd() const;

};

/** 

Kommandos die vom Spieler im Spiel abgesetzt werden koennen. Es darf jeweils
nur ein Kommando gesetzt sein. Nach dem Setzen eines Kommandos wird automatisch 
eine Sperre gesetzt.

Winkel und Koordinaten werden entsprechend den Konventionen aus

mdpstate.h   <- siehe Doku. in dieser Datei

behandelt.
*/
// |-----------------------------------------------------------------|
// |  ,-----.             ,--.      ,-----.            ,--.          |
// | '  .--./,--,--,--. ,-|  |      |  |) /_  ,---.  ,-|  |,--. ,--. |
// | |  |    |        |' .-. |      |  .-.  \| .-. |' .-. | \  '  /  |
// | '  '--'\|  |  |  |\ `-' |,----.|  '--' /' '-' '\ `-' |  \   '   |
// |  `-----'`--`--`--' `---' '----'`------'  `---'  `---' .-'  /    |
// |                                                       `---'     |
// |-----------------------------------------------------------------|
class Cmd_Body:public Cmd_Base
{

private:

    int    type;
    double param_1;
    double param_2;
    ANGLE  param_ang;
    int    priority;

    void check_type( int t ) const;


public:

  ///Typen von Kommandos
    const static int TYPE_NONE     = 0;
    const static int TYPE_MOVETO   = 1;
    const static int TYPE_TURN     = 2;
    const static int TYPE_DASH     = 3;
    const static int TYPE_KICK     = 4;
    const static int TYPE_CATCH    = 5;
    const static int TYPE_TACKLE   = 6;

    Cmd_Body();

    bool clone( Cmd_Body const &cmd );

    int  get_type() const;
    bool is_type( int type );
    int  get_type(   double &power, Angle &angle );
    int  get_type(   double &power, ANGLE &angle );
    int  get_priority() const;

    void set_none();

    void set_moveto( double  x, double  y );
    void get_moveto( double &x, double &y ) const;

    void set_dash(   double  power, int p_priority = 0 );
    void set_dash(   double  power, Angle  angle, int p_priority = 0 );
    void set_dash(   double  power, ANGLE  angle, int p_priority = 0 ); //TG09
    void get_dash(   double &power ) const;
    void get_dash(   double &power, Angle &angle ) const;
    void get_dash(   double &power, ANGLE &angle ) const;

    void set_turn(   Angle  angle );
    void set_turn(   ANGLE const &angle );
    void get_turn(   Angle &angle ) const;
    void get_turn(   ANGLE &angle ) const;

    /**
     * Set tackle as next command
	 *
     * @param angle [in] define kick angle [-180,180] (prior versions of
	 *					 soccer server (<12) interpret this parameter as
	 * 					 power [-100,100]
     */
    void set_tackle (double  angle);
    /**
     * Set tackle as next command
	 *
     * @param angle [in] define kick angle [-180,180] (prior versions of
	 *					 soccer server (<12) interpret this parameter as
	 * 					 power [-100,100]
     * @param foul [in]  foul or ball tackle
     */
    void set_tackle( double  angle, bool foul);
    void set_tackle( ANGLE   angle, bool foul);
    void get_tackle( double &angle, double &foul) const;
    void get_tackle( ANGLE  &angle, double &foul) const;

    void set_kick(   double  power, Angle  angle );
    void set_kick(   double  power, ANGLE  angle );
    void get_kick(   double &power, Angle &angle ) const;
    void get_kick(   double &power, ANGLE &angle ) const;

    void set_catch(  Angle  angle );
    void set_catch(  ANGLE  angle );
    void get_catch(  Angle &angle ) const;
    void get_catch(  ANGLE &angle ) const;


    void check_lock() const;

    friend std::ostream& operator<<( std::ostream &o, const Cmd_Body &cmd );

    void reset();

};

// |----------------------------------------------------------------|
// |  ,-----.             ,--.      ,--.  ,--.             ,--.     |
// | '  .--./,--,--,--. ,-|  |      |  ,'.|  | ,---.  ,---.|  |,-.  |
// | |  |    |        |' .-. |      |  |' '  || .-. :| .--'|     /  |
// | '  '--'\|  |  |  |\ `-' |,----.|  | `   |\   --.\ `--.|  \  \  |
// |  `-----'`--`--`--' `---' '----'`--'  `--' `----' `---'`--'`--' |
// |                                                                |
// |----------------------------------------------------------------|
class Cmd_Neck: public Cmd_Base
{

private:

    ANGLE param_ang;


public:

    Cmd_Neck();


    void set_turn( Angle  angle );
    void set_turn( ANGLE  angle );
    void get_turn( Angle &angle ) const;
    void get_turn( ANGLE &angle ) const;


    friend std::ostream& operator<<( std::ostream &o, const Cmd_Neck &cmd );

    void reset();

};

// |---------------------------------------------------------------|
// |  ,-----.             ,--.   ,--.   ,--.,--.                   |
// | '  .--./,--,--,--. ,-|  |    \  `.'  / `--' ,---. ,--.   ,--. |
// | |  |    |        |' .-. |     \     /  ,--.| .-. :|  |.'.|  | |
// | '  '--'\|  |  |  |\ `-' |,----.\   /   |  |\   --.|   .'.   | |
// |  `-----'`--`--`--' `---' '----' `-'    `--' `----''--'   '--' |
// |                                                               |
// |---------------------------------------------------------------|
class Cmd_View: public Cmd_Base
{

private:

    int view_angle;
    int view_quality;


public:

    const static int VIEW_ANGLE_WIDE   = WIDE;  //WIDE etc. define in globaldef.h
    const static int VIEW_ANGLE_NORMAL = NORMAL;
    const static int VIEW_ANGLE_NARROW = NARROW;
    const static int VIEW_QUALITY_HIGH = HIGH;
    const static int VIEW_QUALITY_LOW  = LOW;

    Cmd_View();


    void set_angle_and_quality( int  ang, int  quality );
    void get_angle_and_quality( int &ang, int &quality ) const;


    void check_lock() const;

    friend std::ostream& operator<<( std::ostream &o, const Cmd_View &cmd );

    void reset();

};

// |---------------------------------------------------------------------|
// |  ,-----.             ,--.      ,------.        ,--.          ,--.   |
// | '  .--./,--,--,--. ,-|  |      |  .--. ' ,---. `--',--,--, ,-'  '-. |
// | |  |    |        |' .-. |      |  '--' || .-. |,--.|      \'-.  .-' |
// | '  '--'\|  |  |  |\ `-' |,----.|  | --' ' '-' '|  ||  ||  |  |  |   |
// |  `-----'`--`--`--' `---' '----'`--'      `---' `--'`--''--'  `--'   |
// |                                                                     |
// |---------------------------------------------------------------------|
class Cmd_Point: public Cmd_Base
{

private:

    bool   off_wanted;
    ANGLE  angle;
    double dist;


public:

    Cmd_Point();


    void set_pointto( double distance, Angle ang );
    void set_pointto( double distance, ANGLE ang );
    void set_pointto_off();
    bool get_pointto_off() const;
    void get_angle_and_dist( Angle &ang, double &distance ) const;
    void get_angle_and_dist( ANGLE &ang, double &distance ) const;


    void check_lock() const;

    friend std::ostream& operator<<( std::ostream &o, const Cmd_Point &cmd );

    void reset();

};

// |----------------------------------------------------------|
// |  ,-----.             ,--.       ,---.                    |
// | '  .--./,--,--,--. ,-|  |      '   .-'  ,--,--.,--. ,--. |
// | |  |    |        |' .-. |      `.  `-. ' ,-.  | \  '  /  |
// | '  '--'\|  |  |  |\ `-' |,----..-'    |\ '-'  |  \   '   |
// |  `-----'`--`--`--' `---' '----'`-----'  `--`--'.-'  /    |
// |                                                `---'     |
// |----------------------------------------------------------|
class Cmd_Say: public Cmd_Base
{

public:

    Cmd_Say();
    /* sets the ball position and velocity. This information can be
       about future ball position and velocity, so the absolute time MUST 
       be given. */
    void set_pass( Vector const &pos, Vector const &vel, int  time );
    bool get_pass( Vector       &pos, Vector       &vel, int &time ) const;
    bool pass_valid() const;

    void set_me_as_ball_holder( Vector const &pos );
    bool get_ball_holder(       Vector       &pos ) const;
    bool ball_holder_valid() const;

    void set_ball( Vector const &pos, Vector const &vel, int  age_pos, int  age_vel ); //age must be < 8
    bool get_ball( Vector       &pos, Vector       &vel, int &age_pos, int &age_vel ) const;
    bool ball_valid() const;

    void set_players( PlayerSet const &pset );
    bool get_player( int idx, Vector &pos, int &team, int &number ) const;
    int  get_players_num() const;

    void set_direct_opponent_assignment( int  assgnmnt );                 //TGdoa
    bool get_direct_opponent_assignment( int &assgnmnt ) const;           //TGdoa
    bool direct_opponent_assignment_valid() const;                        //TGdoa

    void set_pass_request( int  pass_in_n_steps, int  pass_param );       //TGpr
    bool get_pass_request( int &pass_in_n_steps, int &pass_param ) const; //TGpr
    bool pass_request_valid() const;                                      //TGpr

    void set_msg( unsigned char  type, short  p1, short  p2 );
    bool get_msg( unsigned char &type, short &p1, short &p2 ) const;
    bool get_msg( SayMsg &m ) const;
    bool msg_valid() const;


    void check_lock() const;

    friend std::ostream& operator<<( std::ostream &o, const Cmd_Say &cmd );

    void reset();

private:

    struct Pass
    {
        bool   valid;
        Vector ball_pos;
        Vector ball_vel;
        int    time;
    } pass;

    struct Ball_holder
    {
        bool   valid;
        Vector pos;
    } ball_holder;

    struct Ball
    {
        bool   valid;
        Vector ball_pos;
        Vector ball_vel;
        int    age_pos;
        int    age_vel;
    } ball;

    struct Players
    {
        static const int max_num= 3;
        struct _player
        {
            int    number;
            int    team;
            Vector pos;
        } player[max_num];
        int num;
    } players;

    // TGdoa: begin
    struct Direct_opponent_assignment
    {
        bool valid;
        int  assignment;
    } direct_opponent_assignment;
    // TGdoa: end
  
    // TGpr: begin
    struct Pass_request
    {
        bool valid;
        int  pass_in_n_steps;
        int  pass_param;
    } pass_request;
    // TGpr: end

    SayMsg msg;

};

// |----------------------------------------------------------------------------------------------------|
// |  ,-----.             ,--.        ,---.    ,--.    ,--.                   ,--.  ,--.                |
// | '  .--./,--,--,--. ,-|  |       /  O  \ ,-'  '-.,-'  '-. ,---. ,--,--, ,-'  '-.`--' ,---. ,--,--,  |
// | |  |    |        |' .-. |      |  .-.  |'-.  .-''-.  .-'| .-. :|      \'-.  .-',--.| .-. ||      \ |
// | '  '--'\|  |  |  |\ `-' |,----.|  | |  |  |  |    |  |  \   --.|  ||  |  |  |  |  |' '-' '|  ||  | |
// |  `-----'`--`--`--' `---' '----'`--' `--'  `--'    `--'   `----'`--''--'  `--'  `--' `---' `--''--' |
// |                                                                                                    |
// |----------------------------------------------------------------------------------------------------|
class Cmd_Attention: public Cmd_Base
{

private:

    int player;


public:

    Cmd_Attention();


    void set_attentionto_none();
    void set_attentionto( int  p );
    void get_attentionto( int &p ) const;


    void check_lock() const;

//  friend std::ostream& operator<<( std::ostream &o, const Cmd_Attention &cmd );

    void reset();

};

/** Die Klasse Cmd ist die Standardschnittstelle zum Weltmodell
    bzw. Laufzeitumgebung des Agenten. 
    Alles was ein Agent in einem Zeitschritt bestimmen kann, wird in dieser
    Klasse festgehalten.
    Sie ist das Gegestueck zur Klasse WS bzw. WSinfo

                                       |
                  Weltmodell       |WSinfo>
                                       |       Taktikebene
               Laufzeitumgebung        |
                                     <Cmd|
                                       |


   Alle Konventionen bezueglich Koordinaten und Winkelangaben, die in der
   Datei   
   
   ws.h 

   festgehalten sind, gelten damit auch fuer die Klasse Cmd.
*/

// |---------------------------|
// |  ,-----.             ,--. |
// | '  .--./,--,--,--. ,-|  | |
// | |  |    |        |' .-. | |
// | '  '--'\|  |  |  |\ `-' | |
// |  `-----'`--`--`--' `---'  |
// |                           |
// |---------------------------|
class Cmd
{

public:

    Cmd_Body      cmd_body;
    Cmd_Neck      cmd_neck;
    Cmd_View      cmd_view;
    Cmd_Say       cmd_say;
    Cmd_Attention cmd_att;
    Cmd_Point     cmd_point;

    void reset();

    friend std::ostream& operator<<( std::ostream &o, const Cmd &cmd );

};

#endif
