#ifndef _WS_H_
#define _WS_H_

/**

Die Schnittstelle verwendet bei allen Angaben und unabhaengig von der
wirklichen Spielseite des Spielers das unten angegeben Koordinatensystem. Der
Koordinatenursprung befindet sich in der Mitte des Feldes. Die tatsaechliche
Position der Tore und Linien koennen aus den  Angaben ueber Feldmasse abgeleitet
werden. Sie sind aber insofern irrelevant, als dass es bei dieser Schnittstelle
nur um Positionen der beweglichen Objekte geht, unabhaengig von der
tatsaechlichen Feldgroesse. Das gegnerische Tor befinden sich immer rechts, also
in Richtung der X Achse.

      /------------------------------------------------\
      |                       |                        |
      |                       |                        |
      |                       |                        |
      |-------\               |                /-------|
      |       |               |                |       |
      |       |               |                |       |
      |       |               O <- (0,0)       |       | Goal of the opponent
      |       |               |                |       |
      |       |               |                |       |
      |-------/               |                \-------|
      |                       |                        |
      |                       |                        |
      |                       |                        |
      \------------------------------------------------/


            ^ Y axis
           /|\
            |
            |
     -------O-------->  X axis
            |
            |
            |



Winkel werden im Bogenmass angegeben, d.h. jeder Winkel ist aus dem Intervall
[0,2*Pi)


            | 0.5 Pi
            |
   Pi       |          0
     -------O-------->
            |
            |
            | 1.5 Pi
*/

#include "globaldef.h"
#include "Vector.h"
#include "angle.h"
#include "Player.h"
#include "Ball.h"
#include "comm_msg.h"


/** WS = World State
    this is the successor of MDPstate
 */
struct WS
{
    int time;
    int time_of_last_update;
    long ms_time_of_see_delay;
    long ms_time_of_sb;
    long ms_time_of_see;    // -1 if see messages are being ignored

//    char StateInfoString[2000]; //compact representation of state for messaging

    int play_mode;
    int penalty_count;
    int my_team_score;
    int his_team_score;

    int my_goalie_number;
    int his_goalie_number;

    int my_attentionto; // <= 0 if no attention is set

    int current_direct_opponent_assignment; //TGdoa
    int number_of_direct_opponent_updates_from_coach; //TGdoa

#if 0
    struct
    {
        Value dir;
        Value dist;
        int number;
        int seen_at;
        bool tackle_flag;
        int number_of_sensed_opponents;
    } closest_attacker; // the closest player to me; If team is known, then only opponents are considered
#endif

    // ... more information values

    bool synch_mode_ok; // for new synched view controller
    int view_angle;     //takes values WIDE,NORMAL,NARROW defined in globaldef.h
    int view_quality;   //tekes values HIGH,LOW defined in globaldef.h

    struct
    {
        int heart_at;
        int sent_at;
        int from;
    } last_message;
    int last_successful_say_at; // time, when my message was broadcast to others
  
    Ball ball;
    Player my_team[ NUM_PLAYERS + 1 ];
    Player his_team[ NUM_PLAYERS + 1 ];
    int my_team_num;
    int his_team_num;
    SayMsg msg;

    WS();
};

std::ostream& operator<<( std::ostream &outStream, const WS &ws );

#endif
