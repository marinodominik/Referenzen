#ifndef _WS_INFO_H_
#define _WS_INFO_H_

#include "Cmd.h"

/**
    WICHTIG WICHTIG WICHTIG WICHTIG WICHTIG WICHTIG WICHTIG WICHTIG WICHTIG
    WICHTIG WICHTIG WICHTIG WICHTIG WICHTIG WICHTIG WICHTIG WICHTIG WICHTIG
    WICHTIG WICHTIG WICHTIG WICHTIG WICHTIG WICHTIG WICHTIG WICHTIG WICHTIG

    Artur an alle:

    Der folgende Code soll der Nachfolger von mdpInfo werden.
    Aus den Erfahrungen die mit mdpInfo gemacht wurden ist die Idee
    entstanden, dass es keinen Sinn macht, alle nur erdenklichen Routinen
    in mdpInfo zu kodieren, die mit Spielernummern parametrisiert werden.

    Als kleiner Auszug aus den alten mdpInfo Methoden soll dienen:

    static Vector teammate_pos_abs(int number);
    static Vector teammate_vel_abs(int number);
    static double teammate_distance_to(int number, Vector taret);
    static double teammate_distance_to_ball(int number);
    static double teammate_distance_to_me(int number);

    [nur als Anmerkung am Rande: jede der obigen Routinen hat eine "for" Schleife
    implementiert, um uberhaupt den Spieler mit der entsprechenden Nummer zu finden]

    alle obigen Routinen koennen jetzt ersetzt werden durch

    PPlayer p= WSinfo::get_teammate_by_number(number);
    p->pos.distance( pos );

    oder es wird sonstwie auf p->pos, p->vel, p->ang etc.zugegriffen.

    Meist wird man allerdings erst gar nicht Spieler durch Nummern referenzieren!
    Viel wichtiger ist es oftmals, Spieler mit bestimmten Eigenschaften oder
    Positionen zu extrahieren. Oftmals werden Spielermengen als Ergebnistyp
    erwartet, was bisher unmoeglich war. Betrachten wir z.B. folgende Routinen,
    die aus dem alten mdpInfo entnommen wurden:

    static int teammate_closest_to(Vector target);
    static int teammate_closest_to_ball();
    static int teammate_closest_to_me();
    static int teammate_closest_to_ball_wme();
    static int teammate_closest_to_ball_wme_wogoalie();
    static int teammate_closest_to_wme(Vector target);

    Der Nachteil liegt darin, dass man jeweils nur einen Spieler bekommt, wo doch manchmal
    die ersten 2 oder 3 interessant waeren. Also war es bisher unmoeglich, Spieler zu
    extrahieren, die sich in einem bestimmten Bereich befinden. Dazu musste man selbst
    auf die untersten Codeebenen heruntersteigen, und sich mit mehrmals verschachtelten
    "for" Schleifen herumschlagen. Hier einige Beispiele, wie obigen Routinen jetzt ersetzt
    werden koennen:

    WSpset pset= WSinfo::valid_teammates; oder
    WSpset pset= WSinfo::valid_teammates_without_me; oder
    WSpset pset= WSinfo::valid_opponents;
    [ pset+= WSinfo::valid_opponents waere auch moeglich, damit sowohl die teammates
    als auch opponents in die Menge hineinkommen]

    pset.keep_and_sort_closest_players_to_point(3, pos);

    behaelt alle Spieler die am naechsten zur Position [pos] liegen, nach Entfernung sortiert.

    Man kann aber auch jederzeit Spieler eines Bereichs extrahieren, indem man z.B sagt

    pset.keep_players_in_rectangle( Vector(52.5,20), Vector(36.5,-20) );

    so dass alle Spieler im gegnerischen Strafraum erfasst werden (dabei ist es
    unerheblich wie die Menge pset zustande gekommen ist, so dass man alle
    Freiheitsgrade bei deren Zusammenstellung hat!)

    WICHTIG:
    Da der Code in dieser Datei nach bestimmten Design Kriterien erstellt worden ist,
    die fuer einen Benutzer nicht unbedingt auf Anhieb erkennbar sind (wie z.B.
    deferred precomputation etc.), KONSULTIERT mich (=Artur) daher bevor Ihr nicht weiter
    kommt oder meint Routinen hinzufuegen zu muessen, weil sie dringend gebraucht werden.

    Der Code ist sicher noch nicht vollstaendig, so dass ich auf eure Inputs angewiesen bin!!!

    WICHTIG WICHTIG WICHTIG WICHTIG WICHTIG WICHTIG WICHTIG WICHTIG WICHTIG
    WICHTIG WICHTIG WICHTIG WICHTIG WICHTIG WICHTIG WICHTIG WICHTIG WICHTIG
    WICHTIG WICHTIG WICHTIG WICHTIG WICHTIG WICHTIG WICHTIG WICHTIG WICHTIG
 */

/** This class is the replacement for the old mdpInfo class. */
class WSinfo
{
private:
    static char cvCurrentOpponentIdentifier;
public:
    static Cmd *current_cmd;          ///< will be set in client.cpp
    static Cmd *last_cmd;             ///< will be set in client.cpp

    static WS const* ws;
    static WS const* ws_full;

    static PPlayer me;                ///< convenient shorthand for the player itself
    static PPlayer me_full;           ///< convenient shorthand for the player itself in fullstate (me_full==0 iff ws_full==0)

    static PPlayer his_goalie;        ///< can be a zero pointer, if goalie is not valid or not known

    static Ball const* ball;      ///< convenient shorthand for the ball
    static Ball const* ball_full; ///< convenient shorthand for the ball in fullstate (ball_full==0 iff ws_full==0)

    static PlayerSet alive_teammates;
    static PlayerSet alive_teammates_without_me;
    static PlayerSet alive_opponents;

    static PlayerSet valid_teammates;
    static PlayerSet valid_teammates_without_me;
    static PlayerSet valid_opponents;

    struct PlayerInactivities
    {
        int validAtTime[ 2 * NUM_PLAYERS ];
        int minInact[ 2 * NUM_PLAYERS ];
        int maxInact[ 2 * NUM_PLAYERS ];
        PlayerInactivities();
    };
    static PlayerInactivities player_inactivities;

    //JK PASS_MSG_HACK begin
    static bool jk_pass_msg_set;
    static bool jk_pass_msg_rec;
    static char  jk_pass_msg[80];
    static long jk_pass_msg_rec_time;
    static float jk_pass_msg_x;
    static float jk_pass_msg_y;
    //JK PASS_MSG_HACK end

    static int relevant_teammate[11];
    static int num_relevant_teammates;

    static bool init( const WS *worldstate, const WS *worldstate_full ); ///< worldstate_full==0 is possible

    static bool is_my_pos_valid();
    static bool is_teammate_pos_valid( const PPlayer player );
    static bool is_opponent_pos_valid( const PPlayer player );

    static bool is_ball_pos_valid();
    static bool is_ball_kickable_for( const PPlayer player,
                                      const Vector  &ballPos = ball->pos
                                    );
    static bool is_ball_kickable(     const Vector &ballPos = ball->pos );
    static bool is_ball_pos_valid_and_kickable();

    static double get_tackle_probability_for( const PPlayer player,
                                              bool foul = false,
                                              const Vector &ballPos = ball->pos
                                            );
    static double get_tackle_probability( bool foul = false,
                                          const Vector &ballPos = ball->pos
                                        );

    static bool is_player_probably_inactive_after_tackling(
            const PPlayer player );

    static bool is_player_inactive_after_being_fouled( const PPlayer player );

    static void get_player_inactivity_interval_after_tackling(
            const PPlayer player,
            int& mini,
            int& maxi );

    static void get_player_inactivity_interval_after_being_fouled(
            const PPlayer player,
            int& mini,
            int& maxi );

    static void get_player_inactivity_interval(
            const PPlayer player,
            int& mini,
            int& maxi );

    static bool get_teammate( int number, PPlayer &player );
    static PPlayer get_teammate_by_number( int num );
    static PPlayer get_opponent_by_number( int num );

    static int num_teammates_within_circle( const Vector &centre,
                                            const double radius
                                          );

    static double my_team_pos_of_offside_line();
    static double his_team_pos_of_offside_line();

    static PPlayer get_teammate_with_newest_pass_info();
    static PPlayer teammate_closest2ball();

    static void visualize_state();

    static void set_relevant_teammates_default();
    static void set_relevant_teammates( const int  t1 = 0, const int  t2 = 0,
                                        const int  t3 = 0, const int  t4 = 0,
                                        const int  t5 = 0, const int  t6 = 0,
                                        const int  t7 = 0, const int  t8 = 0,
                                        const int  t9 = 0, const int t10 = 0,
                                        const int t11 = 0
                                      );

    static char get_current_opponent_identifier();
    static void set_current_opponent_identifier( char id );

protected:

    /**
     * following values should never be accessed directly. Especially never
     * change the above protected to public. Most of it is initialized in the
     * init(...) method. Take as an example the computation of offside lines.
     * These are not precomputed, but in the init(...) method *_cache_ok is set
     * to false. After the first computation of an offside line the *_cache_ok
     * is set to true, and the cache value can be used in following enquiries
     * (until the new cycle, when the init(...) method is called again)
     */

    static PlayerSet pset_tmp;

    static double  my_team_pos_of_offside_line_cache;
    static bool    my_team_pos_of_offside_line_cache_ok;
    static double  his_team_pos_of_offside_line_cache;
    static bool    his_team_pos_of_offside_line_cache_ok;
    static PPlayer teammate_with_newest_pass_info;
    static bool    teammate_with_newest_pass_info_ok;

    /**
     * cache for keeping all players by their numbers, use the
     * get_teammate_by_number or get_opponent_by_number methods
     * to access the values
     */
    static PPlayer numbered_valid_players[ 2 * NUM_PLAYERS + 1 ];

};

#endif
