/*
 *  Author:   Artur Merke 
 */
#ifndef _SENSORPARSER_H_
#define _SENSORPARSER_H_

#include "sensorbuffer.h"

class SensorParser {
public:
  static bool get_message_type_and_time(const char *, int & type, int & time);
  static bool manual_parse_fullstate(const char *, Msg_fullstate & fullstate);
  static bool manual_parse_fullstate(const char *, Msg_fullstate_v8 & fullstate);
  static bool manual_parse_see(const char *, Msg_see & see, char const* & next);
  static bool manual_parse_init(const char *, Msg_init & init);
  static bool manual_parse_sense_body(const char *, Msg_sense_body & sb, char const* & next);
  static bool manual_parse_hear(const char *, Msg_hear & hear, char const* & next, bool & reset_int);
	static bool manual_parse_card(const char * str, Msg_card &mc, const char *& next);
  static bool manual_parse_teamcomm(const char *, Msg_teamcomm & tc, const char *& next);
  static bool manual_parse_teamcomm(const char *, Msg_teamcomm2 & tc, const char *& next);
  static bool manual_encode_teamcomm(char *, const Msg_teamcomm & tc, char *& end);
  static bool manual_encode_teamcomm(char *, const Msg_teamcomm2 & tc, char *& end);
  // static bool manual_parse_teamcomm_side(const char *, int & side);
  static bool manual_parse_server_param(const char *, Msg_server_param & sp);
  static bool manual_parse_player_param(const char *, Msg_player_param & pp);
  static bool manual_parse_player_type(const char *, Msg_player_type & pt);
  static bool manual_parse_change_player_type(const char *, Msg_change_player_type & cpt);

  static bool manual_parse_my_trainercomm(const char * str);

  static void show_parser_error_point(std::ostream & out, const char * origin, const char * parse_error_point);
 protected:
  static bool manual_parse_my_online_coachcomm(const char *, Msg_my_online_coachcomm & moc, const char * & next);
  static bool manual_parse_my_online_coachcomm_clbased(const char *, Msg_my_online_coachcomm & moc, const char * & next);
  static bool manual_parse_coach_lang_directive(const char *, Msg_my_online_coachcomm & moc, const char * & next);
  static bool manual_parse_play_mode(const char *, PlayMode & pm, int & score, const char *& next);
  static bool manual_parse_view_mode(const char *, int & quality, int & width, const char *& next);

  static const double ENCODE_SCALE;
 protected:
  struct ParseObject {
    static const int UNKNOWN= -1;
    static const int MARKER_LINE= 0;
    static const int MARKER= 1;
    static const int BALL_OBJECT= -2;
    static const int PLAYER_OBJECT= 2;
    static const int PLAYER_MY_TEAM= 2;
    static const int PLAYER_MY_TEAM_GOALIE= 3;
    static const int PLAYER_HIS_TEAM= 4;
    static const int PLAYER_HIS_TEAM_GOALIE= 5;
    static const int PLAYER_UNKNOWN= 6;
    int res;
    int number;
    double x;
    double y;
  };
  static bool manual_parse_see_object(const char *, ParseObject & pobj, const char *& next);

  public: //debug
  
  // pos_x_range * pos_y_range must be < 2^13
  static const int pos_x_range= 55;
  static const int pos_y_range= 36;
  static const int vel_range= 30;
  //TG08: In 2005 we released (almost) our entire source code, including the
  //      classes relevant for communication. Consequently, it is rather
  //      straightforward for any opponent - given that it has had a closer
  //      look into our source code release - to eavesdrop our communication
  //      and to draw a lot of benefits from that.
  //      For these reasons, we simply change the identifiers of our communicated
  //      objects in 2008.
  //      More concrete: We insert the pass_request_id (introduced in 2008 by TG)
  //      as the first identifier usable and, hence, the identifiers of all
  //      other objects are incremented by one.
  static const unsigned int invalid_id= 0;
  static const unsigned int ball_id= 2; //TG08: 1->2
  //difference between player numbers and their communication ids (first player's id is 3 -> his number is 1 => offset is 2)
  static const unsigned int players_to_their_numbers_offset = 2; 
  static const unsigned int pass_info_id= 25; //TG08: 24->25
  static const unsigned int ball_info_id= 26; //TG08: 25->26
  static const unsigned int ball_holder_info_id= 27; //TG08: 26->27
  static const unsigned int msg_id= 28; //TG08: 27->28
  static const unsigned int direct_opponent_assignment_0_id = 29; //TGdoa //TG08: 28->29
  static const unsigned int direct_opponent_assignment_1_id = 30; //TGdoa //TG08: 29->30
  static const unsigned int direct_opponent_assignment_2_id = 31; //TGdoa //tg08: 30->31
  //static const unsigned int direct_opponent_assignment_3_id = 31; //TGdoa
  static const unsigned int pass_request_id = 1; //TGpr
  static const unsigned int max_id= 31;//TGdoa/TGpr: 27

  static const int max_bit_size= 60;
  static const int object_bit_size= 18;
  static const int ball_holder_info_bit_size= 18;
  static const int pass_info_bit_size= 37;
  static const int ball_info_bit_size= 36;
  static const int direct_opponent_assignment_bit_size = 5; //TGdoa
  static const int pass_request_bit_size = 13; //TGpr: 5+4+4

  static bool pos_in_range13bit( Vector pos )
  {
    if( pos.getX() < -pos_x_range || pos.getX() > pos_x_range || pos.getY() < -pos_y_range || pos.getY() > pos_y_range )
      return false;
    return true;
  }

  static bool pos_to_range13bit( Vector pos, unsigned int &max_13_bit )
  {
    //partition -pos_x_range,...,pos_x_range in x dir  : 2*pos_x_range+1 values
    //partition -pos_y_range,...,pos_y_range in y dir  : 2*pos_y_range+1 values

    if( !pos_in_range13bit( pos ) )
      return false;
    
    unsigned int x = int( rint( pos.getX() + pos_x_range ) );
    unsigned int y = int( rint( pos.getY() + pos_y_range ) );

    max_13_bit = x * ( 2 * pos_y_range + 1) + y;

    return true;
  }

  static bool range13bit_to_pos( unsigned int max_13_bit, Vector &pos)
  {
    if( max_13_bit >= ( 2 * pos_x_range + 1 ) * ( 2 * pos_y_range + 1) )
      return false;

    //partition -pos_x_range,...,pos_x_range in x dir  : 2*pos_x_range+1 values
    //partition -pos_y_range,...,pos_y_range in y dir  : 2*pos_y_range+1 values

    pos.setY( ( double ) ( max_13_bit % ( 2 * pos_y_range + 1 ) ) -pos_y_range );
    pos.setX( ( double ) ( max_13_bit / ( 2 * pos_y_range + 1 ) ) -pos_x_range );
    
    return true;
  }

  static bool vel_to_range12bit( Vector vel, unsigned int &max_12_bit )
  {
    vel.mulXby( 10.0 );
    vel.mulYby( 10.0 );

    if( vel.getX() < -vel_range || vel.getX() > vel_range || vel.getY() < -vel_range || vel.getY() > vel_range )
      return false;
    
    unsigned int x = int( rint( vel.getX() + vel_range ) );
    unsigned int y = int( rint( vel.getY() + vel_range ) );

    max_12_bit = x * ( 2 * vel_range + 1 ) + y;

    return true;
  }

  static bool range12bit_to_vel( unsigned int max_12_bit, Vector &vel )
  {
    if( max_12_bit >= ( 2 * vel_range + 1 ) * ( 2 * vel_range + 1) )
      return false;

    vel.setY( ( ( double ) ( max_12_bit % (2 * vel_range + 1 ) ) - vel_range ) / 10.0 );
    vel.setX( ( ( double ) ( max_12_bit / (2 * vel_range + 1 ) ) - vel_range ) / 10.0 );
    
    return true;
  }
};

int str2val(const char * str, NOT_NEEDED & val, const char* & next) ;
bool produce_server_param_parser(char const* param_str);

#endif
