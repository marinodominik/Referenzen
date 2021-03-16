#ifndef _WMTOOLS_H_
#define _WMTOOLS_H_

#include <math.h>
#include <limits.h>
#include "angle.h"
#include "defs.h"
#include "coach.h"
#include "logger.h"

class WMTOOLS {
 public:
  static const double TwoPi;

  static double conv_server_angle_to_angle_0_2Pi(double s_ang) {
    double tmp= (-s_ang * M_PI)/ 180.0;
    tmp= fmod(tmp,TwoPi);
    if (tmp < 0) tmp+= TwoPi;
    return tmp;
  }
  
  
  static int get_stamina_capacity_from_base_encoding(char c) 
  {
    char uchar_base = c - 'A';  
    int base = (int) uchar_base;
    return (base) * PT_STAMINA_CAP_DISCRETIZATION_STEP;
  }
  
  static double get_stamina_capacity_from_pointto(ANGLE pointto) {
      Angle angValue = ( (RUN::side==RUN::side_LEFT) ? pointto.get_value_0_p2PI() 
                                                     : ANGLE(M_PI + pointto.get_value_0_p2PI()).get_value_0_p2PI() );
      // see defs.h for definitions of PT_STAMINA_*
      int base = int(fabs((angValue - PT_STAMINA_CAP_START_ANG) / PT_STAMINA_CAP_ANG_INC));
      int result = base * PT_STAMINA_CAP_DISCRETIZATION_STEP;
      return result;
  }

  static char get_base_encoding_from_stamina_capacity(double cap) {
    int int_base = (int) floor(cap / PT_STAMINA_CAP_DISCRETIZATION_STEP);
    //char start = 'A';
    char uchar_base;
    if ( int_base >= 'Z' - 'A' )
    {
        LOG_FLD(1, "WARNING base encoding of stamina_capacity is messed up!");
        uchar_base = 'Z';
    }
    else
    {
    		uchar_base = 'A' + (char)int_base;
    }
    LOG_FLD(1,"ICH GEBE ZURUECK: "<<(int)uchar_base);
    LOG_FLD(1,"ICH GEBE ZURUECK: "<<uchar_base);
    return uchar_base;
  }
  
  /*
  static double conv_server_x_to_x(double x) { return x; }
  static double conv_server_y_to_y(double y) { return -y; }
  */

  /** \short singned 18 bit integer to 3 byte ascii using 64 fix characters

      this function encodes an integer in the range [2^18-1, ..., 2^18-1]
      other integers are treated as 2^18, which represents the infinity value
      
      the pointer dst should point to a memeory with 3 bytes free after it!
  */
  static bool int18_to_a3x64(int src, char * dst);
  /** \short 3 byte ascii using 64 fix characters to signed integer with max 18 bit
      The value 2^18 indicates an overflow!
      
      the pointer src should point to a memeory with 3 bytes free after it and
      with valid characters.
  */
  static bool a3x64_to_int18(const char * src, int & a);


  /// [0,...,62] -> ascii char
  static bool uint6_to_a64(int src, char * dst);
  
  /// ascii char to [0,...,62], 63 indicates overflow
  static bool a64_to_uint6(const char * src, int & dst);


};
  
#endif
