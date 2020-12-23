//************************************************************************
//*                                                                      *
//*    File Name:      stl_string_regex.h                                *
//*    Author:         Richard Kernan                                    *
//*    Last Modified:  08/01/01                                          *
//*                                                                      *
//************************************************************************
//*                                                                      *
//*    This file contains the function declarations for adding regular   *
//*    expressions to C++ Standard Template Library (STL) string objects *
//*    and C-style character pointer strings (which can be passed to     *
//*    these methods for implicit conversion).                           *
//*                                                                      *
//************************************************************************

#ifndef _STL_STR_REGEX_C_
#define _STL_STR_REGEX_C_

#include <iostream>
#include <string>
#include <vector>
#include <regex.h>
#include <exception>
#include <stdexcept>

using namespace std;

#define  REGX_PRNT 1    // Print error messages
#define  REGX_HIDE 0    // Hide  error messages



/************************************************************************
                Methods at a glance  (details below)
 ************************************************************************
 bool regx_match( const std::string &stl_string, const std::string &pattern )
      throw( bad_alloc, invalid_argument );
 
 bool regx_match( const std::string &stl_string, const std::string &pattern,
                 std::vector<std::string> &sub_strings )
      throw( bad_alloc, invalid_argument );

 ************************************************************************/



//************************************************************************
//*                                                                      *
//*    METHOD:     regx_match( std::string, std::string )                *
//*    PURPOSE:    To provide regular expression searches to Standard    *
//*                Template Library (STL) string objects.                *
//*    ARGUMENTS:  "stl_string" is the STL string object that will be    *
//*                searched.                                             *
//*                "pattern" is the regular expression pattern used in   *
//*                the search.                                           *
//*    RETURNS:    true if there is a regular expression match,          *
//*                false otherwise.                                      *
//*    NOTES:      If there are any errors with the regular expression   *
//*                pattern, this method will throw one of the exceptions *
//*                listed in the exception specification.                *
//*                Other possible exceptions are those that may be       *
//*                thrown by the STL string or vector classes.           *
//*                                                                      *
//************************************************************************
bool regx_match( const std::string &stl_string, const std::string &pattern )
     throw( bad_alloc, invalid_argument );


//************************************************************************
//*                                                                      *
//*    METHOD:     regx_match( std::string, std::string,                 *
//*                            std::vector<std::string> )                *
//*    PURPOSE:    To provide regular expression searches to Standard    *
//*                Template Library (STL) string objects.                *
//*    ARGUMENTS:  "stl_string" is the STL string object that will be    *
//*                searched.                                             *
//*                "pattern" is the regular expression pattern used in   *
//*                the search.                                           *
//*                "sub_strings" is an STL vector of STL string objects  *
//*                that represents any substring matches within the      *
//*                regular expression.  See NOTES for details.           *
//*    RETURNS:    true if there is a regular expression match,          *
//*                false otherwise.                                      *
//*    NOTES:      If there are any errors with the regular expression   *
//*                pattern, this method will throw one of the exceptions *
//*                listed in the exception specification.                *
//*                Other possible exceptions are those that may be       *
//*                thrown by the STL string or vector classes.           *
//*                                                                      *
//*                Details on "sub_strings":                             *
//*                sub_strings[0] = the substring within "stl_string"    *
//*                  that is matched by the entire regular expression.   *
//*                Any other strings in "sub_strings" represent          *
//*                  substrings within "stl_string" that are matched by  *
//*                  parts of the regular expression "pattern" that are  *
//*                  enclosed in parentheses.                            *
//*                For example (run on Linux):                           *
//*                stl_string = "This 234 is jklfjk my 0...9 string"     *
//*                pattern    = "([0-9]+) is(.*?)([0-9]+)"               *
//*                sub_strings[0] = "234 is jklfjk my 0...9"             *
//*                sub_strings[1] = "234"                                *
//*                sub_strings[2] = " jklfjk my 0..."                    *
//*                sub_strings[3] = "9"                                  *
//*                                                                      *
//*                For those of you who are accustomed to Perl syntax,   *
//*                sub_strings[1] = $1                                   *
//*                sub_strings[2] = $2                                   *
//*                etc.                                                  *
//*                                                                      *
//*                This will not compile on the IRIX.gl system at UMBC.  *
//*                I believe the compiler is too old to recognize some   *
//*                of the items in these functions.                      *
//*                                                                      *
//************************************************************************
bool regx_match( const std::string &stl_string, const std::string &pattern,
                 std::vector<std::string> &sub_strings )
     throw( bad_alloc, invalid_argument );



//************************************************************************
//*                                                                      *
//*                     EXAMPLE PROGRAM                                  *
//*                                                                      *
//************************************************************************
/*

#include <iostream>
#include <string>
#include <vector>
#include "stl_string_regex.h"

using namespace std;

int main()
{
    int i = 0;
    vector<string> sub_strings;
    string mystring = "This 234 is jklfjk my 0...9 string";
    string pattern  = "([0-9]+) is(.*?)([0-9]+)";

    try
    {
        if( regx_match( mystring, pattern, sub_strings ) )
        {
            cout << "REGEX MATCH..." << endl;
            for( i = 0; i < sub_strings.size(); i++ )
                cout << "sub_strings[" << i << "] = \"" << sub_strings[i] << "\"" << endl;
        }
        else
            cout << "NO REGEX MATCH" << endl;
    }
    catch( bad_alloc ba )
    {
        cout << ba.what() << endl;
    }
    catch( invalid_argument ia )
    {
        cout << ia.what() << endl;
    }
    catch( ... )
    {
        cout << "some other exception caught" << endl;
    }

    return 0;
}

*/

//************************************************************************
//*             Some helpful regular expressions:                        *
//************************************************************************

//  [0-9]        =>  match any digit
//  [^0-9]       =>  match any non-digit
//  [ \r\t\n\f]  =>  match any whitespace character
//  [^ \r\t\n\f] =>  match any non-whitespace character
//  [a-zA-Z0-9]  =>  match any alphanumeric character
//  [^a-zA-Z0-9] =>  match any non-alphanumeric character
//  x{4}         =>  match exactly 4 x's
//  x{0,4}       =>  match at most 4 x's
//  x{4,6}       =>  match between 4 and 6 x's (inclusive)
//  x{4,}        =>  match at least 4 x's

//************************************************************************

#endif

