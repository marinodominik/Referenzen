//************************************************************************
//*                                                                      *
//*    File Name:      stl_string_regex.cpp                              *
//*    Author:         Richard Kernan                                    *
//*    Last Modified:  08/01/01                                          *
//*                                                                      *
//************************************************************************
//*                                                                      *
//*    This file contains the function definitions for adding regular    *
//*    expressions to C++ Standard Template Library (STL) string objects *
//*    and C-style character pointer strings (which can be passed to     *
//*    these methods for implicit conversion).                           *
//*                                                                      *
//************************************************************************

#include "stl_string_regex.h"

#include <stdlib.h>

using namespace std;

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
bool regx_match( const string &stl_string, const string &pattern )
     throw( bad_alloc, invalid_argument )
{
    regex_t regexp;
    int status = 0;
    char *err_msg = NULL;

    // Compile the regular expression pattern
    if( ( status = regcomp( &regexp, pattern.c_str(), REG_EXTENDED|REG_NOSUB )) != 0 )
    {
        if( status == REG_ESPACE )
            throw bad_alloc();
        else
        {
            // Get the size of the message.
            int err_msg_sz = regerror( status, &regexp, NULL, (size_t) 0 );
            
            // Allocate the memory, print the message,
            // and then free the memory.
            if( ( err_msg = (char *) malloc( err_msg_sz ) ) != NULL )
            {
                regerror( status, &regexp, err_msg, err_msg_sz );
                string error = err_msg;
                free( err_msg );
                err_msg = NULL;
                error += "\nRegular expression = \"";
                error += pattern;
                error += "\"";
                throw invalid_argument( error.c_str() );
            }
            else
            {
                string invalid = "Invalid regular expression:  ";
                invalid += pattern;
                throw invalid_argument( invalid.c_str() );
            }
        }
    }

    // Search for the regular expression in the string
    status = regexec( &regexp, stl_string.c_str(), (size_t) 0, NULL, 0 );

    // Free the memory allocated by
    // the regex.h functions.
    regfree( &regexp );
    if( status == REG_NOMATCH )
        return false;
    else if( status != 0 )
    {
        if( status == REG_ESPACE )
            throw bad_alloc();
        else
        {
            // Get the size of the message.
            int err_msg_sz = regerror( status, &regexp, NULL, (size_t) 0 );
            
            // Allocate the memory, print the message,
            // and then free the memory.
            if( ( err_msg = (char *) malloc( err_msg_sz ) ) != NULL )
            {
                regerror( status, &regexp, err_msg, err_msg_sz );
                string error = err_msg;
                free( err_msg );
                err_msg = NULL;
                error += "\nRegular expression = \"";
                error += pattern;
                error += "\"";
                throw invalid_argument( error.c_str() );
            }
            else
            {
                string invalid = "Invalid regular expression:  ";
                invalid += pattern;
                throw invalid_argument( invalid.c_str() );
            }
        }
    }

    return true;
}



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
bool regx_match( const string &stl_string, const string &pattern,
                 vector<string> &sub_strings )
     throw( bad_alloc, invalid_argument )
{
    regex_t regexp;
    char *err_msg = NULL;
    int status = 0;
    int nsubs  = pattern.size() + 1;

    // Compile the regular expression and check for errors
    if( ( status = regcomp( &regexp, pattern.c_str(), REG_EXTENDED )) != 0 )
    {
        if( status == REG_ESPACE )
            throw bad_alloc();
        else
        {
            // Get the size of the message.
            int err_msg_sz = regerror( status, &regexp, NULL, (size_t) 0 );
            
            // Allocate the memory, print the message,
            // and then free the memory.
            if( ( err_msg = (char *) malloc( err_msg_sz ) ) != NULL )
            {
                regerror( status, &regexp, err_msg, err_msg_sz );
                string error = err_msg;
                free( err_msg );
                err_msg = NULL;
                error += "\nRegular expression = \"";
                error += pattern;
                error += "\"";
                throw invalid_argument( error.c_str() );
            }
            else
            {
                string invalid = "Invalid regular expression:  ";
                invalid += pattern;
                throw invalid_argument( invalid.c_str() );
            }
        }
    }

    if( nsubs > 0 )
    {
        // Allocate space to save the matching substring indices
        regmatch_t *subidx = new regmatch_t[nsubs * sizeof(regmatch_t)];
        if( subidx == NULL )
            nsubs = 0;
        else
        {
            // Allocate space to save the characters in the
            // matching substrings.
            char *substring = new char[stl_string.size() + 1];
            if( substring == NULL )
                nsubs = 0;
            else
            {
                // Perform the search and save the matching substring indices
                status = regexec( &regexp, stl_string.c_str(), (size_t) nsubs, subidx, 0 );
                sub_strings.clear();

                int i = 0;
                for( i = 0; i < nsubs; i++ )
                {
                    // Stop the substring saves when there are no more to save
                    if( subidx[i].rm_so == -1 || subidx[i].rm_eo == -1 )
                        break;

                    int j = 0;
                    int k = 0;
                    // Save the matching substrings to the allocated array
                    for( j = subidx[i].rm_so, k = 0; j < subidx[i].rm_eo; j++, k++ )
                        substring[k] = stl_string.at( j );
                    substring[k] = '\0';

                    // Save the matching substrings in the
                    // vector of strings
                    string str = substring;
                    sub_strings.push_back( str );
                }
                delete[] substring;
            }
            delete[] subidx;
        }
    }
    // If the memory allocation failed, perform the search
    // without saving the substrings.
    if( nsubs <= 0 )
        status = regexec( &regexp, stl_string.c_str(), (size_t) 0, NULL, 0 );

    // Free the memory allocated by
    // the regex.h functions.
    regfree( &regexp );

    if( status == REG_NOMATCH )
        return false;

    if( status != 0 )
    {
        if( status == REG_ESPACE )
            throw bad_alloc();
        else
        {
            // Get the size of the message.
            int err_msg_sz = regerror( status, &regexp, NULL, (size_t) 0 );
            
            // Allocate the memory, print the message,
            // and then free the memory.
            if( ( err_msg = (char *) malloc( err_msg_sz ) ) != NULL )
            {
                regerror( status, &regexp, err_msg, err_msg_sz );
                string error = err_msg;
                free( err_msg );
                err_msg = NULL;
                error += "\nRegular expression = \"";
                error += pattern;
                error += "\"";
                throw invalid_argument( error.c_str() );
            }
            else
            {
                string invalid = "Invalid regular expression:  ";
                invalid += pattern;
                throw invalid_argument( invalid.c_str() );
            }
        }
    }

    return true;
}
