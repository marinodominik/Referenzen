/*
 * Copyright (c) 2002 - , Artur Merke <amerke@ira.uka.de> 
 *
 * This code is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2, or (at your option)
 * any later version.
 *
 * It is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 */
#include "txt_log.h"
#include "txt_filter_bs.h"
#include "str2val.h"
#include "macro_msg.h"


//#define DEBUG(XXX) std::cout << "\ndebug(" << __LINE__ << "):" XXX << flush
#define DEBUG(XXX)

void PosCache::show(std::ostream & out) {
    for (int i=0; i<num; i++)
        out << "\n [" << i << "], time= " << tab[i].time << ", pos= " << tab[i].pos;
}

bool  PosCache::put(int time, size_t pos) {
    if (num < MAX_NUM) {
        tab[num].time= time;
        tab[num].pos= pos;
        num++;
        return true;
    }
    tab[cur].time= time;
    tab[cur].pos= pos;
    cur++;
    if ( cur >= num )
        cur= 0;
    return true;
}

bool PosCache::get(int time, int & res_time, size_t & res_pos) {
    if ( num <= 0 )
        return false;

    int res= -1;  //lower bound
    int res2= -1; //uper bound

    for (int i=0; i<num; i++)
        if (tab[i].time <= time) {
            if (res < 0 || tab[i].time > tab[res].time)
                res= i;
        }
        else
            if (res2 < 0 || tab[i].time < tab[res2].time)
                res2= i;

    //at least res >= 0 or res2 >= 0
    if ( res2 >= 0 ) {
        if ( res < 0 )
            res= res2;
        else if ( time - tab[res].time > 10 &&   //lower bound is more the 10 away
                tab[res2].time - time < time - tab[res].time ) //upper bound is tighter then the lower bound
            res= res2;
    }

    res_pos= tab[res].pos;
    res_time= tab[res].time;
    DEBUG( << "\nsuccessfully cached entry for time= " << time << " res_time= " << res_time << " res_pos= " << res_pos);
    return true;
}

/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/

/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/

NumberedTextNavigator::NumberedTextNavigator() {
    reset();
}

NumberedTextNavigator::~NumberedTextNavigator() {
    close();
    if ( ! buffer_is_shared && line)
        delete[] line;
}

bool NumberedTextNavigator::init_buffer(int size, char * buffer) {
    if (size < 0) {
        if (buffer) {
            ERROR_OUT << "wrong buffer size " << size << ", hint: use get_recommended_buffer_size()";
            return false;
        }
        BUFFER_SIZE= get_recommended_buffer_size();
    }
    else if ( size < get_absolutely_smallest_buffer_size() ) {
        ERROR_OUT << "wrong buffer size= " << size << " hint: use get_recommended_buffer_size()";
        return false;
    }
    else
        BUFFER_SIZE= size;

    BUFFER_LINE_SIZE= BUFFER_SIZE- MAX_ADDRESS_SIZE - 2;

    if (buffer) {
        buffer_is_shared= true;
        line= buffer;
        return true;
    }

    buffer_is_shared= false;
    line= new char[BUFFER_SIZE];
    return true;
}

bool NumberedTextNavigator::open(const char * filename) {
    if ( file ) {
        ERROR_OUT << "\none open file exists already, use close() to close it first";
        return false;
    }
    if ( !(file = fopen (filename, "r"))) {
        //ERROR_OUT << "\ncould not open file " << filename;
        return false;
    }
    return true;
}

bool NumberedTextNavigator::close() {
    if (file) {
        fclose(file);
        return true;
    }
    return false;
}

bool NumberedTextNavigator::show_entries_for_time(int time, TextFilter & ts) {
    if ( !file) {
        ERROR_OUT << "\nno file opened for reading, use open() to initialize a file";
        return false;
    }
    if ( !line) {
        ERROR_OUT << "\nno buffer initialized, use init_buffer(...) to initialize one";
        return false;
    }
    bool res= false;

    //pos_cache.show(cout);
    _binary_search_param param;
    if ( pos_cache.get(time, param.start_time, param.start_pos) ) {//an important optimization
        param.use_start= true;
        if ( param.start_time == time ) {//already at the right time
            res= true;
            param.res_pos= param.start_pos;
        }
    }
    if ( ! res)
        res= binary_search_for_time_entry(file, time, param);

    if ( ! res)
        return false;

    pos_cache.put(time, param.res_pos);

    fseek(file, param.res_pos,SEEK_SET);
    //bool last_char_new_line= false;
    bool last_char_new_line= true;

    while (true) {
        int num_bytes= fread(line, 1, BUFFER_LINE_SIZE, file);
        if ( num_bytes <= 0 )
            return true;

        line[num_bytes]='\0';
        const char *dum= line;
        const char *end= line+num_bytes;

        while ( dum < end ) {
            if ( *dum == '\n' ) {
                last_char_new_line= true;
                dum++;
                continue;
            }

            if (last_char_new_line) {
                if ( end - dum  < MAX_ADDRESS_SIZE * 2 ) //the remaining string is too short, so we could cut the address and type info
                    if ( num_bytes == BUFFER_LINE_SIZE ) {
                        fseek(file, -(end - dum), SEEK_CUR);
                        break;
                    }
                last_char_new_line= false;
                int res_time;
                const char * dum2;
                if ( *dum != ' ' && str2val(dum,res_time,dum2) ) {
                    if ( res_time > time ) {
                        pos_cache.put(res_time, ftell(file) - (end - dum) );
                        return true;
                    }
                    dum= dum2;
                    if ( ts.process_type_info(time,dum,dum2) )
                        dum= dum2;
                    else {
#if 0
                        cout << "\nprocess_type_info retruns false "
                                << "\nmsg[" << dum << "!!!!!!!!!!!!!!!!!!!"
                                << "\nerr[" << dum2 << "!!!!!!!!!!!!!!!!!!!";
#endif
                        ts.process_character('\n');
                    }
                }
                else
                    ts.process_character('\n');
            }
            ts.process_character(*dum);

            dum++;
        }
    }
    return false;
}

bool NumberedTextNavigator::backward_linear_search_for_next_time_entry(FILE * file, _linear_search_param & result) {
    //line must have lenght at least = [MAX_LOG_LINE_SIZE+MAX_ADDRESS_LEN+2];
    result.first_time_jump_occurred= false;
    bool last_char_new_line= false;
    size_t old_backward_pos= ftell(file);
    DEBUG( << "\n ############################ BACKWARD SEARCH at pos= " << old_backward_pos);
    bool keep_running= true;
    bool already_got_first_time= false;
    bool first_backward_read= true;
    size_t total_num_bytes= 0;

    while (keep_running) {
        int num_bytes;
        if (old_backward_pos <= size_t(BUFFER_LINE_SIZE)) { //I'm almost at the begin of the file
            rewind(file);                              //jump to the begin
            num_bytes= fread(line+1,1,old_backward_pos,file);  //don't read the full SIZE, because otherwise you would't search strictly backwards
            *line='\n';                                //prepend a virtual new line at the begin of the file
            num_bytes++;
            keep_running= false;                       //this is the last run, nothing backward to the begin of the file
        }
        else {                                       //there is enough place to run backward
            fseek(file,old_backward_pos - BUFFER_LINE_SIZE, SEEK_SET);  //move the reading window backward
            old_backward_pos= ftell(file);             //remember the position of last reading window
            num_bytes= fread(line, 1, BUFFER_LINE_SIZE, file);
        }

        if ( num_bytes <= 0 ) { //end of file, and nothing found!
            DEBUG(  << "\n@@@@@@@@@@@@ end of file or error");
            return false;
        }
        else if ( result.use_num_bytes_limit ) {
            if ( total_num_bytes > result.num_bytes_limit ) {
                DEBUG( << " reached num_bytes_limit");
                return false;
            }
            total_num_bytes += num_bytes;
        }

        line[num_bytes]='\0';  //important, must be different from ' ' for the backward search, and '\0' is OK for printing (if debugging)
        char * dum = line+ num_bytes-1;
        char * dum_end= line+num_bytes;

        if (first_backward_read) { //if the last char of the whole sequence is a '\n', then search further backward
            first_backward_read= false;
            while (*dum == '\n' && dum >= line)
                dum--;
            if (dum < line)
                continue;
        }

        DEBUG(  << "\nnum_bytes= " << num_bytes << " BACKWARD \ndum=[");// << dum << "]"  << "\nline=[" << line << "]\n" << flush );
        bool buffer_was_prolonged= false;
        char *last_char_new_line_buffer;
        dum++;
        while ( dum > line ) { //start looking for a new line character and an immediately following integer!
            dum--;
            //DEBUG(  << "[" << *dum << "]" );
            if ( *dum == '\n' && dum[1] != ' ') {   //got a new line
                last_char_new_line= true;
                last_char_new_line_buffer= dum+1; //dum+1 is always defined (can be '\0')
                //DEBUG( << "last_char_new_line_buffer=[" << last_char_new_line_buffer << "]");
            }

            if ( ! last_char_new_line )
                continue;

            last_char_new_line= false;

            if (  !buffer_was_prolonged &&
                    (dum_end - last_char_new_line_buffer) < MAX_ADDRESS_SIZE ) {//the remaining string is too short, so we could cut an address
                buffer_was_prolonged= true;
                DEBUG( << "\n$$$$$$$$$$$$$$$$$$$$$$$$$ dum= [" << dum << "]");
                DEBUG( << " 1. num_bytes= " << num_bytes << "(" << BUFFER_LINE_SIZE+MAX_ADDRESS_SIZE << ")");
                int res = fread(dum_end, 1, MAX_ADDRESS_SIZE, file);
                //int res = fread(dum_end, 1, 15, file);
                DEBUG( << " 2. num_bytes= " << num_bytes+res << "(" << BUFFER_LINE_SIZE+MAX_ADDRESS_SIZE << ")");
                dum_end+= res;
                *dum_end='\0';
                DEBUG( << "\n$$$$$$$$$$$$$$$$$$$$$$$$$ dum2= [" << dum << "]");
            }

            //cout << "\nafter new line char num_bytest=" << dum << " RES=" << str2val(dum,res_time);
            int res_time;
            if ( str2val( last_char_new_line_buffer,res_time) ) {  //found a time entry at the begin of a line
                size_t res_pos= ftell(file) - (dum_end - last_char_new_line_buffer);
                if ( already_got_first_time && res_time < result.last_res_time )  //it's a time jump
                    if ( ! result.first_time_jump_occurred
                            || ( result.use_preferred_time_jump && result.last_res_time >= result.preferred_time_jump_time ) ) {
                        DEBUG( << "use " << result.use_preferred_time_jump << " preferred time jump= " << result.preferred_time_jump_time << "now time= " << res_time);
                        result.first_time_jump_occurred= true;
                        /** here we not use the res_pos and time (in contrast to the same part in the
		forward method) because the jump position is always the the position where 
		a new time block begins:
		example:             8 9 9
		                       ^ first time jump
                         */
                        result.first_time_jump_time= result.last_res_time;
                        result.first_time_jump_pos= result.last_res_pos;
                        if ( result.first_time_jump_time == result.preferred_time_jump_time
                                && result.use_preferred_time_jump_for_return ) {
                            result.last_res_time= res_time;
                            result.last_res_pos= res_pos;
                            return true;
                        }
                    } //end of time jump code
                result.last_res_pos= res_pos;
                result.last_res_time= res_time;
                DEBUG(  << "\n$$$ buffer_was_prolonged= " << buffer_was_prolonged << " LAST_RES_TIME= " << result.last_res_time << " LAST_RES_POS= " << result.last_res_pos);
                if ( ! already_got_first_time ) {
                    already_got_first_time= true;
                    result.first_res_pos = result.last_res_pos;
                    result.first_res_time= result.last_res_time;
                    if ( ! result.use_last_time_and_pos ) {
                        DEBUG( << " returning, because ! use_last_time_and_pos" );
                        return true;
                    }
                }
            }
            //DEBUG( << "\nloop end dum=[" << dum << "]");
        }
        if ( already_got_first_time ) {
            DEBUG( << " returning, because already_got_first_time" );
            return true;
        }
    }
    return false;
}

bool NumberedTextNavigator::forward_linear_search_for_next_time_entry(FILE * file, _linear_search_param & result) {
    result.first_time_jump_occurred= false;
    bool last_char_new_line= false;
    size_t tmp= ftell(file);
    if ( tmp == 0 )
        last_char_new_line= true;

    DEBUG( << "\n ############################ FORWARD SEARCH at pos= " << tmp);

    bool already_got_first_time= false;
    size_t total_num_bytes= 0;

    while (true) {
        int num_bytes= fread(line, 1, BUFFER_LINE_SIZE, file);  //just read a number of bytes

        if ( num_bytes <= 0 ) { //end of file, and nothing found!
            DEBUG(  << "\n@@@@@@@@@@@@ end of file or error");
            return false;
        } else if ( result.use_num_bytes_limit ) {
            if ( total_num_bytes > result.num_bytes_limit ) {
                DEBUG(  << "\n@@@@@@@@@@@@ reached num_bytes_limit");
                return false;
            }
            total_num_bytes+= num_bytes;
        }
        line[num_bytes]='\0';  //important, must be different from ' ' for the backward search, and '\0' is OK for printing (if debugging)
        char *dum= line;
        char *dum_end= line+num_bytes;

        DEBUG(  << "\n#### num_bytes= " << num_bytes << "\ndum=[");// << dum << "]" << "  dum_end=[" << dum_end << "]" << "\nline=[" << line << "]"  );
        dum--;
        while ( dum < dum_end ) { //start looking for a new line character and an immediately following integer!
            dum++;
            //DEBUG(  << "[" << *dum << "]" );
            if ( *dum == '\n' ) {   //got a new line
                last_char_new_line= true;
                continue;
            }
            if ( ! last_char_new_line )
                continue;

            last_char_new_line= false;

            if ( *dum == ' ')  //new line must directly be followed by a number!!!
                continue;

            if ( (dum_end - dum) < MAX_ADDRESS_SIZE ) {//the remaining string is too short, so we could cut an address
                if (already_got_first_time)
                    return true;

                if (num_bytes == BUFFER_LINE_SIZE) { //only if not at the end of the file
                    if ( result.use_num_bytes_limit )
                        total_num_bytes -= dum_end - dum;
                    fseek(file, - (dum_end - dum), SEEK_CUR);
                    DEBUG( << "\nfseek " << - (dum_end - dum) << " dum=[" << dum << "]");
                    last_char_new_line= true;
                    break;
                }
            }
            //DEBUG( << "\nNO JUMP BACK");
            //cout << "\nafter new line char num_bytest=" << dum << " RES=" << str2val(dum,res_time);
            int res_time;
            if ( str2val(dum,res_time) ) { //found a time entry at the begin of a line
                size_t res_pos= ftell(file) - num_bytes + (dum-line);
                if ( already_got_first_time && res_time > result.last_res_time )  //it's a time jump
                    if ( ! result.first_time_jump_occurred
                            ||  ( result.use_preferred_time_jump && res_time <= result.preferred_time_jump_time ) ) {
                        DEBUG( << "use " << result.use_preferred_time_jump << " preferred time jump= " << result.preferred_time_jump_time << "now time= " << res_time);
                        result.first_time_jump_occurred= true;
                        result.first_time_jump_time= res_time;
                        result.first_time_jump_pos= res_pos;
                        if ( result.first_time_jump_time == result.preferred_time_jump_time
                                && result.use_preferred_time_jump_for_return ) {
                            result.last_res_time= res_time;
                            result.last_res_pos= res_pos;
                            return true;
                        }
                    } //end of time jump code
                result.last_res_pos= res_pos;
                result.last_res_time= res_time;
                DEBUG(  << "\n$$$ return LAST_RES_TIME= " << result.last_res_time << " LAST_RES_POS= " << result.last_res_pos);
                if ( ! already_got_first_time ) {
                    already_got_first_time= true;
                    result.first_res_pos = result.last_res_pos;
                    result.first_res_time= result.last_res_time;
                    if ( ! result.use_last_time_and_pos ) {
                        DEBUG( << " returning, because ! use_last_time_and_pos" );
                        return true;
                    }
                }
            }
        }
        if (already_got_first_time) {
            DEBUG( << " returning, because already_got_first_time" );
            return true;
        }
    }
    return false;
}

bool NumberedTextNavigator::binary_search_for_time_entry(FILE * file, int targ_time, _binary_search_param & param) {
    DEBUG( << "TEST DEBUG ART\n");

    size_t starting_pos;
    int    starting_time;
    bool res= false;
    if ( param.use_start ) {
        starting_time = param.start_time;
        starting_pos  = param.start_pos;
        res= true;
    }
    else {
        size_t beg_pos= ftell(file);
        _linear_search_param result;
        result.use_last_time_and_pos= false;
        res= forward_linear_search_for_next_time_entry(file, result); //search forward
        if ( ! res ) {
            fseek(file,beg_pos,SEEK_SET);
            res= backward_linear_search_for_next_time_entry(file, result); //search backward
            if ( ! res ) {
                fseek(file,0,SEEK_SET);
                res= forward_linear_search_for_next_time_entry(file, result);
            }
        }
        starting_pos= result.first_res_pos;
        starting_time= result.first_res_time;
    }

    if (!res) {
        DEBUG( << "\ncannot find a beginning sequence");
        return false;
    }

    // --------------------------------------------------------------------------
    if ( starting_time >= targ_time && starting_time <= targ_time + 10 ) { //look seqeuntially backward
        size_t last_pos= starting_pos;
        int last_time= starting_time;
        _linear_search_param result;
        result.use_last_time_and_pos= true;
        result.set_preferred_time_jump(targ_time);
        while (true) {
            DEBUG(  << "\n###### going to " << last_pos ) ;
            fseek(file, last_pos, SEEK_SET);
            res= backward_linear_search_for_next_time_entry(file, result); //search backward

            if ( result.first_time_jump_occurred && result.first_time_jump_time == targ_time ) {
                param.res_pos= result.first_time_jump_pos;
                return true;
            }

            if ( !res || result.first_res_time < targ_time) {
                if (last_time != targ_time)
                    return false;
                param.res_pos= last_pos;
                DEBUG( << "\nres_pos = last_pos = " << param.res_pos);
                return true;
            }
            if ( result.last_res_time >= targ_time ) { //this is only to improve efficiency of searching!
                last_pos= result.last_res_pos;           //this is just to  accelarate linear searching (avoids reading almost the same buffer several times from a file)!!!
                last_time= result.last_res_time;
            }
            else {
                last_pos= result.first_res_pos;
                last_time= result.first_res_time;
            }
        }
    }
    // --------------------------------------------------------------------------
    if ( starting_time < targ_time - 10 || starting_time >= targ_time ) { // do binary search
        DEBUG (  << "\n>>>>>>>>>>>>>>> do binary search");
        //prepare for binary search
//        bool got_time_min;
//        bool got_time_max;
//        int time_min;
//        int time_max;
        size_t pos_max;
        size_t pos_min;
        if (targ_time < starting_time) {
//            time_max= starting_time;
//            got_time_max= true;
//            got_time_min= false;
            pos_max= starting_pos;
            pos_min= 0;
        }
        else {  //in this case always targ_time > starting_time
//            time_min= starting_time;
//            got_time_min= true;
//            got_time_max= false;
            fseek(file,0,SEEK_END);
            pos_min= starting_pos;
            pos_max= ftell(file);
        }

        _linear_search_param result;
        result.use_last_time_and_pos= false; //just need the first entry
        //the following loop keeps invariant that the time after
        //pos_min is always smaller then the time after pos_max
        //Note, that we don't need the actual time_min to be set (or to be known),
        //to keep the above invariant!!!
        while (true) { //binary search
            if ( // got_time_max && got_time_min && time_max - time_min < 10 ||
                    pos_max - pos_min < size_t(BUFFER_LINE_SIZE * 4) ) //stop binary search, proceed linear!
                break;
            size_t pos_mid = pos_min + (pos_max - pos_min) / 2;
            DEBUG( << "\npos    min= " << pos_min << "  mid= " << pos_mid << "  max= " << pos_max);
            DEBUG( << "\ntime   min= " << time_min << "  mid= " << result.first_res_time << "  max= " << time_max << " target= " << targ_time);
            fseek(file,pos_mid,SEEK_SET);
            result.set_num_bytes_limit(pos_max - pos_mid); //this is an optimization to quarantee O(log n), even if there are very few entries
            res= forward_linear_search_for_next_time_entry( file, result );

            if ( ! res ) { //num_bytes_limit reached
                pos_max= pos_mid;
                continue;
            }

            if ( result.first_res_time < targ_time ) {
                pos_min= pos_mid;
//                time_min= result.first_res_time;
//                got_time_min= true;
            }
            else {
                pos_max= pos_mid;
//                time_max= result.first_res_time;
//                got_time_max= true;
            }
        }
        //fseek(file,pos_min,SEEK_SET);
        starting_pos= pos_min;
    }

    fseek(file,starting_pos,SEEK_SET);
    //here the invariant:  [ get_time_at_pos( current position ) < targ_time ] should be valid

    // --------------------------------------------------------------------------
    DEBUG( << "AFTER BINARY SEARCH:    forward linear search for the first occurrence ");
    while (true) {   //search forward for the first occurrence
        _linear_search_param result;
        result.use_last_time_and_pos= true;
        result.set_preferred_time_jump(targ_time);

        bool res= forward_linear_search_for_next_time_entry(file, result);
        if ( ! res) {
            DEBUG( << "error ;-<");
            return false;
        }

        // << "\n########## got time= " << tmp_time << " pos= " << tmp_pos;
        if ( result.first_time_jump_occurred && result.first_time_jump_time == targ_time ) {
            param.res_pos= result.first_time_jump_pos;
            return true;
        }
        if ( result.first_res_time >= targ_time ) {
            if ( result.first_res_time > targ_time) {
                DEBUG(  << "\nno such time " << targ_time);
                return false;
            }
            param.res_pos= result.first_res_pos;
            return true;
        }
        size_t tmp_pos;
        if ( result.last_res_time < targ_time ) //this is just to  accelarate linear searching (avoids reading almost the same buffer several times from a file)!!!
            tmp_pos= result.last_res_pos;
        else
            tmp_pos= result.first_res_pos;

        if (tmp_pos == 0)
            tmp_pos++;
        DEBUG( << " BINARY at linear end, fseek " << tmp_pos);
        fseek(file,tmp_pos,SEEK_SET);
    }
    DEBUG( << "binary_search_for_time_entry:  return 0 (step 5)\n");
    return false;
}

void NumberedTextNavigator::test() {
    _binary_search_param param;
    bool res= binary_search_for_time_entry(file,8,param);
    std::cout << "\nbinary_search_for_time_entry returns: res_pos= " << param.res_pos << " res= " << res;

    //return 0;
#if 1
    TextFilter ts;
#else
    TextFilterCMUlike ts;
    ts.set_normal_mode(1);
    //ts.set_frameview_mode(0);
    //ts.set_error_mode(0);
#endif
    while (1) {
        int time;
        std::cout << "\nget new time: " << std::flush;
        std::cin >> time;
#if 0
        if (time>=0) {
            bool res= binary_search_for_time_entry(file,time,res_pos);
            std::cout << "\nbinary_search_for_time_entry returns res_pos= " << res_pos << " RES= " << res << " ftell(file)= " << ftell(file) ;
        }
        else {
            fseek(file,-time,SEEK_SET);
            _linear_search_param result;
            result.use_last_time_and_pos= true;
            bool res= backward_linear_search_for_next_time_entry(file, result);
            std::cout << "\n-- pos_ART returns res= " << res
                    << "\nfirst_res_pos= " << result.first_res_pos << " first_res_time= " << result.first_res_time
                    << "\n last_res_pos= " <<  result.last_res_pos << "  last_res_time= " << result.last_res_time;
        }
#else
        std::cout << "\n===================================="
                << "\nentry for time " << time << "\n";
        std::cout << "\n-----------\nold routine:\n";
        //show_all_entries_for_time(cout,time);
        std::cout << "\n-----------\nnew routine:";

        //res= binary_search_for_time_entry(file,time,param);
        //cout << "\nbinary_search_for_time_entry returns res_pos= " << param.res_pos << " RES= " << res << " ftell(file)= " << ftell(file) ;
        res= show_entries_for_time(time,ts);
        if ( ! res )
            std::cout << "\n nothing found for time " << time;

        for (time=1; false; time++) {
            std::cout << "\ntime=" <<time;
            show_entries_for_time(64,ts);
        }
#endif
    }
}

#if 0
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/

int main(int argc, char ** argv) {
    NumberedTextNavigator nav;
    if (argc<=1) {
        WARNING_OUT << "\nno argument specified";
        return -1;
    }
    nav.open(argv[1]);
    nav.test();
    nav.close();
    return 0;
}
#endif
