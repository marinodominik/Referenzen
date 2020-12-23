/**
 * Created by dominik on 29.05.2017.
 */

package NewPatient;
import java.time.Year;

/**
 * Created by dominik on 11.05.2017.
 */
public class NewPatientFillCombonents {

    public static String[] fillMonth() {
        String[] month = {"January", "February", "March", "April", "May", "June", "July", "August", "October", "November", "December"};
        return month;
    }

    public static Integer[] fillDate(){
        Integer[] date = new Integer[31];
        int j=0;
        for(int i=1; i<32; i++){
            date[j] = i;
            j++;
        }

        return date;
    }

    public static Integer[] fillYear(){
        int currentYear = Year.now().getValue();
        int j=1900;
        int k = currentYear-j;
        Integer[] year = new Integer[k+1];
        for(int i=0; i<= k; i++){
            year[i]= j;
            j++;
        }

        return year;
    }

    public static String[] fillGender() {
        String[] gender = {"Male", "Female"};
        return gender;
    }

    public static Integer[] fillHeight() {
        Integer[] height = new Integer[151];
        int j=0;
        for(int i=50; i<201; i++){
            height[j]=i;
            j++;
        }

        return height;
    }

    public static Integer[] fillWeight() {
        Integer[] weight = new Integer[250];
        int j=0;

        for(int i=1; i<251;i++){
            weight[j]=i;
            j++;
        }
        return weight;
    }

    public static String[] fillActivity() {
        String[] activity = {"low", "medium", "high"};
        return activity;
    }

    public static String[] fillHealthInsurence() {
        String[] healthInsurence = {"private", "legal"};
        return healthInsurence;
    }
}
