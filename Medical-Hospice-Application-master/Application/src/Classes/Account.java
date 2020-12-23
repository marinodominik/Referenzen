package Classes;

/**
 * Created by dominik on 01.06.2017.
 */
public class Account {
    public static String username;
    public static String password;
    public String job;


    public static String getUsername() {
        return username;
    }

    public static void setUsername(String userName) {
        username = userName;
    }

    public static String getPassword() {
        return password;
    }

    public static void setPassword(String passWord) {
        password = passWord;
    }

    public String getJob() {
        return job;
    }

    public void setJob(String job) {
        this.job = job;
    }

}
