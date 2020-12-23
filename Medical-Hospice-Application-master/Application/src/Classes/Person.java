package Classes;

import java.util.Date;

/**
 * Created by dominik on 18.05.2017.
 */
public class Person {
    public String name;
    public String firstName;
    public Date birth;

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getFirstName() {
        return firstName;
    }

    public void setFirstName(String firstName) {
        this.firstName = firstName;
    }

    public Date getBirth() {
        return birth;
    }

    public void setBirth(Date birth) {
        this.birth = birth;
    }


}
