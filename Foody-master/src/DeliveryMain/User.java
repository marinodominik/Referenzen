package DeliveryMain;


public class User {

    private String address;
    private String city;
    private int zip;
    private int telephone;
    private String email;
    private String password;

    public User( String address, String city, int zip, int telephone, String email, String password) {
        this.address = address;
        this.city = city;
        this.zip = zip;
        this.telephone = telephone;
        this.email = email;
        this.password = password;
    }


    public String getAddress() {
        return address;
    }

    public void setAddress(String address) {
        this.address = address;
    }

    public String getCity() {
        return city;
    }

    public void setCity(String city) {
        this.city = city;
    }

    public int getZip() {
        return zip;
    }

    public void setZip(int zip) {
        this.zip = zip;
    }

    public int getTelephone() {
        return telephone;
    }

    public void setTelephone(int telephone) {
        this.telephone = telephone;
    }

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }

    public String getPassword() {
        return password;
    }

    public void setPassword(String password) {
        this.password = password;
    }
}
