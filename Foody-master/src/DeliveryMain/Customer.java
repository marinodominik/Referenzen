package DeliveryMain;


import java.util.ArrayList;
import java.util.List;

public class Customer extends User {
    private String firstName;
    private String lastName;
    private List<Order> orderList = new ArrayList<>();
    private int customerID;

    /*
     * Customer constructor with empty order list
     */
    public Customer( String address, String city, int zip, int telephone, String email, String password,
                    String firstName, String lastName, int customerID) {
        super(address, city, zip, telephone, email, password);
        this.firstName = firstName;
        this.lastName = lastName;
        this.customerID = customerID;
    }

    /*
    * Customer constructor with non-empty order list
    */
    public Customer(String address, String city, int zip, int telephone, String email, String password,
                    String firstName, String lastName, List<Order> orderList, int customerID) {
        super(address, city, zip, telephone, email, password);
        this.firstName = firstName;
        this.lastName = lastName;
        this.orderList = orderList;
        this.customerID = customerID;
        this.orderList = orderList;
    }

    public String getFirstName() {
        return firstName;
    }

    public void setFirstName(String firstName) {
        this.firstName = firstName;
    }

    public String getLastName() {
        return lastName;
    }

    public void setLastName(String lastName) {
        this.lastName = lastName;
    }

    public List<Order> getOrderList() {
        return orderList;
    }

    public void setOrderList(List<Order> orderList) {
        this.orderList = orderList;
    }

    public int getCustomerID() {
        return customerID;
    }

    public void setCustomerID(int customerID) {
        this.customerID = customerID;
    }

    public void addOrderToList(Order order){
        this.orderList.add(order);
    }
}
