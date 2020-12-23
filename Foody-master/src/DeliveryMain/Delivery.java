package DeliveryMain;

import java.util.ArrayList;
import java.util.List;

public class Delivery extends User {
    private int deliveryID;
    private List<Order> orderList = new ArrayList<>();;

    public Delivery(String address, String city, int zip, int telephone, String email, String password, int deliveryID, List<Order> orderList) {
        super( address, city, zip, telephone, email, password);
        this.deliveryID = deliveryID;
        this.orderList = orderList;
    }

    public Delivery(String street, String address, String city, int zip, int telephone, String email, String password, int deliveryID) {
        super( address, city, zip, telephone, email, password);
        this.deliveryID = deliveryID;
    }

    public int getDeliveryID() {
        return deliveryID;
    }

    public void setDeliveryID(int deliveryID) {
        this.deliveryID = deliveryID;
    }

    public List<Order> getOrderList() {
        return orderList;
    }

    public void setOrderList(List<Order> orderList) {
        this.orderList = orderList;
    }

    public void addOrderToList(Order order){
        this.orderList.add(order);
    }
}
