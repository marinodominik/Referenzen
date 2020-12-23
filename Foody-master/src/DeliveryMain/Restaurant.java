package DeliveryMain;

import java.util.ArrayList;
import java.util.List;

public class Restaurant extends User {
    private int restaurantID;
    private String restaurantName;
    private List<Order> orderList = new ArrayList<>();
    private List<Rating> ratingList = new ArrayList<>();
    private Menu menu;
    private Membership membership;

    /*
     * Constructor with all lists
     */
    public Restaurant(String address, String city, int zip, int telephone, String email, String password,
                      int restaurantID, String restaurantName, Membership membership, List<Order> orderList, List<Rating> ratingList, Menu menu) {
        super( address, city, zip, telephone, email, password);
        this.restaurantID = restaurantID;
        this.restaurantName = restaurantName;
        this.membership = membership;
        this.orderList = orderList;
        this.ratingList = ratingList;
        this.menu = menu;
    }

    /*
     * Constructor without lists
     */
    public Restaurant(String address, String city, int zip, int telephone, String email, String password,
                      int restaurantID, String restaurantName, Membership membership, Menu menu) {
        super(address, city, zip, telephone, email, password);
        this.restaurantID = restaurantID;
        this.restaurantName = restaurantName;
        this.membership = membership;
        this.menu = menu;
    }

    public int getRestaurantID() {
        return restaurantID;
    }

    public void setRestaurantID(int restaurantID) {
        this.restaurantID = restaurantID;
    }

    public String getRestaurantName() {
        return restaurantName;
    }

    public void setRestaurantName(String restaurantName) {
        this.restaurantName = restaurantName;
    }

    public Membership getMembership() {
        return membership;
    }

    public void setMembership(Membership membership) {
        this.membership = membership;
    }

    public List<Order> getOrderList() {
        return orderList;
    }

    public void setOrderList(List<Order> orderList) {
        this.orderList = orderList;
    }

    public List<Rating> getRatingList() {
        return ratingList;
    }

    public void setRatingList(List<Rating> ratingList) {
        this.ratingList = ratingList;
    }

    public Menu getMenu() {
        return menu;
    }

    public void setMenu(Menu menu) {
        this.menu = menu;
    }

    public void addOrderToList(Order order){
        this.orderList.add(order);
    }

    public void addRatingToList(Rating rating){
        this.ratingList.add(rating);
    }

    public String getHTMLString(){
        String html = "<div class = \"restaurant-container\">";
        html += "<h1><a href =\"menu.jsp?id=" + Integer.toString(this.getRestaurantID()) + "\">" + this.restaurantName + "</a></h1>";
        return html;
    }
}
