package DeliveryMain;


import java.util.ArrayList;
import java.util.List;

public class Order {
    private int orderID;
    private Payment paymentMethod;
    private double amountToPay;
    private int customerID;
    private int restaurantID;
    private boolean inProduction;
    private boolean isReadyForDelivery;
    private boolean orderIsOnTheWay;
    private boolean orderDelivered;
    private Rating rating;
    private List<Meal> mealList = new ArrayList<>();

    /**
     * Constructor without Lists and orderID, to be used by *humans
     * @param paymentMethod
     * @param customerID
     * @param restaurantID
     */
    public Order(Payment paymentMethod, int customerID, int restaurantID) {
        this.orderID = 0;
        this.paymentMethod = paymentMethod;
        this.amountToPay = 0.00;
        this.customerID = customerID;
        this.restaurantID = restaurantID;

        this.inProduction = false;
        this.orderIsOnTheWay = false;
        this.isReadyForDelivery = false;
        this.orderDelivered = false;
        this.rating = null;
    }

    /**
     * Constructor with all possible parameters for storing data from database
     * @param orderID
     * @param paymentMethod
     * @param amountToPay
     * @param customerID
     * @param restaurantID
     * @param status
     * @param rating
     * @param mealList
     */
    public Order(int orderID, Payment paymentMethod, double amountToPay, int customerID, int restaurantID,
                 int status, Rating rating, List<Meal> mealList) {
        this.orderID = orderID;
        this.paymentMethod = paymentMethod;
        this.amountToPay = amountToPay;
        this.customerID = customerID;
        this.restaurantID = restaurantID;
        this.rating = rating;
        handleOrderStatus(status);
        this.mealList = mealList;
    }

    public int getOrderID() {
        return orderID;
    }

    public void setOrderID(int orderID) {
        this.orderID = orderID;
    }

    public Payment getPaymentMethod() {
        return paymentMethod;
    }

    public void setPaymentMethod(Payment paymentMethod) {
        this.paymentMethod = paymentMethod;
    }

    public double getAmountToPay() {
        return amountToPay;
    }

    public void setAmountToPay(double amountToPay) {
        this.amountToPay = amountToPay;
    }

    public int getCustomerID() {
        return customerID;
    }

    public void setCustomerID(int customerID) {
        this.customerID = customerID;
    }

    public int getRestaurantID() {
        return restaurantID;
    }

    public void setRestaurantID(int restaurantID) {
        this.restaurantID = restaurantID;
    }

    public boolean isInProduction() {
        return inProduction;
    }

    public void setInProduction(boolean inProduction) {
        this.inProduction = inProduction;
    }

    public boolean isReadyForDelivery() {
        return isReadyForDelivery;
    }

    public void setReadyForDelivery(boolean readyForDelivery) {
        isReadyForDelivery = readyForDelivery;
    }

    public boolean isOrderIsOnTheWay() {
        return orderIsOnTheWay;
    }

    public void setOrderIsOnTheWay(boolean orderIsOnTheWay) {
        this.orderIsOnTheWay = orderIsOnTheWay;
    }

    public boolean isOrderDelivered() {
        return orderDelivered;
    }

    public void setOrderDelivered(boolean orderDelivered) {
        this.orderDelivered = orderDelivered;
    }

    public Rating getRating() {
        return rating;
    }


    public void setRating(Rating rating) {
        this.rating = rating;

    }

    /**
     * Calculates also amount to pay
     * @param meal
     */
    public void addMealToOrderList(Meal meal){
        mealList.add(meal);
        this.amountToPay += meal.getMealPrice();
    }

    public List<Meal> getMealList(){
        return mealList;
    }

    private void handleOrderStatus(int status){
        switch (status){
            case 0: inProduction = false; isReadyForDelivery = false; orderIsOnTheWay = false; orderDelivered = false;
            case 1: inProduction = true; isReadyForDelivery = false; orderIsOnTheWay = false; orderDelivered = false;
            case 2: inProduction = true; isReadyForDelivery = true; orderIsOnTheWay = false; orderDelivered = false;
            case 3: inProduction = true; isReadyForDelivery = true; orderIsOnTheWay = true; orderDelivered = false;
            case 4: inProduction = true; isReadyForDelivery = true; orderIsOnTheWay = true; orderDelivered = true;
        }
    }

    public int getStatus(){
        int ret = 0;

        if(inProduction && !isReadyForDelivery() && !orderIsOnTheWay && !orderDelivered) ret = 1;
        else if (inProduction && isReadyForDelivery() && !orderIsOnTheWay && !orderDelivered) ret = 2;
        else if (inProduction && isReadyForDelivery() && orderIsOnTheWay && !orderDelivered) ret = 3;
        else if (inProduction && isReadyForDelivery() && orderIsOnTheWay && orderDelivered) ret = 4;

        return ret;
    }
}
