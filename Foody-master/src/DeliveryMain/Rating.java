package DeliveryMain;

public class Rating {
    private int ratingID;
    private int stars;
    private String comment;
    private int customerID;
    private int restaurantID;
    private int orderID;

    public Rating(int ratingID, int stars, String comment, int customerID, int restaurantID, int orderID) {
        this.ratingID = ratingID;
        this.stars = stars;
        this.comment = comment;
        this.customerID = customerID;
        this.restaurantID = restaurantID;
        this.orderID = orderID;
    }

    public int getRatingID() {
        return ratingID;
    }

    public void setRatingID(int ratingID) {
        this.ratingID = ratingID;
    }

    public int getStars() {
        return stars;
    }

    public void setStars(int stars) {
        this.stars = stars;
    }

    public String getComment() {
        return comment;
    }

    public void setComment(String comment) {
        this.comment = comment;
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

    public int getOrderID() {
        return orderID;
    }

    public void setOrderID(int orderID) {
        this.orderID = orderID;
    }
}
