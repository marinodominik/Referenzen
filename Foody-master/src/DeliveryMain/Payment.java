package DeliveryMain;


public class Payment {
    private String paymentMethod;
    private int paymentID;
    private int userID;
    private int orderID;

    public Payment(String paymentMethod, int paymentID, int userID, int orderID) {
        this.paymentMethod = paymentMethod;
        this.paymentID = paymentID;
        this.userID = userID;
        this.orderID = orderID;
    }

    public String getPaymentMethod() {
        return paymentMethod;
    }

    public void setPaymentMethod(String paymentMethod) {
        this.paymentMethod = paymentMethod;
    }

    public int getPaymentID() {
        return paymentID;
    }

    public void setPaymentID(int paymentID) {
        this.paymentID = paymentID;
    }

    public int getUserID() {
        return userID;
    }

    public void setUserID(int userID) {
        this.userID = userID;
    }

    public int getOrderID() {
        return orderID;
    }

    public void setOrderID(int orderID) {
        this.orderID = orderID;
    }
}
