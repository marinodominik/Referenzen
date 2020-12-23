package DeliveryMain;

/**
 * Created by Erlandas on 02/06/2017.
 */
public class Meal {
    private int restaurantID;
    private int mealID;
    private String mealName;
    private String mealDescription;
    private double mealPrice;

    public Meal(int restaurantID, int mealID, String mealName, String mealDescription, double mealPrice) {
        this.restaurantID = restaurantID;
        this.mealID = mealID;
        this.mealName = mealName;
        this.mealDescription = mealDescription;
        this.mealPrice = mealPrice;
    }

    public int getRestaurant() {
        return restaurantID;
    }

    public void setRestaurant(Restaurant restaurant) {
        this.restaurantID = restaurantID;
    }

    public int getMealID() {
        return mealID;
    }

    public void setMealID(int mealID) {
        this.mealID = mealID;
    }

    public String getMealName() {
        return mealName;
    }

    public void setMealName(String mealName) {
        this.mealName = mealName;
    }

    public String getMealDescription() {
        return mealDescription;
    }

    public void setMealDescription(String mealDescription) {
        this.mealDescription = mealDescription;
    }

    public double getMealPrice() {
        return mealPrice;
    }

    public void setMealPrice(double mealPrice) {
        this.mealPrice = mealPrice;
    }

    public String getHTML(){
        String html = "<div>";
        html += "<h1 onclick = \"addToCart("+ Integer.toString(this.getMealID()) +")\">" + getMealName() + "</h1>";
        html += "<p>" + getMealDescription() + "</p>";
        html += "<p>" + getMealPrice() + "€</p>";
        html += "</div>";
        return html;
    }
}
