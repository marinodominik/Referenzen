package DeliveryMain;

import java.util.List;


public class Test {
    public static void main(String[] args){
        List<Restaurant> restaurantList = Main.getRestaurantListByZip(12345);
        for(Restaurant r : restaurantList){
            System.out.println("Rest. name: " + r.getRestaurantName() + " Payment: " + r.getMembership().getClass().getName());
        }

        List<Order> orderList = Main.getOrderListByCustomerID(1);
        Customer customerByID = Main.getCustomerByCustomerID(1);
        System.out.println("Customer: " + customerByID.getFirstName() + " got these meals at his last order: ");
        for(Meal meal : customerByID.getOrderList().get(0).getMealList()){
            System.out.println(meal.getMealName());
        }
        System.out.println("For this order he has paid: " + customerByID.getOrderList().get(0).getAmountToPay());

        Customer customerByEmail = Main.getCustomerByCustomerEmail("nima@muster.com");
        //System.out.println(customerByEmail.getFirstName());

        Restaurant restaurantByID = Main.getRestaurantByRestaurantID(1);
        //System.out.println(restaurantByID.getRestaurantName());

        Payment paymentByOrderID = Main.getPaymentByOrderID(1);
        Menu menubyRestaurantID = Main.getMenuByRestaurantID(1);
        //System.out.println(menubyRestaurantID.getDishlist().get(0).getMealName());

        List<Rating> ratingsByCustomerID = Main.getRatingListByCustomerID(1);
        Rating ratingByOrderID = Main.getRatingByOrderID(1);
        List<Rating> ratingsByRestaurantID = Main.getRatingListByRestaurantID(1);
        List<Order> getOrdersByRestaurantID = Main.getOrdersByRestaurantID(1);
        List<Meal> getMealsByOrderID = Main.getMealsOfTheOrderByOrderID(1);
        Meal meal = Main.getMealByMealID(1);
        System.out.println("Got meal: " + meal.getMealName());
        // placing an order is working
        //Order order = new Order(12,new Payment("VISA", 12, 1, 12),1,1);
        //order.addMealToOrderList(new Meal(1,1,"Spaghetti Bolungnese","Tomatensoße mit Hackfleisch und Gewürzen",7.99));
        //order.addMealToOrderList(new Meal(1,2,"Lasange","Normale Lasange nicht soscheiß ein veganer ",12.56));
        //Main.placeAnOrder(order);

        //System.out.println(order.getAmountToPay());
    }
}
