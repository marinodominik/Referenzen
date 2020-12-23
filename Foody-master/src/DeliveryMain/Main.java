package DeliveryMain;

import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.List;


public class Main {




    public static List<Order> getOrderListByCustomerID(int customerID){
        Connector connector = new Connector();
        List<Order> orderList = new ArrayList<>();
        ResultSet rs = connector.query("SELECT * FROM Orders WHERE customerID = '" + customerID + "';");
        try {
            while(rs.next()){
                orderList.add(new Order(
                                        rs.getInt("ordersID"),
                                        getPaymentByOrderID(rs.getInt("ordersID")),
                                        rs.getDouble("amount"),
                                        rs.getInt("customerID"),
                                        rs.getInt("restaurantID") ,
                                        rs.getInt("status"),
                                        getRatingByOrderID(rs.getInt("ordersID")),
                                        getMealsOfTheOrderByOrderID(rs.getInt("ordersID"))));
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
        return orderList;
    }

    public static Customer getCustomerByCustomerEmail(String email){
        Connector connector = new Connector();
        ResultSet rs = connector.query("SELECT * FROM Customer WHERE email = '" + email + "';");
        Customer customer = null;
        try {
            if(rs.next()){
                customer = new Customer(rs.getString("address"),
                        rs.getString("city"),
                        rs.getInt("zip"),
                        rs.getInt("telephone"),
                        rs.getString("email"),
                        rs.getString("password"),
                        rs.getString("firstname"),
                        rs.getString("lastname"),
                        getOrderListByCustomerID(rs.getInt("customerID")),
                        rs.getInt("customerID"));
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
        return customer;
    }

    public static Customer getCustomerByCustomerID(int customerID){
        Connector connector = new Connector();
        Customer customer = null;
        ResultSet rs = connector.query("SELECT * FROM Customer WHERE customerID = '" + customerID + "';");
        try {
            if(rs.next()){
                customer = new Customer(rs.getString("address"),
                        rs.getString("city"),
                        rs.getInt("zip"),
                        rs.getInt("telephone"),
                        rs.getString("email"),
                        rs.getString("password"),
                        rs.getString("firstname"),
                        rs.getString("lastname"),
                        getOrderListByCustomerID(customerID),
                        rs.getInt("customerID"));
            }

        } catch (SQLException e) {
            e.printStackTrace();
        }
        return customer;
    }
/*
    //Redundant because same method as getRestaurantListByZip?
    public static Restaurant getRestaurantByZip(int zip){
        Connector connector = new Connector();
        ResultSet rs = connector.query("SELECT * FROM Restaurant WHERE zip = '" + zip + "';");
        Restaurant restaurant = null;
        try {
            restaurant = new Restaurant(
                    rs.getString("adress"),
                    rs.getString("city"),
                    rs.getInt("zip"),
                    rs.getInt("telephone"),
                    rs.getString("email"),
                    rs.getString("password"),
                    rs.getInt("restaurantID"),
                    rs.getString("restaurantName"),
                    getMembership(rs.getString("membership")), // <--------------------------
                    getOrdersByRestaurantID(rs.getInt("restaurantID")),
                    getRatingListByRestaurantID(rs.getInt("restaurantID")),
                    getMenuByRestaurantID(rs.getInt("restaurantID")));
        } catch (SQLException e) {
            e.printStackTrace();
        }
        return restaurant;
    }
*/
    public static Restaurant getRestaurantByRestaurantID(int restaurantID) {
        Connector connector = new Connector();
        Restaurant restaurant = null;
        ResultSet rs = connector.query("SELECT  * FROM Restaurant WHERE restaurantID = '" + restaurantID + "';");
        try {
            if(rs.next()){
                restaurant = new Restaurant(
                        rs.getString("address"),
                        rs.getString("city"),
                        rs.getInt("zip"),
                        rs.getInt("phonenumber"),
                        rs.getString("email"),
                        rs.getString("password"),
                        rs.getInt("restaurantID"),
                        rs.getString("restaurantName"),
                        getMembership(rs.getString("membership")),
                        getOrdersByRestaurantID(restaurantID),
                        getRatingListByRestaurantID(restaurantID),
                        getMenuByRestaurantID(restaurantID));
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }

        return restaurant;
    }

    public static List<Restaurant> getRestaurantListByZip(int zip){
        Connector connector = new Connector();
        List<Restaurant> restaurantList = new ArrayList<>();
        ResultSet rs = connector.query("SELECT * FROM Restaurant WHERE zip = '" + zip + "';");
        try {
            while(rs.next()){
                restaurantList.add( new Restaurant(
                        rs.getString("address"),
                        rs.getString("city"),
                        rs.getInt("zip"),
                        rs.getInt("phonenumber"),
                        rs.getString("email"),
                        rs.getString("password"),
                        rs.getInt("restaurantID"),
                        rs.getString("restaurantName"),
                        getMembership(rs.getString("membership")), // <--------------------------
                        getOrdersByRestaurantID(rs.getInt("restaurantID")),
                        getRatingListByRestaurantID(rs.getInt("restaurantID")),
                        getMenuByRestaurantID(rs.getInt("restaurantID"))));
            }

        } catch (SQLException e) {
            e.printStackTrace();
        }
        return restaurantList;
    }


    public static Payment getPaymentByOrderID(int orderID){
        Connector connector = new Connector();
        Payment payment = null;
        ResultSet rs = connector.query("SELECT * FROM Payment WHERE ordersID = '" + orderID + "';");
        try {
            if(rs.next()){
                payment = new Payment(
                        rs.getString("paymentMethod"),
                        rs.getInt("paymentID"),
                        rs.getInt("customerID"),
                        rs.getInt("ordersID"));
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }

        return payment;
    }


    public static Membership getMembership(String type){
        Membership membership = null;

        switch (type){
            case "regular" : membership = new Regular();
            case "display only" : membership = new DisplayOnly();
            case "premium" : membership = new Premium();
        }
        return membership;
    }




    public static Menu getMenuByRestaurantID(int restaurantID){
        Menu menu = null;
        Connector connector = new Connector();
        ResultSet rs = connector.query("SELECT  * FROM Meal WHERE restaurantID = " + Integer.toString(restaurantID) + ";");
        try {
            menu = new Menu();
            while(rs.next()){
                menu.addDishToMenu(new Meal(restaurantID,
                                            rs.getInt("mealID"),
                                            rs.getString("mealName"),
                                            rs.getString("description"),
                                            rs.getDouble("price")));
            }

        } catch (SQLException e) {
            e.printStackTrace();
        }
        return menu;
    }

    public static List<Rating> getRatingListByCustomerID(int customerID){
        List<Rating> ratingList = null;
        Connector connector = new Connector();
        ResultSet rs = connector.query("SELECT * FROM Rating WHERE customerID = '" + customerID + "';");
        try {
            while(rs.next()){
                ratingList.add( new Rating(
                        rs.getInt("ratingID"),
                        rs.getInt("stars"),
                        rs.getString("comment"),
                        rs.getInt("customerID"),
                        rs.getInt("restaurantID"),
                        rs.getInt("ordersID")));
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
        return ratingList;
    }

    public static Rating getRatingByOrderID(int ordersID){
        Rating rating = null;
        Connector connector = new Connector();
        ResultSet rs = connector.query("SELECT * FROM Rating WHERE ordersID = '" + ordersID + "';");
        try {
            if(rs.next()){
                rating = new Rating(
                        rs.getInt("ratingID"),
                        rs.getInt("stars"),
                        rs.getString("comment"),
                        rs.getInt("customerID"),
                        rs.getInt("restaurantID"),
                        rs.getInt("ordersID"));
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
        return rating;
    }

    public static List<Rating> getRatingListByRestaurantID(int restaurantID){
        List<Rating> ratinglist = new ArrayList<>();
        Connector connector = new Connector();
        ResultSet rs = connector.query("SELECT * FROM Rating WHERE restaurantID = '" + restaurantID + "';");
        try {
            while(rs.next()){
                ratinglist.add( new Rating(
                        rs.getInt("ratingID"),
                        rs.getInt("stars"),
                        rs.getString("comment"),
                        rs.getInt("customerID"),
                        rs.getInt("restaurantID"),
                        rs.getInt("orderID")));
            }

        } catch (SQLException e) {
            e.printStackTrace();
        }
        return ratinglist;
    }
/*
    public static Rating getRatingByRestaurantID(int restaurantID){
        Rating rating = null;
        Connector connector = new Connector();
        ResultSet rs = connector.query("SELECT * FROM Rating WHERE restaurantID = '" + restaurantID + "';");
        try {
            rating = new Rating(
                    rs.getInt("ratingID"),
                    rs.getInt("stars"),
                    rs.getString("comment"),
                    rs.getInt("customerID"),
                    rs.getInt("restaurantID"),
                    rs.getInt("orderID"));

        } catch (SQLException e) {
            e.printStackTrace();
        }
        return rating;
    }
*/
    public static List<Order> getOrdersByRestaurantID(int restaurantID){
        List<Order> orderList = new ArrayList<>();
        Connector connector = new Connector();
        ResultSet rs = connector.query("SELECT * FROM Orders  WHERE restaurantID = '" + restaurantID + "';");
        try {
            while(rs.next()){
                orderList.add(new Order(
                        rs.getInt("ordersID"),
                        getPaymentByOrderID(rs.getInt("ordersID")),
                        rs.getDouble("amount"),
                        rs.getInt("customerID"),
                        rs.getInt("restaurantID") ,
                        rs.getInt("status"),
                        getRatingByOrderID(rs.getInt("ordersID")),
                        getMealsOfTheOrderByOrderID(rs.getInt("ordersID"))));
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
        return orderList;
    }

    public static List<Meal> getMealsOfTheOrderByOrderID(int orderID){
        List<Meal> mealList = new ArrayList<>();
        Connector connector = new Connector();
        ResultSet rs = connector.query("SELECT restaurantID, mealID, mealName, description, price FROM MealOfOrder NATURAL JOIN Orders NATURAL JOIN Meal WHERE ordersID = '" + orderID + "';");
        try {
            while(rs.next()){
                mealList.add(new Meal(rs.getInt("restaurantID"),
                                        rs.getInt("mealID"),
                                        rs.getString("mealName"),
                                        rs.getString("description"),
                                        rs.getDouble("price")));
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
        return mealList;
    }

    public static Meal getMealByMealID(int mealID){
        Meal meal = null;
        Connector connector = new Connector();
        ResultSet rs = connector.query("SELECT * FROM Meal WHERE mealID = '"+mealID+"';");
        try{
            while(rs.next()){
                meal = new Meal(rs.getInt("restaurantID"), rs.getInt("mealID"), rs.getString("mealName"),
                        rs.getString("description"), rs.getDouble("price"));
            }
        }catch (SQLException e){
            e.printStackTrace();
        }
        return meal;
    }

    public static void placeAnOrder(Order order){
        Connector connector = new Connector();
        Payment payment = order.getPaymentMethod();

        // saving payment data, without OrdersID
        String query = "INSERT INTO Payment(customerID, paymentMethod, ordersID) VALUES('"+order.getCustomerID()+"', '"+payment.getPaymentMethod()+"', null);";
        // saving Orders data
        String query2 = "INSERT INTO Orders(restaurantID, paymentID, customerID, amount, ratingID, status) " +
                       "VALUES ('"+order.getRestaurantID()+"', (SELECT paymentID FROM Payment WHERE paymentID = (SELECT max(paymentID) FROM Payment)), '"+order.getCustomerID()+"', " +
                       " '"+order.getAmountToPay()+"', null, '"+order.getStatus()+"' ); ";
        // saving ordersID to existing payment row
        String query3 = "UPDATE Payment SET OrdersID = (SELECT ordersID FROM Orders WHERE ordersID = (select max(ordersID) FROM Orders));";

        connector.executeUpdate(query);
        connector.executeUpdate(query2);
        connector.executeUpdate(query3);

        // saving meals to orders
        for(Meal meal : order.getMealList()){
            String query4 = "INSERT INTO MealOfOrder(mealID, ordersID) VALUES('"+meal.getMealID()+"', (SELECT ordersID FROM Orders WHERE ordersID = (select max(ordersID) FROM Orders)));";
            connector.executeUpdate(query4);
        }
    }
}
