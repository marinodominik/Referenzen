<%@ page import="DeliveryMain.Main" %>
<%@ page import="DeliveryMain.Menu" %>
<%@ page import="DeliveryMain.Meal" %>
<%@ page import="java.util.List" %>
<%@ page import="DeliveryMain.Restaurant" %><%--
  Created by IntelliJ IDEA.
  User: nicolasdarr
  Date: 11.06.17
  Time: 11:32
  To change this template use File | Settings | File Templates.
--%>
<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<html>
<head>
    <title>Foody</title>
    <link rel="stylesheet" type="text/css" href="css/main.css">
    <link rel="stylesheet" type="text/css" href="css/bootstrap.css">
    <script src="js/jquery.js"></script>
    <script src="js/jquery_cookie.js"></script>
    <script>
        var cart = [];
        function addToCart(id) {
            cart.push(id);
            alert("You selected the item with id: " + id);
        }
        function submit(){
            $.removeCookie("cart");
            var json_cart = JSON.stringify(cart);
            var now = new Date();
            var time = now.getTime();
            var expireTime = time + 1000 * 36000;
            now.setTime(expireTime);
            $.cookie("cart",json_cart,{ expires : 1, path :'/' });
            alert(json_cart);
            window.location = "/order.jsp";
        }

    </script>
</head>
<body>
<!--Header-->
<div class="seperator">
    <h1>FOODY</h1>
</div>
<!--Restaurants output-->
<%
    //get restaurants
    int id = Integer.parseInt(request.getParameter("id"));
    Menu menu = Main.getMenuByRestaurantID(id);
    List<Meal> meals;
    meals = menu.getDishlist();

%>
<div class="container">
    <% if (meals.isEmpty()) {
        out.println("No results found");
    } else {
        out.println("<p>We found " + Integer.toString(meals.size()) + " meals for this restaurant</p>");
        for (Meal m : meals) {
            out.println(m.getHTML());
        }
    }
    %>
    <button onclick="submit()">Submit</button>
</div>
<!--Footer-->
<div class="seperator">
</div>
</body>
</html>
