<%@ page import="DeliveryMain.Main" %><%--
  Created by IntelliJ IDEA.
  User: nicolasdarr
  Date: 08.06.17
  Time: 22:02
  To change this template use File | Settings | File Templates.
--%>
<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<%-- Created by IntelliJ IDEA. --%>
<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<%@ page import="DeliveryMain.*" %>
<%@ page import="java.util.List" %>
<html>
    <head>
        <title>Foody</title>
        <link rel="stylesheet" type="text/css" href = "css/main.css">
        <link rel="stylesheet" type="text/css" href = "css/bootstrap.css">
    </head>
    <body>
    <!--Header-->
        <div class="seperator">
            <h1>FOODY</h1>
        </div>
        <!--Restaurants output-->
        <%
            //get restaurants
            int zip = Integer.parseInt(request.getParameter("zip"));
            List<Restaurant> restaurants = Main.getRestaurantListByZip(zip);

        %>
        <div class="container">
            <%  if(restaurants.isEmpty()){
                    out.println("No results found");
                }
                else{
                    out.println("<p>We found " + Integer.toString(restaurants.size()) + " restaurants nearby you</p>");
                    for(Restaurant r: restaurants){
                        out.println(r.getHTMLString());
                    }
                }
            %>
        </div>
        <!--Footer-->
        <div class="seperator">
        </div>
    </body>
</html>
