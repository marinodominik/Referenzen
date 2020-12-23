<%-- Created by IntelliJ IDEA. --%>
<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<%@ page import="DeliveryMain.*" %>
<%@ page import="java.util.LinkedList" %>
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
    <!--Search field-->
    <div class="search-container container">
      <form class="search-form col-md-8 col-md-offset-2" action="restaurants.jsp" method="get">
        <input id="search_field" name="zip" class="input-lg col-xs-10" oninvalid="setCustomValidity('Enter a correct zip')" type="number" placeholder="Enter your zip here">
        <button type="submit"  class="btn-lg col-xs-2">Search</button>
      </form>
    </div>
    <!--Footer-->
    <div class="seperator">
    </div>
  </body>
</html>