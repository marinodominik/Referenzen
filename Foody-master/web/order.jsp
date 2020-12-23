
<%@ page import="org.json.simple.JSONArray" %>
<%@ page import="java.util.Iterator" %>
<%@ page import="org.json.simple.parser.JSONParser" %>
<%@ page import="java.util.LinkedList" %>
<%@ page import="java.net.URLDecoder" %>

<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<html>
<head>
    <title>Foody</title>
    <link rel="stylesheet" type="text/css" href="css/main.css">
    <link rel="stylesheet" type="text/css" href="css/bootstrap.css">
    <script src="js/jquery.js"></script>
    <script src="js/jquery_cookie.js"></script>
    <script>
        $(function () {
            if($.cookie("cart")===""){
                window.location = "/index.jsp";
            }
        });
    </script>
</head>
<body>
<!--Header-->
<div class="seperator">
    <h1>FOODY</h1>
</div>
<!--Get Meals by cart-->
<%
    Cookie cookies[] = request.getCookies();
    if(cookies.length > 0){
        out.println(Integer.toString(cookies.length) + " cookies found!");
        for(int i = 0; i < cookies.length; i++){
            out.println(cookies[i].getName());
        }
    }
    else{
        out.println("No cookies found!");
    }
    Cookie cookie_cart = null;
    for(Cookie c: cookies){
        if(c.getName().equals("cart")){
            cookie_cart = c;
            out.println("<p>"+c.toString()+"</p>");
            out.println("<p>"+c.getValue()+"</p>");
            break;
        }
    }
    if(cookie_cart!=null){
        String str_cart = cookie_cart.getValue();
        str_cart = URLDecoder.decode(str_cart, "UTF-8");
        out.println(str_cart);
        LinkedList<Long> cart = new LinkedList<>();
        JSONParser parser = new JSONParser();
        JSONArray json_cart = (JSONArray) parser.parse(str_cart);
        Iterator<Long> iterator = json_cart.iterator();
        while (iterator.hasNext()) {
            Long id = iterator.next();
            long cache = id;
            int meal_id = (int) cache;
            cart.add(id);
        }
        for(Long l: cart){
            out.println("<p>"+Long.toString(l)+"</p>");
        }
    }
    else{
        out.println("Cookie not found!");
    }
%>

<div class="container">

    <!---<button onclick="submit()">Submit</button>--->
</div>
<!--Footer-->
<div class="seperator">
</div>
</body>
</html>

