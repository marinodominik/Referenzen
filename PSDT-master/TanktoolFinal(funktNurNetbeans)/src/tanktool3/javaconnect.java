/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tanktool3;
import java.sql.*;
import javax.swing.*;

/**
 *
 * @author Emre
 */
public class javaconnect {
    Connection conn =null;
    
    
    public static Connection ConnecrDb(){
        
        try{
            /*
            Connect to the jdbc driver
            */
            Class.forName("com.mysql.jdbc.Driver");
            
            //login to the database in mysql
                Connection conn = DriverManager.getConnection("jdbc:mysql://localhost/progex","root","YOUR-DB-PASSWORD");
            
            //return the connection
            return conn;
        }catch(Exception e){
            
            //if there is no connection print in a dialog window an exception
            JOptionPane.showMessageDialog(null,e);
            return null;
            
        }
    }
}
