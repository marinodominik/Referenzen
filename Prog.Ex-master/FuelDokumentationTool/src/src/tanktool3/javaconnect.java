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
            Class.forName("com.mysql.jdbc.Driver");
            Connection conn = DriverManager.getConnection("jdbc:mysql://localhost/progEx","root","mysqlpassword");
            
            return conn;
        }catch(Exception e){
            JOptionPane.showMessageDialog(null,e);
            return null;
            
        }
    }
}
