package DeliveryMain;

import com.mysql.jdbc.*;
import java.sql.*;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.Statement;


/**
 * Used to establish a connection and query on DB
 */
public class Connector{

    private String user = "root";
    private String password = "root";
    public Connection connectToDB(){
        Connection conn = null;
        try {
            Class.forName("com.mysql.jdbc.Driver");
            conn = DriverManager.getConnection("jdbc:mysql://52.57.10.17:3306/SWE-D", user, password);
        } catch (SQLException e) {
            e.printStackTrace();
        }
        catch (ClassNotFoundException e){
            e.printStackTrace();
        }
        return conn;
    }

    public ResultSet query(String query){
        Connection connection = connectToDB();
        ResultSet resultSet = null;
        try {
            PreparedStatement preparedStatement = connection.prepareStatement(query);
            resultSet = preparedStatement.executeQuery();
        } catch (SQLException e) {
            e.printStackTrace();
        }
        return resultSet;
    }

    public void executeUpdate(String query){
        Connection connection = connectToDB();
        Statement statement = null;
        try {
            statement = connection.createStatement();
            statement.executeUpdate(query);
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
}
