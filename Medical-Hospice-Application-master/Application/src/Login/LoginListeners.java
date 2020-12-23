package Login;
import Database.Connector;
import Classes.Account;
import NewPatient.NewPatientBuilder;

import javax.swing.*;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.sql.*;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.sql.Connection;

/**
 * Created by dominik on 11.05.2017.
 */
public class LoginListeners implements ActionListener {

    public void loginAccsses(JLabel connect) {
        String query = "select * from Account where username = ? and password = ?";
        try {

            Connection connection = new Connector().connection();
            PreparedStatement pst = connection.prepareStatement(query);
            pst.setString(1, Account.getUsername());
            pst.setString(2, Account.getPassword());

            ResultSet rs = pst.executeQuery();

            if (rs.next()) {
                JFrame NewPatientInterfaceOpen = new NewPatientBuilder();
            } else{
                JOptionPane.showMessageDialog(null, "Account not Found");
                connect.setText("Username or Password is not correct");
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }


    @Override
    public void actionPerformed(ActionEvent e) {

    }
}
