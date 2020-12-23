package Login;

import Classes.Account;

import javax.swing.*;
import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;

/**
 * Created by dominik on 11.05.2017.
 */


public class LoginBuilder extends JFrame {

    JLabel user;
    JLabel password;
    JLabel connection;
    JButton loginAccess;
    JTextField logName;
    JPasswordField logPassword;

    public LoginBuilder(){
        setVisible(true);
        //Define different schriftzüge
        Font schrift = new Font("", Font.BOLD + Font.ITALIC, 20);

        //labels
        user = new JLabel("User: ");
        user.setBounds(30, 50, 80, 60);
        user.setFont(schrift);
        add(user);

        password = new JLabel("Password: ");
        password.setBounds(30, 80, 120,60);
        password.setFont(schrift);
        add(password);

        connection = new JLabel("Status");
        connection.setBounds(150, 40 , 120, 20);
        add(connection);

        //Buttons
        loginAccess = new JButton("Login");
        loginAccess.setBounds(105, 160, 70, 30);
        loginAccess.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                new LoginListeners().loginAccsses(connection);
            }
        });
        add(loginAccess);

        //TextField
        logName = new JTextField();
        logName.addActionListener(new LoginListeners(){
            @Override
            public void actionPerformed(ActionEvent event){
                String query = logName.getText().toString();
                Account.setUsername(query);

                System.out.println("logname: "+Account.getUsername());;
            }

        });
        logName.setBounds(150,70,110,20);
        add(logName);

        //PasswordField
        logPassword = new JPasswordField();
        logPassword.addActionListener(new LoginListeners(){
            @Override
            public void actionPerformed(ActionEvent event){
                String query = logPassword.getText().toString();
                Account.setPassword(query);

                System.out.println("Password: "+Account.getPassword());
            }
        });
        logPassword.setBounds(150, 100, 110, 20);
        add(logPassword);
    }

}
