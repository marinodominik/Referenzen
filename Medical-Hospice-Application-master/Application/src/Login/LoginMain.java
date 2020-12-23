package Login;

import NewPatient.NewPatientBuilder;
import javax.swing.*;


public class LoginMain {

    public static void main(String[] args) {
        JFrame LoginInterface = new LoginBuilder();
       // LoginInterface.setVisible(true);
        LoginInterface.setSize(280, 260);
        LoginInterface.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        LoginInterface.setLocationRelativeTo(null);
        LoginInterface.setTitle("Login");
        LoginInterface.setResizable(false);
        LoginInterface.setLayout(null);
    }
}
