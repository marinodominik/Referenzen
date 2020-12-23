package NewPatient;

import javax.swing.*;
import java.awt.*;

/**
 * Created by dominik on 11.05.2017.
 */
public class NewPatientBuilder extends JFrame {
    JLabel firstname;
    JLabel lastname;
    JLabel birth;
    JLabel gender;
    JLabel emergenceContact;
    JLabel homeaddress;
    JLabel weight;
    JLabel height;
    JLabel activity;
    JLabel healthInsurance;
    JLabel healthInsuranceArt;
    JLabel phone;
    JLabel medicaments;
    JLabel illness;
    JLabel roomNr;

    JButton finish;
    JButton addMedicaments;
    JButton addIllness;

    JTextField textLastname;
    JTextField textFirstname;
    JTextField textPhone;
    JTextField textcityaddress;
    JTextField textPLZ;
    JTextField textHomeaddress;
    JTextField textEmergencyConLastname;
    JTextField textEmergencyConFirstname;
    JTextField textHealthInsurencekind;
    JTextField textMedicaments;
    JTextField textIllness;


    JComboBox comboDay;
    JComboBox comboMonth;
    JComboBox comboYear;
    JComboBox comboGender;
    JComboBox comboRoom;
    JComboBox comboWeight;
    JComboBox comboHeight;
    JComboBox comboActivity;
    JComboBox comboHealthInsurence;



    public NewPatientBuilder(){
        setVisible(true);
        setSize(650, 400);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLocationRelativeTo(null);
        setTitle("Add Patient");
        setResizable(false);
        setLayout(null);

        //schriftarten
        Font schrift = new Font("", Font.BOLD + Font.ITALIC, 12);


        //labels
        lastname = new JLabel("Lastname: ");
        lastname.setBounds(30, 30, 80, 60);
        lastname.setFont(schrift);
        add(lastname);

        firstname = new JLabel("Firstname: ");
        firstname.setBounds(130, 30, 80, 60);
        firstname.setFont(schrift);
        add(firstname);

        birth = new JLabel("Birth: ");
        birth.setBounds(30, 90, 80, 60);
        birth.setFont(schrift);
        add(birth);

        gender = new JLabel("Gender: ");
        gender.setBounds(30, 150, 80, 60);
        gender.setFont(schrift);
        add(gender);

        emergenceContact = new JLabel("Emergence Contact: ");
        emergenceContact.setBounds(30, 230, 140, 60);
        emergenceContact.setFont(schrift);
        add(emergenceContact);

        homeaddress = new JLabel("Homeaddress: ");
        homeaddress.setBounds(30, 290, 100, 60);
        homeaddress.setFont(schrift);
        add(homeaddress);

        weight = new JLabel("Weight (in KG): ");
        weight.setBounds(340, 30, 100, 60);
        weight.setFont(schrift);
        add(weight);

        height = new JLabel("Height (in cm): ");
        height.setBounds(440, 30, 100, 60);
        height.setFont(schrift);
        add(height);

        activity = new JLabel("Activity: ");
        activity.setBounds(340, 90, 80, 60);
        activity.setFont(schrift);
        add(activity);

        phone = new JLabel("Phone: ");
        phone.setBounds(440, 90, 80, 60);
        phone.setFont(schrift);
        add(phone);

        healthInsurance = new JLabel ("Health Insur.: ");
        healthInsurance.setBounds(340, 150, 120, 60);
        healthInsurance.setFont(schrift);
        add(healthInsurance);

        healthInsuranceArt = new JLabel("Health Insur. k: ");
        healthInsuranceArt.setBounds(440, 150, 120, 60);
        healthInsuranceArt.setFont(schrift);
        add(healthInsuranceArt);

        medicaments = new JLabel("Medicaments:");
        medicaments.setBounds(340,230,120, 30);
        medicaments.setFont(schrift);
        add(medicaments);

        illness = new JLabel("Illness:");
        illness.setBounds(340, 265, 120, 30);
        illness.setFont(schrift);
        add(illness);

        roomNr = new JLabel("Room Nr:");
        roomNr.setBounds(130, 165, 120, 30);
        roomNr.setFont(schrift);
        add(roomNr);

        //Button
        finish = new JButton(("Finish"));
        finish.setBounds(500, 320, 120, 30);
        finish.setFont(schrift);
        add(finish);

        addMedicaments = new JButton("ADD");
        addMedicaments.setBounds(540, 235, 60, 25);
        add(addMedicaments);

        addIllness = new JButton("ADD");
        addIllness.setBounds(540,270,60,25);
        add(addIllness);

        //textfield
        textLastname = new JTextField();
        textLastname.setBounds(30, 70, 90, 25);
        add(textLastname);

        textFirstname = new JTextField();
        textFirstname.setBounds(130, 70, 90, 25);
        add(textFirstname);

        textEmergencyConLastname = new JTextField("Lastname");
        textEmergencyConLastname.setBounds(30, 270, 90, 25);
        add(textEmergencyConLastname);

        textEmergencyConFirstname = new JTextField("Firstname");
        textEmergencyConFirstname.setBounds(130, 270 , 90, 25);
        add(textEmergencyConFirstname);

        textHomeaddress = new JTextField("Strasse");
        textHomeaddress.setBounds(30, 335, 90 ,25);
        add(textHomeaddress);

        textPLZ = new JTextField("PLZ");
        textPLZ.setBounds(130, 335, 90, 25);
        add(textPLZ);

        textcityaddress = new JTextField("City");
        textcityaddress.setBounds(130,305,90,25);
        add(textcityaddress);


        textPhone = new JTextField();
        textPhone.setBounds(440, 130, 90, 25);
        add(textPhone);

        textHealthInsurencekind = new JTextField();
        textHealthInsurencekind.setBounds(440, 190, 90, 25);
        add(textHealthInsurencekind);

        textMedicaments = new JTextField();
        textMedicaments.setBounds(440, 235, 90, 25);
        add(textMedicaments);

        textIllness = new JTextField();
        textIllness.setBounds(440, 270, 90,25);
        add(textIllness);
        //combobox

        comboMonth = new JComboBox<>(NewPatientFillCombonents.fillMonth());
        comboMonth.setBounds(130, 105, 90, 25);
        add(comboMonth);

        comboDay = new JComboBox<Integer>(NewPatientFillCombonents.fillDate());
        comboDay.setBounds(30, 135, 90, 25);
        add(comboDay);

        comboYear = new JComboBox<Integer>(NewPatientFillCombonents.fillYear());
        comboYear.setBounds(130, 135, 90, 25);
        add(comboYear);

        comboGender = new JComboBox(NewPatientFillCombonents.fillGender());
        comboGender.setBounds(30, 190, 90, 25);
        add(comboGender);

        comboRoom = new JComboBox();
        comboRoom.setBounds(130, 190, 90, 25);
        add(comboRoom);

        comboHeight = new JComboBox(NewPatientFillCombonents.fillHeight());
        comboHeight.setBounds(340, 70, 90, 25);
        add(comboHeight);

        comboWeight = new JComboBox(NewPatientFillCombonents.fillWeight());
        comboWeight.setBounds(440, 70,90,25);
        add(comboWeight);

        comboActivity = new JComboBox(NewPatientFillCombonents.fillActivity());
        comboActivity.setBounds(340, 130, 90, 25);
        add(comboActivity);

        comboHealthInsurence = new JComboBox(NewPatientFillCombonents.fillHealthInsurence());
        comboHealthInsurence.setBounds(340, 190, 90, 25);
        add(comboHealthInsurence);

    }
}

