# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Heart_Sound_Identification_System.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.
# The illutration picture is provided by <a href="http://www.freepik.com">Designed by Freepik</a>

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import*
from PyQt5.QtGui import*
from PyQt5.QtCore import*
from PyQt5.uic import loadUi
from firebase import firebase
from ErrorUI import ErrorActivity
from Interface2_AudioAcquisition import AcquisitionGUI
import threading
import sys
import time


class LoginSignUpActivity(QtGui.QMainWindow):
    """
    A class for login page and sign up page of the application

    ...

    Methods
    -------
    login()
        Shift or display login page
    signup()
        Shift or display to sign up page
    resizeWidget()
        Resize the size of widget
    loginVerification()
        Retrieve the login information for verification
    signupVerification()
        Retrieve the sign up information for verification
    firebase_login_signup_verification(isLogin=None)
        Verify the information via database in Firebase
    """
    def __init__(self):
        super().__init__()
        #Link to the UI file named LoginandSignup.ui
        loadUi("LoginandSignup.ui",self)
        """
        Initialization of the variables of both login page and sign up page

        Attributes
        -----------
        self.login_username : the login username
        self.login_password : the login password
        self.signup_firstname : the sign up first name
        self.signup_lastname : the sign up last name
        self.signup_age : the sign up age
        self.signup_gender : the sign up gender
        self.signup_weight : the sign up weight
        self.signup_height : the sign up height
        self.signup_username : the sign up username 
        self.signup_password : the sign up password
        self.information_collection : collect the user information after login
        """
        self.login_username = ""
        self.login_password = ""
        self.signup_firstname=""
        self.signup_lastname=""
        self.signup_age=0
        self.signup_gender=""
        self.signup_weight=""
        self.signup_height=""
        self.signup_username=""
        self.signup_password=""
        self.information_collection = []
        #Load the image
        oImage = QImage("Icon/5005356.jpg")
        #define the scale percentage
        scale_percent = 30
        original_width = oImage.width()#extract width of the original image
        original_height = oImage.height()#extract height of the original image

        #Scale the image
        image_width = int(original_width * scale_percent / 100)#scale image horizontally
        image_height = int(original_width * (scale_percent) / 100)#scale image horizontally

        #Scale the original image
        sImage = oImage.scaled(QSize(image_width, image_height))
        #Set the scaled image to label
        self.label_Design.setPixmap(QtGui.QPixmap(sImage))

        #Load the image
        oImage = QImage("Icon/heartHeader.png")
        #Scale the image
        sImage = oImage.scaled(QSize(image_width, image_height))

        #Set the scaled image to label
        self.label_icon.setPixmap(QtGui.QPixmap(oImage))
        
        #Align the content of the label to right
        self.label_icon.setAlignment(Qt.AlignRight)

        #Set the header of the system        
        headerMessageSystem = """<p align="center"><span style="font-size:16pt; verticle-align:super; color:#E74C3C;">Heart Sound Detection</span></p>"""
        self.label_systemWord.setText(headerMessageSystem)
        self.label_subHead.setText("\n")
        #Set the gratitude pharse
        headerMessage = """<p align="center"><span style="font-size:24pt; verticle-align:super; color:#E74C3C;"><strong>Welcome Back</strong></span></p>"""
        self.label_welcome.setText("\n"+headerMessage)
        self.label_pharse1.setText("o Diagnose your heart")
        self.label_pharse2.setText("\no Discover your health")
        self.label_pharse3.setText("\no Share with your friends")
        #Design the words of the label
        self.label_pharse1.setStyleSheet("QLabel {font: 14pt 'Arial'; "+
                                         "color: #884EA0;}")
        self.label_pharse2.setStyleSheet("QLabel {font: 14pt 'Arial'; "+
                                         "color: #884EA0;}")
        self.label_pharse3.setStyleSheet("QLabel {font: 14pt 'Arial'; "+
                                         "color: #884EA0;}")
        #Align the content of the label to center
        self.label_pharse1.setAlignment(Qt.AlignLeft)
        self.label_pharse2.setAlignment(Qt.AlignLeft)
        self.label_pharse3.setAlignment(Qt.AlignLeft)
        #Set the text on login and sign up button
        self.pushButton_login.setText("Login")
        self.pushButton_signup.setText(" / Sign up")
        #Design the login and sign up button
        self.pushButton_login.setStyleSheet("QPushButton{color:#00B2FF;" 
                                      'font: 18pt "Nirmala UI";' +
                                      "border-radius: 15px;" +
                                      "text-align: right;" +
                                      "background-color: rgba(255,255,255,0%);}")
        self.pushButton_signup.setStyleSheet("QPushButton{color:#AAB7B8;" 
                                      'font: 18pt "Nirmala UI";' +
                                      "border-radius: 15px;" +
                                      "text-align: left;" +
                                      "background-color: rgba(255,255,255,0%);}")
        #design the frame of contain the login and sign up details
        self.frame_text_login.setStyleSheet("QFrame{border-style: solid; border-radius: 20px; background-color:rgb(26, 82, 118);}")
        self.frame_text_signup.setStyleSheet("QFrame{border-style: solid; border-radius: 20px; background-color:rgb(26, 82, 118);}")
        #design the label of username and password
        usernameMessage = """<p align="center"><span style="font-size:14pt; verticle-align:super; color:#ECF0F1;"><strong>Username</strong></span></p>"""
        passwordMessage = """<p align="center"><span style="font-size:14pt; verticle-align:super; color:#ECF0F1;"><strong>Password</strong></span></p>"""
        #set the text to the label
        self.label_username_login.setText(usernameMessage)
        self.label_password_login.setText(passwordMessage)
        
        #Design the edit text for login 
        self.lineEdit_username_login.setStyleSheet("QLineEdit{color:#17202A;" 
                                      'font: 16pt "Nirmala UI";' +
                                      "border-radius: 15px;" +
                                      "background-color: rgb(255,255,255);}")

        self.lineEdit_password_login.setStyleSheet("QLineEdit{color:#17202A;" 
                                      'font: 16pt "Nirmala UI";' +
                                      "border-radius: 15px;" +
                                      "background-color: rgb(255,255,255);}")

        #set the content for password text editor into echo mode 
        self.lineEdit_password_login.setEchoMode(QLineEdit.Password)
        #Design the edit text for sign up
        self.lineEdit_firstName.setStyleSheet("QLineEdit{color:#17202A;" 
                                      'font: 12pt "Nirmala UI";' +
                                      "border-radius: 15px;" +
                                      "background-color: rgb(255,255,255);}")
        self.lineEdit_lastName.setStyleSheet("QLineEdit{color:#17202A;" 
                                      'font: 12pt "Nirmala UI";' +
                                      "border-radius: 15px;" +
                                      "background-color: rgb(255,255,255);}")
        self.lineEdit_age.setStyleSheet("QLineEdit{color:#17202A;" 
                                      'font: 12pt "Nirmala UI";' +
                                      "border-radius: 15px;" +
                                      "background-color: rgb(255,255,255);}")
        self.lineEdit_height.setStyleSheet("QLineEdit{color:#17202A;" 
                                      'font: 12pt "Nirmala UI";' +
                                      "border-radius: 15px;" +
                                      "background-color: rgb(255,255,255);}")
        self.lineEdit_weight.setStyleSheet("QLineEdit{color:#17202A;" 
                                      'font: 12pt "Nirmala UI";' +
                                      "border-radius: 15px;" +
                                      "background-color: rgb(255,255,255);}")
        self.lineEdit_username.setStyleSheet("QLineEdit{color:#17202A;" 
                                      'font: 12pt "Nirmala UI";' +
                                      "border-radius: 15px;" +
                                      "background-color: rgb(255,255,255);}")
        self.lineEdit_password.setStyleSheet("QLineEdit{color:#17202A;" 
                                      'font: 12pt "Nirmala UI";' +
                                      "border-radius: 15px;" +
                                      "background-color: rgb(255,255,255);}")
        #set the content for password text editor into echo mode 
        self.lineEdit_password.setEchoMode(QLineEdit.Password)
        #design the label for each text editor
        firstnameLabel = """<p align="center"><span style="font-size:14pt; verticle-align:super; color:#ECF0F1;"><strong>First Name </strong></span></p>"""
        lastnameLabel = """<p align="center"><span style="font-size:14pt; verticle-align:super; color:#ECF0F1;"><strong>Last Name </strong></span></p>"""
        ageLabel = """<p align="center"><span style="font-size:14pt; verticle-align:super; color:#ECF0F1;"><strong>Age </strong></span></p>"""
        genderLabel = """<p align="center"><span style="font-size:14pt; verticle-align:super; color:#ECF0F1;"><strong>Gender </strong></span></p>"""
        heightLabel = """<p align="center"><span style="font-size:14pt; verticle-align:super; color:#ECF0F1;"><strong>Height </strong></span></p>"""
        weightLabel = """<p align="center"><span style="font-size:14pt; verticle-align:super; color:#ECF0F1;"><strong>Weight </strong></span></p>"""
        usernameLabel = """<p align="center"><span style="font-size:14pt; verticle-align:super; color:#ECF0F1;"><strong>Username </strong></span></p>"""
        passwordLabel = """<p align="center"><span style="font-size:14pt; verticle-align:super; color:#ECF0F1;"><strong>Password </strong></span></p>"""
        #set the text to the label
        self.label_firstName.setText(firstnameLabel)
        self.label_lastName.setText(lastnameLabel)
        self.label_age.setText(ageLabel)
        self.label_gender.setText(genderLabel)
        self.label_height.setText(heightLabel)
        self.label_weight.setText(weightLabel)
        self.label_username.setText(usernameLabel)
        self.label_password.setText(passwordLabel)
        self.radioButton_male.setText("Male")
        self.radioButton_female.setText("Female")
        #set the radio button
        self.radioButton_male.setStyleSheet("QRadioButton{color:#ECF0F1;" 
                                      'font: 14pt "Nirmala UI";' +
                                      "background-color: rgb(26,82,118);}")
        self.radioButton_female.setStyleSheet("QRadioButton{color:#ECF0F1;" 
                                      'font: 14pt "Nirmala UI";' +
                                      "background-color: rgb(26,82,118);}")
        #design the button for login and sign up
        self.pushButton_login_2.setStyleSheet("QPushButton{color:#ECF0F1;" 
                                      'font: 14pt "Nirmala UI";' +
                                      "border-style : solid;" +
                                      "border-radius: 15px;" +
                                      "background-color: rgb(212, 172, 13);}")

        self.pushButton_signup_2.setStyleSheet("QPushButton{color:#ECF0F1;" 
                                      'font: 14pt "Nirmala UI";' +
                                      "border-style : solid;" +
                                      "border-radius: 15px;" +
                                      "background-color: rgb(212, 172, 13);}")
        #define the shadow effect
        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(30)
        self.shadow.setXOffset(0)
        self.shadow.setYOffset(0)
        self.shadow.setColor(QColor(0,0,0,100))
        self.stackedWidget.setGraphicsEffect(self.shadow)
        #design the label of error message
        self.label_errorMessage.setStyleSheet("QLabel {font: 14pt 'Arial'; "+
                                         "color:#E74C3C;}")

        self.label_signupErrorMessage.setStyleSheet("QLabel {font: 14pt 'Arial'; "+
                                         "color:#E74C3C;}")
        #link the function to the buttons
        self.pushButton_login.clicked.connect(self.login)
        self.pushButton_signup.clicked.connect(self.signup)
        self.pushButton_login_2.clicked.connect(self.loginVerification)
        self.pushButton_signup_2.clicked.connect(self.signupVerification)

    def login(self):
        """
        Shift the widget page to login page
        """
        self.resizeWidget()
        self.stackedWidget.setCurrentWidget(self.page_login)

    def signup(self):
        """
        Shift the widget page to sign up page
        """
        self.resizeWidget()
        self.stackedWidget.setCurrentWidget(self.page_signup)

    def resizeWidget(self):
        """
        Define the resize animation using in out cubic
        """
        #access the original width and height of the stacked widget component
        width = self.stackedWidget.frameGeometry().width()
        height = self.stackedWidget.frameGeometry().height()
        #resize the stacked widget component
        self.anim = QPropertyAnimation(self.stackedWidget, b"size")
        self.anim.setEasingCurve(QEasingCurve.InOutCubic)
        self.anim.setStartValue(QSize(0, 0))
        self.anim.setEndValue(QSize(width, height))
        self.anim.setDuration(1000)
        self.anim.start()

    def loginVerification(self):
        """
        Extract the username and the password entered by the user
        Pass the username and password to verifiy the identity
        """
        #extract the texts
        self.login_username = self.lineEdit_username_login.text()
        self.login_password = self.lineEdit_password_login.text()
        #pass for verification
        self.firebase_login_signup_verification(isLogin=True)

    def signupVerification(self):
        """
        Extract the sign up details of the user
        The details include first name, last name, age, gender, weight, height,
        username, and password
        Pass the details for verification
        """
        #extract the texts
        self.signup_firstname=self.lineEdit_firstName.text()
        self.signup_lastname=self.lineEdit_lastName.text()
        self.signup_age=self.lineEdit_age.text()
        self.signup_weight=self.lineEdit_weight.text()
        self.signup_height=self.lineEdit_height.text()
        self.signup_username=self.lineEdit_username.text()
        self.signup_password=self.lineEdit_password.text()
        #set the gender of the user according to the radio button selected
        if self.radioButton_female.isChecked():
            self.signup_gender = "Female"
        elif self.radioButton_male.isChecked():
            self.signup_gender = "Male"
        else:
            self.signup_gender = None
        #pass for verification
        self.firebase_login_signup_verification(isLogin=False)


    def firebase_login_signup_verification(self, isLogin=None):
        """
        Verify the login and the sign up details
        New window to access the functionatity of the system is shown if the details are valid

        Parameter
        ---------
        isLogin : Boolean
            To identify between login and sign up page
        """
        try:
        
            #access the firebase realtime database using URL link
            user_dataset = firebase.FirebaseApplication('https://heartsounddb-default-rtdb.firebaseio.com/', None)
            #if isLogin is true else false
            if isLogin == True:
                #set the error message if the username or password are empty
                if len(self.login_username) == 0 or len(self.login_password) == 0:
                    self.label_errorMessage.setText("Your username and password cannot be empty!!!")
                #if the username and password is not empty
                else:
                    #retrieve the data from User table
                    user_data = user_dataset.get('/heartsounddb-default-rtdb/User/','')
                    #access all the values from the User table
                    user_value = user_data.values()
                    #initialize the boolean to validate the login
                    login_valid = False
                    #iterate through all the value
                    for i in user_value:
                        #retrieve all the information of the user if Username and Password are matched
                        if (i['Username'] == self.login_username) and (i['Password'] == self.login_password):
                            if (i['Username'] != '') and (i['Password'] != ''):
                                self.information_collection.append(i['First Name'])
                                self.information_collection.append(i['Last Name'])
                                self.information_collection.append(i['Age'])
                                self.information_collection.append(i['Gender'])
                                self.information_collection.append(i['Weight'])
                                self.information_collection.append(i['Height'])
                                self.information_collection.append(i['Username'])
                                #login is valid
                                login_valid = True
                                break
                    #Start the new window if login is valid, otherwise set the error message
                    if login_valid == True:
                        self.acquisitionThread = AcquisitionGUI(self.information_collection)
                        self.acquisitionThread.run()
                        #close the login and sign up pages
                        self.close()

                    else:
                        self.label_errorMessage.setText("Your username or password does not match!\n Please enter the correct username and password!")
            #sign up validation
            else:
                #set the error message if first name is empty
                if len(self.signup_firstname) == 0:
                    self.label_signupErrorMessage.setText("Please provide your first name!")
                    return False
                #set the error message if last name is empty
                if len(self.signup_lastname) == 0:
                    self.label_signupErrorMessage.setText("Please provide your last name!")
                    return False
                #set the error message if age is empty or non numerical
                if len(self.signup_age) == 0:
                    self.label_signupErrorMessage.setText("Please provide your age!")
                    return False
                elif self.signup_age.isdigit() == False: 
                    self.label_signupErrorMessage.setText("Your age is invalid!")
                    return False
                #set the error message if weight is empty or non numerical
                if len(self.signup_weight) == 0:
                    self.label_signupErrorMessage.setText("Please provide your weigth!")
                    return False
                elif self.signup_weight.count('.') == 1 or self.signup_weight.isdigit():
                    if self.signup_weight.replace('.','').isdigit() == False:
                        self.label_signupErrorMessage.setText("Your weigth is invalid!")
                        return False
                else:
                    self.label_signupErrorMessage.setText("Your weigth is invalid!")
                    return False
                #set the error message if height is empty or non mumerical     
                if len(self.signup_height) == 0:
                    self.label_signupErrorMessage.setText("Please provide your height!")
                    return False
                elif self.signup_height.count('.') == 1 or self.signup_height.isdigit():
                    if self.signup_height.replace('.','').isdigit() == False:
                        self.label_signupErrorMessage.setText("Your heigth is invalid!")
                        return False
                else:
                    self.label_signupErrorMessage.setText("Your height is invalid!")
                    return False
                #set the error message if username is empty or is duplicated
                if len(self.signup_username) == 0:
                    self.label_signupErrorMessage.setText("Please provide your username!")
                    return False
                else:
                    user_data = user_dataset.get('/heartsounddb-default-rtdb/User/','')
                    user_value = user_data.values()
                    for i in user_value:
                        if (i['Username'] == self.signup_username):
                            self.label_signupErrorMessage.setText("This username has been registered before!\nPlease try out a new one!")
                            return False
                #set the error message if the password is empty
                if len(self.signup_password) == 0:
                    self.label_signupErrorMessage.setText("Please provide your password!")
                    return False
                #set the gender if gender is None
                if self.signup_gender == None:
                    self.label_signupErrorMessage.setText("Please select a gender!")
                    return False
                #assign the valid details                                                                                                                                                                                                                     
                single_signup = { 'First Name':self.signup_firstname,
                              'Last Name':self.signup_lastname,
                              'Age':self.signup_age,
                              'Gender':self.signup_gender,
                              'Weight':self.signup_weight,
                              'Height':self.signup_height,
                              'Username': self.signup_username,
                              'Password': self.signup_password,
                            }
                #save the details to database
                user_dataset.post('/heartsounddb-default-rtdb/User/',single_signup)
                #design the error message label
                self.label_signupErrorMessage.setStyleSheet("QLabel {font: 14pt 'Arial'; "+
                                             "color:#58D68D;}")
                #set the error message
                self.label_signupErrorMessage.setText("Sign Up successfully!")
                #save the details of the user
                self.information_collection.append(self.signup_firstname)
                self.information_collection.append(self.signup_lastname)
                self.information_collection.append(self.signup_age)
                self.information_collection.append(self.signup_gender)
                self.information_collection.append(self.signup_weight)
                self.information_collection.append(self.signup_height)
                self.information_collection.append(self.signup_username)
                #initialize the new window
                self.acquisitionThread = AcquisitionGUI(self.information_collection)
                self.acquisitionThread.run()
                
                #close the login and sign up window
                self.close()
        except:
            #Display the error message window
            self.error = ErrorActivity(0)
            self.error.show()
            print('Internet is disconnected')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = LoginSignUpActivity()
    ui.show()
    sys.exit(app.exec_())
