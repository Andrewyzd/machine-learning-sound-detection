from ErrorUI import ErrorActivity, ErrorUI
from threading import *
import time
import serial

class SerialProcess(Thread):
    """
    A class for real-time ECG signal transmission 

    ...

    Attributes
    ----------
    queue_time : Qqueue
        The queue to store the time
    queue_signal : Qqueue
        The queue to store the real-time ECG sensor
    check_connection : list
        The list to check for the connection error

    Methods
    -------
    run()
        Start the thread for real-time transmission
    mapSignal(value)
        Map the signal from sensor with audio file signal
    closePort()
        Terminate the real-time sensor port
    """
    def __init__(self, queue_time, queue_signal, check_connection):
        """
        Parameters
        ----------
        queue_time : Qqueue
            The queue to store the time
        queue_signal : Qqueue
            The queue to store the real-time ECG sensor
        check_connection : list
            The list to check for the connection error
        """
        super(SerialProcess, self).__init__()
        self.exit = Event()
        self.queue_time = queue_time
        self.queue_signal = queue_signal
        self.check_connection = check_connection
          
    def run(self, e):
        """
        Start the COM4 to retrieve the ECG signal of heart sound through ECG sensor

        Parameter
        --------
        e: threading.Event()
            The object of the threading event
            
        Raises
        ------
        self.error = ErrorActivity()
            The interface to illustrate the error between sensor connectivity

        Finally
        -------
        self.closePort()
            close the connected port
        """
        self.init_time = time.time()
        try:
            self.arduino = serial.Serial(port="COM4", baudrate=9600)
            current_time = time.time() - self.init_time
            #Start receiving the signal from sensor
            while self.arduino.isOpen() and not e.isSet():
                #Read the signal
                value = str(self.arduino.readline())
                #Clean the signal
                value = value.translate({ord(i): None for i in "br\\'n"})
                #Convert the signal to float
                value = float(value)
                #Read the current time
                current_time = time.time() - self.init_time
                #Map the signal within [-1,1]
                value = self.mapSignal(value)
                #Display the signal
                print(value)
                #Save the signal and time concurrently
                self.queue_time.put([time.time() - self.init_time])
                self.queue_signal.put(value)
        except ValueError:
            print('Sensor value error, ', value)
        except:
            #Pass False to list for error connection indicator
            self.check_connection.insert(0, False)
            print('Sensor is not connected')
        finally:
            self.closePort()
            
    def mapSignal(self, value):
        """
        Map the signal value between 1 and -1

        Parameters
        ----------
        value: float
            The signal from the sensor

        Returns
        -------
        mapped
            The mapped real-time signal value which is between -1 and 1
        
        """
        # Return the mapped value
        return((value / 800.0) * 2) - 1
        
    def closePort(self):
        """
        Terminate the port of the sensor
        """
        self.exit.set()
