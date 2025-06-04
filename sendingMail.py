import smtplib



from email.mime.multipart import MIMEMultipart



from email.mime.text import MIMEText



from email.mime.image import MIMEImage





def sendingMail():



# Email setup



  sender = '22h51a6677@cmrcet.ac.in'



  sender_pass = 'idff twot bloy lftb' # Replace with the actual App Password



  receivers = ['22h51a6689@cmrcet.ac.in', '22h51a6668@cmrcet.ac.in', '22h51a6694@cmrcet.ac.in','22h51a66c5@cmrcet.ac.in','22h51a66a6@cmrcet.ac.in','22h51a66a1@cmrcet.ac.in','22h51a6669@cmrcet.ac.in','22h51a6683@cmrcet.ac.in']







  # Create the email message



  message = MIMEMultipart()



  message['From'] = sender



  message['To'] = ', '.join(receivers) # Join multiple recipients as a single string



  message['Subject'] = 'Test Mail!! And Email Osthe GROUP lo Confirm Cheyyandi!! ~ From Mukesh'







  # Attach the email body



  body = "testing mail"



  message.attach(MIMEText(body, 'plain'))







  # Attach the image

  image_path = r"C:\Users\Pranay\OneDrive\Desktop\test3\yolov5-fire-detection\yolov5\image.jpg"

#    = 'yolov5-fire-detection\yolov5\image.jpg' # Replace with the actual path to the image



  with open(image_path, 'rb') as img_file:



    img = MIMEImage(img_file.read())

    img.add_header('Content-Disposition', 'attachment', filename='image.jpg') # Change filename as needed

    message.attach(img)

  # Send the email



  with smtplib.SMTP_SSL('smtp.gmail.com', 465) as connection:

    connection.login(user=sender, password=sender_pass)

    connection.sendmail(sender, receivers, message.as_string())

  print("Email with image attachment sent successfully to multiple recipients!")