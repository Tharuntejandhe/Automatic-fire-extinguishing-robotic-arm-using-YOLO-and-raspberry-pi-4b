import FreeCAD as App
import FreeCADGui
import time

# Open the FreeCAD document
def getMovement():
    print("Function Called")
    doc = App.openDocument("C:/Users/ander/OneDrive/Desktop/rotating_disc1.FCStd")
    doc.recompute()

    # Select the disc object
    disc = doc.getObject("Body001")  # Replace "Body" with your disc's name in the model tree

    # Rotate the disc in steps
    for angle in range(0, 360, 10):  # Rotates in 10-degree steps
        disc.Placement.Rotation = App.Rotation(App.Vector(0, 0, 1), angle)
        doc.recompute()
        time.sleep(0.1)  # Pause for 0.1 seconds to visualize rotation

    # Save the rotated state if needed
    doc.save()