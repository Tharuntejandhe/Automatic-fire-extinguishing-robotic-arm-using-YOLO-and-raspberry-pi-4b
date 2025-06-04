# Automatic-fire-extinguishing-robotic-arm-using-YOLO-and-raspberry-pi-4b
Nirva-Agni is an AI-powered autonomous fire detection and extinguishing system using YOLOv8, custom CNN, and a 2-axis robotic arm. It detects, tracks, and suppresses fire using water and CO₂, while sending real-time alerts—built for smart safety, funded by iCreate, and awarded by IIT Hyderabad.

# 🔥 Nirva-Agni: AI-Powered Autonomous Fire Detection & Extinguishing System 🚒

[![Watch the Demo](https://img.youtube.com/vi/YOUR_VIDEO_ID_HERE/0.jpg)](https://youtu.be/ppQ95qvBmH8)
> 🎥 Click above to watch the full system demo on YouTube

---

## 🌟 Overview

**Nirva-Agni** is an AI-driven, autonomous fire detection and response system designed to combat indoor fire hazards with high accuracy, speed, and intelligence. Built at the intersection of **deep learning**, **robotics**, and **edge vision**, the system identifies, tracks, and extinguishes fire outbreaks using real-time detection and actuation mechanisms. It’s built to **save lives, protect infrastructure**, and operate without human intervention.

🛡️ **Built for:** Smart Buildings | Data Centers | Warehouses | Industrial Plants | Defense Infrastructure

---

## 🚀 Key Features

- 🎯 **AI-based Fire Detection**: Custom CNN + YOLOv8 model trained on 10,000+ images, achieving 80% accuracy
- 🔭 **Real-time Flame Tracking**: Dual-axis robotic arm follows and targets fire origin dynamically
- 💧 **Smart Fire Suppression**: Sprays water in early stages, releases CO₂ if fire persists beyond 3 minutes
- 📸 **Visual Alerts**: Captures fire frame and sends image preview to authorities via GSM
- 🧠 **Edge Processing**: Runs locally on Raspberry Pi or NVIDIA Jetson (offline compatible)
- 🏆 **Award-Winning**: 3rd place at **IIT Hyderabad AI X Ideathon & Hackathon** (221 teams)
- 💰 **iCreate Funded Project**

---

## 🧠 AI Model Details

| Model        | Purpose           | Accuracy | Dataset           |
|--------------|-------------------|----------|--------------------|
| Custom CNN   | Initial Fire Classification | 80%     | 10,000+ Fire/Smoke Images |
| YOLOv8       | Real-Time Object Detection & Tracking | High     | Filtered COCO + Fire Classes |

---

## ⚙️ Hardware Components

- 🔥 Flame Detection Camera Module (Raspberry Pi Camera / USB Cam)
- ⚙️ 2-Axis Robotic Arm (Ceiling Mounted)
- 💧 Water Pump + Sprayer Nozzle
- 🧯 CO₂ Cylinder with Electromagnetic Trigger
- 📲 GSM Module for SMS Alerts
- 💡 LED Indicators, Buzzer, Servo Motors

---

## 🛠️ Tech Stack

- `Python`, `OpenCV`, `PyTorch`, `YOLOv8`
- `Raspberry Pi` / `Jetson Nano`
- `Arduino` (servo & actuator control)
- `Custom CNN architecture` for fire classification
- `ONNX` model conversion for edge deployment
- GSM + Serial Communication for alerting

---

## 🔁 System Workflow

1. **Frame Capture** from camera
2. **Fire Detection** using YOLOv8 + CNN
3. **Tracking & Localization** of flame
4. **Arm Movement** toward fire source
5. **Water spray** for ≤3 min fire; **CO₂ release** if fire persists
6. **Capture Image + GSM Alert** with location/time
7. **Return to Origin** once fire is extinguished

---

## 📽️ Live Demo

🔗 [Watch the Full Video on YouTube »](https://www.youtube.com/watch?v=YOUR_VIDEO_ID_HERE)

> 🔔 Don't forget to like, share, and subscribe for more AI + Robotics innovations!

---

## 🏆 Achievements

- 🥉 **3rd Place** – IIT Hyderabad AI X Hackathon (221+ teams)
- 🎯 **Finalist** – BVRIT PROMETHAN Ideathon (110+ teams)
- 🥉 **3rd Prize** – VISAI International Project Expo 2024
- 💸 **Funded by iCreate** for innovation in safety & robotics

---

## 📂 Folder Structure

```
nirva-agni/
├── models/               # YOLOv8 + Custom CNN models
├── fire_detection/       # Python code for real-time fire detection
├── robotic_arm_control/  # Arduino + Serial control code
├── hardware_docs/        # Circuit diagrams, specs
├── dataset/              # Sample images (10k+ fire/no-fire images)
├── gsm_alert/            # SMS alert scripts
└── README.md             # This file
```

---

## 🤖 Future Scope

- 🔌 Integration with building automation systems (IoT)
- 🌐 Cloud Dashboard for alert logs and analytics
- 🧭 Autonomous mobile base for outdoor fire response
- 📡 Satellite image-based wildfire prediction

---

## 👨‍💻 Contributors

- **Tharun Tej** – AI & Robotics Lead  
- **Team Nirva-Agni** – [Mukesh, Pranay, Anirudh, Sathvik, Mohith, Varun, Sandeep]

---

## 📬 Contact

📧 Email: [tharuntejandhe@gmail.com]  
🌐 Portfolio: [https://nirvaagni-oeg1.onrender.com/]  
🤝 Collaboration & Partnerships Welcome!

---

## 📄 License

MIT License – use it, modify it, improve it – **just don't let fires burn!**

> 🔥 *“Technology should be the first to respond when disaster strikes.” – Nirva-Agni Team*
