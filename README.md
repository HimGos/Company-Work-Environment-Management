# Company Work Environment Management

## Project Description 🗒️

The Company Work Environment Management project is a system designed to monitor and optimize the workplace environment by tracking various employee activities. It utilizes facial recognition technology to enhance safety, security, and employee productivity. By analyzing facial expressions and behavior, it helps ensure a conducive work environment.

## Project Demo 🎥

![GIF](https://github.com/HimGos/Company-Work-Environment-Management/blob/master/demo.gif) 

## Installation 📲

To use this project, follow these steps:

1. Clone the repository to your local device:

```bash
git clone https://github.com/HimGos/Company-Work-Environment-Management.git
```
## Usage 🔌

Using this project is very easy and intuitive. Just like in Demo Preview: 
- On the first page, you will be prompted a video frame which access your webcam and ask you to capture image. The captured image then passes through a machine learning model which runs in background and in very quickly shows result on next page.
- On the second page, if it's a known person then it shows 'Start Shift' button otherwise it shows 'Go Back Home' to Unknown person. After hitting 'Start Shift' button, we move to the next page.
- On the third page, webcam runs again and tracks our head position, eyes and hand position to detect various activities (mentioned in [Features](#features) ). At the bottom of this page, user will see 'End Shift' button, clicking on it will take user to the next page.
- This last page shows stats to the user. Under the stats table there are 2 buttons 'Start Shift' & 'Send Data'. Clicking on 'Send Data' button sends the data to server and clicking on the other button takes user to the first page again for the next work shift.

## Features 📝

Managing the company's work environment is crucial for resource optimization and employee productivity. The project's key features include:

- Detecting if the right person is sitting in front of the screen.
- Tracking whether a person is looking at the screen or looking away.
- Detecting if a person is using a phone during work.
- Detecting if a user is falling asleep.
- Recording the duration of the above activities and sending the data in JSON format to a server.

## Contribute 🤓

To contribute or support the project, please contact me at [himgos@gmail.com](mailto:himgos@gmail.com).

## License 🔰

This project is licensed under the [GPL-2.0 License](LICENSE).

## Credits 

This project relies on the following libraries and tools:

- [OpenCV](https://opencv.org/) 
- [Mediapipe](https://mediapipe.dev/)
- [Flask](https://flask.palletsprojects.com/en/3.0.x/#)

## Contact Information 👏

For any inquiries or questions, you can reach me at 📧 [himgos@gmail.com](mailto:himgos@gmail.com). 

---

Leave a star ⭐ in GitHub, share the repo 📲 and share this guide if you found this helpful.

