# Emotional expressive machine
時にはビデオ会議で顔を出したくない日はありませんか？  
なんとこれを使用すればあなたに代わってビデオ会議で感情を表現したり、顔に合ったリアクションを表示してくれます！！  

Do you sometimes have days when you don't want to show up for a video conference?  
What a great way to express emotions and display face-to-face reactions in a video conference for you!  

# Features
* Pythonとリアルタイムビデオを使用してあなたの顔の感情を読み取り、リアクションを表示します  
* ソフトウェアライブラリ「Dlib」を使用しており顔の傾きによって画像の傾きも変わります  
* OBS studioと合わせて使用すればビデオ会議でスムーズに機能の切り替えが可能になります   
 **機能は以下の三つです**
* Key[Z]：感情に合わせた画像を表示します
* Key[x]：カメラを使用した時に自分の顔に感情に合わせた画像を貼り付けます
* Key[c]：感情に合わせたリアクションを顔の周りに表示します
* Kry[q]：ソフトを終了します
  - Keyは反応するまで長押ししてください
---
* Using python and real time video to read emotions from the face and display effects on the video It is.  
* It uses the software library "Dlib" and the tilt of the image changes according to the tilt of the face.  
* When used in conjunction with OBS studio, it makes the transition to and from video conferencing a breeze.  
 **There are three features**
* Key[z]：Displays an emotionally relevant image.
* Key[x]：Paste an emotionally relevant image on your face when you use the camera
* Key[c]：Display emotional reactions around your face.
* Kry[q]：Exit the software.
  - Press and hold the Key until it reacts.


# Requirement
* Python3.5
* keras 2.0.5
* tensorflow 1.1.0
* opencv-python 3.2.0.6
* pandas 0.19.1
* PIL 7.1.2
* numpy 1.13.3
* scipy 1.1.0
* h5py 2.7.0
* statistics 1.0.3.5
* Dlib 19.20.0

Each library may work with versions other than those listed above.

# Installation
>## Running with Miniconda
>**Environmental Architecture**   
>Do the following in order.
~~~  bash
$ conda create -n ENVIROMENT-NAME python=3.5
$ conda activate ENVIROMENT-NAME
$ pip install -r REQUIREMENTS.txt
~~~


# Usage
>Basic Usage Introduction
>>**Download the necessary folders.**
~~~ bash
$ git clone https://github.com/remi0628/Express_emotions.git
~~~
>>**Run the file.**
~~~ bash
$ cd Express_emotions/src
$ python main.py
~~~
Let's see your face on the camera.

# Note
Please refer to the installation method from other sites when installing Dlib.


# Author
* Author
  - Shimane Chihiro
  - Okada Ituki
* Belonging
  - Tokyo University of Technology Department of Media
* E-mail
  - m011813471@edu.teu.ac.jp [Shimane]
  - m011906297@edu.teu.ac.jp [Okada]

# The underlying program
[Face classification and detection.](https://github.com/oarriaga/face_classification)
>Real-time face detection and emotion/gender classification using fer2013/IMDB datasets with a keras CNN model and openCV.  
* IMDB gender classification test accuracy: 96%.  
* fer2013 emotion classification test accuracy: 66%.

[OpenCVで画像上に別の画像を描画する](https://note.com/npaka/n/nddb33be1b782)
>OpenCVで、背景画像上に透過画像をオーバーレイさせるのは、意外と面倒なので備忘録的に残します。

# License
The image uses a free いらすとや.  
Please follow the license on which this program is based.  
Places to see. **The underlying program**
